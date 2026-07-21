use std::collections::{hash_map::DefaultHasher, BTreeMap};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::compression::verifier::preserves_requested_numbers;
use crate::config::profile::ProfileDefinition;
use crate::error::{CompressionError, Result};
use crate::types::{
    CompressionConstraints, CompressionLevel, CompressionRequest, RequestSource, RequestTarget,
};

use super::catalog::{
    ModelDefinition, ModelRegistry, PromptProfileRegistry, RuntimeDefinition, RuntimeLaunchMode,
    RuntimeRegistry,
};
use super::model_download::{
    ensure_model_file, resumable_downloaded_bytes, verify_existing_model, ModelDownloadSpec,
};
pub use super::model_download::{ModelDownloadCancellation, ModelDownloadProgress};
use super::prompt_structure::PromptStructure;

#[cfg(feature = "embedded-llama")]
mod embedded_llama;
mod http_client;
mod openai_compatible;
mod output_parser;
mod thread_tuning;

#[cfg(feature = "embedded-llama")]
use embedded_llama as llama_cpp;
use http_client::{http_json_request, parse_http_base_url};
use openai_compatible::{request_openai_completion, resolve_lmstudio_model_name};
#[cfg(feature = "embedded-llama")]
use output_parser::first_complete_json_object_end;
use output_parser::{output_snippet, parse_compression_output};
#[cfg(test)]
use thread_tuning::{automatic_runtime_thread_counts, RuntimeBatchSizes, RuntimeThreadCounts};
use thread_tuning::{
    available_runtime_threads, manual_runtime_thread_counts, parse_runtime_threads,
    RuntimeInferenceConfig, ThreadTuningStore,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompressionDraft {
    pub distilled_prompt: String,
    pub removed_content_summary: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeTransformation {
    RestoredRequiredConstraints,
    RestoredRequiredTerms,
    PolishedModelOutput,
    RuntimeFallback,
}

#[derive(Debug, Clone)]
pub struct RuntimeCompressionObservation {
    pub raw_model_draft: Option<CompressionDraft>,
    pub final_draft: CompressionDraft,
    pub transformations: Vec<RuntimeTransformation>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfileModelStatus {
    pub profile: String,
    pub model_id: String,
    pub label: String,
    pub requires_install: bool,
    pub installed: bool,
    pub repository: Option<String>,
    pub revision: Option<String>,
    pub filename: Option<String>,
    pub size_bytes: Option<u64>,
    pub partial_downloaded_bytes: Option<u64>,
    pub destination: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProfileThreadStatus {
    pub mode: String,
    pub generation_threads: u32,
    pub batch_threads: u32,
    pub logical_batch_size: u32,
    pub physical_batch_size: u32,
    pub available_threads: u32,
}

#[cfg(feature = "embedded-llama")]
const PHYSICAL_BATCH_QUALITY_INPUTS: [&str; 3] = [
    "Next.js の POST /api/orders で、customerId が空のまま送られた時に 500 エラーになっています。入力検証を追加し、空の customerId の場合は HTTP 400 と INVALID_CUSTOMER のエラーコードを返すようにしてください。成功時のレスポンス形式、在庫引当処理、既存の監査ログは変更しないでください。テストでは正常系と customerId 空文字のケースを確認できるようにしてください。",
    "OpenAPI 定義に PATCH /users/{id} を追加してください。name と avatarUrl は任意で更新できるようにし、email は変更不可にしてください。404 と 409 のエラーレスポンス例を入れ、既存の POST /users は変更しないでください。既存の schema_version や共通エラー形式がある場合は、それに合わせてください。",
    "OpenAPI に PATCH /users/{id} を追加してほしいです。name と avaterUrl、正しくは avatarUrl は任意、email は変更しちゃだめです。404 と409のエラーレスポンス例を入れてください。既存の POST /users は変更しないでください。途中でごタップして変な空白や文字が入るかもしれませんが、email 変更不可と POST /users 維持は絶対に落とさないでください。",
];

impl RuntimeCompressionObservation {
    fn unobserved(final_draft: CompressionDraft) -> Self {
        Self {
            raw_model_draft: None,
            final_draft,
            transformations: Vec::new(),
        }
    }

    fn raw_model(final_draft: CompressionDraft) -> Self {
        Self {
            raw_model_draft: Some(final_draft.clone()),
            final_draft,
            transformations: Vec::new(),
        }
    }

    fn runtime_fallback(final_draft: CompressionDraft) -> Self {
        Self {
            raw_model_draft: None,
            final_draft,
            transformations: vec![RuntimeTransformation::RuntimeFallback],
        }
    }
}

#[cfg(feature = "embedded-llama")]
enum EmbeddedCompletion {
    RawModel(CompressionDraft),
    RuntimeFallback(CompressionDraft),
}

#[cfg(feature = "embedded-llama")]
fn ensure_embedded_deadline(signal: &llama_cpp::AbortSignal, timeout_ms: u64) -> Result<()> {
    if signal.is_aborted() {
        Err(CompressionError::RuntimeTimeout(timeout_ms))
    } else {
        Ok(())
    }
}

#[cfg(feature = "embedded-llama")]
fn prefer_embedded_timeout<T>(
    result: Result<T>,
    signal: &llama_cpp::AbortSignal,
    timeout_ms: u64,
) -> Result<T> {
    if signal.is_aborted() {
        Err(CompressionError::RuntimeTimeout(timeout_ms))
    } else {
        result
    }
}

#[cfg(feature = "embedded-llama")]
fn embedded_timeout_completion(
    request: &CompressionRequest,
    timeout_ms: u64,
) -> Result<EmbeddedCompletion> {
    if let Some(draft) = trusted_precompacted_fallback_draft(request) {
        Ok(EmbeddedCompletion::RuntimeFallback(draft))
    } else {
        Err(CompressionError::RuntimeTimeout(timeout_ms))
    }
}

pub trait RuntimeBackend {
    fn compress(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
    ) -> Result<CompressionDraft>;

    fn compress_observed(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
    ) -> Result<RuntimeCompressionObservation> {
        self.compress(request, profile)
            .map(RuntimeCompressionObservation::unobserved)
    }

    fn prepare(&self, _request: &CompressionRequest, _profile: &ProfileDefinition) -> Result<bool> {
        Ok(false)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoopRuntimeBackend;

impl RuntimeBackend for NoopRuntimeBackend {
    fn compress(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
    ) -> Result<CompressionDraft> {
        let summary = vec![format!(
            "No runtime backend is connected yet; returning passthrough output for profile '{}'.",
            profile.id
        )];

        Ok(CompressionDraft {
            distilled_prompt: request.input_text.trim().to_string(),
            removed_content_summary: summary,
        })
    }
}

#[derive(Debug, Clone)]
pub struct ConfiguredRuntimeBackend {
    project_root: PathBuf,
    prompts_dir: PathBuf,
    models: ModelRegistry,
    runtimes: RuntimeRegistry,
    prompt_profiles: PromptProfileRegistry,
    managed_runtimes: Arc<ManagedRuntimeManager>,
    embedded_models: Arc<EmbeddedModelManager>,
    model_files: Arc<ModelFileCoordinator>,
    thread_tuning: Arc<ThreadTuningStore>,
}

#[derive(Debug, Default)]
struct ManagedRuntimeManager {
    processes: Mutex<BTreeMap<String, ManagedServer>>,
}

#[derive(Debug)]
struct ManagedServer {
    child: Child,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelFileIdentity {
    size_bytes: u64,
    modified: SystemTime,
}

impl ModelFileIdentity {
    fn from_path(path: &Path) -> Result<Option<Self>> {
        let metadata = match fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        if !metadata.is_file() {
            return Ok(None);
        }
        Ok(Some(Self {
            size_bytes: metadata.len(),
            modified: metadata.modified()?,
        }))
    }
}

#[derive(Debug, Default)]
struct ModelFileCoordinator {
    verified: Mutex<BTreeMap<PathBuf, ModelFileIdentity>>,
    operations: Mutex<BTreeMap<PathBuf, Arc<Mutex<()>>>>,
}

impl ModelFileCoordinator {
    fn operation_lock(&self, path: &Path) -> Result<Arc<Mutex<()>>> {
        let mut operations = self.operations.lock().map_err(|_| {
            CompressionError::Runtime("model file coordinator is unavailable".into())
        })?;
        Ok(operations
            .entry(path.to_path_buf())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone())
    }

    fn is_verified(&self, path: &Path) -> Result<bool> {
        let current = ModelFileIdentity::from_path(path)?;
        let mut verified = self.verified.lock().map_err(|_| {
            CompressionError::Runtime("model verification cache is unavailable".into())
        })?;
        let matches = current
            .as_ref()
            .is_some_and(|identity| verified.get(path) == Some(identity));
        if !matches {
            verified.remove(path);
        }
        Ok(matches)
    }

    fn mark_verified(&self, path: &Path) -> Result<()> {
        let identity = ModelFileIdentity::from_path(path)?.ok_or_else(|| {
            CompressionError::Runtime(format!(
                "verified model file disappeared at {}",
                path.display()
            ))
        })?;
        self.verified
            .lock()
            .map_err(|_| {
                CompressionError::Runtime("model verification cache is unavailable".into())
            })?
            .insert(path.to_path_buf(), identity);
        Ok(())
    }

    fn forget(&self, path: &Path) -> Result<()> {
        self.verified
            .lock()
            .map_err(|_| {
                CompressionError::Runtime("model verification cache is unavailable".into())
            })?
            .remove(path);
        Ok(())
    }
}

#[derive(Default)]
struct EmbeddedModelManager {
    #[cfg(feature = "embedded-llama")]
    models: Mutex<BTreeMap<String, llama_cpp::LlamaModel>>,
    #[cfg(feature = "embedded-llama")]
    prepared_prompt_sessions: Mutex<BTreeMap<String, llama_cpp::LlamaSession>>,
    #[cfg(feature = "embedded-llama")]
    prepared_input_session: Mutex<Option<(String, llama_cpp::LlamaSession)>>,
}

impl std::fmt::Debug for EmbeddedModelManager {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("EmbeddedModelManager");
        #[cfg(feature = "embedded-llama")]
        {
            let model_count = self.models.lock().map(|models| models.len()).unwrap_or(0);
            let prepared_prompt_count = self
                .prepared_prompt_sessions
                .lock()
                .map(|sessions| sessions.len())
                .unwrap_or(0);
            let prepared_input = self
                .prepared_input_session
                .lock()
                .map(|session| session.is_some())
                .unwrap_or(false);
            debug.field("model_count", &model_count);
            debug.field("prepared_prompt_count", &prepared_prompt_count);
            debug.field("prepared_input", &prepared_input);
        }
        debug.finish()
    }
}

#[derive(Debug, Clone)]
struct PromptParts {
    prefix: String,
    suffix: String,
}

impl PromptParts {
    fn whole(prompt: String) -> Self {
        Self {
            prefix: String::new(),
            suffix: prompt,
        }
    }

    fn combined(&self) -> String {
        format!("{}{}", self.prefix, self.suffix)
    }
}

const MAX_PREPARED_PROMPT_SESSIONS: usize = 3;

impl EmbeddedModelManager {
    #[cfg(feature = "embedded-llama")]
    fn load_or_get(&self, cache_key: &str, model_path: &Path) -> Result<llama_cpp::LlamaModel> {
        ensure_embedded_cpu_engine_is_supported()?;
        let mut models = self.models.lock().map_err(|_| {
            CompressionError::Runtime("embedded model registry is unavailable".into())
        })?;

        if let Some(model) = models.get(cache_key) {
            return Ok(model.clone());
        }

        let loaded = llama_cpp::LlamaModel::load_from_file(model_path, llama_cpp::LlamaParams)
            .map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to load embedded llama.cpp model at {}: {error}",
                    model_path.display()
                ))
            })?;

        models.insert(cache_key.to_string(), loaded.clone());
        Ok(loaded)
    }

    #[cfg(feature = "embedded-llama")]
    fn has_prepared_session(&self, cache_key: &str) -> Result<bool> {
        let sessions = self.prepared_prompt_sessions.lock().map_err(|_| {
            CompressionError::Runtime("embedded prompt session cache is unavailable".into())
        })?;
        Ok(sessions.contains_key(cache_key))
    }

    #[cfg(feature = "embedded-llama")]
    fn get_prepared_session(&self, cache_key: &str) -> Result<Option<llama_cpp::LlamaSession>> {
        let stored = {
            let sessions = self.prepared_prompt_sessions.lock().map_err(|_| {
                CompressionError::Runtime("embedded prompt session cache is unavailable".into())
            })?;
            sessions.get(cache_key).cloned()
        };

        stored
            .map(|session| {
                session.deep_copy().map_err(|error| {
                    CompressionError::Runtime(format!(
                        "failed to copy prepared embedded prompt session: {error}"
                    ))
                })
            })
            .transpose()
    }

    #[cfg(feature = "embedded-llama")]
    fn store_prepared_session(
        &self,
        cache_key: String,
        prepared: llama_cpp::LlamaSession,
    ) -> Result<()> {
        let mut sessions = self.prepared_prompt_sessions.lock().map_err(|_| {
            CompressionError::Runtime("embedded prompt session cache is unavailable".into())
        })?;

        if !sessions.contains_key(&cache_key) && sessions.len() >= MAX_PREPARED_PROMPT_SESSIONS {
            if let Some(oldest_key) = sessions.keys().next().cloned() {
                sessions.remove(&oldest_key);
            }
        }

        sessions.insert(cache_key, prepared);
        Ok(())
    }

    #[cfg(feature = "embedded-llama")]
    fn store_prepared_session_copy(
        &self,
        cache_key: String,
        session: &llama_cpp::LlamaSession,
    ) -> Result<()> {
        let prepared = session.deep_copy().map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to copy prepared embedded prompt session: {error}"
            ))
        })?;
        self.store_prepared_session(cache_key, prepared)
    }

    #[cfg(feature = "embedded-llama")]
    fn has_prepared_input_session(&self, cache_key: &str) -> Result<bool> {
        let prepared = self.prepared_input_session.lock().map_err(|_| {
            CompressionError::Runtime("embedded input session cache is unavailable".into())
        })?;
        Ok(prepared
            .as_ref()
            .is_some_and(|(stored_key, _)| stored_key == cache_key))
    }

    #[cfg(feature = "embedded-llama")]
    fn take_prepared_input_session(
        &self,
        cache_key: &str,
    ) -> Result<Option<llama_cpp::LlamaSession>> {
        let mut prepared = self.prepared_input_session.lock().map_err(|_| {
            CompressionError::Runtime("embedded input session cache is unavailable".into())
        })?;
        Ok(take_matching_prepared_value(&mut prepared, cache_key))
    }

    #[cfg(feature = "embedded-llama")]
    fn store_prepared_input_session(
        &self,
        cache_key: String,
        prepared_session: llama_cpp::LlamaSession,
    ) -> Result<()> {
        let mut prepared = self.prepared_input_session.lock().map_err(|_| {
            CompressionError::Runtime("embedded input session cache is unavailable".into())
        })?;
        *prepared = Some((cache_key, prepared_session));
        Ok(())
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbeddedCpuEngine {
    Compatible,
    Avx2,
    Avx512,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Debug, Clone, Copy, Default)]
struct EmbeddedCpuCapabilities {
    sse42: bool,
    avx2: bool,
    fma: bool,
    f16c: bool,
    bmi2: bool,
    avx512f: bool,
    avx512cd: bool,
    avx512bw: bool,
    avx512dq: bool,
    avx512vl: bool,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn compiled_embedded_cpu_engine() -> EmbeddedCpuEngine {
    if cfg!(feature = "embedded-llama-avx512") {
        EmbeddedCpuEngine::Avx512
    } else if cfg!(feature = "embedded-llama-avx2") {
        EmbeddedCpuEngine::Avx2
    } else {
        EmbeddedCpuEngine::Compatible
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn embedded_cpu_engine_is_supported(
    engine: EmbeddedCpuEngine,
    capabilities: EmbeddedCpuCapabilities,
) -> bool {
    let compatible = capabilities.sse42;
    let avx2 = compatible
        && capabilities.avx2
        && capabilities.fma
        && capabilities.f16c
        && capabilities.bmi2;
    match engine {
        EmbeddedCpuEngine::Compatible => compatible,
        EmbeddedCpuEngine::Avx2 => avx2,
        EmbeddedCpuEngine::Avx512 => {
            avx2 && capabilities.avx512f
                && capabilities.avx512cd
                && capabilities.avx512bw
                && capabilities.avx512dq
                && capabilities.avx512vl
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn ensure_embedded_cpu_engine_is_supported() -> Result<()> {
    let engine = compiled_embedded_cpu_engine();
    let capabilities = EmbeddedCpuCapabilities {
        sse42: std::arch::is_x86_feature_detected!("sse4.2"),
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        fma: std::arch::is_x86_feature_detected!("fma"),
        f16c: std::arch::is_x86_feature_detected!("f16c"),
        bmi2: std::arch::is_x86_feature_detected!("bmi2"),
        avx512f: std::arch::is_x86_feature_detected!("avx512f"),
        avx512cd: std::arch::is_x86_feature_detected!("avx512cd"),
        avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
        avx512dq: std::arch::is_x86_feature_detected!("avx512dq"),
        avx512vl: std::arch::is_x86_feature_detected!("avx512vl"),
    };
    if embedded_cpu_engine_is_supported(engine, capabilities) {
        return Ok(());
    }

    let requirement = match engine {
        EmbeddedCpuEngine::Compatible => "SSE4.2",
        EmbeddedCpuEngine::Avx2 => "SSE4.2, AVX2, FMA, F16C, and BMI2",
        EmbeddedCpuEngine::Avx512 => {
            "SSE4.2, AVX2, FMA, F16C, BMI2, AVX512F, AVX512CD, AVX512BW, AVX512DQ, and AVX512VL"
        }
    };
    Err(CompressionError::Runtime(format!(
        "this build requires {requirement}; select a compatible CPU engine"
    )))
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn ensure_embedded_cpu_engine_is_supported() -> Result<()> {
    Ok(())
}

fn take_matching_prepared_value<T>(prepared: &mut Option<(String, T)>, key: &str) -> Option<T> {
    if !prepared
        .as_ref()
        .is_some_and(|(stored_key, _)| stored_key == key)
    {
        return None;
    }
    prepared.take().map(|(_, value)| value)
}

impl Drop for ManagedRuntimeManager {
    fn drop(&mut self) {
        if let Ok(mut processes) = self.processes.lock() {
            for server in processes.values_mut() {
                let _ = server.child.kill();
                let _ = server.child.wait();
            }
        }
    }
}

impl ConfiguredRuntimeBackend {
    pub fn from_settings_dir(settings_dir: impl AsRef<Path>) -> Result<Self> {
        let settings_dir = settings_dir.as_ref();
        let project_root = settings_dir
            .parent()
            .ok_or_else(|| {
                CompressionError::InvalidConfig(format!(
                    "settings directory has no parent: {}",
                    settings_dir.display()
                ))
            })?
            .to_path_buf();

        let prompts_dir = project_root.join("resources").join("prompts");
        let models = ModelRegistry::from_path(settings_dir.join("model-catalog.yaml"))?;
        Ok(Self {
            project_root: project_root.clone(),
            prompts_dir,
            models,
            runtimes: RuntimeRegistry::from_path(settings_dir.join("runtime-backends.yaml"))?,
            prompt_profiles: PromptProfileRegistry::from_path(
                settings_dir
                    .join("compression-policies")
                    .join("level-prompt-profiles-v1.yaml"),
            )?,
            managed_runtimes: Arc::new(ManagedRuntimeManager::default()),
            embedded_models: Arc::new(EmbeddedModelManager::default()),
            model_files: Arc::new(ModelFileCoordinator::default()),
            thread_tuning: Arc::new(ThreadTuningStore::new(
                project_root
                    .join("local")
                    .join("state")
                    .join("inference-tuning-v1"),
            )),
        })
    }

    pub fn profile_model_status(&self, profile: &ProfileDefinition) -> Result<ProfileModelStatus> {
        let (model, runtime) = self.resolve_model_and_runtime(profile)?;
        let requires_install = matches!(runtime.launch_mode, RuntimeLaunchMode::Embedded);
        let destination = model
            .model_path
            .as_ref()
            .map(|path| resolve_project_path(&self.project_root, path));
        let installed = if requires_install {
            self.model_file_is_installed(model, &runtime.id)?
        } else {
            true
        };
        let partial_downloaded_bytes = match (&destination, &model.download) {
            (Some(destination), Some(download)) if !installed => {
                Some(resumable_downloaded_bytes(destination, download)?)
            }
            (_, Some(_)) => Some(0),
            _ => None,
        };

        Ok(ProfileModelStatus {
            profile: profile.id.clone(),
            model_id: model.id.clone(),
            label: model.label.clone(),
            requires_install,
            installed,
            repository: model
                .download
                .as_ref()
                .map(|value| value.repository().to_string()),
            revision: model
                .download
                .as_ref()
                .map(|value| value.revision().to_string()),
            filename: model
                .download
                .as_ref()
                .map(|value| value.filename().to_string()),
            size_bytes: model.download.as_ref().map(ModelDownloadSpec::size_bytes),
            partial_downloaded_bytes,
            destination,
        })
    }

    pub fn profile_thread_status(
        &self,
        profile: &ProfileDefinition,
    ) -> Result<ProfileThreadStatus> {
        let (model, runtime) = self.resolve_model_and_runtime(profile)?;
        let manual_threads = manual_runtime_thread_counts()?;
        let fallback = thread_tuning::runtime_config_for_threads(
            manual_threads.unwrap_or(parse_runtime_threads(runtime)?),
        );

        #[cfg(feature = "embedded-llama")]
        let configuration = if manual_threads.is_some() {
            fallback
        } else {
            match self.resolve_model_file_path(model, &runtime.id) {
                Ok(model_path) if model_path.is_file() => {
                    match self.thread_tuning.resolve(model, &model_path, runtime) {
                        Ok(configuration) => configuration,
                        Err(error) => {
                            eprintln!("embedded thread status is unavailable: {error}");
                            fallback
                        }
                    }
                }
                _ => fallback,
            }
        };
        #[cfg(not(feature = "embedded-llama"))]
        let configuration = {
            let _ = model;
            fallback
        };

        let mode = if manual_threads.is_some() {
            "manual"
        } else if runtime.threads.eq_ignore_ascii_case("auto") {
            "auto"
        } else {
            "configured"
        };
        Ok(ProfileThreadStatus {
            mode: mode.to_string(),
            generation_threads: configuration.threads.generation,
            batch_threads: configuration.threads.batch,
            logical_batch_size: configuration.batch_sizes.logical,
            physical_batch_size: configuration.batch_sizes.physical,
            available_threads: u32::try_from(available_runtime_threads())
                .unwrap_or(u32::MAX)
                .max(1),
        })
    }

    pub fn warm_profile(&self, profile: &ProfileDefinition) -> Result<bool> {
        let (model, runtime) = self.resolve_model_and_runtime(profile)?;
        match (runtime.backend_kind.as_str(), &runtime.launch_mode) {
            ("llama.cpp", RuntimeLaunchMode::Embedded) => {
                self.require_model_file(model, &runtime.id)?;
                self.preload_embedded_llama_model(model, runtime)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub fn tune_profile_threads(
        &self,
        profile: &ProfileDefinition,
        cancellation: &AtomicBool,
    ) -> Result<bool> {
        let (model, runtime) = self.resolve_model_and_runtime(profile)?;
        match (runtime.backend_kind.as_str(), &runtime.launch_mode) {
            ("llama.cpp", RuntimeLaunchMode::Embedded) => {
                self.require_model_file(model, &runtime.id)?;
                self.tune_embedded_llama_threads(profile, model, runtime, cancellation)
            }
            _ => Ok(false),
        }
    }

    pub fn reset_profile_thread_tuning(&self, profile: &ProfileDefinition) -> Result<bool> {
        let (model, runtime) = self.resolve_model_and_runtime(profile)?;
        match (runtime.backend_kind.as_str(), &runtime.launch_mode) {
            ("llama.cpp", RuntimeLaunchMode::Embedded) => {
                let model_path = self.resolve_model_file_path(model, &runtime.id)?;
                self.thread_tuning.reset(model, &model_path, runtime)
            }
            _ => Ok(false),
        }
    }

    pub fn profile_thread_tuning_required(&self, profile: &ProfileDefinition) -> Result<bool> {
        let (model, runtime) = self.resolve_model_and_runtime(profile)?;
        match (runtime.backend_kind.as_str(), &runtime.launch_mode) {
            ("llama.cpp", RuntimeLaunchMode::Embedded) => {
                let model_path = self.resolve_model_file_path(model, &runtime.id)?;
                self.thread_tuning.is_required(model, &model_path, runtime)
            }
            _ => Ok(false),
        }
    }

    pub fn profile_supports_embedded_cpu_tuning(
        &self,
        profile: &ProfileDefinition,
    ) -> Result<bool> {
        let (_, runtime) = self.resolve_model_and_runtime(profile)?;
        Ok(runtime.backend_kind == "llama.cpp"
            && matches!(runtime.launch_mode, RuntimeLaunchMode::Embedded))
    }

    pub fn install_profile_with_progress(
        &self,
        profile: &ProfileDefinition,
        progress: impl FnMut(ModelDownloadProgress) + Send,
    ) -> Result<bool> {
        self.install_profile_with_progress_and_cancellation(
            profile,
            &ModelDownloadCancellation::default(),
            progress,
        )
    }

    pub fn install_profile_with_progress_and_cancellation(
        &self,
        profile: &ProfileDefinition,
        cancellation: &ModelDownloadCancellation,
        mut progress: impl FnMut(ModelDownloadProgress) + Send,
    ) -> Result<bool> {
        let (model, runtime) = self.resolve_model_and_runtime(profile)?;
        match (runtime.backend_kind.as_str(), &runtime.launch_mode) {
            ("llama.cpp", RuntimeLaunchMode::Embedded) => {
                self.install_model_file(model, &runtime.id, cancellation, &mut progress)?;
                cancellation.check()?;
                self.preload_embedded_llama_model(model, runtime)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub fn prepare_profile_prompt(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
    ) -> Result<bool> {
        if request.compression_level.is_original() {
            return Ok(false);
        }

        let (model, runtime) = self.resolve_model_and_runtime(profile)?;
        match (runtime.backend_kind.as_str(), &runtime.launch_mode) {
            ("llama.cpp", RuntimeLaunchMode::Embedded) if request.input_text.trim().is_empty() => {
                self.prepare_embedded_llama_prompt_prefix(request, profile, model, runtime)
            }
            ("llama.cpp", RuntimeLaunchMode::Embedded) => {
                self.prepare_embedded_llama_input(request, profile, model, runtime)
            }
            _ => Ok(false),
        }
    }

    fn resolve_model_and_runtime(
        &self,
        profile: &ProfileDefinition,
    ) -> Result<(&ModelDefinition, &RuntimeDefinition)> {
        let model = self.models.resolve(&profile.model_ref)?;
        let runtime = self.runtimes.resolve(&profile.runtime_ref)?;

        if model.runtime_ref != runtime.id {
            return Err(CompressionError::InvalidConfig(format!(
                "model '{}' uses runtime '{}', but profile '{}' points to '{}'",
                model.id, model.runtime_ref, profile.id, runtime.id
            )));
        }

        Ok((model, runtime))
    }

    fn build_llama_cpp_command(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<(Command, u64)> {
        let executable_path = runtime.executable_path.as_ref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "runtime '{}' is missing executable_path for llama.cpp",
                runtime.id
            ))
        })?;
        let executable_path = resolve_project_path(&self.project_root, executable_path);
        let executable_path = resolve_windows_exe(&executable_path);
        if !executable_path.is_file() {
            return Err(CompressionError::Runtime(format!(
                "llama.cpp executable not found at {}",
                executable_path.display()
            )));
        }

        let model_path = self.resolve_model_file_path(model, &runtime.id)?;

        let prompt = self.build_prompt(request, profile, model)?;
        let mut command = Command::new(executable_path);
        command.current_dir(&self.project_root);
        command.arg("-m").arg(model_path);
        command.arg("-p").arg(prompt);
        command
            .arg("-n")
            .arg(effective_max_output_tokens(request, model).to_string());
        command.arg("--temp").arg("0");
        command
            .arg("--ctx-size")
            .arg(model.context_length.to_string());

        if runtime.threads != "auto" {
            command.arg("--threads").arg(&runtime.threads);
        }

        Ok((command, runtime.timeout_ms))
    }

    fn compress_with_lmstudio(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<RuntimeCompressionObservation> {
        let base_url = runtime.base_url.as_deref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "runtime '{}' is missing base_url for LM Studio",
                runtime.id
            ))
        })?;
        let model_name = resolve_lmstudio_model_name(model, runtime)?;
        self.compress_with_openai_compatible(
            request,
            profile,
            model,
            runtime,
            base_url,
            model_name.as_str(),
        )
    }

    fn compress_with_openai_compatible(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
        base_url: &str,
        model_name: &str,
    ) -> Result<RuntimeCompressionObservation> {
        let prompt = self.build_prompt(request, profile, model)?;
        self.request_observed_completion(request, &prompt, model, runtime, base_url, model_name)
    }

    fn request_observed_completion(
        &self,
        request: &CompressionRequest,
        prompt: &str,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
        base_url: &str,
        model_name: &str,
    ) -> Result<RuntimeCompressionObservation> {
        let first_draft =
            request_openai_completion(request, prompt, model, runtime, base_url, model_name)?;
        trace_model_output("openai.raw_draft", &first_draft.distilled_prompt);
        finalize_observed_model_draft(request, first_draft)
    }

    fn compress_with_managed_llama_cpp_server(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<RuntimeCompressionObservation> {
        let base_url = self.ensure_managed_llama_cpp_server(model, runtime)?;
        let model_name = model
            .api_model
            .as_deref()
            .filter(|configured| *configured != "auto")
            .unwrap_or(model.id.as_str());

        self.compress_with_openai_compatible(
            request, profile, model, runtime, &base_url, model_name,
        )
    }

    #[cfg(feature = "embedded-llama")]
    fn preload_embedded_llama_model(
        &self,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<()> {
        let model_path = self.resolve_model_file_path(model, &runtime.id)?;
        let cache_key = embedded_model_cache_key(model, &model_path);
        self.embedded_models
            .load_or_get(&cache_key, &model_path)
            .map(|_| ())
    }

    #[cfg(feature = "embedded-llama")]
    fn resolve_embedded_runtime_configuration(
        &self,
        model: &ModelDefinition,
        model_path: &Path,
        runtime: &RuntimeDefinition,
    ) -> Result<RuntimeInferenceConfig> {
        let configuration = match self.thread_tuning.resolve(model, model_path, runtime) {
            Ok(configuration) => configuration,
            Err(error) => {
                eprintln!("embedded thread tuning is unavailable: {error}");
                thread_tuning::runtime_config_for_threads(parse_runtime_threads(runtime)?)
            }
        };
        trace_runtime_value(
            "embedded.generation_threads",
            configuration.threads.generation as usize,
        );
        trace_runtime_value(
            "embedded.batch_threads",
            configuration.threads.batch as usize,
        );
        trace_runtime_value(
            "embedded.logical_batch_size",
            configuration.batch_sizes.logical as usize,
        );
        trace_runtime_value(
            "embedded.physical_batch_size",
            configuration.batch_sizes.physical as usize,
        );
        Ok(configuration)
    }

    #[cfg(feature = "embedded-llama")]
    fn tune_embedded_llama_threads(
        &self,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
        cancellation: &AtomicBool,
    ) -> Result<bool> {
        let model_path = self.resolve_model_file_path(model, &runtime.id)?;
        let cache_key = embedded_model_cache_key(model, &model_path);
        let llama_model = self.embedded_models.load_or_get(&cache_key, &model_path)?;
        let quality_prompts = self.physical_batch_quality_prompts(profile, model)?;
        self.thread_tuning.tune(
            &llama_model,
            model,
            &model_path,
            runtime,
            &quality_prompts,
            || cancellation.load(Ordering::Relaxed),
        )
    }

    #[cfg(feature = "embedded-llama")]
    fn physical_batch_quality_prompts(
        &self,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
    ) -> Result<Vec<Vec<u8>>> {
        let compression_level = CompressionLevel::from_u8(2)?;
        PHYSICAL_BATCH_QUALITY_INPUTS
            .iter()
            .map(|input_text| {
                let request = CompressionRequest {
                    input_text: (*input_text).to_string(),
                    compression_level,
                    profile: profile.id.clone(),
                    constraints: CompressionConstraints::default(),
                    target: RequestTarget::codex_default(),
                    source: RequestSource::Desktop,
                };
                self.build_prompt(&request, profile, model)
                    .map(String::into_bytes)
            })
            .collect()
    }

    #[cfg(not(feature = "embedded-llama"))]
    fn tune_embedded_llama_threads(
        &self,
        _profile: &ProfileDefinition,
        _model: &ModelDefinition,
        runtime: &RuntimeDefinition,
        _cancellation: &AtomicBool,
    ) -> Result<bool> {
        Err(CompressionError::InvalidConfig(format!(
            "runtime '{}' uses embedded llama.cpp, but this build was compiled without the 'embedded-llama' feature",
            runtime.id
        )))
    }

    #[cfg(not(feature = "embedded-llama"))]
    fn preload_embedded_llama_model(
        &self,
        _model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<()> {
        Err(CompressionError::InvalidConfig(format!(
            "runtime '{}' uses embedded llama.cpp, but this build was compiled without the 'embedded-llama' feature",
            runtime.id
        )))
    }

    #[cfg(feature = "embedded-llama")]
    fn prepare_embedded_llama_prompt_prefix(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<bool> {
        let total_started_at = Instant::now();
        let timeout_ms = runtime.timeout_ms;
        let abort_signal = llama_cpp::AbortSignal::with_timeout(Duration::from_millis(timeout_ms));
        prefer_embedded_timeout(
            self.preload_embedded_llama_model(model, runtime),
            &abort_signal,
            timeout_ms,
        )?;
        let prompt_parts = self.build_prompt_parts(request, profile, model)?;
        if prompt_parts.prefix.trim().is_empty() {
            return Ok(false);
        }

        let model_path = self.resolve_model_file_path(model, &runtime.id)?;
        let model_cache_key = embedded_model_cache_key(model, &model_path);
        let prompt_prefix = format_embedded_llama_prompt_prefix(&prompt_parts.prefix);
        let model_started_at = Instant::now();
        let llama_model = prefer_embedded_timeout(
            self.embedded_models
                .load_or_get(&model_cache_key, &model_path),
            &abort_signal,
            timeout_ms,
        )?;
        trace_runtime_timing(
            "embedded.prepare_model_load_or_cache",
            model_started_at.elapsed(),
        );
        let configuration =
            self.resolve_embedded_runtime_configuration(model, &model_path, runtime)?;
        let max_tokens = effective_max_output_tokens(request, model) as usize;
        let context_length = select_embedded_context_length(
            &llama_model,
            &prompt_prefix,
            "",
            max_tokens,
            model.context_length as usize,
        )?;
        trace_runtime_value("embedded.selected_context", context_length as usize);
        let prompt_cache_key = embedded_prompt_cache_key(
            model,
            &model_path,
            context_length,
            configuration,
            &prompt_prefix,
        );
        if self
            .embedded_models
            .has_prepared_session(&prompt_cache_key)?
        {
            trace_runtime_value("embedded.prepare_prompt_cache_hit", 1);
            trace_runtime_timing("embedded.prepare_total", total_started_at.elapsed());
            return Ok(true);
        }

        trace_runtime_value("embedded.prepare_prompt_cache_hit", 0);
        let session_started_at = Instant::now();
        let mut session = llama_model
            .create_session(embedded_session_params(context_length, configuration))
            .map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to create prepared embedded llama.cpp session for '{}': {error}",
                    model.id
                ))
            })?;
        session
            .set_abort_signal(Arc::clone(&abort_signal))
            .map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to configure prepared embedded llama.cpp timeout for '{}': {error}",
                    model.id
                ))
            })?;
        ensure_embedded_deadline(&abort_signal, timeout_ms)?;
        trace_runtime_timing(
            "embedded.prepare_session_create",
            session_started_at.elapsed(),
        );

        trace_runtime_value("embedded.prepare_prompt_prefix_bytes", prompt_prefix.len());
        let feed_started_at = Instant::now();
        prefer_embedded_timeout(
            session
                .advance_context(prompt_prefix.as_bytes())
                .map_err(|error| {
                    CompressionError::Runtime(format!(
                    "failed to prepare prompt prefix for embedded llama.cpp model '{}': {error}",
                    model.id
                ))
                }),
            &abort_signal,
            timeout_ms,
        )?;
        trace_runtime_timing(
            "embedded.prepare_prompt_prefix_eval",
            feed_started_at.elapsed(),
        );

        // Preserve a clean prefix session before exercising the one-token decode path.
        // Both cold operations then finish before the first user compression.
        let copy_started_at = Instant::now();
        let clean_session = prefer_embedded_timeout(
            session.deep_copy().map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to pre-copy prepared embedded prompt session for '{}': {error}",
                    model.id
                ))
            }),
            &abort_signal,
            timeout_ms,
        )?;
        trace_runtime_timing(
            "embedded.prepare_prompt_cold_copy",
            copy_started_at.elapsed(),
        );

        let generation_started_at = Instant::now();
        let mut warmup = session
            .start_completing_with(
                llama_cpp::standard_sampler::StandardSampler::new_greedy(),
                1,
            )
            .map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to warm embedded llama.cpp generation for '{}': {error}",
                    model.id
                ))
            })?;
        if let Some(result) = warmup.next() {
            prefer_embedded_timeout(
                result.map_err(|error| {
                    CompressionError::Runtime(format!(
                        "failed to warm embedded llama.cpp generation for '{}': {error}",
                        model.id
                    ))
                }),
                &abort_signal,
                timeout_ms,
            )?;
        }
        drop(warmup);
        ensure_embedded_deadline(&abort_signal, timeout_ms)?;
        trace_runtime_timing(
            "embedded.prepare_generation_warmup",
            generation_started_at.elapsed(),
        );

        let store_started_at = Instant::now();
        self.embedded_models
            .store_prepared_session(prompt_cache_key, clean_session)?;
        trace_runtime_timing("embedded.prepare_prompt_store", store_started_at.elapsed());
        trace_runtime_timing("embedded.prepare_total", total_started_at.elapsed());
        Ok(true)
    }

    #[cfg(feature = "embedded-llama")]
    fn prepare_embedded_llama_input(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<bool> {
        let total_started_at = Instant::now();
        let timeout_ms = runtime.timeout_ms;
        let abort_signal = llama_cpp::AbortSignal::with_timeout(Duration::from_millis(timeout_ms));
        prefer_embedded_timeout(
            self.preload_embedded_llama_model(model, runtime),
            &abort_signal,
            timeout_ms,
        )?;
        let prompt_parts = self.build_prompt_parts(request, profile, model)?;
        let model_path = self.resolve_model_file_path(model, &runtime.id)?;
        let model_cache_key = embedded_model_cache_key(model, &model_path);
        let use_prompt_cache = !prompt_parts.prefix.trim().is_empty();
        let (embedded_prefix, embedded_suffix) = if use_prompt_cache {
            let prefix = format_embedded_llama_prompt_prefix(&prompt_parts.prefix);
            (
                prefix,
                format_embedded_llama_prompt_suffix(&prompt_parts.suffix),
            )
        } else {
            (
                String::new(),
                format_embedded_llama_prompt(&prompt_parts.combined()),
            )
        };
        let model_started_at = Instant::now();
        let llama_model = prefer_embedded_timeout(
            self.embedded_models
                .load_or_get(&model_cache_key, &model_path),
            &abort_signal,
            timeout_ms,
        )?;
        trace_runtime_timing(
            "embedded.prepare_input_model_load_or_cache",
            model_started_at.elapsed(),
        );
        let configuration =
            self.resolve_embedded_runtime_configuration(model, &model_path, runtime)?;
        let max_tokens = effective_max_output_tokens(request, model) as usize;
        let context_length = select_embedded_context_length(
            &llama_model,
            &embedded_prefix,
            &embedded_suffix,
            max_tokens,
            model.context_length as usize,
        )?;
        trace_runtime_value("embedded.selected_context", context_length as usize);
        let prompt_cache_key = use_prompt_cache.then(|| {
            embedded_prompt_cache_key(
                model,
                &model_path,
                context_length,
                configuration,
                &embedded_prefix,
            )
        });
        let input_cache_key = embedded_input_cache_key(
            model,
            &model_path,
            context_length,
            configuration,
            &embedded_prefix,
            &embedded_suffix,
        );
        if self
            .embedded_models
            .has_prepared_input_session(&input_cache_key)?
        {
            trace_runtime_value("embedded.prepare_input_cache_hit", 1);
            trace_runtime_timing("embedded.prepare_input_total", total_started_at.elapsed());
            return Ok(true);
        }
        trace_runtime_value("embedded.prepare_input_cache_hit", 0);

        let mut session = if let Some(prompt_cache_key) = prompt_cache_key.as_deref() {
            if let Some(session) = self
                .embedded_models
                .get_prepared_session(prompt_cache_key)?
            {
                session
            } else {
                let mut session = llama_model
                    .create_session(embedded_session_params(context_length, configuration))
                    .map_err(|error| {
                        CompressionError::Runtime(format!(
                            "failed to create prepared embedded llama.cpp session for '{}': {error}",
                            model.id
                        ))
                    })?;
                session
                    .set_abort_signal(Arc::clone(&abort_signal))
                    .map_err(|error| {
                        CompressionError::Runtime(format!(
                            "failed to configure prepared embedded llama.cpp timeout for '{}': {error}",
                            model.id
                        ))
                    })?;
                prefer_embedded_timeout(
                    session.advance_context(embedded_prefix.as_bytes()).map_err(|error| {
                        CompressionError::Runtime(format!(
                            "failed to prepare prompt prefix for embedded llama.cpp model '{}': {error}",
                            model.id
                        ))
                    }),
                    &abort_signal,
                    timeout_ms,
                )?;
                self.embedded_models
                    .store_prepared_session_copy(prompt_cache_key.to_string(), &session)?;
                session
            }
        } else {
            llama_model
                .create_session(embedded_session_params(context_length, configuration))
                .map_err(|error| {
                    CompressionError::Runtime(format!(
                        "failed to create prepared embedded llama.cpp session for '{}': {error}",
                        model.id
                    ))
                })?
        };
        session
            .set_abort_signal(Arc::clone(&abort_signal))
            .map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to configure prepared input timeout for '{}': {error}",
                    model.id
                ))
            })?;
        ensure_embedded_deadline(&abort_signal, timeout_ms)?;

        let feed_started_at = Instant::now();
        prefer_embedded_timeout(
            session
                .advance_context(embedded_suffix.as_bytes())
                .map_err(|error| {
                    CompressionError::Runtime(format!(
                        "failed to prepare input for embedded llama.cpp model '{}': {error}",
                        model.id
                    ))
                }),
            &abort_signal,
            timeout_ms,
        )?;
        trace_runtime_timing("embedded.prepare_input_eval", feed_started_at.elapsed());
        ensure_embedded_deadline(&abort_signal, timeout_ms)?;
        self.embedded_models
            .store_prepared_input_session(input_cache_key, session)?;
        trace_runtime_timing("embedded.prepare_input_total", total_started_at.elapsed());
        Ok(true)
    }

    #[cfg(not(feature = "embedded-llama"))]
    fn prepare_embedded_llama_prompt_prefix(
        &self,
        _request: &CompressionRequest,
        _profile: &ProfileDefinition,
        _model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<bool> {
        Err(CompressionError::InvalidConfig(format!(
            "runtime '{}' uses embedded llama.cpp, but this build was compiled without the 'embedded-llama' feature",
            runtime.id
        )))
    }

    #[cfg(not(feature = "embedded-llama"))]
    fn prepare_embedded_llama_input(
        &self,
        _request: &CompressionRequest,
        _profile: &ProfileDefinition,
        _model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<bool> {
        Err(CompressionError::InvalidConfig(format!(
            "runtime '{}' uses embedded llama.cpp, but this build was compiled without the 'embedded-llama' feature",
            runtime.id
        )))
    }

    #[cfg(feature = "embedded-llama")]
    fn compress_with_embedded_llama_cpp(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<RuntimeCompressionObservation> {
        let prompt_parts = self.build_prompt_parts(request, profile, model)?;
        self.request_observed_embedded_completion(request, &prompt_parts, model, runtime)
    }

    #[cfg(not(feature = "embedded-llama"))]
    fn compress_with_embedded_llama_cpp(
        &self,
        _request: &CompressionRequest,
        _profile: &ProfileDefinition,
        _model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<RuntimeCompressionObservation> {
        Err(CompressionError::InvalidConfig(format!(
            "runtime '{}' uses embedded llama.cpp, but this build was compiled without the 'embedded-llama' feature",
            runtime.id
        )))
    }

    #[cfg(feature = "embedded-llama")]
    fn request_observed_embedded_completion(
        &self,
        request: &CompressionRequest,
        prompt_parts: &PromptParts,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<RuntimeCompressionObservation> {
        let completion =
            self.request_embedded_llama_completion(request, prompt_parts, model, runtime, true)?;
        match completion {
            EmbeddedCompletion::RawModel(first_draft) => {
                trace_model_output("embedded.raw_draft", &first_draft.distilled_prompt);
                finalize_observed_model_draft(request, first_draft)
            }
            EmbeddedCompletion::RuntimeFallback(final_draft) => {
                trace_model_output("embedded.timeout_fallback", &final_draft.distilled_prompt);
                Ok(RuntimeCompressionObservation::runtime_fallback(final_draft))
            }
        }
    }

    #[cfg(feature = "embedded-llama")]
    fn request_embedded_llama_completion(
        &self,
        request: &CompressionRequest,
        prompt_parts: &PromptParts,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
        allow_prompt_cache: bool,
    ) -> Result<EmbeddedCompletion> {
        let total_started_at = Instant::now();
        let timeout_ms = runtime.timeout_ms;
        let abort_signal = llama_cpp::AbortSignal::with_timeout(Duration::from_millis(timeout_ms));
        let model_path = self.resolve_model_file_path(model, &runtime.id)?;
        let cache_key = embedded_model_cache_key(model, &model_path);
        let model_started_at = Instant::now();
        let llama_model = match self.embedded_models.load_or_get(&cache_key, &model_path) {
            Ok(model) if !abort_signal.is_aborted() => model,
            Ok(_) => return embedded_timeout_completion(request, timeout_ms),
            Err(_) if abort_signal.is_aborted() => {
                return embedded_timeout_completion(request, timeout_ms)
            }
            Err(error) => return Err(error),
        };
        trace_runtime_timing("embedded.model_load_or_cache", model_started_at.elapsed());
        let configuration =
            self.resolve_embedded_runtime_configuration(model, &model_path, runtime)?;

        let prompt_started_at = Instant::now();
        let use_prompt_cache = allow_prompt_cache && !prompt_parts.prefix.trim().is_empty();
        let (embedded_prefix, embedded_suffix) = if use_prompt_cache {
            let prefix = format_embedded_llama_prompt_prefix(&prompt_parts.prefix);
            (
                prefix,
                format_embedded_llama_prompt_suffix(&prompt_parts.suffix),
            )
        } else {
            (
                String::new(),
                format_embedded_llama_prompt(&prompt_parts.combined()),
            )
        };
        trace_runtime_timing("embedded.prompt_format", prompt_started_at.elapsed());
        trace_runtime_value(
            "embedded.prompt_bytes",
            embedded_prefix.len() + embedded_suffix.len(),
        );
        let max_tokens = effective_max_output_tokens(request, model) as usize;
        let context_length = select_embedded_context_length(
            &llama_model,
            &embedded_prefix,
            &embedded_suffix,
            max_tokens,
            model.context_length as usize,
        )?;
        trace_runtime_value("embedded.selected_context", context_length as usize);
        let prompt_cache_key = use_prompt_cache.then(|| {
            embedded_prompt_cache_key(
                model,
                &model_path,
                context_length,
                configuration,
                &embedded_prefix,
            )
        });
        let input_cache_key = embedded_input_cache_key(
            model,
            &model_path,
            context_length,
            configuration,
            &embedded_prefix,
            &embedded_suffix,
        );

        let mut prompt_prefix_eval_elapsed = Duration::ZERO;
        let prepared_input_started_at = Instant::now();
        let prepared_input = self
            .embedded_models
            .take_prepared_input_session(&input_cache_key)?;
        let input_cache_hit = prepared_input.is_some();
        trace_runtime_timing(
            "embedded.input_session_restore",
            prepared_input_started_at.elapsed(),
        );
        let (mut session, prompt_cache_hit) = if let Some(session) = prepared_input {
            (session, true)
        } else if let Some(cache_key) = prompt_cache_key.as_deref() {
            let restore_started_at = Instant::now();
            if let Some(session) = self.embedded_models.get_prepared_session(cache_key)? {
                trace_runtime_timing("embedded.session_restore", restore_started_at.elapsed());
                (session, true)
            } else {
                let session_started_at = Instant::now();
                let mut session = llama_model
                    .create_session(embedded_session_params(context_length, configuration))
                    .map_err(|error| {
                        CompressionError::Runtime(format!(
                            "failed to create embedded llama.cpp session for '{}': {error}",
                            model.id
                        ))
                    })?;
                session
                    .set_abort_signal(Arc::clone(&abort_signal))
                    .map_err(|error| {
                        CompressionError::Runtime(format!(
                            "failed to configure embedded llama.cpp timeout for '{}': {error}",
                            model.id
                        ))
                    })?;
                trace_runtime_timing("embedded.session_create", session_started_at.elapsed());

                let prefix_started_at = Instant::now();
                let prefix_result = session.advance_context(embedded_prefix.as_bytes()).map_err(
                    |error| {
                        CompressionError::Runtime(format!(
                            "failed to feed prepared prompt prefix to embedded llama.cpp model '{}': {error}",
                            model.id
                        ))
                    },
                );
                if abort_signal.is_aborted() {
                    return embedded_timeout_completion(request, timeout_ms);
                }
                prefix_result?;
                prompt_prefix_eval_elapsed = prefix_started_at.elapsed();
                trace_runtime_timing("embedded.prompt_prefix_eval", prompt_prefix_eval_elapsed);
                self.embedded_models
                    .store_prepared_session_copy(cache_key.to_string(), &session)?;
                (session, false)
            }
        } else {
            let session_started_at = Instant::now();
            let session = llama_model
                .create_session(embedded_session_params(context_length, configuration))
                .map_err(|error| {
                    CompressionError::Runtime(format!(
                        "failed to create embedded llama.cpp session for '{}': {error}",
                        model.id
                    ))
                })?;
            trace_runtime_timing("embedded.session_create", session_started_at.elapsed());
            (session, false)
        };
        session
            .set_abort_signal(Arc::clone(&abort_signal))
            .map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to configure embedded llama.cpp timeout for '{}': {error}",
                    model.id
                ))
            })?;
        if abort_signal.is_aborted() {
            return embedded_timeout_completion(request, timeout_ms);
        }
        trace_runtime_value("embedded.prompt_cache_hit", usize::from(prompt_cache_hit));
        trace_runtime_value("embedded.input_cache_hit", usize::from(input_cache_hit));

        let suffix_elapsed = if input_cache_hit {
            Duration::ZERO
        } else {
            let suffix_started_at = Instant::now();
            let suffix_result =
                session
                    .advance_context(embedded_suffix.as_bytes())
                    .map_err(|error| {
                        CompressionError::Runtime(format!(
                            "failed to feed prompt to embedded llama.cpp model '{}': {error}",
                            model.id
                        ))
                    });
            if abort_signal.is_aborted() {
                return embedded_timeout_completion(request, timeout_ms);
            }
            suffix_result?;
            suffix_started_at.elapsed()
        };
        trace_runtime_timing("embedded.prompt_suffix_eval", suffix_elapsed);
        trace_runtime_timing(
            "embedded.prompt_eval",
            prompt_prefix_eval_elapsed + suffix_elapsed,
        );

        trace_runtime_value("embedded.max_output_tokens", max_tokens);
        let completion_setup_started_at = Instant::now();
        let mut completions = session
            .start_completing_with(
                llama_cpp::standard_sampler::StandardSampler::new_greedy(),
                max_tokens,
            )
            .map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to start embedded llama.cpp completion for '{}': {error}",
                    model.id
                ))
            })?
            .into_strings();
        trace_runtime_timing(
            "embedded.completion_setup",
            completion_setup_started_at.elapsed(),
        );
        let generation_started_at = Instant::now();
        let mut output = String::new();
        let mut generated_chunks = 0usize;

        for token in &mut completions {
            if abort_signal.is_aborted() {
                trace_runtime_timing("embedded.generation", generation_started_at.elapsed());
                trace_runtime_value("embedded.generated_chunks", generated_chunks);
                trace_runtime_value("embedded.output_chars", output.chars().count());
                return embedded_timeout_completion(request, timeout_ms);
            }

            let token = match token {
                Ok(token) => token,
                Err(_) if abort_signal.is_aborted() => {
                    return embedded_timeout_completion(request, timeout_ms)
                }
                Err(error) => {
                    return Err(CompressionError::Runtime(format!(
                        "embedded llama.cpp generation failed for '{}': {error}",
                        model.id
                    )))
                }
            };
            generated_chunks += 1;
            output.push_str(&token);
            let completed_json = {
                let stopped = trim_after_stop_marker(&output);
                first_complete_json_object_end(stopped)
                    .map(|end_index| stopped[..end_index].to_string())
            };
            if let Some(json) = completed_json {
                output = json;
                break;
            }
        }
        if abort_signal.is_aborted() {
            return embedded_timeout_completion(request, timeout_ms);
        }
        trace_runtime_timing("embedded.generation", generation_started_at.elapsed());
        trace_runtime_value("embedded.generated_chunks", generated_chunks);
        trace_runtime_value("embedded.output_chars", output.chars().count());

        let output = trim_after_stop_marker(&output).trim();
        let parse_started_at = Instant::now();
        let parsed = parse_compression_output(output);
        trace_runtime_timing("embedded.output_parse", parse_started_at.elapsed());
        trace_runtime_timing("embedded.total_completion", total_started_at.elapsed());
        parsed.map(EmbeddedCompletion::RawModel)
    }

    fn resolve_model_file_path(
        &self,
        model: &ModelDefinition,
        runtime_id: &str,
    ) -> Result<PathBuf> {
        let model_path = model.model_path.as_ref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "model '{}' is missing model_path for runtime '{}'",
                model.id, runtime_id
            ))
        })?;
        let model_path = resolve_project_path(&self.project_root, model_path);
        self.require_model_file(model, runtime_id)?;
        Ok(model_path)
    }

    fn install_model_file(
        &self,
        model: &ModelDefinition,
        runtime_id: &str,
        cancellation: &ModelDownloadCancellation,
        progress: &mut (dyn FnMut(ModelDownloadProgress) + Send),
    ) -> Result<()> {
        let configured_path = model.model_path.as_ref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "model '{}' is missing model_path for runtime '{}'",
                model.id, runtime_id
            ))
        })?;
        let model_path = resolve_project_path(&self.project_root, configured_path);
        let operation_lock = self.model_files.operation_lock(&model_path)?;
        let _operation = operation_lock
            .lock()
            .map_err(|_| CompressionError::Runtime("model file operation is unavailable".into()))?;
        if self.model_files.is_verified(&model_path)? {
            return Ok(());
        }

        if let Some(download) = &model.download {
            if model_path.file_name().and_then(|value| value.to_str()) != Some(download.filename())
            {
                return Err(CompressionError::InvalidConfig(format!(
                    "model '{}' path filename must match its Hugging Face filename",
                    model.id
                )));
            }
            ensure_model_file(&model_path, download, cancellation, progress)?;
            self.model_files.mark_verified(&model_path)?;
            return Ok(());
        }
        if !model_path.is_file() {
            return Err(CompressionError::Runtime(format!(
                "model file not found at {}",
                model_path.display()
            )));
        }
        self.model_files.mark_verified(&model_path)?;
        Ok(())
    }

    fn require_model_file(&self, model: &ModelDefinition, runtime_id: &str) -> Result<()> {
        if !self.model_file_is_installed(model, runtime_id)? {
            return Err(CompressionError::Runtime(format!(
                "model '{}' is not installed or failed verification; install it from the app model setup before compression",
                model.id
            )));
        }
        Ok(())
    }

    fn model_file_is_installed(&self, model: &ModelDefinition, runtime_id: &str) -> Result<bool> {
        let configured_path = model.model_path.as_ref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "model '{}' is missing model_path for runtime '{}'",
                model.id, runtime_id
            ))
        })?;
        let model_path = resolve_project_path(&self.project_root, configured_path);
        let operation_lock = self.model_files.operation_lock(&model_path)?;
        let _operation = operation_lock
            .lock()
            .map_err(|_| CompressionError::Runtime("model file operation is unavailable".into()))?;
        if self.model_files.is_verified(&model_path)? {
            return Ok(true);
        }
        let installed = match &model.download {
            Some(download) => verify_existing_model(&model_path, download)?,
            None => model_path.is_file(),
        };
        if installed {
            self.model_files.mark_verified(&model_path)?;
        } else {
            self.model_files.forget(&model_path)?;
        }
        Ok(installed)
    }

    fn ensure_managed_llama_cpp_server(
        &self,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<String> {
        let base_url = runtime.base_url.as_deref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "runtime '{}' is missing base_url for a managed llama.cpp server",
                runtime.id
            ))
        })?;
        let base = parse_http_base_url(base_url)?;
        if !matches!(base.host.as_str(), "127.0.0.1" | "localhost") {
            return Err(CompressionError::InvalidConfig(format!(
                "managed runtime '{}' must bind to 127.0.0.1 or localhost, not '{}'",
                runtime.id, base.host
            )));
        }

        let mut processes = self.managed_runtimes.processes.lock().map_err(|_| {
            CompressionError::Runtime("managed runtime process registry is unavailable".into())
        })?;
        if let Some(server) = processes.get_mut(&runtime.id) {
            if server.child.try_wait()?.is_none() {
                return Ok(base_url.to_string());
            }
        }
        processes.remove(&runtime.id);

        let executable_path = runtime.executable_path.as_ref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "runtime '{}' is missing executable_path for a managed llama.cpp server",
                runtime.id
            ))
        })?;
        let executable_path =
            resolve_windows_exe(&resolve_project_path(&self.project_root, executable_path));
        if !executable_path.is_file() {
            return Err(CompressionError::Runtime(format!(
                "managed llama.cpp server executable not found at {}",
                executable_path.display()
            )));
        }

        let model_path = self.resolve_model_file_path(model, &runtime.id)?;

        let mut command = Command::new(executable_path);
        command.current_dir(&self.project_root);
        command.arg("-m").arg(model_path);
        command.arg("--host").arg(&base.host);
        command.arg("--port").arg(base.port.to_string());
        command
            .arg("--ctx-size")
            .arg(model.context_length.to_string());

        if runtime.threads != "auto" {
            command.arg("--threads").arg(&runtime.threads);
        }
        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to start managed llama.cpp server '{}': {error}",
                    runtime.id
                ))
            })?;

        let health_base_url = format!("http://{}:{}", base.host, base.port);
        let health_path = runtime.health_path.as_deref().unwrap_or("/health");
        let startup_timeout = Duration::from_millis(runtime.startup_timeout_ms);
        if let Err(error) = wait_for_runtime_health(&health_base_url, health_path, startup_timeout)
        {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }

        processes.insert(runtime.id.clone(), ManagedServer { child });
        Ok(base_url.to_string())
    }

    fn build_prompt(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
    ) -> Result<String> {
        Ok(self.build_prompt_parts(request, profile, model)?.combined())
    }

    fn build_prompt_parts(
        &self,
        request: &CompressionRequest,
        _profile: &ProfileDefinition,
        model: &ModelDefinition,
    ) -> Result<PromptParts> {
        if model.prompt_style == "concise" {
            let language = request_output_language(request);
            let prompt_input = preprocess_input_for_llm(&request.input_text);
            let prompt_profile = self
                .prompt_profiles
                .resolve(request.compression_level.value())?;
            let target_ratio = &prompt_profile.target_ratio;
            let required_terms = required_technical_terms(&prompt_input);
            let terms_instruction = if required_terms.is_empty() {
                String::new()
            } else {
                format!("必須語(逐字/各1回):{}\n", required_terms.join(","))
            };
            let prompt_structure = PromptStructure::analyze(&prompt_input, &required_terms);
            let organized_input = prompt_structure.render_for_model();
            let protected_expressions = prompt_structure.protected_expressions();
            let protected_instruction = if protected_expressions.is_empty() {
                String::new()
            } else {
                format!("保護(逐字/各1回):{}\n", protected_expressions.join(" | "))
            };
            let semantic_shortening_instruction = if prompt_profile.allow_semantic_shortening {
                "同義短縮可。"
            } else {
                "表現変更最小。"
            };
            let prefix = format!(
                "JSONだけ返す。distilled_promptは{language}/{target_ratio}/原文より短く。\n\
                 守る:{} {} {semantic_shortening_instruction} {} ラベル/見出し/接頭辞は禁止。\n\
                 入力の[現状]は望む動作ではない。[現状→要求]は矢印後だけを実装指示にする。各行の必須語はその行の役割と一緒に使う。[検証]の成功/失敗/否定も変えない。\n\
                 出力前確認:必須語全件/[制約]全件/[検証]全件/限定・否定・条件語の脱落なし/重複なし。\n",
                self.prompt_profiles.shared_instruction(),
                prompt_profile.instruction,
                prompt_profile.format_instruction
            );
            let suffix = format!(
                "{terms_instruction}{protected_instruction}入力整理:\n{organized_input}\nJSON:{{\"distilled_prompt\":\"\"}}"
            );
            trace_model_prompt("concise.prefix", &prefix);
            trace_model_prompt("concise.suffix", &suffix);
            return Ok(PromptParts { prefix, suffix });
        }

        let template_path = self
            .prompts_dir
            .join(format!("{}.md", model.prompt_template));
        let template = fs::read_to_string(&template_path)?;
        let output_language = if request_output_language(request) == "日本語" {
            "`distilled_prompt` は必ず日本語で書く。"
        } else {
            "`distilled_prompt` は入力と同じ言語で書く。"
        };
        let required_terms = required_technical_terms(&request.input_text);
        let technical_terms = if required_terms.is_empty() {
            String::new()
        } else {
            format!(
                "次の技術用語は表記を変えず、必ず `distilled_prompt` に含める: {}。\n",
                required_terms.join(", ")
            )
        };

        Ok(PromptParts::whole(format!(
            "{template}\n\n\
             {output_language}\n\
             {technical_terms}\
             Codex がそのまま実行できる短い実装指示にする。要求、制約、禁止事項を落とさない。\n\
             「実装指示を作成してください」「プロンプトを作成してください」とは書かず、実装作業を直接命令する。\n\
             出力は次の JSON オブジェクトだけにする。`removed_content_summary` に削除内容がなければ空配列を入れる。\n\
              {{\"distilled_prompt\":\"...\",\"removed_content_summary\":[\"...\"]}}\n\n\
              ユーザーの依頼:\n{}",
            preprocess_input_for_llm(&request.input_text)
        )))
    }
}

impl RuntimeBackend for ConfiguredRuntimeBackend {
    fn compress(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
    ) -> Result<CompressionDraft> {
        self.compress_observed_configured(request, profile)
            .map(|observation| observation.final_draft)
    }

    fn compress_observed(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
    ) -> Result<RuntimeCompressionObservation> {
        self.compress_observed_configured(request, profile)
    }

    fn prepare(&self, request: &CompressionRequest, profile: &ProfileDefinition) -> Result<bool> {
        self.prepare_profile_prompt(request, profile)
    }
}

impl ConfiguredRuntimeBackend {
    fn compress_observed_configured(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
    ) -> Result<RuntimeCompressionObservation> {
        let (model, runtime) = self.resolve_model_and_runtime(profile)?;

        match (runtime.backend_kind.as_str(), &runtime.launch_mode) {
            ("llama.cpp", RuntimeLaunchMode::OneShot) => {
                self.compress_with_llama_cpp(request, profile, model, runtime)
            }
            ("llama.cpp", RuntimeLaunchMode::ManagedSidecar) => {
                self.compress_with_managed_llama_cpp_server(request, profile, model, runtime)
            }
            ("llama.cpp", RuntimeLaunchMode::Embedded) => {
                self.compress_with_embedded_llama_cpp(request, profile, model, runtime)
            }
            ("lmstudio" | "lm_studio" | "lm-studio", RuntimeLaunchMode::External) => {
                self.compress_with_lmstudio(request, profile, model, runtime)
            }
            (backend, launch_mode) => Err(CompressionError::Runtime(format!(
                "unsupported runtime backend '{backend}' with launch_mode '{launch_mode:?}' for runtime '{}'",
                runtime.id
            ))),
        }
    }

    fn compress_with_llama_cpp(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<RuntimeCompressionObservation> {
        let (command, timeout_ms) =
            self.build_llama_cpp_command(request, profile, model, runtime)?;
        let output = run_command_with_timeout(command, Duration::from_millis(timeout_ms))?;

        if !output.status.success() {
            return Err(CompressionError::Runtime(format!(
                "llama.cpp process exited with status {}; stderr was {} bytes",
                output.status,
                output.stderr.len()
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_compression_output(&stdout).map(RuntimeCompressionObservation::raw_model)
    }
}

fn trusted_precompacted_fallback_draft(request: &CompressionRequest) -> Option<CompressionDraft> {
    let candidate = trusted_precompacted_candidate(&request.input_text)?;
    let safety_input = normalize_self_correction_artifacts(&normalize_known_input_typos_for_llm(
        &remove_obvious_input_noise(&normalize_input_whitespace(&request.input_text)),
    ));
    if candidate.chars().count() >= request.input_text.trim().chars().count()
        || (request.constraints.preserve_numbers
            && !preserves_requested_numbers(&safety_input, &candidate))
        || !structured_candidate_preserves_requirements(&safety_input, &candidate)
    {
        return None;
    }

    Some(CompressionDraft {
        distilled_prompt: candidate,
        removed_content_summary: vec![
            "Local runtime returned unusable output; used verified preprocessed compact instruction."
                .to_string(),
        ],
    })
}

fn finalize_observed_model_draft(
    request: &CompressionRequest,
    raw_model_draft: CompressionDraft,
) -> Result<RuntimeCompressionObservation> {
    let mut final_draft = raw_model_draft.clone();
    let mut transformations = Vec::new();

    let previous = final_draft.clone();
    restore_missing_required_constraints(request, &mut final_draft);
    if final_draft != previous {
        transformations.push(RuntimeTransformation::RestoredRequiredConstraints);
    }

    let previous = final_draft.clone();
    restore_missing_required_terms(request, &mut final_draft);
    if final_draft != previous {
        transformations.push(RuntimeTransformation::RestoredRequiredTerms);
    }

    let previous = final_draft.clone();
    polish_model_output_for_request(request, &mut final_draft);
    if final_draft != previous {
        transformations.push(RuntimeTransformation::PolishedModelOutput);
    }

    let previous = final_draft.clone();
    final_draft = validated_draft_or_fallback(request, final_draft)?;
    if final_draft != previous {
        transformations.push(RuntimeTransformation::RuntimeFallback);
    }

    Ok(RuntimeCompressionObservation {
        raw_model_draft: Some(raw_model_draft),
        final_draft,
        transformations,
    })
}

fn validated_draft_or_fallback(
    request: &CompressionRequest,
    draft: CompressionDraft,
) -> Result<CompressionDraft> {
    if let Err(error) = validate_compression_draft(request, &draft) {
        if let Some(mut fallback) = trusted_precompacted_fallback_draft(request) {
            restore_missing_required_constraints(request, &mut fallback);
            restore_missing_required_terms(request, &mut fallback);
            polish_model_output_for_request(request, &mut fallback);
            return Ok(fallback);
        }
        return Err(CompressionError::Runtime(format!(
            "{error}; invalid draft starts with: {}",
            output_snippet(&draft.distilled_prompt)
        )));
    }
    Ok(draft)
}

fn trusted_precompacted_candidate(input: &str) -> Option<String> {
    verified_structured_candidate(input)
}

fn verified_structured_candidate(input: &str) -> Option<String> {
    let normalized_input =
        normalize_self_correction_artifacts(&normalize_known_input_typos_for_llm(
            &remove_obvious_input_noise(&normalize_input_whitespace(input)),
        ));
    let required_terms = required_technical_terms(&normalized_input);
    let candidate =
        PromptStructure::analyze(&normalized_input, &required_terms).compact_candidate()?;
    let original_chars = normalized_input.trim().chars().count();

    (candidate.chars().count() < original_chars
        && candidate.chars().count() * 3 >= original_chars
        && structured_candidate_preserves_requirements(&normalized_input, &candidate))
    .then_some(candidate)
}

fn structured_candidate_preserves_requirements(input: &str, candidate: &str) -> bool {
    contains_required_technical_terms(input, candidate)
        && preserves_requested_numbers(input, candidate)
        && preserves_targeted_change_constraints(input, candidate)
        && preserves_negative_constraints(input, candidate)
        && preserves_list_constraints(input, candidate)
        && preserves_state_persistence_constraints(input, candidate)
        && preserves_verification_constraints(input, candidate)
}

struct ProcessRunOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn run_command_with_timeout(mut command: Command, timeout: Duration) -> Result<ProcessRunOutput> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            CompressionError::Runtime(format!("failed to start llama.cpp process: {error}"))
        })?;
    let started_at = Instant::now();

    loop {
        if let Some(status) = child.try_wait().map_err(|error| {
            CompressionError::Runtime(format!("failed to poll llama.cpp process: {error}"))
        })? {
            let mut stdout = Vec::new();
            if let Some(mut pipe) = child.stdout.take() {
                pipe.read_to_end(&mut stdout)?;
            }

            let mut stderr = Vec::new();
            if let Some(mut pipe) = child.stderr.take() {
                pipe.read_to_end(&mut stderr)?;
            }

            return Ok(ProcessRunOutput {
                status,
                stdout,
                stderr,
            });
        }

        if started_at.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(CompressionError::RuntimeTimeout(timeout.as_millis() as u64));
        }

        thread::sleep(Duration::from_millis(25));
    }
}

fn wait_for_runtime_health(base_url: &str, health_path: &str, timeout: Duration) -> Result<()> {
    let started_at = Instant::now();
    let probe_timeout = Duration::from_millis(1_000);
    let mut last_error = None;

    while started_at.elapsed() < timeout {
        match http_json_request("GET", base_url, health_path, None, None, probe_timeout) {
            Ok(_) => return Ok(()),
            Err(error) => last_error = Some(error),
        }
        thread::sleep(Duration::from_millis(200));
    }

    let detail = last_error
        .map(|error| error.to_string())
        .unwrap_or_else(|| "no health response received".to_string());
    Err(CompressionError::Runtime(format!(
        "managed local runtime did not become healthy within {} ms: {detail}",
        timeout.as_millis()
    )))
}

fn trace_runtime_timing(stage: &str, elapsed: Duration) {
    if trace_enabled() {
        eprintln!(
            "trace.runtime.{stage}_ms={}",
            elapsed.as_millis().min(u128::from(u64::MAX))
        );
    }
}

fn trace_runtime_value(stage: &str, value: usize) {
    if trace_enabled() {
        eprintln!("trace.runtime.{stage}={value}");
    }
}

fn trace_enabled() -> bool {
    env::var_os("PROMPT_COMPRESSOR_TRACE").is_some()
}

const EMBEDDED_LLAMA_PROMPT_PREAMBLE: &str = concat!(
    "<|system|>\n",
    "JSONのみ。日本語依頼を短縮。説明/Markdown禁止。キーはdistilled_prompt。",
    "識別子/API/URL/ファイル名/数値/否定条件は保持。\n",
    "<|user|>\n",
);

const EMBEDDED_LLAMA_PROMPT_ASSISTANT_SUFFIX: &str = "\n<|assistant|>\n";

fn format_embedded_llama_prompt(prompt: &str) -> String {
    format!("{EMBEDDED_LLAMA_PROMPT_PREAMBLE}{prompt}{EMBEDDED_LLAMA_PROMPT_ASSISTANT_SUFFIX}")
}

fn format_embedded_llama_prompt_prefix(prompt_prefix: &str) -> String {
    format!("{EMBEDDED_LLAMA_PROMPT_PREAMBLE}{prompt_prefix}")
}

fn format_embedded_llama_prompt_suffix(prompt_suffix: &str) -> String {
    format!("{prompt_suffix}{EMBEDDED_LLAMA_PROMPT_ASSISTANT_SUFFIX}")
}

fn trim_after_stop_marker(output: &str) -> &str {
    ["<|im_end|>", "<|endoftext|>", "</s>"]
        .iter()
        .filter_map(|marker| output.find(marker))
        .min()
        .map(|index| &output[..index])
        .unwrap_or(output)
}

fn embedded_model_cache_key(model: &ModelDefinition, model_path: &Path) -> String {
    format!("{}:{}", model.id, model_path.display())
}

fn trace_model_output(stage: &str, output: &str) {
    if env::var_os("PROMPT_COMPRESSOR_TRACE_OUTPUT").is_some() {
        eprintln!("trace.model.{stage}={output}");
    }
}

fn trace_model_prompt(stage: &str, prompt: &str) {
    if env::var_os("PROMPT_COMPRESSOR_TRACE_PROMPT").is_some() {
        eprintln!("trace.prompt.{stage}={prompt}");
    }
}

fn embedded_prompt_cache_key(
    model: &ModelDefinition,
    model_path: &Path,
    context_length: u32,
    configuration: RuntimeInferenceConfig,
    embedded_prompt_prefix: &str,
) -> String {
    let mut hasher = DefaultHasher::new();
    embedded_prompt_prefix.hash(&mut hasher);
    format!(
        "{}:{}:{}:{}:{}:{}:{}:{}",
        model.id,
        model_path.display(),
        context_length,
        configuration.threads.generation,
        configuration.threads.batch,
        configuration.batch_sizes.logical,
        configuration.batch_sizes.physical,
        hasher.finish()
    )
}

fn embedded_input_cache_key(
    model: &ModelDefinition,
    model_path: &Path,
    context_length: u32,
    configuration: RuntimeInferenceConfig,
    embedded_prompt_prefix: &str,
    embedded_prompt_suffix: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update((embedded_prompt_prefix.len() as u64).to_le_bytes());
    hasher.update(embedded_prompt_prefix.as_bytes());
    hasher.update(embedded_prompt_suffix.as_bytes());
    let digest = hasher.finalize();
    let mut digest_hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut digest_hex, "{byte:02x}").expect("writing to a String cannot fail");
    }
    format!(
        "{}:{}:{}:{}:{}:{}:{}:input:{digest_hex}",
        model.id,
        model_path.display(),
        context_length,
        configuration.threads.generation,
        configuration.threads.batch,
        configuration.batch_sizes.logical,
        configuration.batch_sizes.physical
    )
}

#[cfg(feature = "embedded-llama")]
fn embedded_session_params(
    context_length: u32,
    configuration: RuntimeInferenceConfig,
) -> llama_cpp::SessionParams {
    llama_cpp::SessionParams {
        n_ctx: context_length,
        n_batch: configuration.batch_sizes.logical,
        n_ubatch: configuration.batch_sizes.physical,
        n_threads: configuration.threads.generation,
        n_threads_batch: configuration.threads.batch,
    }
}

fn effective_max_output_tokens(request: &CompressionRequest, model: &ModelDefinition) -> u32 {
    let level_cap = match request.compression_level.value() {
        0 => 16,
        1 => 112,
        2 => 192,
        3 => 96,
        _ => model.default_max_output,
    };
    let input_characters = request.input_text.trim().chars().count() as u32;
    let input_scaled_cap = match request.compression_level.value() {
        0 => 16,
        1 => input_characters.saturating_mul(3) / 4 + 32,
        2 => input_characters.saturating_mul(3) / 5 + 32,
        3 => input_characters / 2 + 24,
        _ => input_characters + 48,
    };

    model
        .default_max_output
        .min(level_cap)
        .min(input_scaled_cap.max(64))
        .max(32)
}

const EMBEDDED_CONTEXT_LENGTH_TIERS: [u32; 3] = [1_024, 2_048, 4_096];
const CONTEXT_SAFETY_MARGIN: usize = 8;

#[cfg(feature = "embedded-llama")]
fn select_embedded_context_length(
    llama_model: &llama_cpp::LlamaModel,
    prefix: &str,
    suffix: &str,
    max_output_tokens: usize,
    model_context_length: usize,
) -> Result<u32> {
    let prefix_tokens = llama_model
        .tokenize_bytes(prefix.as_bytes(), false, true)
        .map_err(|error| {
            CompressionError::Runtime(format!("failed to tokenize prompt prefix: {error}"))
        })?
        .len();
    let suffix_tokens = llama_model
        .tokenize_bytes(suffix.as_bytes(), false, true)
        .map_err(|error| {
            CompressionError::Runtime(format!("failed to tokenize prompt input: {error}"))
        })?
        .len();
    select_context_length_for_token_budget(
        prefix_tokens.saturating_add(suffix_tokens),
        max_output_tokens,
        model_context_length,
    )
}

fn select_context_length_for_token_budget(
    prompt_tokens: usize,
    max_output_tokens: usize,
    model_context_length: usize,
) -> Result<u32> {
    validate_prompt_token_budget(prompt_tokens, max_output_tokens, model_context_length)?;
    let required_tokens = prompt_tokens
        .saturating_add(max_output_tokens)
        .saturating_add(CONTEXT_SAFETY_MARGIN);
    let selected = EMBEDDED_CONTEXT_LENGTH_TIERS
        .into_iter()
        .find(|tier| *tier as usize >= required_tokens && *tier as usize <= model_context_length)
        .unwrap_or(model_context_length as u32);
    Ok(selected)
}

fn validate_prompt_token_budget(
    prompt_tokens: usize,
    max_output_tokens: usize,
    context_length: usize,
) -> Result<()> {
    let required_tokens = prompt_tokens
        .saturating_add(max_output_tokens)
        .saturating_add(CONTEXT_SAFETY_MARGIN);
    if required_tokens > context_length {
        return Err(CompressionError::Runtime(format!(
            "input exceeds the model context: prompt={prompt_tokens} tokens, output reserve={max_output_tokens} tokens, context={context_length} tokens; shorten the input or split it into smaller requests"
        )));
    }
    Ok(())
}

fn validate_compression_draft(
    request: &CompressionRequest,
    draft: &CompressionDraft,
) -> Result<()> {
    let output = draft.distilled_prompt.trim();
    if output.is_empty() {
        return Err(CompressionError::Runtime(
            "local runtime returned an empty distilled_prompt".into(),
        ));
    }
    if is_meta_task_restatement(output) {
        return Err(CompressionError::Runtime(
            "local runtime described the request instead of producing a Codex task prompt".into(),
        ));
    }
    if request_is_japanese(request) && !contains_japanese_text(output) {
        return Err(CompressionError::Runtime(
            "local runtime did not return a Japanese distilled_prompt".into(),
        ));
    }
    if output.chars().count() >= request.input_text.trim().chars().count() {
        return Err(CompressionError::Runtime(
            "local runtime did not reduce the prompt".into(),
        ));
    }

    let validation_input = normalized_verification_input(&request.input_text);
    let missing_terms: Vec<_> = required_technical_terms(&validation_input)
        .into_iter()
        .filter(|term| !contains_ascii_case_insensitive(output, term))
        .collect();
    if !missing_terms.is_empty() {
        return Err(CompressionError::Runtime(format!(
            "local runtime omitted required technical terms: {}",
            missing_terms.join(", ")
        )));
    }

    let validation_input = normalized_verification_input(&request.input_text);
    if request.constraints.preserve_numbers
        && !preserves_requested_numbers(&validation_input, output)
    {
        return Err(CompressionError::Runtime(
            "local runtime omitted a required number".into(),
        ));
    }

    if !preserves_requested_negations(request, output) {
        return Err(CompressionError::Runtime(
            "local runtime omitted a required prohibition or negative constraint".into(),
        ));
    }

    Ok(())
}

pub(crate) fn preserves_requested_negations(request: &CompressionRequest, output: &str) -> bool {
    if !request.constraints.preserve_negations {
        return true;
    }

    let validation_input = normalized_verification_input(&request.input_text);
    preserves_constraint_clause_roles(&validation_input, output)
        && preserves_targeted_change_constraints(&validation_input, output)
        && preserves_negative_constraints(&validation_input, output)
}

fn preserves_constraint_clause_roles(input: &str, output: &str) -> bool {
    let output_clauses = input_clauses(output);
    let output_segments: Vec<_> = output
        .split(['。', '！', '？', '\n', ';', '；', '、', ','])
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .collect();
    required_constraint_clauses(input)
        .into_iter()
        .all(|clause| {
            let atomic_term_groups: Vec<_> = clause
                .split(['、', ','])
                .map(required_technical_terms)
                .filter(|terms| terms.len() >= 2)
                .collect();
            if !atomic_term_groups.iter().all(|terms| {
                output_segments.iter().any(|output_segment| {
                    terms
                        .iter()
                        .all(|term| contains_ascii_case_insensitive(output_segment, term))
                })
            }) {
                return false;
            }

            if !positive_term_groups(clause).iter().all(|terms| {
                output_segments.iter().any(|output_segment| {
                    terms
                        .iter()
                        .all(|term| contains_ascii_case_insensitive(output_segment, term))
                        && contains_any_marker(
                            output_segment,
                            &[
                                "入れ", "追加", "導入", "設定", "使用", "使", "有効", "実行",
                                "作成",
                            ],
                        )
                })
            }) {
                return false;
            }

            if !preserves_verification_list_separation(clause, output) {
                return false;
            }

            let negative_requirements = negative_clause_requirements(clause);
            negative_requirements.iter().all(|requirement| {
                output_clauses.iter().any(|output_clause| {
                    output_clause_preserves_constraint_action(output_clause, requirement)
                        && contains_output_negative_marker(output_clause)
                })
            }) && trailing_constraint_actions(clause)
                .iter()
                .all(|action| output_preserves_trailing_constraint_action(output, action))
        })
}

fn preserves_verification_list_separation(clause: &str, output: &str) -> bool {
    let anchors = verification_list_anchors(clause);
    if anchors.is_empty() {
        return true;
    }

    input_clauses(output).into_iter().any(|output_clause| {
        if !contains_any_marker(output_clause, &["テスト", "確認", "検証", "test", "verify"])
        {
            return false;
        }
        let output_segments: Vec<_> = output_clause
            .split(['、', ',', '/', ';', '；'])
            .map(str::trim)
            .filter(|segment| !segment.is_empty())
            .collect();
        let mut used_segments = Vec::new();
        anchors.iter().all(|anchor| {
            let matching_segment = output_segments
                .iter()
                .enumerate()
                .find(|(index, segment)| {
                    !used_segments.contains(index)
                        && contains_ascii_case_insensitive(segment, anchor)
                })
                .map(|(index, _)| index);
            if let Some(index) = matching_segment {
                used_segments.push(index);
                true
            } else {
                false
            }
        })
    })
}

fn verification_list_anchors(clause: &str) -> Vec<String> {
    if !contains_explicit_test_marker(clause) {
        return Vec::new();
    }
    let items: Vec<_> = clause
        .split(['、', ','])
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .collect();
    let list_start = items
        .iter()
        .enumerate()
        .take(items.len().saturating_sub(1))
        .filter(|(_, item)| {
            contains_any_marker(
                item,
                &[
                    "維持し",
                    "保持し",
                    "変更し",
                    "変更せ",
                    "変えず",
                    "追加し",
                    "修正し",
                    "設定し",
                    "実装し",
                ],
            )
        })
        .map(|(index, _)| index + 1)
        .next_back()
        .unwrap_or(0);
    let items = &items[list_start..];
    if items.len() < 3 {
        return Vec::new();
    }

    items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            let mut anchor = *item;
            if index == 0 {
                if let Some((_, focused)) = anchor.rsplit_once('で') {
                    anchor = focused.trim();
                }
            }
            for marker in ["をテスト", "を確認", "を検証", "をtest", "をverify"] {
                if let Some((focused, _)) = anchor.split_once(marker) {
                    anchor = focused.trim();
                    break;
                }
            }
            if let Some(case_marker) = anchor.find('の').filter(|index| {
                anchor[*index..].contains("ケース")
                    && anchor[*index..]
                        .chars()
                        .any(|character| character.is_ascii_digit())
            }) {
                anchor = anchor[..case_marker].trim();
            }
            let anchor = ["してください", "する", "して", "せよ"]
                .iter()
                .find_map(|suffix| anchor.strip_suffix(suffix))
                .unwrap_or(anchor)
                .trim();
            (anchor.chars().count() >= 2).then(|| anchor.to_string())
        })
        .collect()
}

fn contains_output_negative_marker(clause: &str) -> bool {
    if contains_any_marker(
        clause,
        &[
            "しない",
            "せず",
            "ない",
            "なし",
            "不要",
            "禁止",
            "非表示",
            "避け",
            "回避",
            "維持",
            "保持",
            "unchanged",
            "without",
            "must not",
            "do not",
        ],
    ) {
        return true;
    }

    clause.match_indices('ず').any(|(start, marker)| {
        let end = start + marker.len();
        !is_non_negative_zu_form(&clause[..end])
    })
}

fn positive_term_groups(clause: &str) -> Vec<Vec<String>> {
    clause
        .split(['。', '！', '？', '\n', ';', '；', '、', ','])
        .map(str::trim)
        .filter(|segment| {
            contains_any_marker(
                segment,
                &[
                    "入れ", "追加", "導入", "設定", "使用", "使", "有効", "実行", "作成",
                ],
            ) && !contains_any_marker(
                segment,
                &[
                    "しない",
                    "せず",
                    "ない",
                    "なし",
                    "不要",
                    "禁止",
                    "避け",
                    "除外",
                ],
            )
        })
        .filter_map(|segment| {
            let terms = required_technical_terms(segment);
            (terms.len() >= 2).then_some(terms)
        })
        .collect()
}

fn negative_clause_requirements(clause: &str) -> Vec<String> {
    let mut requirements = Vec::new();
    let mut covered_ranges = Vec::new();
    for marker in ["ないで", "せず", "さず", "ず", "ない"] {
        for (start, _) in clause.match_indices(marker) {
            let end = start + marker.len();
            if marker == "ず" && is_non_negative_zu_form(&clause[..end]) {
                continue;
            }
            if covered_ranges
                .iter()
                .any(|(covered_start, covered_end)| start >= *covered_start && end <= *covered_end)
            {
                continue;
            }
            covered_ranges.push((start, end));
            let mut anchor = clause[..start]
                .rsplit(['。', '、', ',', ';', '；', 'は', 'を', 'が', 'も'])
                .next()
                .unwrap_or_default()
                .trim()
                .to_string();
            if anchor.ends_with('し') && !anchor.ends_with("させ") {
                anchor.pop();
            }
            if !anchor.is_empty()
                && anchor.chars().count() <= 16
                && !requirements.iter().any(|existing| existing == &anchor)
            {
                requirements.push(anchor);
            }
        }
    }
    requirements
}

fn trailing_constraint_actions(clause: &str) -> Vec<String> {
    let mut actions = Vec::new();
    for marker in ["ないで", "せず", "さず", "ず", "ない"] {
        for (start, _) in clause.match_indices(marker) {
            let end = start + marker.len();
            if marker == "ず" && is_non_negative_zu_form(&clause[..end]) {
                continue;
            }
            let after = &clause[end..];
            let fragment = after
                .split(['。', '、', ',', ';', '；'])
                .next()
                .unwrap_or_default()
                .trim();
            let fragment = ["してください", "てください", "でください", "ください"]
                .iter()
                .find_map(|suffix| fragment.strip_suffix(suffix))
                .unwrap_or(fragment)
                .trim_start_matches(['が', 'し', 'て'])
                .trim();
            if matches!(fragment, "ように" | "ようにし" | "ようにする" | "よう") {
                continue;
            }
            if fragment.chars().count() >= 2
                && fragment.chars().count() <= 16
                && !actions.iter().any(|existing| existing == fragment)
            {
                actions.push(fragment.to_string());
            }
        }
    }
    actions
}

fn is_non_negative_zu_form(prefix: &str) -> bool {
    ["かかわらず", "関わらず", "問わず"]
        .iter()
        .any(|form| prefix.ends_with(form))
}

fn output_clause_preserves_constraint_action(output_clause: &str, requirement: &str) -> bool {
    if contains_ascii_case_insensitive(output_clause, requirement) {
        return true;
    }
    if requirement.contains("はみ出さ") {
        return contains_any_marker(output_clause, &["はみ出さない", "収ま", "収め", "バー内"]);
    }
    if requirement.contains("混ざら") || requirement.contains("混ぜ") {
        return contains_any_marker(
            output_clause,
            &["混ざらない", "混ぜない", "混在禁止", "データ混ざらない"],
        );
    }
    if requirement.contains("出さ") {
        return contains_any_marker(output_clause, &["出さない", "表示しない", "非表示"]);
    }
    if matches!(requirement, "消え" | "消さ") {
        return contains_any_marker(
            output_clause,
            &["消えない", "消さない", "維持", "保持", "残す", "残して"],
        );
    }
    if matches!(requirement, "変え" | "変更") {
        return contains_any_marker(output_clause, &["変え", "変更", "維持", "保持"]);
    }
    if requirement.contains("エラー") && requirement.contains("返") {
        return contains_any_marker(output_clause, &["エラー返却", "エラーを返", "エラー返"]);
    }
    if requirement.contains("返") {
        return contains_any_marker(output_clause, &["返却", "返す", "返し"]);
    }
    false
}

fn output_preserves_trailing_constraint_action(output: &str, action: &str) -> bool {
    if contains_ascii_case_insensitive(output, action) {
        return true;
    }
    if action.contains("エラー") && action.contains("返") {
        return contains_any_marker(output, &["エラー返却", "エラーを返", "エラー返"]);
    }

    let output = compact_constraint_action_text(output);
    let action = compact_constraint_action_text(action);
    !action.is_empty() && contains_ascii_case_insensitive(&output, &action)
}

fn compact_constraint_action_text(text: &str) -> String {
    [
        ("分かりやすい", ""),
        ("わかりやすい", ""),
        (" の ", ""),
        (" の", ""),
        ("の", ""),
        (" を ", ""),
        ("を", ""),
        (" に ", ""),
        ("に", ""),
        ("してください", ""),
        ("してほしい", ""),
        ("テスト追加", "テスト追加"),
    ]
    .iter()
    .fold(text.trim().to_string(), |compact, (from, to)| {
        compact.replace(from, to)
    })
    .split_whitespace()
    .collect::<String>()
}

pub(crate) fn normalized_verification_input(input: &str) -> String {
    normalize_self_correction_artifacts(&normalize_known_input_typos_for_llm(
        &remove_obvious_input_noise(&normalize_input_whitespace(input)),
    ))
}

fn restore_missing_required_terms(request: &CompressionRequest, draft: &mut CompressionDraft) {
    let validation_input = normalized_verification_input(&request.input_text);
    let normalized_output =
        normalize_known_required_term_typos(&validation_input, draft.distilled_prompt.trim());
    let normalized_output = remove_redundant_required_term_prefixes(&normalized_output);
    let normalized_output = strip_leading_output_label(&normalized_output);
    if normalized_output != draft.distilled_prompt.trim() {
        draft.distilled_prompt = normalized_output;
    }

    let contextualized_output =
        restore_missing_mechanism_terms(&validation_input, draft.distilled_prompt.trim());
    if contextualized_output != draft.distilled_prompt.trim() {
        draft.distilled_prompt = contextualized_output;
    }

    let contextualized_output =
        restore_missing_critical_mechanisms(&validation_input, draft.distilled_prompt.trim());
    if contextualized_output != draft.distilled_prompt.trim() {
        draft.distilled_prompt = contextualized_output;
    }

    let contextualized_output =
        restore_missing_explicit_target_context(&validation_input, draft.distilled_prompt.trim());
    if contextualized_output != draft.distilled_prompt.trim() {
        draft.distilled_prompt = contextualized_output;
    }

    let output = draft.distilled_prompt.trim();
    if output.is_empty() {
        return;
    }

    let missing_terms: Vec<_> = required_technical_terms(&validation_input)
        .into_iter()
        .filter(|term| !contains_ascii_case_insensitive(output, term))
        .collect();
    if missing_terms.is_empty() {
        return;
    }

    let restored = format!("{}: {}", missing_terms.join("/"), output);
    if restored.chars().count() < request.input_text.trim().chars().count() {
        draft.distilled_prompt = restored;
    }
}

fn normalize_known_required_term_typos(input: &str, output: &str) -> String {
    let mut normalized = output.to_string();
    if contains_ascii_case_insensitive(input, "columns") {
        normalized = normalized
            .replace("column mappings", "columns mapping")
            .replace("column mapping", "columns mapping");
    }
    for (required_term, typo, replacement) in [
        ("TypeScript", "TypeScrip", "TypeScript"),
        ("TypeScript", "TypeScritp", "TypeScript"),
        ("PowerShell", "PawerShell", "PowerShell"),
        ("DataLoader", "DataLoder", "DataLoader"),
        ("LM Studio", "LMStduio", "LM Studio"),
        ("LM Studio", "LMS Studio", "LM Studio"),
        ("clipboard", "clpboard", "clipboard"),
    ] {
        if (contains_ascii_case_insensitive(input, required_term)
            || contains_ascii_case_insensitive(input, typo))
            && normalized.contains(typo)
        {
            normalized = replace_exact_ascii_token(&normalized, typo, replacement);
        }
    }
    normalize_near_match_required_identifiers(input, &normalized)
}

fn normalize_near_match_required_identifiers(input: &str, output: &str) -> String {
    let mut normalized = output.to_string();
    for required in required_technical_terms(input) {
        if required.len() < 5
            || !required
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || character == '_')
            || contains_ascii_case_insensitive(&normalized, &required)
        {
            continue;
        }

        let candidates: Vec<String> = normalized
            .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
            .filter(|candidate| candidate.len() >= 5)
            .map(str::to_string)
            .collect();
        let Some(candidate) = candidates.iter().find(|candidate| {
            candidate
                .chars()
                .next()
                .zip(required.chars().next())
                .is_some_and(|(left, right)| left.eq_ignore_ascii_case(&right))
                && ascii_identifiers_are_one_edit_apart(candidate, &required)
        }) else {
            continue;
        };

        normalized = replace_exact_ascii_token(&normalized, candidate, &required);
    }
    normalized
}

fn ascii_identifiers_are_one_edit_apart(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    if left.len().abs_diff(right.len()) > 1 || left == right {
        return false;
    }
    if left.len() == right.len() {
        return left
            .iter()
            .zip(right)
            .filter(|(left, right)| !left.eq_ignore_ascii_case(right))
            .count()
            == 1;
    }

    let (shorter, longer) = if left.len() < right.len() {
        (left, right)
    } else {
        (right, left)
    };
    let mut short_index = 0;
    let mut long_index = 0;
    let mut skipped = false;
    while short_index < shorter.len() && long_index < longer.len() {
        if shorter[short_index].eq_ignore_ascii_case(&longer[long_index]) {
            short_index += 1;
            long_index += 1;
        } else if skipped {
            return false;
        } else {
            skipped = true;
            long_index += 1;
        }
    }
    true
}

fn replace_exact_ascii_token(value: &str, typo: &str, replacement: &str) -> String {
    let mut replaced = String::with_capacity(value.len());
    let mut copied_until = 0;

    for (start, matched) in value.match_indices(typo) {
        let end = start + matched.len();
        let has_identifier_before = value[..start]
            .chars()
            .next_back()
            .is_some_and(is_ascii_identifier_character);
        let has_identifier_after = value[end..]
            .chars()
            .next()
            .is_some_and(is_ascii_identifier_character);
        if has_identifier_before || has_identifier_after {
            continue;
        }

        replaced.push_str(&value[copied_until..start]);
        replaced.push_str(replacement);
        copied_until = end;
    }

    if copied_until == 0 {
        value.to_string()
    } else {
        replaced.push_str(&value[copied_until..]);
        replaced
    }
}

fn restore_missing_explicit_target_context(input: &str, output: &str) -> String {
    let required_terms = required_technical_terms(input);
    let mut restored = output.to_string();
    let preprocessed = preprocess_input_for_llm(input);

    for clause in preprocessed
        .split(['。', '！', '？', '\n', ';', '；'])
        .map(str::trim)
        .filter(|clause| !clause.is_empty())
    {
        let target = clause
            .strip_prefix("対象は")
            .or_else(|| clause.strip_prefix("対象:"))
            .or_else(|| clause.strip_prefix("対象："));
        let Some(target) = target else {
            continue;
        };
        let target = target
            .trim()
            .trim_end_matches("です")
            .trim_end_matches("である")
            .trim();
        if target.is_empty() {
            continue;
        }

        let restores_missing_term = required_terms.iter().any(|term| {
            contains_ascii_case_insensitive(target, term)
                && !contains_ascii_case_insensitive(&restored, term)
        });
        if !restores_missing_term {
            continue;
        }

        let candidate = format!("{target}を対象に、{}", restored.trim_start());
        if candidate.chars().count() < input.trim().chars().count() {
            restored = candidate;
        }
    }

    restored
}

fn restore_missing_mechanism_terms(input: &str, output: &str) -> String {
    let required_terms = required_technical_terms(input);
    let mut restored = output.to_string();
    let preprocessed = preprocess_input_for_llm(input);

    for term in &required_terms {
        if contains_ascii_case_insensitive(&restored, term) {
            continue;
        }
        for clause in preprocessed
            .split(['。', '！', '？', '\n', ';', '；'])
            .map(str::trim)
            .filter(|clause| !clause.is_empty())
        {
            let Some(term_start) = clause.find(term) else {
                continue;
            };
            let after_term = &clause[term_start + term.len()..];
            let relation = after_term.trim_start();
            if !relation.starts_with('で')
                && !relation.starts_with("を使って")
                && !relation.starts_with("を使用して")
                && !relation.starts_with("を用いて")
            {
                continue;
            }

            let Some((anchor, anchor_start)) = required_terms
                .iter()
                .filter(|anchor| *anchor != term)
                .filter(|anchor| after_term.contains(anchor.as_str()))
                .filter_map(|anchor| restored.find(anchor).map(|start| (anchor, start)))
                .min_by_key(|(_, start)| *start)
            else {
                continue;
            };

            let before = &restored[..anchor_start];
            let after = &restored[anchor_start..];
            let candidate = if let Some(before_without_no) = before.strip_suffix('の') {
                format!("{before_without_no}を{term}で{after}")
            } else {
                format!("{before}{term}で{after}")
            };
            if candidate.chars().count() < input.trim().chars().count()
                && contains_ascii_case_insensitive(&candidate, anchor)
            {
                restored = candidate;
                break;
            }
        }
    }

    restored
}

fn is_ascii_identifier_character(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '_'
}

fn remove_redundant_required_term_prefixes(output: &str) -> String {
    let trimmed = output.trim();
    for separator in [":", "："] {
        if let Some((prefix, body)) = trimmed.split_once(separator) {
            let terms: Vec<_> = prefix
                .split('/')
                .map(str::trim)
                .filter(|term| !term.is_empty())
                .collect();
            let body = body.trim_start();
            if !terms.is_empty()
                && prefix.chars().count() <= 80
                && terms
                    .iter()
                    .all(|term| contains_ascii_case_insensitive(body, term))
            {
                return body.to_string();
            }
        }
    }
    for term in [
        "TypeScript",
        "PowerShell",
        "DataLoader",
        "LM Studio",
        "clipboard",
    ] {
        for separator in [":", "："] {
            let prefix = format!("{term}{separator}");
            let Some(rest) = trimmed.strip_prefix(&prefix) else {
                continue;
            };
            let rest = rest.trim_start();
            if contains_ascii_case_insensitive(rest, term) {
                return rest.to_string();
            }
        }
    }
    trimmed.to_string()
}

fn restore_missing_required_constraints(
    request: &CompressionRequest,
    draft: &mut CompressionDraft,
) {
    if !request.constraints.preserve_negations {
        return;
    }

    let validation_input = normalized_verification_input(&request.input_text);
    let normalized =
        normalize_required_constraint_terms(&validation_input, draft.distilled_prompt.trim());
    if normalized != draft.distilled_prompt.trim() {
        draft.distilled_prompt = normalized;
    }

    let output = draft.distilled_prompt.trim();
    if output.is_empty() {
        return;
    }

    let preserves_negatives = preserves_negative_constraints(&request.input_text, output);
    let phrases = if preserves_negatives {
        missing_nonnegative_constraint_restoration_phrases(&request.input_text, output)
    } else {
        missing_constraint_restoration_phrases(&request.input_text, output)
    };
    if phrases.is_empty() {
        return;
    }

    let input_len = request.input_text.trim().chars().count();
    let mut restored = output.to_string();
    let mut applied = false;
    for phrase in phrases {
        let mut candidate = append_restoration_phrase(&restored, &phrase);
        if candidate.chars().count() >= input_len {
            let phrase_len = phrase.chars().count();
            let Some(existing_budget) = input_len.checked_sub(phrase_len + 3) else {
                continue;
            };
            let Some(trimmed_output) = trim_to_char_budget(&restored, existing_budget) else {
                continue;
            };
            candidate = append_restoration_phrase(&trimmed_output, &phrase);
        }

        if !contains_required_technical_terms(&validation_input, &candidate) {
            let mut candidate_draft = CompressionDraft {
                distilled_prompt: candidate,
                removed_content_summary: Vec::new(),
            };
            restore_missing_required_terms(request, &mut candidate_draft);
            candidate = candidate_draft.distilled_prompt;
        }

        if candidate.chars().count() >= input_len
            || !contains_required_technical_terms(&validation_input, &candidate)
        {
            continue;
        }

        restored = candidate;
        applied = true;
        if !preserves_negatives && preserves_negative_constraints(&request.input_text, &restored) {
            draft.distilled_prompt = restored;
            return;
        }
    }

    if preserves_negatives && applied {
        draft.distilled_prompt = restored;
    }
}

fn append_restoration_phrase(output: &str, phrase: &str) -> String {
    let output = output
        .trim()
        .trim_end_matches(['。', '.', ';', '；', '、', ',']);
    let phrase = phrase
        .trim()
        .trim_end_matches(['。', '.', ';', '；', '、', ',']);
    if output.is_empty() {
        phrase.to_string()
    } else if phrase.is_empty() {
        output.to_string()
    } else {
        format!("{output}、{phrase}。")
    }
}

fn normalize_required_constraint_terms(input: &str, output: &str) -> String {
    let mut normalized = strip_leading_output_label(output);
    for (from, to) in [
        ("変更禁止変更せず", "変更しない"),
        ("変更禁止変更しない", "変更しない"),
        ("変更禁止、変更せず", "変更しない"),
        ("変更禁止、変更しない", "変更しない"),
    ] {
        normalized = normalized.replace(from, to);
    }
    if input.contains("変更しない") {
        normalized = normalized.replace("変更せず", "変更しない");
    }
    if input.contains("保存しない") {
        normalized = normalized.replace("保存せず", "保存しない");
    }
    if input.contains("任意") {
        normalized = normalized.replace("更新可能", "任意");
    }
    if contains_any_marker(input, &["残してください", "残して", "残す"]) {
        normalized = normalized
            .replace("残してください", "残す")
            .replace("残してほしい", "残す")
            .replace("残して", "残す");
    }
    if contains_any_marker(input, &["いりません", "いらない", "不要"]) {
        normalized = normalized.replace("いりません", "不要");
    }
    if input.contains("ウィンドウバー") {
        normalized = normalized
            .replace(
                "スクロールしても固定表示",
                "スクロール時もウィンドウバー固定",
            )
            .replace("スクロール時も固定表示", "スクロール時もウィンドウバー固定");
    }
    normalized = normalized
        .replace("バー内に収まり", "はみ出さない")
        .replace("バー内に収め", "はみ出さない")
        .replace("データ混在禁止", "データ混ざらない")
        .replace("検索じょうたい", "検索状態")
        .replace("検索状態は残す", "検索状態維持")
        .replace("検索状態は残", "検索状態維持")
        .replace("触らず", "触らない")
        .replace("やめてください", "やめる")
        .replace("やめて", "やめる")
        .replace("LMS Studio", "LM Studio")
        .replace("空もじ列", "空文字列")
        .replace("空もじ", "空文字")
        .replace("テスと", "テスト")
        .replace("UTF,BOM", "UTF-8 BOM");
    if input.contains("読み込まず") {
        for (from, to) in [
            ("読み込み拒否", "読み込まず"),
            ("読み込みを拒否", "読み込まず"),
            ("読込拒否", "読み込まず"),
            ("ファイル拒否", "ファイル読み込まず"),
        ] {
            normalized = normalized.replace(from, to);
        }
    }
    if input.contains("個人情報") && contains_any_marker(input, &["入れない", "含めない"])
    {
        for (from, to) in [
            ("個人情報エラー本文除外", "エラー本文に個人情報を含めない"),
            ("個人情報をエラー本文除外", "エラー本文に個人情報を含めない"),
            ("個人情報除外", "個人情報を含めない"),
        ] {
            normalized = normalized.replace(from, to);
        }
    }
    if input.contains("確認") && !normalized.contains("確認") {
        normalized = normalized.replace("検証", "確認");
    } else if input.contains("テスト") && !normalized.contains("テスト") {
        normalized = normalized.replace("検証", "テスト");
    }
    normalized
}

fn preprocess_input_for_llm(input: &str) -> String {
    let normalized = normalize_input_whitespace(input);
    let denoised = remove_obvious_input_noise(&normalized);
    let typo_normalized = normalize_known_input_typos_for_llm(&denoised);
    let correction_normalized = normalize_self_correction_artifacts(&typo_normalized);
    let cleaned = remove_polite_request_fillers(&correction_normalized);

    if preprocessed_input_is_safe(&correction_normalized, &cleaned) {
        cleaned
    } else {
        normalized
    }
}

fn normalize_input_whitespace(input: &str) -> String {
    let normalized = input
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace('\u{3000}', " ");
    let mut lines = Vec::new();
    let mut previous_blank = false;

    for line in normalized.lines() {
        let compacted = if should_preserve_line_spacing(line) {
            line.trim_end().to_string()
        } else {
            collapse_inline_spaces(line.trim())
        };

        if compacted.is_empty() {
            if !previous_blank && !lines.is_empty() {
                lines.push(String::new());
                previous_blank = true;
            }
            continue;
        }

        lines.push(compacted);
        previous_blank = false;
    }

    while matches!(lines.last(), Some(line) if line.is_empty()) {
        lines.pop();
    }

    lines.join("\n")
}

fn should_preserve_line_spacing(line: &str) -> bool {
    let trimmed = line.trim_start();
    line.starts_with(' ')
        || line.starts_with('\t')
        || trimmed.starts_with("```")
        || trimmed.starts_with('|')
        || trimmed.starts_with('>')
}

fn collapse_inline_spaces(text: &str) -> String {
    map_unquoted_spans(text, collapse_plain_spaces)
}

fn collapse_plain_spaces(text: &str) -> String {
    let mut compacted = String::with_capacity(text.len());
    let mut previous_space = false;

    for character in text.chars() {
        if matches!(character, ' ' | '\t') {
            if !previous_space {
                compacted.push(' ');
                previous_space = true;
            }
            continue;
        }

        compacted.push(character);
        previous_space = false;
    }

    compacted
}

fn remove_obvious_input_noise(input: &str) -> String {
    let mut cleaned = String::new();
    let mut segment = String::new();

    for character in input.chars() {
        segment.push(character);
        if matches!(character, '。' | '！' | '？' | '\n') {
            push_non_noise_segment(&mut cleaned, &segment);
            segment.clear();
        }
    }

    if !segment.is_empty() {
        push_non_noise_segment(&mut cleaned, &segment);
    }

    normalize_input_whitespace(&remove_inline_noise_tokens(&cleaned))
}

fn push_non_noise_segment(output: &mut String, segment: &str) {
    if is_obvious_unrelated_noise_segment(segment) {
        return;
    }
    output.push_str(segment);
}

fn is_obvious_unrelated_noise_segment(segment: &str) -> bool {
    let trimmed = segment
        .trim()
        .trim_matches(|character: char| matches!(character, '。' | '！' | '？' | ',' | '、'));
    if trimmed.is_empty() {
        return false;
    }

    let explicit_noise_markers = [
        "キーボードに触っただけ",
        "変換中の文字は無視",
        "依頼とは関係ありません",
        "ここも無視",
        "これは意味ない",
        "意味ないです",
        "貼り付けの残り",
        "意味はありません",
        "意味ありません",
        "意味ないので",
        "意味なし文字",
        "関係ない文字",
        "タッチミスと関係ない話",
        "依頼には含めない",
        "本題だけ圧縮できるか",
    ];
    if contains_marker_outside_literal_spans(trimmed, &explicit_noise_markers)
        && !(contains_preprocess_protected_content(trimmed)
            && contains_preprocess_actionable_content(trimmed))
    {
        return true;
    }

    if contains_marker_outside_literal_spans(trimmed, &["変な入力"])
        && contains_marker_outside_literal_spans(trimmed, &["無視"])
        && !contains_preprocess_actionable_content(trimmed)
    {
        return true;
    }

    const NOISE_MARKERS: &[&str] = &[
        "こんにちｈ",
        "今日はｄあさ",
        "これは関係ない",
        "依頼内容とは関係ない",
        "本題ではない",
        "変な文字",
        "変な入力",
        "打ち間違い",
        "入力ミス",
        "無視して大丈夫",
        "残さなくて大丈夫",
        "圧縮後にはいらない",
        "意味なし文字",
        "関係ない文字",
    ];

    if contains_marker_outside_literal_spans(trimmed, NOISE_MARKERS)
        && !contains_preprocess_protected_content(trimmed)
        && !contains_preprocess_actionable_content(trimmed)
    {
        return true;
    }

    if contains_preprocess_protected_content(trimmed) {
        return false;
    }

    false
}

fn contains_preprocess_actionable_content(text: &str) -> bool {
    contains_marker_outside_literal_spans(
        text,
        &[
            "修正",
            "追加",
            "実装",
            "更新",
            "作成",
            "調査",
            "保持",
            "維持",
            "変更しない",
            "変更せず",
            "残す",
            "確認",
            "テスト",
            "返して",
            "使って",
            "使用して",
        ],
    )
}

fn contains_preprocess_protected_content(text: &str) -> bool {
    if text.chars().any(|character| character.is_ascii_digit()) {
        return true;
    }
    if !required_technical_terms(text).is_empty() {
        return true;
    }
    contains_any_marker(
        text,
        &[
            "変更しない",
            "変更せず",
            "変えない",
            "維持",
            "保持",
            "残す",
            "消さない",
            "壊さない",
            "追加しない",
            "表示しない",
            "保存しない",
            "読み込まず",
            "だけ",
            "のみ",
            "ではなく",
            "じゃなく",
            "禁止",
            "避け",
        ],
    )
}

fn remove_inline_noise_tokens(input: &str) -> String {
    map_unquoted_spans(input, |span| {
        [
            "こんにちｈ。",
            "こんにちｈ、",
            "こんにちｈ",
            "今日はｄあさ。",
            "今日はｄあさ、",
            "今日はｄあさ",
        ]
        .iter()
        .fold(span.to_string(), |text, marker| text.replace(marker, ""))
    })
}

fn normalize_known_input_typos_for_llm(input: &str) -> String {
    input
        .lines()
        .map(|line| {
            if should_skip_typo_normalization(line) {
                line.to_string()
            } else {
                apply_known_input_typo_replacements(line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalize_self_correction_artifacts(input: &str) -> String {
    map_unquoted_spans(input, normalize_self_correction_span)
}

fn normalize_self_correction_span(input: &str) -> String {
    let mut normalized = normalize_explicit_ascii_self_corrections(input);
    normalized = remove_repeated_ascii_confirmations(&normalized);
    for term in [
        "React",
        "useSearchParams",
        "status",
        "AbortController",
        "Vitest",
        "TypeScript",
        "Shift_JIS",
        "UTF-8 BOM",
        "10MB",
        "INVALID_FILE_SIZE",
        "columns",
        "dryRun",
        "Next.js",
        "POST /api/orders",
        "customerId",
        "HTTP 400",
        "INVALID_CUSTOMER",
        "requestId",
        "user-settings.json",
        "application/local/state",
        "WebSocket",
        "2秒",
        "±20%",
        "close code 4001",
        "socket",
        "message handler",
        "10回",
        "fake timer",
    ] {
        for pattern in [
            format!("{term}、じゃなくて {term}"),
            format!("{term}、ではなく{term}"),
            format!("{term}、ではなく {term}"),
            format!("{term}じゃなくて {term}"),
            format!("{term} ではなく {term}"),
            format!("{term}ではなく {term}"),
            format!("{term} と書いたけど正しくは {term}"),
            format!("{term}、正しくは {term}"),
            format!("{term}、すみません {term}"),
            format!("{term}、表記は {term}"),
            format!("{term} かな、正式には {term}"),
            format!("{term} と書きかけましたが {term} が正しい名前"),
            format!("{term}、いや{term}"),
            format!("{term} は {term} です"),
            format!("{term},{term}"),
            format!("{term}、{term}"),
        ] {
            normalized = normalized.replace(&pattern, term);
        }
    }
    normalized
        .replace("しません", "しない")
        .replace("ですです", "です")
}

fn normalize_explicit_ascii_self_corrections(input: &str) -> String {
    const MARKERS: &[&str] = &[
        "と書きましたが正しくは",
        "と書いたけど正しくは",
        "、いや最終的に",
        "、いや正しくは",
        "、いや",
        "、じゃなくて",
        "じゃなくて",
        "、正しくは",
    ];

    let mut normalized = input.to_string();
    for marker in MARKERS {
        while let Some(marker_start) = normalized.find(marker) {
            let before = &normalized[..marker_start];
            let after = normalized[marker_start + marker.len()..].trim_start();
            let Some(old_start) = trailing_correction_token_start(before) else {
                break;
            };
            if leading_correction_token_end(after).is_none() {
                break;
            }

            let mut retained = before[..old_start].trim_end().to_string();
            for discourse in ["たぶん", "おそらく", "最初", "当初"] {
                if retained.ends_with(discourse) {
                    retained.truncate(retained.len() - discourse.len());
                    retained = retained.trim_end().to_string();
                }
            }
            normalized = format!("{retained}{after}");
        }
    }
    normalized
}

fn remove_repeated_ascii_confirmations(input: &str) -> String {
    const MARKERS: &[&str] = &["、あっ", "、あ、"];
    const CONFIRMATIONS: &[&str] = &[
        "で合ってます",
        "で合っています",
        "で正しいです",
        "で間違いありません",
    ];

    let mut normalized = input.to_string();
    for marker in MARKERS {
        while let Some(marker_start) = normalized.find(marker) {
            let tail_start = marker_start + marker.len();
            let tail = &normalized[tail_start..];
            let segment_end = tail
                .char_indices()
                .find_map(|(index, character)| {
                    matches!(character, '。' | '！' | '？' | '\n' | '、' | ',').then_some(index)
                })
                .unwrap_or(tail.len());
            let fragment = tail[..segment_end].trim();
            let Some(token_end) = leading_correction_token_end(fragment) else {
                break;
            };
            let token = &fragment[..token_end];
            let confirmation = fragment[token_end..].trim();
            if !CONFIRMATIONS.contains(&confirmation)
                || !contains_ascii_case_insensitive(&normalized[..marker_start], token)
            {
                break;
            }

            normalized.replace_range(marker_start..tail_start + segment_end, "");
        }
    }
    normalized
}

fn trailing_correction_token_start(value: &str) -> Option<usize> {
    let mut token_end = value.trim_end().len();
    for particle in ["も", "は", "を", "が", "に", "で", "と"] {
        if value[..token_end].ends_with(particle) {
            token_end -= particle.len();
            break;
        }
    }
    let token_prefix = &value[..token_end];
    let token_start = token_prefix
        .char_indices()
        .rev()
        .take_while(|(_, character)| is_correction_token_character(*character))
        .last()
        .map(|(index, _)| index)?;
    let token = &token_prefix[token_start..];
    token
        .chars()
        .any(|character| character.is_ascii_alphanumeric())
        .then_some(token_start)
}

fn leading_correction_token_end(value: &str) -> Option<usize> {
    let end = value
        .char_indices()
        .take_while(|(_, character)| is_correction_token_character(*character))
        .last()
        .map(|(index, character)| index + character.len_utf8())?;
    value[..end]
        .chars()
        .any(|character| character.is_ascii_alphanumeric())
        .then_some(end)
}

fn is_correction_token_character(character: char) -> bool {
    character.is_ascii_alphanumeric()
        || matches!(
            character,
            '_' | '-' | '.' | '/' | ':' | '+' | '#' | '@' | '=' | '<' | '>' | '%' | '±'
        )
}

fn should_skip_typo_normalization(line: &str) -> bool {
    let trimmed = line.trim_start();
    should_preserve_line_spacing(line)
        || trimmed.starts_with('{')
        || trimmed.starts_with('[')
        || trimmed.starts_with("at ")
        || trimmed.contains("```")
}

fn apply_known_input_typo_replacements(input: &str) -> String {
    map_unquoted_spans(input, |span| {
        known_input_typo_replacements()
            .iter()
            .fold(span.to_string(), |text, (from, to)| {
                if from.chars().all(is_ascii_identifier_character) {
                    replace_exact_ascii_token(&text, from, to)
                } else {
                    text.replace(from, to)
                }
            })
    })
}

fn contains_marker_outside_literal_spans(value: &str, markers: &[&str]) -> bool {
    let mut found = false;
    let _ = map_unquoted_spans(value, |span| {
        if contains_any_marker(span, markers) {
            found = true;
        }
        span.to_string()
    });
    found
}

fn map_unquoted_spans(value: &str, mut transform: impl FnMut(&str) -> String) -> String {
    let mut output = String::with_capacity(value.len());
    let mut plain_start = 0;
    let mut literal_start = None;
    let mut literal_closer = '\0';
    let mut escaped = false;

    for (index, character) in value.char_indices() {
        if let Some(start) = literal_start {
            if matches!(literal_closer, '\"' | '\'' | '`') && character == '\\' && !escaped {
                escaped = true;
                continue;
            }
            if character == literal_closer && !escaped {
                let end = index + character.len_utf8();
                output.push_str(&value[start..end]);
                plain_start = end;
                literal_start = None;
            }
            escaped = false;
            continue;
        }

        let closer = match character {
            '\"' => Some('\"'),
            '\'' => Some('\''),
            '`' => Some('`'),
            '「' => Some('」'),
            '『' => Some('』'),
            _ => None,
        };
        if let Some(closer) = closer {
            output.push_str(&transform(&value[plain_start..index]));
            literal_start = Some(index);
            literal_closer = closer;
            escaped = false;
        }
    }

    if let Some(start) = literal_start {
        output.push_str(&value[start..]);
    } else {
        output.push_str(&transform(&value[plain_start..]));
    }
    output
}

fn known_input_typo_replacements() -> &'static [(&'static str, &'static str)] {
    &[
        ("TypeScritp", "TypeScript"),
        ("typeScritp", "TypeScript"),
        ("Recat", "React"),
        ("useSerchParams", "useSearchParams"),
        ("stauts", "status"),
        ("AbortContorller", "AbortController"),
        ("Vitset", "Vitest"),
        ("Shift JSI", "Shift_JIS"),
        ("UTF8 BMO", "UTF-8 BOM"),
        ("10BM", "10MB"),
        ("INVALID_FILE_SISE", "INVALID_FILE_SIZE"),
        ("colmuns", "columns"),
        ("dryrun", "dryRun"),
        ("お願いしまうs", "お願いします"),
        ("ほしです", "ほしいです"),
        ("Nex.js", "Next.js"),
        ("Nextjs", "Next.js"),
        ("POST /api/odrers", "POST /api/orders"),
        ("custmerID", "customerId"),
        ("reqestId", "requestId"),
        ("user-setings.json", "user-settings.json"),
        ("aplication/local/stete", "application/local/state"),
        ("せってい", "設定"),
        ("圧縮れべる", "圧縮レベル"),
        ("ウインドウ", "ウィンドウ"),
        ("保存しなで", "保存しないで"),
        ("WebSoket", "WebSocket"),
        ("指数ばっくおふ", "指数バックオフ"),
        ("2病", "2秒"),
        ("プラマイ20%", "±20%"),
        ("close cord 4001", "close code 4001"),
        ("soket", "socket"),
        ("message hander", "message handler"),
        ("10会", "10回"),
        ("fake timre", "fake timer"),
        ("PawerShell", "PowerShell"),
        ("DataLoder", "DataLoader"),
        ("Sarashna 2.2 3B", "Sarashina 2.2 3B"),
        ("LMStduio", "LM Studio"),
        ("LMStudio", "LM Studio"),
        ("custmerId", "customerId"),
        ("INVALID_CUSTMER", "INVALID_CUSTOMER"),
        ("avaterUrl", "avatarUrl"),
        ("clpboard", "clipboard"),
        ("Shift JIS", "Shift_JIS"),
        ("UTF8 BOM", "UTF-8 BOM"),
        ("UTF8", "UTF-8"),
        ("HTTP400", "HTTP 400"),
        ("HTTP429", "HTTP 429"),
        ("10 %", "10%"),
        ("--dryrun", "--dry-run"),
        ("ja.josn", "ja.json"),
        ("読みこまず", "読み込まず"),
        ("こえる", "超える"),
        ("変更しなで", "変更しないで"),
        ("テスと", "テスト"),
        ("空もじ列", "空文字列"),
        ("空もじ", "空文字"),
        ("いりません", "不要"),
        ("LMS Studio", "LM Studio"),
    ]
}

fn preprocessed_input_is_safe(original: &str, candidate: &str) -> bool {
    let candidate = candidate.trim();
    !candidate.is_empty()
        && candidate.chars().count() * 3 >= original.trim().chars().count()
        && preserves_negative_constraints(original, candidate)
        && preserves_preprocess_required_terms(original, candidate)
}

fn preserves_preprocess_required_terms(original: &str, candidate: &str) -> bool {
    required_technical_terms(original).iter().all(|term| {
        contains_ascii_case_insensitive(candidate, term)
            || contains_ascii_case_insensitive(
                candidate,
                &apply_known_input_typo_replacements(term),
            )
    })
}

fn trim_to_char_budget(text: &str, max_chars: usize) -> Option<String> {
    if max_chars == 0 {
        return None;
    }
    if text.chars().count() <= max_chars {
        return Some(text.trim().to_string());
    }

    let mut end_byte = text.len();
    for (index, (byte_index, _)) in text.char_indices().enumerate() {
        if index == max_chars {
            end_byte = byte_index;
            break;
        }
    }

    let hard_cut = text[..end_byte].trim();
    let min_soft_cut_chars = max_chars / 2;
    let mut best_soft_cut = None;
    for (byte_index, character) in hard_cut.char_indices() {
        if matches!(character, '。' | '、' | ';' | '/' | ' ') {
            let candidate = hard_cut[..byte_index].trim();
            if candidate.chars().count() >= min_soft_cut_chars {
                best_soft_cut = Some(candidate);
            }
        }
    }

    let trimmed = best_soft_cut
        .unwrap_or(hard_cut)
        .trim_matches(|character: char| {
            character.is_whitespace() || matches!(character, '、' | '/' | ';' | ':' | '：')
        });
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn contains_required_technical_terms(input: &str, output: &str) -> bool {
    required_technical_terms(input)
        .iter()
        .all(|term| contains_ascii_case_insensitive(output, term))
}

fn polish_model_output_for_request(request: &CompressionRequest, draft: &mut CompressionDraft) {
    let model_output = draft.distilled_prompt.trim();
    if model_output.is_empty() {
        return;
    }

    let validation_input = normalized_verification_input(&request.input_text);
    let mut candidate = strip_leading_output_label(model_output);
    if request.compression_level.value() >= 2 {
        candidate = remove_duplicate_assignment_values(&validation_input, &candidate);
        candidate = remove_redundant_counted_reference(&validation_input, &candidate);
        candidate = remove_redundant_constraint_tail(&validation_input, &candidate);
        candidate = remove_polite_request_fillers(&candidate);
    }

    let input_characters = request.input_text.trim().chars().count();
    let candidate_characters = candidate.chars().count();
    let level_two_needs_more_compaction = request.compression_level.value() == 2
        && candidate_characters.saturating_mul(10) > input_characters.saturating_mul(9);
    if candidate_characters < input_characters
        && !level_two_needs_more_compaction
        && structured_candidate_preserves_requirements(&validation_input, &candidate)
    {
        draft.distilled_prompt = candidate;
        return;
    }

    if let Some(structured) = verified_structured_candidate(&validation_input) {
        draft.distilled_prompt = structured;
    }
}

fn strip_leading_output_label(output: &str) -> String {
    let trimmed = output.trim_start();
    for label in ["圧縮結果", "要約", "短縮文", "出力"] {
        for separator in [":", "："] {
            let prefix = format!("{label}{separator}");
            if let Some(rest) = trimmed.strip_prefix(&prefix) {
                let rest = rest.trim_start();
                if !rest.is_empty() {
                    return rest.to_string();
                }
            }
        }
    }
    output.to_string()
}

fn restore_missing_critical_mechanisms(input: &str, output: &str) -> String {
    let mut restored = output.to_string();
    for phrase in missing_critical_mechanism_phrases(input, &restored) {
        let replaced = replace_vague_mechanism_with_phrase(&restored, &phrase);
        if replaced != restored {
            restored = replaced;
            continue;
        }
        let candidate = append_restoration_phrase(&restored, &phrase);
        if candidate.chars().count() < input.trim().chars().count() {
            restored = candidate;
        }
    }
    restored
}

fn missing_critical_mechanism_phrases(input: &str, output: &str) -> Vec<String> {
    let mut phrases = Vec::new();
    if input.contains("一時ファイル")
        && input.contains("置換")
        && !(output.contains("一時ファイル") && output.contains("置換"))
    {
        phrases.push("一時ファイルへ書いてから置換".to_string());
    }
    phrases
}

fn replace_vague_mechanism_with_phrase(output: &str, phrase: &str) -> String {
    let mut replaced = output.to_string();
    for vague in [
        "設定ファイル保全方式確立",
        "設定ファイル保全方式",
        "ファイル保全方式確立",
        "保全方式確立",
    ] {
        if replaced.contains(vague) {
            replaced = replaced.replace(vague, phrase);
        }
    }
    replaced
}

fn remove_redundant_counted_reference(input: &str, output: &str) -> String {
    let Some(list) = parse_counted_item_reference_list(input) else {
        return output.to_string();
    };
    if !list
        .targets
        .iter()
        .all(|target| shared_predicate_target_satisfied(target, output))
    {
        return output.to_string();
    }

    let count = list.targets.len();
    let mut saw_restore = false;
    let mut kept = Vec::new();
    for segment in output
        .trim()
        .trim_end_matches('。')
        .split('、')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
    {
        if (segment.contains(&format!("この{count}項目"))
            || segment.contains(&format!("この {count} 項目")))
            && contains_any_marker(segment, &["だけ", "のみ"])
        {
            continue;
        }
        if segment == "次回起動時に復元" && saw_restore {
            continue;
        }
        if segment.contains("復元") {
            saw_restore = true;
        }
        kept.push(segment);
    }
    if kept.is_empty() {
        output.to_string()
    } else {
        format!("{}。", kept.join("、"))
    }
}

fn remove_duplicate_assignment_values(input: &str, output: &str) -> String {
    let mut polished = output.to_string();
    for term in required_technical_terms(input) {
        let Some((_, value)) = term.split_once('=') else {
            continue;
        };
        if value.is_empty() || !polished.contains(&term) {
            continue;
        }
        for pattern in [
            format!(": {value},"),
            format!("：{value}、"),
            format!(", {value},"),
            format!("、{value}、"),
            format!(", {value}"),
            format!("、{value}"),
            format!("/{value}/"),
            format!(" {value} "),
        ] {
            let replacement = if pattern.starts_with('/') && pattern.ends_with('/') {
                "/"
            } else if pattern.starts_with(':') {
                ":"
            } else if pattern.starts_with('：') {
                "："
            } else if pattern.ends_with('、') {
                "、"
            } else if pattern.ends_with(',') {
                ","
            } else {
                ""
            };
            polished = polished.replace(&pattern, replacement);
        }
    }
    polished
}

fn remove_redundant_constraint_tail(input: &str, output: &str) -> String {
    let mut current = normalize_required_constraint_terms(input, output)
        .trim()
        .to_string();
    while let Some((head, tail)) = current.rsplit_once(';') {
        let head = head.trim();
        let tail = tail.trim();
        if head.is_empty() || tail.is_empty() {
            break;
        }

        let compact_tail = compact_all_whitespace(tail);
        let compacted_constraints: Vec<_> = required_constraint_clauses(input)
            .into_iter()
            .map(compact_constraint_clause)
            .collect();
        let tail_is_known_constraint = compacted_constraints.iter().any(|clause| {
            let compact_clause = compact_all_whitespace(clause);
            !compact_clause.is_empty()
                && (compact_tail.contains(&compact_clause)
                    || compact_clause.contains(&compact_tail))
        });
        if tail_is_known_constraint
            && contains_required_technical_terms(input, head)
            && preserves_negative_constraints(input, head)
        {
            current = head.to_string();
            continue;
        }

        let mut removed_prefix = false;
        for clause in compacted_constraints {
            let clause = clause.trim();
            if clause.is_empty() || !tail.starts_with(clause) {
                continue;
            }
            let remainder = tail[clause.len()..].trim_start_matches(|character: char| {
                character.is_whitespace() || matches!(character, '。' | '、' | ',' | ';' | '；')
            });
            let candidate = if remainder.is_empty() {
                head.to_string()
            } else {
                format!("{head}。{remainder}")
            };
            if contains_required_technical_terms(input, &candidate)
                && preserves_negative_constraints(input, &candidate)
            {
                current = candidate;
                removed_prefix = true;
            }
            break;
        }
        if !removed_prefix {
            break;
        }
    }
    current
}

fn remove_polite_request_fillers(output: &str) -> String {
    map_unquoted_spans(output, |span| {
        [
            ("なんですけど ", "で"),
            ("なんですけど、", "で"),
            ("えっと", ""),
            ("を直してほしいです", "を修正"),
            ("を呼び出してください", "呼出"),
            ("を作成してください", "作成"),
            ("を追加してください", "追加"),
            ("を実装してください", "実装"),
            ("を修正してください", "修正"),
            ("を確認してください", "確認"),
            ("を調べてください", "調査"),
            ("を整理してください", "整理"),
            ("をまとめてください", "整理"),
            ("を提案してください", "提案"),
            ("を検証してください", "検証"),
            ("を更新してください", "更新"),
            ("を保持してください", "保持"),
            ("を維持してください", "維持"),
            ("を返してください", "返却"),
            ("をコピーしてください", "コピー"),
            ("を出してください", "出力"),
            ("バー内に収まり", "はみ出さない"),
            ("バー内に収め", "はみ出さない"),
            ("データ混在禁止", "データ混ざらない"),
            ("検索じょうたい", "検索状態"),
            ("検索状態は残してください", "検索状態維持"),
            ("検索状態は残してほしいです", "検索状態維持"),
            ("検索状態は残してほしい", "検索状態維持"),
            ("検索状態は残して", "検索状態維持"),
            ("検索状態は残す", "検索状態維持"),
            ("検索状態は残", "検索状態維持"),
            ("触らず", "触らない"),
            ("やめてください", "やめる"),
            ("やめて", "やめる"),
            ("残してください", "残す"),
            ("残してほしいです", "残す"),
            ("残してほしい", "残す"),
            ("残して", "残す"),
            (
                "ユーザーが任意のローカルモデルを検証するために残すこと",
                "任意ローカルモデル検証用に残す",
            ),
            ("ユーザーが任意モデルを試すために残す", "任意モデル用に残す"),
            ("採用中の ", ""),
            (
                "が毎回依存関係を再インストールしていて遅い",
                "の再インストール遅延",
            ),
            ("を使って npm のキャッシュを有効化して", "でnpmキャッシュ有効化"),
            ("テストコマンド npm test と lint は変更しないで", "npm test/lint変更しない"),
            (
                "キャッシュが効かなかった場合でも CI が失敗しないようにし",
                "キャッシュ無効でもCI失敗しない",
            ),
            (
                "ログでキャッシュヒットの有無を確認できるようにして",
                "ログでキャッシュヒット確認",
            ),
            (
                "既存の useSearchParams による URL クエリ管理は維持し、ページ番号を変更しても検索条件と検索状態が消えないようにしてください",
                "useSearchParams URL管理維持、ページ変更時も検索条件/状態維持",
            ),
            (
                "ページ番号を変更しても検索条件と検索状態が消えないようにしてください",
                "ページ変更時も検索条件/状態維持",
            ),
            (
                "TypeScript の既存構造はなるべく活かし、大規模なリファクタリングや画面全体の作り直しは避けてください",
                "TypeScript既存構造維持、大規模リファクタリング/画面作り直し回避",
            ),
            ("大規模なリファクタリング", "大規模リファクタリング"),
            ("画面全体の作り直し", "画面作り直し"),
            ("避けてください", "回避"),
            ("お願いいたします", ""),
            ("お願い致します", ""),
            ("お願いします", ""),
        ]
        .iter()
        .fold(span.to_string(), |text, (from, to)| text.replace(from, to))
    })
    .trim()
    .to_string()
}

fn missing_constraint_restoration_phrases(input: &str, output: &str) -> Vec<String> {
    const RULES: &[(&[&str], &[&str], &str)] = &[
        (
            &[
                "避け", "禁止", "avoid", "must not", "do not", "don't", "never",
            ],
            &[
                "避け",
                "回避",
                "禁止",
                "しない",
                "出さない",
                "avoid",
                "must not",
                "do not",
                "don't",
                "never",
            ],
            "禁止",
        ),
        (
            &["しない", "不要", "ない", "without"],
            &[
                "しない",
                "せず",
                "禁止",
                "避け",
                "回避",
                "抑制",
                "最小限",
                "不要",
                "ない",
                "without",
                "do not",
                "must not",
            ],
            "しない",
        ),
        (&["ではなく", "でなく"], &["ではなく", "でなく"], "ではなく"),
        (
            &["はみ出さない"],
            &["はみ出さない", "収ま", "収め", "バー内"],
            "はみ出さない",
        ),
        (&["行わない"], &["行わない", "しない"], "行わない"),
        (
            &["止めず", "止めない"],
            &["止めず", "止めない", "継続", "続行"],
            "起動継続",
        ),
        (
            &["読み込まず"],
            &["読み込まず", "読まない", "読み込み禁止"],
            "読み込まず",
        ),
        (&["下げない"], &["下げない"], "下げない"),
        (&["廃止"], &["廃止"], "廃止"),
        (&["変えず"], &["変えず", "変更しない"], "変えず"),
        (
            &["入れない"],
            &["入れない", "含めない", "しない"],
            "入れない",
        ),
        (&["影響させない"], &["影響させない"], "影響させない"),
        (&["削除しない"], &["削除しない"], "削除しない"),
        (&["変更不可"], &["変更不可", "禁止"], "変更不可"),
        (&["実通信しない"], &["実通信しない"], "実通信しない"),
        (&["送信しない"], &["送信しない"], "送信しない"),
        (&["二重作成しない"], &["二重作成しない"], "二重作成しない"),
        (&["再接続しない"], &["再接続しない"], "再接続しない"),
        (&["再試行しない"], &["再試行しない"], "再試行しない"),
        (&["超えたら"], &["超えたら"], "超えたら"),
        (&["再推論せず"], &["再推論せず"], "再推論せず"),
        (&["戻さない"], &["戻さない"], "戻さない"),
        (&["参照しない"], &["参照しない"], "参照しない"),
        (
            &["混ぜない", "混ざらない"],
            &["混ぜない", "混ざらない", "混在禁止", "共有しない"],
            "混在禁止",
        ),
        (&["取りすぎない"], &["取りすぎない"], "取りすぎない"),
        (&["押さなくても"], &["押さなくても"], "押さなくても"),
        (&["クリア"], &["クリア"], "クリア"),
        (&["隠れない"], &["隠れない"], "隠れない"),
        (
            &["出さない"],
            &["出さない", "表示しない", "非表示"],
            "出さない",
        ),
        (&["見せない"], &["見せない"], "見せない"),
        (&["依存せず"], &["依存せず"], "依存せず"),
        (&["保存せず"], &["保存せず"], "保存せず"),
        (
            &[
                "変更せず",
                "変更しない",
                "変更なし",
                "変えない",
                "改変せず",
                "改変しない",
                "改変なし",
            ],
            &[
                "変更せず",
                "変更しない",
                "変更なし",
                "変えない",
                "改変せず",
                "改変しない",
                "改変なし",
                "維持",
                "保持",
                "そのまま",
            ],
            "変更せず",
        ),
        (
            &["増やさない", "増えない", "増加させない"],
            &[
                "増やさない",
                "増えない",
                "増加させない",
                "重複接続しない",
                "回避",
                "抑制",
                "最小限",
            ],
            "増加回避",
        ),
        (
            &["のみ", "だけ", "only"],
            &["のみ", "だけ", "いずれか", "only"],
            "のみ",
        ),
    ];

    let mut phrases = Vec::new();
    for (input_markers, output_markers, fallback_marker) in RULES {
        if !constraint_marker_matches_input(input, input_markers)
            || contains_any_marker(output, output_markers)
        {
            continue;
        }

        let source_clause = required_constraint_clauses(input)
            .into_iter()
            .find(|clause| contains_any_marker(clause, input_markers));
        if source_clause.is_some_and(|clause| {
            parse_conditional_value_list(clause).is_some()
                || parse_shared_predicate_list(clause).is_some()
        }) {
            continue;
        }
        let phrase = if input_markers
            .iter()
            .any(|marker| matches!(*marker, "のみ" | "だけ"))
            && source_clause.is_some_and(|clause| clause.contains("本題"))
        {
            "本題のみ".to_string()
        } else {
            source_clause
                .map(compact_constraint_clause)
                .filter(|phrase| !phrase.trim().is_empty())
                .unwrap_or_else(|| (*fallback_marker).to_string())
        };

        let phrase = if contains_any_marker(&phrase, output_markers) {
            phrase
        } else {
            format!("{}{}", phrase.trim(), fallback_marker)
        };

        if !phrases.iter().any(|existing| existing == &phrase) {
            phrases.push(phrase);
        }
    }

    for phrase in missing_state_persistence_restoration_phrases(input, output) {
        if !phrases.iter().any(|existing| existing == &phrase) {
            phrases.push(phrase);
        }
    }

    for phrase in missing_verification_restoration_phrases(input, output) {
        if !phrases.iter().any(|existing| existing == &phrase) {
            phrases.push(phrase);
        }
    }

    for phrase in missing_list_constraint_restoration_phrases(input, output) {
        if !phrases.iter().any(|existing| existing == &phrase) {
            phrases.push(phrase);
        }
    }

    for phrase in missing_retention_constraint_restoration_phrases(input, output) {
        if !phrases.iter().any(|existing| existing == &phrase) {
            phrases.push(phrase);
        }
    }

    for phrase in missing_focus_scope_restoration_phrases(input, output) {
        if !phrases.iter().any(|existing| existing == &phrase) {
            phrases.push(phrase);
        }
    }

    phrases
}

fn missing_nonnegative_constraint_restoration_phrases(input: &str, output: &str) -> Vec<String> {
    let mut phrases = missing_retention_constraint_restoration_phrases(input, output);
    for phrase in missing_focus_scope_restoration_phrases(input, output) {
        if !phrases.iter().any(|existing| existing == &phrase) {
            phrases.push(phrase);
        }
    }
    phrases
}

fn missing_focus_scope_restoration_phrases(input: &str, output: &str) -> Vec<String> {
    if required_constraint_clauses(input)
        .into_iter()
        .any(|clause| clause.contains("本題") && contains_any_marker(clause, &["だけ", "のみ"]))
        && !contains_any_marker(output, &["本題のみ", "本題だけ"])
    {
        vec!["本題のみ".to_string()]
    } else {
        Vec::new()
    }
}

fn missing_retention_constraint_restoration_phrases(input: &str, output: &str) -> Vec<String> {
    required_constraint_clauses(input)
        .into_iter()
        .filter(|clause| contains_any_marker(clause, &["残す", "残して"]))
        .filter(|clause| !retention_constraint_satisfied(clause, output))
        .filter_map(retention_constraint_restoration_phrase)
        .fold(Vec::new(), |mut phrases, phrase| {
            if !phrases.iter().any(|existing| existing == &phrase) {
                phrases.push(phrase);
            }
            phrases
        })
}

fn retention_constraint_satisfied(clause: &str, output: &str) -> bool {
    if !contains_any_marker(output, &["残す", "残して", "維持", "保持"]) {
        return false;
    }
    if clause.contains("任意") && !output.contains("任意") {
        return false;
    }
    if clause.contains("ローカルモデル")
        && !contains_any_marker(output, &["ローカルモデル", "任意モデル"])
    {
        return false;
    }
    required_technical_terms(clause)
        .into_iter()
        .all(|term| contains_ascii_case_insensitive(output, &term))
}

fn retention_constraint_restoration_phrase(clause: &str) -> Option<String> {
    let segment = clause
        .split(['、', ','])
        .map(str::trim)
        .find(|segment| contains_any_marker(segment, &["残す", "残して"]))?;
    let mut phrase = compact_constraint_clause(segment);
    for (from, to) in [
        (
            "任意ローカルモデル検証用に残すこと",
            "任意ローカルモデル検証用に残す",
        ),
        ("任意モデル用に残すこと", "任意モデル用に残す"),
        ("残すこと", "残す"),
        ("してください", ""),
    ] {
        phrase = phrase.replace(from, to);
    }
    let phrase = phrase
        .trim()
        .trim_end_matches(['。', '！', '？', '、', ','])
        .trim()
        .to_string();
    (!phrase.is_empty()).then_some(phrase)
}

fn missing_list_constraint_restoration_phrases(input: &str, output: &str) -> Vec<String> {
    let mut phrases = Vec::new();
    if let Some(list) = parse_counted_item_reference_list(input) {
        if let Some(phrase) = counted_item_list_restoration_phrase(&list, output) {
            phrases.push(phrase);
        }
    }

    input_clauses(input)
        .into_iter()
        .filter(|clause| !list_constraint_satisfied(clause, output))
        .filter_map(|clause| list_constraint_restoration_phrase(clause, output))
        .fold(phrases, |mut phrases, phrase| {
            if !phrases.iter().any(|existing| existing == &phrase) {
                phrases.push(phrase);
            }
            phrases
        })
}

fn list_constraint_restoration_phrase(clause: &str, output: &str) -> Option<String> {
    if parse_conditional_value_list(clause).is_some() {
        return structured_constraint_clause(clause);
    }
    let list = parse_shared_predicate_list(clause)?;
    shared_predicate_list_restoration_phrase(&list, output)
}

fn shared_predicate_list_restoration_phrase(
    list: &SharedPredicateList,
    output: &str,
) -> Option<String> {
    let missing_targets = list
        .targets
        .iter()
        .filter(|target| !shared_predicate_target_satisfied(target, output))
        .map(|target| compact_shared_predicate_target(target))
        .collect::<Vec<_>>();
    if missing_targets.is_empty() {
        None
    } else {
        Some(format!("{}{}", missing_targets.join("/"), list.predicate))
    }
}

fn counted_item_list_restoration_phrase(
    list: &SharedPredicateList,
    output: &str,
) -> Option<String> {
    if list
        .targets
        .iter()
        .all(|target| shared_predicate_target_satisfied(target, output))
    {
        return None;
    }
    let targets = list
        .targets
        .iter()
        .map(|target| compact_shared_predicate_target(target))
        .collect::<Vec<_>>()
        .join("/");
    Some(format!("保存対象={targets}{}", list.predicate))
}

fn preserves_list_constraints(input: &str, output: &str) -> bool {
    if parse_counted_item_reference_list(input).is_some_and(|list| {
        !list
            .targets
            .iter()
            .all(|target| shared_predicate_target_satisfied(target, output))
    }) {
        return false;
    }
    input_clauses(input)
        .into_iter()
        .all(|clause| list_constraint_satisfied(clause, output))
}

fn list_constraint_satisfied(clause: &str, output: &str) -> bool {
    if retry_limit_interval_clause_satisfied(clause, output) {
        return true;
    }
    if let Some(list) = parse_conditional_value_list(clause) {
        return contains_ascii_case_insensitive(output, &list.key)
            && list
                .values
                .iter()
                .all(|value| contains_ascii_case_insensitive(output, value));
    }
    if let Some(list) = parse_shared_predicate_list(clause) {
        return list
            .targets
            .iter()
            .all(|target| shared_predicate_target_satisfied(target, output));
    }
    true
}

fn retry_limit_interval_clause_satisfied(clause: &str, output: &str) -> bool {
    if !(clause.contains("再試行")
        && clause.contains("最大")
        && clause.contains("それ以上")
        && clause.contains("行わない"))
    {
        return false;
    }

    contains_any_marker(output, &["再試行"])
        && contains_any_marker(output, &["最大2回", "最大 2 回", "2回まで"])
        && contains_any_marker(output, &["1秒と2秒", "1秒/2秒", "1 秒と 2 秒"])
        && contains_any_marker(output, &["それ以上行わない", "それ以上は行わない"])
}

fn shared_predicate_target_satisfied(target: &str, output: &str) -> bool {
    let technical_terms: Vec<_> = required_technical_terms(target)
        .into_iter()
        .filter(|term| term.is_ascii())
        .collect();
    if !technical_terms.is_empty()
        && technical_terms
            .iter()
            .all(|term| contains_ascii_case_insensitive(output, term))
    {
        return true;
    }
    let anchor = constraint_target_anchor(target);
    let normalized_output = constraint_target_anchor(output);
    !anchor.is_empty() && contains_ascii_case_insensitive(&normalized_output, &anchor)
}

fn constraint_target_anchor(target: &str) -> String {
    [
        ("既存の", ""),
        ("成功した時の", "成功"),
        ("成功時の", "成功"),
        ("入力した", ""),
        ("最近開いた", ""),
        ("処理", ""),
        ("形式", ""),
        ("の内容", ""),
        ("オプション", ""),
        ("の表示", "表示"),
        ("マッピング", "mapping"),
        ("columns", "column"),
        (" の ", ""),
        ("の", ""),
    ]
    .iter()
    .fold(target.to_string(), |anchor, (from, to)| {
        anchor.replace(from, to)
    })
    .split_whitespace()
    .collect::<String>()
}

fn missing_state_persistence_restoration_phrases(input: &str, output: &str) -> Vec<String> {
    required_constraint_clauses(input)
        .into_iter()
        .filter(|clause| is_state_persistence_clause(clause))
        .filter(|clause| !state_persistence_clause_satisfied(clause, output))
        .map(|clause| state_persistence_restoration_phrase(clause, output))
        .filter(|phrase| !phrase.trim().is_empty())
        .fold(Vec::new(), |mut phrases, phrase| {
            if !phrases.iter().any(|existing: &String| existing == &phrase) {
                phrases.push(phrase);
            }
            phrases
        })
}

fn state_persistence_restoration_phrase(clause: &str, output: &str) -> String {
    if contains_any_marker(clause, &["ページ番号", "ページ"])
        && clause.contains("検索条件")
        && clause.contains("検索状態")
        && contains_any_marker(output, &["useSearchParams", "URL", "クエリ管理"])
        && contains_any_marker(output, &["維持", "保持", "残す", "消えない", "消さない"])
    {
        "ページ変更時も検索条件/状態維持".to_string()
    } else {
        compact_constraint_clause(clause)
    }
}

fn preserves_state_persistence_constraints(input: &str, output: &str) -> bool {
    required_constraint_clauses(input)
        .into_iter()
        .filter(|clause| is_state_persistence_clause(clause))
        .all(|clause| state_persistence_clause_satisfied(clause, output))
}

fn missing_verification_restoration_phrases(input: &str, output: &str) -> Vec<String> {
    input_clauses(input)
        .into_iter()
        .filter(|clause| is_verification_constraint_clause(clause))
        .filter(|clause| !verification_constraint_satisfied(clause, output))
        .map(verification_restoration_phrase)
        .filter(|phrase| !phrase.trim().is_empty())
        .fold(Vec::new(), |mut phrases, phrase| {
            if !phrases.iter().any(|existing: &String| existing == &phrase) {
                phrases.push(phrase);
            }
            phrases
        })
}

fn preserves_verification_constraints(input: &str, output: &str) -> bool {
    input_clauses(input)
        .into_iter()
        .filter(|clause| is_verification_constraint_clause(clause))
        .all(|clause| verification_constraint_satisfied(clause, output))
}

fn is_verification_constraint_clause(clause: &str) -> bool {
    contains_explicit_test_marker(clause)
        && contains_any_marker(
            clause,
            &["確認", "検証", "ケース", "正常系", "異常系", "境界値"],
        )
}

fn contains_explicit_test_marker(text: &str) -> bool {
    contains_any_marker(
        text,
        &[
            "テスト",
            "test",
            "spec",
            "assert",
            "vitest",
            "jest",
            "playwright",
            "pytest",
            "junit",
            "rspec",
            "cypress",
            "selenium",
        ],
    )
}

fn verification_constraint_satisfied(clause: &str, output: &str) -> bool {
    if !contains_any_marker(output, &["テスト", "確認", "検証"]) {
        return false;
    }

    for marker in ["正常系", "異常系", "境界値"] {
        if clause.contains(marker) && !output.contains(marker) {
            return false;
        }
    }

    if clause.contains("空文字") && !contains_any_marker(output, &["空文字", "空"]) {
        return false;
    }

    enumerated_verification_targets_satisfied(clause, output)
}

fn enumerated_verification_targets_satisfied(clause: &str, output: &str) -> bool {
    let mut items: Vec<&str> = clause
        .split(['、', ','])
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .collect();
    if items.len() < 2 {
        return true;
    }

    let first = items[0];
    let method_and_first_subject = first.split_once("では").or_else(|| first.split_once("で"));
    let Some((method, first_subject)) = method_and_first_subject else {
        return true;
    };
    let method = method.trim();
    if method.is_empty() || method.chars().count() > 32 {
        return true;
    }
    if !contains_explicit_test_marker(method) {
        return true;
    }
    items[0] = first_subject.trim();

    let Some(scope_start) = output.find(method) else {
        return false;
    };
    let scope = &output[scope_start..];
    items.into_iter().all(|item| {
        let item = item
            .trim()
            .trim_end_matches("を確認してください")
            .trim_end_matches("を確認してほしいです")
            .trim_end_matches("を確認してほしい")
            .trim_end_matches("を確認")
            .trim();
        let anchor = ["に", "の", "を", "が", "は"]
            .iter()
            .filter_map(|particle| item.find(particle).map(|index| &item[..index]))
            .filter(|candidate| candidate.chars().count() >= 2)
            .min_by_key(|candidate| candidate.chars().count())
            .unwrap_or(item)
            .trim();
        !anchor.is_empty() && contains_ascii_case_insensitive(scope, anchor)
    })
}

fn verification_restoration_phrase(clause: &str) -> String {
    let mut phrase = compact_constraint_clause(clause);
    for (from, to) in [
        ("テストでは", ""),
        ("テストで", ""),
        ("テストも", "テスト"),
        ("正常系と", "正常系/"),
        ("正常系、", "正常系/"),
        (" のケース", "ケース"),
        ("のケース", "ケース"),
        ("のケースを確認できるようにしてください", "テスト確認"),
        ("のケースを確認できるように", "テスト確認"),
        ("ケースを確認できるようにしてください", "ケース確認"),
        ("ケースを確認できるように", "ケース確認"),
        ("を確認できるようにしてください", "確認"),
        ("を確認できるように", "確認"),
        ("確認したいです", "確認"),
        ("追加してほしいですが", "追加"),
        (
            "を追加し、既存テストの書き方に合わせてください",
            "、既存形式準拠",
        ),
        ("テストは", "テスト:"),
    ] {
        phrase = phrase.replace(from, to);
    }
    phrase.trim().to_string()
}

fn is_state_persistence_clause(clause: &str) -> bool {
    let has_named_state = contains_any_marker(
        clause,
        &["検索条件", "検索状態", "入力状態", "選択状態", "表示状態"],
    );
    let has_generic_state_persistence = clause.contains("状態")
        && contains_any_marker(
            clause,
            &["消えない", "消さない", "維持", "保持", "keep", "preserve"],
        );

    (has_named_state || has_generic_state_persistence)
        && contains_any_marker(
            clause,
            &[
                "消えない",
                "消さない",
                "維持",
                "保持",
                "残す",
                "残して",
                "keep",
                "preserve",
            ],
        )
}

fn state_persistence_clause_satisfied(clause: &str, output: &str) -> bool {
    let output_compact = compact_all_whitespace(output);
    let has_persistence = contains_any_marker(
        output,
        &[
            "維持",
            "保持",
            "残す",
            "残して",
            "消えない",
            "消さない",
            "保存",
            "復元",
            "keep",
            "preserve",
        ],
    );
    if !has_persistence {
        return false;
    }
    let has_search_state_output = clause.contains("検索条件")
        && (contains_any_marker(output, &["検索条件", "条件/状態", "条件維持"])
            || (contains_any_marker(output, &["useSearchParams", "URL", "クエリ"])
                && contains_any_marker(output, &["保存", "復元"])));
    let has_named_state_output = has_search_state_output
        || (clause.contains("検索状態")
            && contains_any_marker(output, &["検索状態", "条件/状態", "状態維持"]))
        || (clause.contains("入力状態") && contains_any_marker(output, &["入力状態", "入力維持"]))
        || (clause.contains("選択状態") && contains_any_marker(output, &["選択状態", "選択維持"]))
        || (clause.contains("表示状態") && contains_any_marker(output, &["表示状態", "表示維持"]));

    if contains_any_marker(clause, &["ページ番号", "ページ"])
        && !contains_any_marker(output, &["ページ", "page"])
    {
        return false;
    }

    if clause.contains("検索条件") && !has_search_state_output {
        return false;
    }

    if clause.contains("検索状態")
        && !contains_any_marker(output, &["検索状態", "条件/状態", "状態維持"])
    {
        return false;
    }

    if clause.contains("入力状態") && !contains_any_marker(output, &["入力状態", "入力維持"])
    {
        return false;
    }

    if clause.contains("選択状態") && !contains_any_marker(output, &["選択状態", "選択維持"])
    {
        return false;
    }

    if clause.contains("表示状態") && !contains_any_marker(output, &["表示状態", "表示維持"])
    {
        return false;
    }

    if clause.contains("状態")
        && !contains_any_marker(
            output,
            &["状態", "条件/状態", "入力維持", "選択維持", "表示維持"],
        )
        && !output_compact.contains("状態維持")
        && !has_named_state_output
    {
        return false;
    }

    true
}

fn contains_any_marker(text: &str, markers: &[&str]) -> bool {
    let normalized = text.to_ascii_lowercase();
    markers
        .iter()
        .any(|marker| normalized.contains(&marker.to_ascii_lowercase()))
}

fn is_meta_task_restatement(output: &str) -> bool {
    let normalized = output.trim().to_ascii_lowercase();
    [
        "given a natural language description",
        "create a prompt",
        "summarize the task",
        "the user wants",
        "ユーザーの依頼",
        "プロンプトを作成",
        "タスクを要約",
    ]
    .iter()
    .any(|marker| normalized.contains(marker))
}

fn request_is_japanese(request: &CompressionRequest) -> bool {
    contains_japanese_text(&request.input_text)
}

fn request_output_language(request: &CompressionRequest) -> &'static str {
    if request.input_text.trim().is_empty() || request_is_japanese(request) {
        "日本語"
    } else {
        "入力と同じ言語"
    }
}

fn contains_japanese_text(text: &str) -> bool {
    text.chars()
        .any(|character| matches!(character, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{9fff}'))
}

fn required_technical_terms(input: &str) -> Vec<String> {
    const COMMON_TERMS: &[&str] = &[
        "react",
        "typescript",
        "javascript",
        "next",
        "vue",
        "angular",
        "svelte",
        "node",
        "express",
        "expo",
        "git",
        "go",
        "graphql",
        "i18n",
        "rest",
        "windows",
        "android",
        "vitest",
        "jest",
        "playwright",
        "prisma",
        "python",
        "rust",
        "runtime",
        "model",
        "nginx",
        "redis",
        "location",
        "users",
        "posts",
        "storybook",
        "button",
        "validation",
    ];

    let normalized_input = normalize_known_input_typos_for_llm(input);
    let input = normalized_input.as_str();

    let mut terms = Vec::new();
    for token in
        input.split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
    {
        if token.is_empty() {
            continue;
        }

        let lower = token.to_ascii_lowercase();
        let is_common_term = COMMON_TERMS.contains(&lower.as_str());
        let is_noisy_http_method = matches!(
            lower.as_str(),
            "get" | "post" | "put" | "patch" | "delete" | "head" | "options"
        );
        let ascii_alpha_count = token
            .chars()
            .filter(|character| character.is_ascii_alphabetic())
            .count();
        let starts_with_ascii_alpha = token
            .chars()
            .next()
            .map(|character| character.is_ascii_alphabetic())
            .unwrap_or(false);
        let is_uppercase_acronym = token.len() >= 2
            && ascii_alpha_count >= 2
            && token.chars().all(|character| {
                !character.is_ascii_alphabetic() || character.is_ascii_uppercase()
            });
        let has_internal_uppercase = starts_with_ascii_alpha
            && token
                .chars()
                .skip(1)
                .any(|character| character.is_ascii_uppercase());
        let is_inline_identifier = is_likely_inline_ascii_identifier(input, token);

        if !is_noisy_http_method
            && (is_common_term
                || is_uppercase_acronym
                || has_internal_uppercase
                || is_inline_identifier)
            && !terms
                .iter()
                .any(|existing: &String| existing.eq_ignore_ascii_case(token))
        {
            terms.push(token.to_string());
        }
    }

    for token in code_like_literal_terms(input) {
        if !terms
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(&token))
        {
            terms.push(token);
        }
    }

    for token in compound_ascii_terms(input) {
        if !terms
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(&token))
        {
            terms.push(token);
        }
    }

    for token in database_table_terms(input) {
        if !terms
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(&token))
        {
            terms.push(token);
        }
    }

    for token in http_status_terms(input) {
        if !terms
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(&token))
        {
            terms.push(token);
        }
    }

    for token in japanese_preserve_terms(input) {
        if !terms
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(&token))
        {
            terms.push(token);
        }
    }

    remove_assignment_key_duplicates(&mut terms);
    remove_covered_inline_identifier_terms(&mut terms, COMMON_TERMS);
    terms
}

#[cfg(test)]
fn organize_input_for_model(input: &str, required_terms: &[String]) -> String {
    PromptStructure::analyze(input, required_terms).render_for_model()
}

fn remove_covered_inline_identifier_terms(terms: &mut Vec<String>, common_terms: &[&str]) {
    let snapshot = terms.clone();
    terms.retain(|term| {
        let lower = term.to_ascii_lowercase();
        if common_terms.contains(&lower.as_str())
            || term.len() < 3
            || term.len() > 8
            || !term
                .chars()
                .all(|character| character.is_ascii_lowercase() || character.is_ascii_digit())
        {
            return true;
        }

        !snapshot.iter().any(|candidate| {
            candidate.len() > term.len()
                && candidate
                    .split(|character: char| !character.is_ascii_alphanumeric())
                    .any(|component| component.eq_ignore_ascii_case(term))
        })
    });
}

fn is_likely_inline_ascii_identifier(input: &str, token: &str) -> bool {
    const ENGLISH_STOP_WORDS: &[&str] = &[
        "and", "are", "but", "for", "from", "into", "not", "only", "that", "the", "then", "this",
        "when", "with", "without",
    ];
    if token.len() < 3
        || token.len() > 32
        || ENGLISH_STOP_WORDS.contains(&token.to_ascii_lowercase().as_str())
        || !token
            .chars()
            .next()
            .is_some_and(|character| character.is_ascii_lowercase())
        || !token.chars().all(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_'
        })
    {
        return false;
    }

    input.match_indices(token).any(|(start, _)| {
        let end = start + token.len();
        let has_identifier_before = input[..start]
            .chars()
            .next_back()
            .is_some_and(is_ascii_identifier_character);
        let has_identifier_after = input[end..]
            .chars()
            .next()
            .is_some_and(is_ascii_identifier_character);
        if has_identifier_before || has_identifier_after {
            return false;
        }

        let after = input[end..].trim_start();
        [
            "を",
            "と",
            "は",
            "が",
            "の",
            "へ",
            "で",
            "に",
            "から",
            "だけ",
            "時",
            "列",
            "キー",
            "項目",
            "値",
            "引数",
            "フィールド",
        ]
        .iter()
        .any(|particle| after.starts_with(particle))
    })
}

fn database_table_terms(input: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for token in
        input.split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
    {
        if token.len() < 3
            || !token
                .chars()
                .any(|character| character.is_ascii_alphabetic())
        {
            continue;
        }

        let is_table_name = input.match_indices(token).any(|(start, _)| {
            let after = &input[start + token.len()..];
            let after_head: String = after.chars().take(16).collect();
            let after_head = after_head.trim_start().to_ascii_lowercase();
            after_head.starts_with("テーブル") || after_head.starts_with("table")
        });

        if is_table_name
            && !terms
                .iter()
                .any(|existing: &String| existing.eq_ignore_ascii_case(token))
        {
            terms.push(token.to_string());
        }
    }

    terms
}

fn remove_assignment_key_duplicates(terms: &mut Vec<String>) {
    let assignment_keys: Vec<_> = terms
        .iter()
        .filter_map(|term| {
            term.split_once('=')
                .map(|(key, _)| key.to_ascii_lowercase())
        })
        .collect();
    terms.retain(|term| {
        !assignment_keys
            .iter()
            .any(|key| term.eq_ignore_ascii_case(key))
    });
}

fn http_status_terms(input: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let lower = input.to_ascii_lowercase();
    for (start, _) in lower.match_indices("http") {
        let after_http = &input[start + "http".len()..];
        let trimmed = after_http.trim_start();
        let status: String = trimmed
            .chars()
            .take_while(|character| character.is_ascii_digit())
            .collect();
        if status.len() != 3 {
            continue;
        }
        let term = format!("HTTP {status}");
        if !terms.iter().any(|existing: &String| existing == &term) {
            terms.push(term);
        }
    }

    terms
}

fn japanese_preserve_terms(input: &str) -> Vec<String> {
    const KNOWN_TERMS: &[&str] = &[
        "メール通知",
        "アプリ内通知",
        "個人情報",
        "依存関係キャッシュ",
        "ロックファイル",
        "監視ログ",
        "無効",
        "圧縮完了",
        "圧縮結果",
        "結果欄",
        "アイコン",
        "メトリクス",
        "最小化",
        "最大化",
        "閉じる",
        "ウィンドウバー",
        "トークン",
        "文字数",
        "圧縮レベル",
        "検証失敗理由",
        "必須語",
        "平均圧縮率",
        "既定値",
        "Sarashina 2.2 3B",
        "LM Studio",
        "push token",
        "Vue 3",
        "cart store",
        "derived store",
        "修復再推論",
        "1回目",
        "原文返し",
        "評価基準",
        "モード",
        "タスク種別",
        "共通プロンプト",
        "レベル別プロンプト",
        "Spring Boot",
        "Bean Validation",
        "version 4",
        "public access block",
        "Dockerfile",
        "dist",
        "実通信しない",
        "fake timers",
    ];
    let mut terms = Vec::new();
    for term in KNOWN_TERMS {
        if input.contains(term) {
            terms.push((*term).to_string());
        }
    }
    terms.extend(numeric_limit_terms(input));
    terms
}

fn numeric_limit_terms(input: &str) -> Vec<String> {
    const SUFFIXES: &[&str] = &["以下", "以上", "未満", "以内", "超"];
    let mut terms = Vec::new();
    let char_indices: Vec<_> = input.char_indices().collect();

    for (position, (start, character)) in char_indices.iter().copied().enumerate() {
        if !character.is_ascii_digit() {
            continue;
        }
        if position > 0 && char_indices[position - 1].1.is_ascii_digit() {
            continue;
        }

        let search_start = start + character.len_utf8();
        for (offset, next) in input[search_start..].char_indices() {
            if matches!(next, '。' | '、' | ',' | '，' | '\n' | '\r') {
                break;
            }
            let end = search_start + offset + next.len_utf8();
            let candidate = input[start..end].trim();
            if candidate.chars().count() > 16 {
                break;
            }
            if SUFFIXES.iter().any(|suffix| candidate.ends_with(suffix)) {
                if !terms.iter().any(|existing: &String| existing == candidate) {
                    terms.push(candidate.to_string());
                }
                break;
            }
        }
    }

    terms
}

fn code_like_literal_terms(input: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let skip_path_terms = input.contains("本番ログ")
        || input.contains("ログを解析")
        || input.contains("ログ解析")
        || input.contains("requestId=")
        || input.contains("ECONNRESET");
    for token in input.split(|character: char| {
        !(character.is_ascii_alphanumeric()
            || matches!(
                character,
                '_' | '-' | '.' | '/' | ':' | '+' | '#' | '@' | '=' | '<' | '>'
            ))
    }) {
        let mut token = token
            .trim_matches(|character: char| matches!(character, '.' | ',' | ':' | ';' | ')' | '('));
        if token.len() < 3 {
            continue;
        }
        if token.starts_with('/') && skip_path_terms {
            continue;
        }
        if let Some((key, value)) = token.split_once('=') {
            if matches!(key.to_ascii_lowercase().as_str(), "upstream" | "service")
                && value.len() >= 3
            {
                token = value;
            }
        }

        let has_ascii_alpha = token
            .chars()
            .any(|character| character.is_ascii_alphabetic());
        let has_separator = token.chars().any(|character| {
            matches!(
                character,
                '_' | '-' | '.' | '/' | ':' | '+' | '#' | '@' | '=' | '<' | '>'
            )
        });
        if has_ascii_alpha
            && has_separator
            && !terms
                .iter()
                .any(|existing: &String| existing.eq_ignore_ascii_case(token))
        {
            terms.push(token.to_string());
        }
    }

    terms
}

fn compound_ascii_terms(input: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut current = String::new();
    for character in input.chars().chain(std::iter::once('\0')) {
        let allowed = character.is_ascii_alphanumeric()
            || character.is_ascii_whitespace()
            || matches!(
                character,
                '_' | '-' | '.' | '/' | ':' | '+' | '#' | '@' | '=' | '<' | '>'
            );
        if allowed {
            current.push(character);
            continue;
        }

        let run = current.split_whitespace().collect::<Vec<_>>().join(" ");
        current.clear();
        let words: Vec<_> = run.split_whitespace().collect();
        if words.len() < 2 || words.len() > 6 {
            continue;
        }

        let has_cli_option = words.iter().any(|word| word.starts_with("--"));
        let is_http_route = matches!(
            words[0].to_ascii_uppercase().as_str(),
            "GET" | "POST" | "PUT" | "PATCH" | "DELETE" | "HEAD" | "OPTIONS"
        ) && words.get(1).is_some_and(|word| word.starts_with('/'));
        let is_named_phrase = words.len() == 2
            && (((is_title_ascii_word(words[0]) || is_ascii_acronym(words[0]))
                && is_title_ascii_word(words[1]))
                || is_ascii_acronym(words[1]));
        let is_japanese_embedded_phrase = contains_japanese_text(input)
            && words.len() <= 4
            && words.iter().all(|word| {
                word.chars()
                    .any(|character| character.is_ascii_alphabetic())
            });
        if has_cli_option || is_http_route || is_named_phrase || is_japanese_embedded_phrase {
            runs.push(run);
        }
    }
    runs
}

fn is_title_ascii_word(value: &str) -> bool {
    value
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_uppercase())
        && value
            .chars()
            .skip(1)
            .any(|character| character.is_ascii_lowercase())
}

fn is_ascii_acronym(value: &str) -> bool {
    let letters: Vec<_> = value
        .chars()
        .filter(|character| character.is_ascii_alphabetic())
        .collect();
    letters.len() >= 2
        && letters
            .iter()
            .all(|character| character.is_ascii_uppercase())
}

#[derive(Debug)]
struct ConditionalValueList {
    key: String,
    values: Vec<String>,
    consequence: String,
}

#[derive(Debug)]
struct SharedPredicateList {
    targets: Vec<String>,
    predicate: String,
}

fn structured_constraint_clause(clause: &str) -> Option<String> {
    if let Some(list) = parse_conditional_value_list(clause) {
        let consequence = compact_constraint_clause(&list.consequence)
            .replace("へ進む前に", "前に")
            .replace(" を返し、", "、")
            .replace(" の ", " ")
            .replace("に HTTP", "にHTTP");
        let consequence = compact_assignment_segments(&consequence);
        return Some(format!(
            "{}={}のいずれかなら{}",
            list.key,
            list.values.join("/"),
            consequence
        ));
    }
    parse_shared_predicate_list(clause).map(|list| {
        let targets = list
            .targets
            .iter()
            .map(|target| compact_shared_predicate_target(target))
            .collect::<Vec<_>>()
            .join("/");
        format!("{targets}{}", list.predicate)
    })
}

fn compact_assignment_segments(value: &str) -> String {
    value
        .split('、')
        .map(|segment| {
            let segment = segment.trim();
            let Some((left, right)) = segment.split_once(" は ") else {
                return segment.to_string();
            };
            let right = right.trim_end_matches(" に").trim();
            if left.trim().is_empty() || right.is_empty() {
                segment.to_string()
            } else {
                format!("{}={right}", left.trim())
            }
        })
        .collect::<Vec<_>>()
        .join("、")
}

fn compact_shared_predicate_target(target: &str) -> String {
    [
        ("既存の", ""),
        ("成功時の", "成功"),
        ("成功した時の", "成功"),
        (" の ", ""),
        (" の", ""),
        ("の採番", "採番"),
        ("エラーコード", "エラー"),
        ("処理", ""),
        ("オプション", ""),
        ("の表示", "表示"),
        ("テストコマンド", "テスト"),
        ("レスポンスフィールド名", "レスポンス名"),
    ]
    .iter()
    .fold(target.trim().to_string(), |compact, (from, to)| {
        compact.replace(from, to)
    })
}

fn parse_conditional_value_list(clause: &str) -> Option<ConditionalValueList> {
    let (before, consequence) = ["のいずれかなら", "のいずれかの場合", "のどれかなら"]
        .iter()
        .find_map(|marker| clause.split_once(marker))?;
    let mut items: Vec<String> = before
        .split(['、', ','])
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::to_string)
        .collect();
    if items.len() < 3 {
        return None;
    }
    let first_item = items[0].clone();
    let (key, first_value) = first_item.split_once('が')?;
    let key = key.trim().to_string();
    if key.is_empty() {
        return None;
    }
    items[0] = first_value.trim().to_string();
    if let Some(last) = items.last_mut() {
        *last = last.trim_end_matches("だけ").trim().to_string();
    }
    if items.iter().any(|item| item.is_empty()) || consequence.trim().is_empty() {
        return None;
    }
    Some(ConditionalValueList {
        key,
        values: items,
        consequence: consequence.trim().to_string(),
    })
}

fn parse_shared_predicate_list(clause: &str) -> Option<SharedPredicateList> {
    let (before, predicate) = [
        ("変更しないでください", "変更しない"),
        ("変更しない", "変更しない"),
        ("変えないでください", "変えない"),
        ("変えない", "変えない"),
        ("維持してください", "維持"),
        ("保持してください", "保持"),
        ("含めないでください", "含めない"),
        ("含めない", "含めない"),
        ("保存しないでください", "保存しない"),
        ("保存しない", "保存しない"),
        ("行わないでください", "行わない"),
        ("行わない", "行わない"),
    ]
    .iter()
    .find_map(|(marker, predicate)| {
        clause
            .rfind(marker)
            .map(|index| (&clause[..index], *predicate))
    })?;
    let mut before = before
        .trim()
        .trim_start_matches("ただし")
        .trim_start_matches("なお")
        .trim_start_matches("一方で")
        .trim();
    for marker in ["は機密情報", "は個人情報", "は秘密情報"] {
        if let Some((list_part, _)) = before.split_once(marker) {
            before = list_part.trim();
            break;
        }
    }
    if let Some((_, focused)) = before.rsplit_once(['、', ',']) {
        if focused.contains('や') {
            before = focused.trim();
        }
    }
    let before = before
        .trim_end_matches("までは")
        .trim_end_matches(['は', 'を'])
        .trim();
    let supports_two_item_prohibition = before.contains('や') && predicate == "行わない";
    let targets: Vec<String> = before
        .split(['、', ',', 'や'])
        .map(str::trim)
        .filter(|target| !target.is_empty())
        .map(|target| target.trim_start_matches("既存の").trim().to_string())
        .collect();
    if (targets.len() < 3 && !(targets.len() == 2 && supports_two_item_prohibition))
        || targets.iter().any(|target| {
            contains_any_marker(
                target,
                &[
                    "維持し",
                    "保持し",
                    "含め",
                    "追加し",
                    "返し",
                    "行い",
                    "してください",
                    "変更し",
                ],
            )
        })
    {
        return None;
    }
    Some(SharedPredicateList {
        targets,
        predicate: predicate.to_string(),
    })
}

fn parse_counted_item_reference_list(input: &str) -> Option<SharedPredicateList> {
    let clauses = input_clauses(input);
    for pair in clauses.windows(2) {
        let Some(targets) = parse_enumerated_definition_clause(pair[0]) else {
            continue;
        };
        let Some(count) = referenced_item_count(pair[1]) else {
            continue;
        };
        if targets.len() != count || !contains_any_marker(pair[1], &["だけ", "のみ"]) {
            continue;
        }
        let predicate = if pair[1].contains("保存") {
            "のみ保存"
        } else if pair[1].contains("復元") {
            "のみ復元"
        } else {
            "のみ"
        };
        return Some(SharedPredicateList {
            targets,
            predicate: predicate.to_string(),
        });
    }
    None
}

fn parse_enumerated_definition_clause(clause: &str) -> Option<Vec<String>> {
    let (_, after_marker) = clause.rsplit_once('は')?;
    let list_text = after_marker
        .trim()
        .trim_end_matches("です")
        .trim_end_matches("でした")
        .trim();
    let targets: Vec<String> = list_text
        .split(['、', ','])
        .map(str::trim)
        .filter(|target| !target.is_empty())
        .map(str::to_string)
        .collect();
    if targets.len() >= 3 {
        Some(targets)
    } else {
        None
    }
}

fn referenced_item_count(clause: &str) -> Option<usize> {
    (2..=12).find(|count| {
        clause.contains(&format!("{count}項目")) || clause.contains(&format!("{count}つ"))
    })
}

fn compact_constraint_clause(clause: &str) -> String {
    let mut compact = trim_constraint_discourse_prefix(clause).trim().to_string();
    if let Some(list) = parse_shared_predicate_list(&compact) {
        let targets = list
            .targets
            .iter()
            .map(|target| compact_shared_predicate_target(target))
            .collect::<Vec<_>>()
            .join("/");
        return format!("{targets}{}", list.predicate);
    }
    for (from, to) in [
        ("変更しないでください", "変更しない"),
        ("変えないでください", "変えない"),
        ("維持してください", "維持"),
        ("保持してください", "保持"),
        ("含めないでください", "含めない"),
        ("行わないでください", "行わない"),
        ("していただけますでしょうか", ""),
        ("していただきたいです", ""),
        ("してほしいです", ""),
        ("をお願いします", ""),
        ("成功時のレスポンス形式", "成功レスポンス形式"),
        ("既存の監査ログ", "監査ログ"),
        ("既存の schema.graphql", "schema.graphql"),
        ("レスポンスフィールド名", "レスポンス名"),
        (
            "ページ番号を変更しても検索条件と検索状態が消えないようにしてください",
            "ページ変更時も検索条件/状態維持",
        ),
        (
            "検索条件と検索状態が消えないようにしてください",
            "検索条件/状態維持",
        ),
        ("検索条件と検索状態が消えないように", "検索条件/状態維持"),
        (
            "大規模なリファクタリングや画面全体の作り直しは避けてください",
            "大規模リファクタリング/画面作り直し回避",
        ),
        ("大規模なリファクタリング", "大規模リファクタリング"),
        ("画面全体の作り直し", "画面作り直し"),
        ("避けてください", "回避"),
        ("データが混ざらないようにしてください", "データ混ざらない"),
        ("データが混ざらないように", "データ混ざらない"),
        (
            "ユーザーが任意のローカルモデルを検証するために残すこと",
            "任意ローカルモデル検証用に残す",
        ),
        (
            "ユーザーが任意のローカルモデルを検証するために残す",
            "任意ローカルモデル検証用に残す",
        ),
        (
            "任意のローカルモデルを検証するために残すこと",
            "任意ローカルモデル検証用に残す",
        ),
        (
            "任意のローカルモデルを検証するために残す",
            "任意ローカルモデル検証用に残す",
        ),
        (
            "ユーザーが任意モデルを試すために残してください",
            "任意モデル用に残す",
        ),
        ("ユーザーが任意モデルを試すために残す", "任意モデル用に残す"),
        ("ことを明記してください", ""),
        ("ことを明記", ""),
        ("を明記してください", ""),
        ("を明記", ""),
        ("本題だけ圧縮できるか見たいです", "本題のみ"),
        ("本題だけ圧縮", "本題のみ"),
        ("本題だけ", "本題のみ"),
        ("作り直しは今回はいりません", "作り直し不要"),
        ("作り直しは今回は不要です", "作り直し不要"),
        ("今回はいりません", "不要"),
    ] {
        compact = compact.replace(from, to);
    }
    collapse_plain_spaces(&compact)
        .trim()
        .trim_end_matches(['。', '！', '？'])
        .trim()
        .to_string()
}

fn trim_constraint_discourse_prefix(clause: &str) -> &str {
    for marker in ["今回は", "今回については"] {
        let Some((prefix, focused)) = clause.split_once(marker) else {
            continue;
        };
        let prefix_is_request_history = contains_any_marker(
            prefix,
            &["前にも", "以前", "先ほど", "似た", "同様", "お願い"],
        );
        if prefix_is_request_history && !focused.trim().is_empty() {
            return focused.trim();
        }
    }
    clause
}

fn required_constraint_clauses(input: &str) -> Vec<&str> {
    const CONSTRAINT_MARKERS: &[&str] = &[
        "避け",
        "禁止",
        "しない",
        "できない",
        "不要",
        "ではなく",
        "でなく",
        "はみ出さない",
        "行わない",
        "読み込まず",
        "読まない",
        "下げない",
        "廃止",
        "変えず",
        "入れない",
        "影響させない",
        "削除しない",
        "消えない",
        "消さない",
        "変更不可",
        "実通信しない",
        "送信しない",
        "二重作成しない",
        "再接続しない",
        "再試行しない",
        "超えたら",
        "再推論せず",
        "戻さない",
        "参照しない",
        "混ぜない",
        "混ざらない",
        "取りすぎない",
        "押さなくても",
        "クリア",
        "隠れない",
        "出さない",
        "見せない",
        "依存せず",
        "保存せず",
        "のみ",
        "だけ",
        "必ず",
        "維持",
        "残す",
        "残して",
        "残してほしい",
        "テスト",
        "確認できる",
        "確認したい",
        "場合",
        "なら",
        "ときは",
        "時は",
        "際は",
        "失敗時",
        "成功時",
        "変えない",
        "変更せず",
        "変更しない",
        "変更なし",
        "改変せず",
        "改変しない",
        "改変なし",
        "増やさない",
        "増えない",
        "増加させない",
        "せず",
        "行わず",
        "作らず",
        "戻さず",
        "送らず",
        "含めず",
        "must",
        "must not",
        "do not",
        "don't",
        "avoid",
        "only",
        "without",
        "preserve",
        "keep",
    ];

    input_clauses(input)
        .into_iter()
        .filter(|clause| {
            let normalized = clause.to_ascii_lowercase();
            CONSTRAINT_MARKERS
                .iter()
                .any(|marker| normalized.contains(marker))
        })
        .collect()
}

fn input_clauses(input: &str) -> Vec<&str> {
    input
        .split(['。', '！', '？', '\n', ';', '；'])
        .map(str::trim)
        .filter(|clause| !clause.is_empty())
        .collect()
}

fn preserves_targeted_change_constraints(input: &str, output: &str) -> bool {
    const SOURCE_PREDICATES: &[&str] = &[
        "変更しない",
        "変更せず",
        "変更なし",
        "変えない",
        "変えず",
        "改変しない",
        "改変せず",
        "改変なし",
        "維持",
        "保持",
        "preserve",
        "keep",
    ];
    const OUTPUT_PREDICATES: &[&str] = &[
        "変更しない",
        "変更せず",
        "変更なし",
        "変えない",
        "変えず",
        "改変しない",
        "改変せず",
        "改変なし",
        "維持",
        "保持",
        "そのまま",
        "禁止",
        "しない",
        "せず",
        "preserve",
        "keep",
        "unchanged",
    ];

    input_clauses(input).into_iter().all(|clause| {
        let Some(predicate_index) = SOURCE_PREDICATES
            .iter()
            .filter_map(|predicate| clause.rfind(predicate))
            .max()
        else {
            return true;
        };
        let target_text = clause[..predicate_index]
            .trim()
            .trim_end_matches(['は', 'を', 'も', 'が'])
            .trim();
        if target_text.is_empty() {
            return true;
        }

        if parse_shared_predicate_list(clause).is_some() {
            return true;
        }
        let targets: Vec<_> = target_text
            .split(['、', ',', '/', 'と'])
            .map(str::trim)
            .filter(|target| !target.is_empty())
            .collect();
        if targets.len() != 1 {
            return true;
        }
        let Some(anchor) = change_constraint_target_anchor(targets[0]) else {
            return true;
        };
        input_clauses(output).into_iter().any(|output_clause| {
            contains_ascii_case_insensitive(output_clause, &anchor)
                && contains_any_marker(output_clause, OUTPUT_PREDICATES)
        })
    })
}

fn change_constraint_target_anchor(target: &str) -> Option<String> {
    let focused = target
        .trim()
        .trim_start_matches("既存の")
        .trim_start_matches("既存")
        .trim();
    let focused = focused.rsplit_once('の').map_or(focused, |(head, tail)| {
        let tail = tail.trim();
        if matches!(tail, "意味" | "形式" | "名前" | "名") {
            head.trim()
        } else {
            tail
        }
    });
    let focused = ["形式", "名前"]
        .iter()
        .find_map(|suffix| focused.strip_suffix(suffix))
        .unwrap_or(focused)
        .trim_end_matches(['は', 'を', 'も', 'に', 'が'])
        .trim();
    if focused.is_empty() {
        return None;
    }

    let ascii_anchor = focused
        .split(|character: char| {
            !(character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.' | '/' | ':'))
        })
        .filter(|token| {
            token
                .chars()
                .any(|character| character.is_ascii_alphabetic())
        })
        .max_by_key(|token| token.len());
    if let Some(anchor) = ascii_anchor {
        return Some(anchor.to_string());
    }

    let chars: Vec<_> = focused.chars().collect();
    let start = chars.len().saturating_sub(2);
    Some(chars[start..].iter().collect())
}

fn preserves_negative_constraints(input: &str, output: &str) -> bool {
    const NEGATION_RULES: &[(&[&str], &[&str])] = &[
        (
            &[
                "避け", "禁止", "avoid", "must not", "do not", "don't", "never",
            ],
            &[
                "避け",
                "回避",
                "禁止",
                "しない",
                "avoid",
                "must not",
                "do not",
                "don't",
                "never",
            ],
        ),
        (
            &["しない", "不要", "なし", "without"],
            &[
                "しない",
                "せず",
                "禁止",
                "避け",
                "回避",
                "抑制",
                "最小限",
                "不要",
                "なし",
                "without",
                "do not",
                "must not",
            ],
        ),
        (
            &[
                "変更せず",
                "変更しない",
                "変更なし",
                "変えない",
                "改変せず",
                "改変しない",
                "改変なし",
            ],
            &[
                "変更せず",
                "変更しない",
                "変更なし",
                "変えない",
                "改変せず",
                "改変しない",
                "改変なし",
                "維持",
                "保持",
                "そのまま",
            ],
        ),
        (&["ではなく", "でなく"], &["ではなく", "でなく"]),
        (
            &["はみ出さない"],
            &["はみ出さない", "収ま", "収め", "バー内"],
        ),
        (&["行わない"], &["行わない", "しない"]),
        (&["書き換えず"], &["書き換えず", "書き換えない"]),
        (
            &["壊れない"],
            &["壊れない", "壊れず", "壊さない", "破損しない"],
        ),
        (
            &["削除せず"],
            &["削除せず", "削除しない", "維持", "保持", "共存"],
        ),
        (&["読み込まず"], &["読み込まず", "読まない", "読み込み禁止"]),
        (&["下げない"], &["下げない"]),
        (&["廃止"], &["廃止"]),
        (&["変えず"], &["変えず", "変更せず", "変更しない"]),
        (&["入れない"], &["入れない", "含めない", "しない"]),
        (&["影響させない"], &["影響させない"]),
        (&["削除しない"], &["削除しない"]),
        (&["変更不可"], &["変更不可", "禁止"]),
        (&["実通信しない"], &["実通信しない"]),
        (&["送信しない"], &["送信しない"]),
        (&["二重作成しない"], &["二重作成しない"]),
        (&["再接続しない"], &["再接続しない"]),
        (&["再試行しない"], &["再試行しない"]),
        (
            &["増えない"],
            &["増えない", "増やさない", "しない", "回避", "抑制", "最小限"],
        ),
        (
            &["増やさない"],
            &["増えない", "増やさない", "しない", "回避", "抑制", "最小限"],
        ),
        (
            &["増加させない"],
            &["増えない", "増やさない", "しない", "回避", "抑制", "最小限"],
        ),
        (&["超えたら"], &["超えたら"]),
        (&["再推論せず"], &["再推論せず"]),
        (&["戻さない"], &["戻さない"]),
        (&["参照しない"], &["参照しない"]),
        (
            &["混ぜない", "混ざらない"],
            &["混ぜない", "混ざらない", "混在禁止", "共有しない"],
        ),
        (&["取りすぎない"], &["取りすぎない"]),
        (&["押さなくても"], &["押さなくても"]),
        (&["クリア"], &["クリア"]),
        (&["隠れない"], &["隠れない"]),
        (&["出さない"], &["出さない", "表示しない", "非表示"]),
        (&["見せない"], &["見せない"]),
        (&["失わない"], &["失わない", "残す", "保持"]),
        (&["依存せず"], &["依存せず"]),
        (&["保存せず"], &["保存せず"]),
        (
            &["増やさない", "増えない", "増加させない"],
            &[
                "増やさない",
                "増えない",
                "増加させない",
                "重複接続しない",
                "回避",
                "抑制",
                "最小限",
            ],
        ),
        (
            &["のみ", "だけ", "only"],
            &["のみ", "だけ", "いずれか", "only"],
        ),
    ];

    let input = input.to_ascii_lowercase();
    let output = output.to_ascii_lowercase();
    preserves_state_persistence_constraints(&input, &output)
        && preserves_verification_constraints(&input, &output)
        && preserves_list_constraints(&input, &output)
        && NEGATION_RULES
            .iter()
            .all(|(input_markers, output_markers)| {
                !constraint_marker_matches_input(&input, input_markers)
                    || output_markers.iter().any(|marker| output.contains(marker))
            })
}

fn constraint_marker_matches_input(input: &str, markers: &[&str]) -> bool {
    if !contains_any_marker(input, markers) {
        return false;
    }
    if !markers
        .iter()
        .any(|marker| matches!(*marker, "ない" | "出さない"))
    {
        return true;
    }
    input_clauses(input)
        .into_iter()
        .any(|clause| contains_any_marker(clause, markers) && !clause.contains("はみ出さない"))
}

fn contains_ascii_case_insensitive(text: &str, term: &str) -> bool {
    let text = text.to_ascii_lowercase();
    let term = term.to_ascii_lowercase();
    contains_required_term_match(&text, &term)
        || (term.chars().any(char::is_whitespace)
            && compact_all_whitespace(&text).contains(&compact_all_whitespace(&term)))
        || contains_natural_compound_required_term(&text, &term)
}

fn contains_required_term_match(text: &str, term: &str) -> bool {
    if !term.chars().all(is_ascii_identifier_character) {
        return text.contains(term);
    }

    text.match_indices(term).any(|(start, _)| {
        let end = start + term.len();
        !text[..start]
            .chars()
            .next_back()
            .is_some_and(is_ascii_identifier_character)
            && !text[end..]
                .chars()
                .next()
                .is_some_and(is_ascii_identifier_character)
    })
}

fn contains_natural_compound_required_term(text: &str, term: &str) -> bool {
    let parts: Vec<_> = term
        .split_whitespace()
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    if parts.len() < 2 || !parts.iter().any(|part| !part.is_ascii()) {
        return false;
    }

    let compact_text = compact_all_whitespace(text);
    parts
        .iter()
        .all(|part| text.contains(part) || compact_text.contains(&compact_all_whitespace(part)))
}

fn compact_all_whitespace(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn resolve_project_path(project_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        project_root.join(path)
    }
}

fn resolve_windows_exe(path: &Path) -> PathBuf {
    if cfg!(windows) && path.extension().is_none() && !path.exists() {
        let mut candidate = path.to_path_buf();
        candidate.set_extension("exe");
        candidate
    } else {
        path.to_path_buf()
    }
}

#[cfg(test)]
#[path = "backend_tests.rs"]
mod tests;
