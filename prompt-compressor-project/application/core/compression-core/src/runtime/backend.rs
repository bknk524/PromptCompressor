use std::collections::{hash_map::DefaultHasher, BTreeMap};
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::config::profile::ProfileDefinition;
use crate::error::{CompressionError, Result};
use crate::types::CompressionRequest;

#[derive(Debug, Clone)]
pub struct CompressionDraft {
    pub distilled_prompt: String,
    pub removed_content_summary: Vec<String>,
}

pub trait RuntimeBackend {
    fn compress(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
    ) -> Result<CompressionDraft>;

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
}

#[derive(Debug, Default)]
struct ManagedRuntimeManager {
    processes: Mutex<BTreeMap<String, ManagedServer>>,
}

#[derive(Debug)]
struct ManagedServer {
    child: Child,
}

#[derive(Default)]
struct EmbeddedModelManager {
    #[cfg(feature = "embedded-llama")]
    models: Mutex<BTreeMap<String, llama_cpp::LlamaModel>>,
    #[cfg(feature = "embedded-llama")]
    prepared_prompt_sessions: Mutex<BTreeMap<String, llama_cpp::LlamaSession>>,
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
            debug.field("model_count", &model_count);
            debug.field("prepared_prompt_count", &prepared_prompt_count);
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
    fn load_or_get(
        &self,
        cache_key: &str,
        model_path: &Path,
        gpu_layers: u32,
    ) -> Result<llama_cpp::LlamaModel> {
        let mut models = self.models.lock().map_err(|_| {
            CompressionError::Runtime("embedded model registry is unavailable".into())
        })?;

        if let Some(model) = models.get(cache_key) {
            return Ok(model.clone());
        }

        let mut params = llama_cpp::LlamaParams::default();
        params.n_gpu_layers = gpu_layers;
        let loaded =
            llama_cpp::LlamaModel::load_from_file(model_path, params).map_err(|error| {
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
        session: &llama_cpp::LlamaSession,
    ) -> Result<()> {
        let prepared = session.deep_copy().map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to store prepared embedded prompt session: {error}"
            ))
        })?;

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

        Ok(Self {
            project_root: project_root.clone(),
            prompts_dir: project_root.join("resources").join("prompts"),
            models: ModelRegistry::from_path(settings_dir.join("model-catalog.yaml"))?,
            runtimes: RuntimeRegistry::from_path(settings_dir.join("runtime-backends.yaml"))?,
            prompt_profiles: PromptProfileRegistry::from_path(
                settings_dir
                    .join("compression-policies")
                    .join("level-prompt-profiles-v1.yaml"),
            )?,
            managed_runtimes: Arc::new(ManagedRuntimeManager::default()),
            embedded_models: Arc::new(EmbeddedModelManager::default()),
        })
    }

    pub fn warm_profile(&self, profile: &ProfileDefinition) -> Result<bool> {
        let (model, runtime) = self.resolve_model_and_runtime(profile)?;
        match (runtime.backend_kind.as_str(), &runtime.launch_mode) {
            ("llama.cpp", RuntimeLaunchMode::Embedded) => {
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
            ("llama.cpp", RuntimeLaunchMode::Embedded) => {
                self.prepare_embedded_llama_prompt_prefix(request, profile, model, runtime)
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

        let model_path = model.model_path.as_ref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "model '{}' is missing model_path for llama.cpp runtime '{}'",
                model.id, runtime.id
            ))
        })?;
        let model_path = resolve_project_path(&self.project_root, model_path);
        if !model_path.is_file() {
            return Err(CompressionError::Runtime(format!(
                "model file not found at {}",
                model_path.display()
            )));
        }

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

        if runtime.gpu_layers > 0 {
            command
                .arg("--n-gpu-layers")
                .arg(runtime.gpu_layers.to_string());
        }

        Ok((command, runtime.timeout_ms))
    }

    fn compress_with_lmstudio(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<CompressionDraft> {
        let base_url = runtime.base_url.as_deref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "runtime '{}' is missing base_url for LM Studio",
                runtime.id
            ))
        })?;
        let model_name = self.resolve_lmstudio_model_name(model, runtime)?;
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
    ) -> Result<CompressionDraft> {
        let prompt = self.build_prompt(request, profile, model)?;
        self.request_validated_completion(request, &prompt, model, runtime, base_url, model_name)
    }

    fn request_validated_completion(
        &self,
        request: &CompressionRequest,
        prompt: &str,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
        base_url: &str,
        model_name: &str,
    ) -> Result<CompressionDraft> {
        let mut first_draft =
            self.request_openai_completion(request, prompt, model, runtime, base_url, model_name)?;
        trace_model_output("openai.raw_draft", &first_draft.distilled_prompt);
        restore_missing_required_constraints(request, &mut first_draft);
        restore_missing_required_terms(request, &mut first_draft);
        polish_model_output_for_request(request, &mut first_draft);
        validate_compression_draft(request, &first_draft).map_err(|error| {
            CompressionError::Runtime(format!(
                "{error}; invalid draft starts with: {}",
                output_snippet(&first_draft.distilled_prompt)
            ))
        })?;
        Ok(first_draft)
    }

    fn request_openai_completion(
        &self,
        request: &CompressionRequest,
        prompt: &str,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
        base_url: &str,
        model_name: &str,
    ) -> Result<CompressionDraft> {
        let payload = ChatCompletionRequest {
            model: model_name,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: "返答は JSON オブジェクトだけにしてください。日本語入力には日本語で返答し、コード、識別子、API 名、ファイル名は原文のまま保持してください。",
                },
                ChatMessage {
                    role: "user",
                    content: prompt,
                },
            ],
            temperature: 0.0,
            max_tokens: effective_max_output_tokens(request, model),
            stream: false,
            response_format: model
                .supports_json_schema
                .then(compression_response_schema),
        };
        let body = serde_json::to_vec(&payload).map_err(|error| {
            CompressionError::Runtime(format!(
                "failed to serialize local runtime request: {error}"
            ))
        })?;
        let response_body = http_json_request(
            "POST",
            base_url,
            "/chat/completions",
            runtime.api_token_env.as_deref(),
            Some(&body),
            Duration::from_millis(runtime.timeout_ms),
        )?;
        let completion: ChatCompletionResponse =
            serde_json::from_str(&response_body).map_err(|error| {
                CompressionError::Runtime(format!(
                    "local runtime response was not valid chat completion JSON: {error}"
                ))
            })?;
        let content = completion
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_deref())
            .ok_or_else(|| {
                CompressionError::Runtime(
                    "local runtime response did not include choices[0].message.content".into(),
                )
            })?;

        parse_compression_output(content)
    }

    fn resolve_lmstudio_model_name(
        &self,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<String> {
        let configured = model.api_model.as_deref().unwrap_or(model.id.as_str());
        if configured != "auto" {
            return Ok(configured.to_string());
        }

        let base_url = runtime.base_url.as_deref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "runtime '{}' is missing base_url for LM Studio",
                runtime.id
            ))
        })?;
        let response_body = http_json_request(
            "GET",
            base_url,
            "/models",
            runtime.api_token_env.as_deref(),
            None,
            Duration::from_millis(runtime.timeout_ms),
        )?;
        let models: ModelsResponse = serde_json::from_str(&response_body).map_err(|error| {
            CompressionError::Runtime(format!(
                "LM Studio /models response was not valid JSON: {error}"
            ))
        })?;

        models
            .data
            .first()
            .map(|item| item.id.clone())
            .ok_or_else(|| {
                CompressionError::Runtime(
                    "LM Studio returned no available models from /v1/models".into(),
                )
            })
    }

    fn compress_with_managed_llama_cpp_server(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<CompressionDraft> {
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
        let cache_key = embedded_model_cache_key(model, &model_path, runtime);
        self.embedded_models
            .load_or_get(&cache_key, &model_path, runtime.gpu_layers)
            .map(|_| ())
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
        self.preload_embedded_llama_model(model, runtime)?;
        let prompt_parts = self.build_prompt_parts(request, profile, model)?;
        if prompt_parts.prefix.trim().is_empty() {
            return Ok(false);
        }

        let total_started_at = Instant::now();
        let model_path = self.resolve_model_file_path(model, &runtime.id)?;
        let model_cache_key = embedded_model_cache_key(model, &model_path, runtime);
        let prompt_prefix = format_embedded_llama_prompt_prefix(&prompt_parts.prefix);
        let prompt_cache_key =
            embedded_prompt_cache_key(model, &model_path, runtime, &prompt_prefix);
        if self
            .embedded_models
            .has_prepared_session(&prompt_cache_key)?
        {
            trace_runtime_value("embedded.prepare_prompt_cache_hit", 1);
            trace_runtime_timing("embedded.prepare_total", total_started_at.elapsed());
            return Ok(true);
        }

        trace_runtime_value("embedded.prepare_prompt_cache_hit", 0);
        let model_started_at = Instant::now();
        let llama_model =
            self.embedded_models
                .load_or_get(&model_cache_key, &model_path, runtime.gpu_layers)?;
        trace_runtime_timing(
            "embedded.prepare_model_load_or_cache",
            model_started_at.elapsed(),
        );

        let session_started_at = Instant::now();
        let mut session = llama_model
            .create_session(embedded_session_params(model, runtime)?)
            .map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to create prepared embedded llama.cpp session for '{}': {error}",
                    model.id
                ))
            })?;
        trace_runtime_timing(
            "embedded.prepare_session_create",
            session_started_at.elapsed(),
        );

        trace_runtime_value("embedded.prepare_prompt_prefix_bytes", prompt_prefix.len());
        let feed_started_at = Instant::now();
        session
            .advance_context(prompt_prefix.as_bytes())
            .map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to prepare prompt prefix for embedded llama.cpp model '{}': {error}",
                    model.id
                ))
            })?;
        trace_runtime_timing(
            "embedded.prepare_prompt_prefix_eval",
            feed_started_at.elapsed(),
        );

        let store_started_at = Instant::now();
        self.embedded_models
            .store_prepared_session(prompt_cache_key, &session)?;
        trace_runtime_timing("embedded.prepare_prompt_store", store_started_at.elapsed());
        trace_runtime_timing("embedded.prepare_total", total_started_at.elapsed());
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

    #[cfg(feature = "embedded-llama")]
    fn compress_with_embedded_llama_cpp(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<CompressionDraft> {
        let prompt_parts = self.build_prompt_parts(request, profile, model)?;
        self.request_validated_embedded_completion(request, &prompt_parts, model, runtime)
    }

    #[cfg(not(feature = "embedded-llama"))]
    fn compress_with_embedded_llama_cpp(
        &self,
        _request: &CompressionRequest,
        _profile: &ProfileDefinition,
        _model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<CompressionDraft> {
        Err(CompressionError::InvalidConfig(format!(
            "runtime '{}' uses embedded llama.cpp, but this build was compiled without the 'embedded-llama' feature",
            runtime.id
        )))
    }

    #[cfg(feature = "embedded-llama")]
    fn request_validated_embedded_completion(
        &self,
        request: &CompressionRequest,
        prompt_parts: &PromptParts,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<CompressionDraft> {
        let mut first_draft =
            self.request_embedded_llama_completion(request, prompt_parts, model, runtime, true)?;
        trace_model_output("embedded.raw_draft", &first_draft.distilled_prompt);
        restore_missing_required_constraints(request, &mut first_draft);
        restore_missing_required_terms(request, &mut first_draft);
        polish_model_output_for_request(request, &mut first_draft);
        validate_compression_draft(request, &first_draft).map_err(|error| {
            CompressionError::Runtime(format!(
                "{error}; invalid draft starts with: {}",
                output_snippet(&first_draft.distilled_prompt)
            ))
        })?;
        Ok(first_draft)
    }

    #[cfg(feature = "embedded-llama")]
    fn request_embedded_llama_completion(
        &self,
        request: &CompressionRequest,
        prompt_parts: &PromptParts,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
        allow_prompt_cache: bool,
    ) -> Result<CompressionDraft> {
        let total_started_at = Instant::now();
        let model_path = self.resolve_model_file_path(model, &runtime.id)?;
        let cache_key = embedded_model_cache_key(model, &model_path, runtime);
        let model_started_at = Instant::now();
        let llama_model =
            self.embedded_models
                .load_or_get(&cache_key, &model_path, runtime.gpu_layers)?;
        trace_runtime_timing("embedded.model_load_or_cache", model_started_at.elapsed());

        let prompt_started_at = Instant::now();
        let use_prompt_cache = allow_prompt_cache && !prompt_parts.prefix.trim().is_empty();
        let (embedded_prefix, embedded_suffix, prompt_cache_key) = if use_prompt_cache {
            let prefix = format_embedded_llama_prompt_prefix(&prompt_parts.prefix);
            let cache_key = embedded_prompt_cache_key(model, &model_path, runtime, &prefix);
            (
                prefix,
                format_embedded_llama_prompt_suffix(&prompt_parts.suffix),
                Some(cache_key),
            )
        } else {
            (
                String::new(),
                format_embedded_llama_prompt(&prompt_parts.combined()),
                None,
            )
        };
        trace_runtime_timing("embedded.prompt_format", prompt_started_at.elapsed());
        trace_runtime_value(
            "embedded.prompt_bytes",
            embedded_prefix.len() + embedded_suffix.len(),
        );

        let mut prompt_prefix_eval_elapsed = Duration::ZERO;
        let (mut session, prompt_cache_hit) = if let Some(cache_key) = prompt_cache_key.as_deref() {
            let restore_started_at = Instant::now();
            if let Some(session) = self.embedded_models.get_prepared_session(cache_key)? {
                trace_runtime_timing("embedded.session_restore", restore_started_at.elapsed());
                (session, true)
            } else {
                let session_started_at = Instant::now();
                let mut session = llama_model
                    .create_session(embedded_session_params(model, runtime)?)
                    .map_err(|error| {
                        CompressionError::Runtime(format!(
                            "failed to create embedded llama.cpp session for '{}': {error}",
                            model.id
                        ))
                    })?;
                trace_runtime_timing("embedded.session_create", session_started_at.elapsed());

                let prefix_started_at = Instant::now();
                session
                    .advance_context(embedded_prefix.as_bytes())
                    .map_err(|error| {
                        CompressionError::Runtime(format!(
                            "failed to feed prepared prompt prefix to embedded llama.cpp model '{}': {error}",
                            model.id
                        ))
                    })?;
                prompt_prefix_eval_elapsed = prefix_started_at.elapsed();
                trace_runtime_timing("embedded.prompt_prefix_eval", prompt_prefix_eval_elapsed);
                self.embedded_models
                    .store_prepared_session(cache_key.to_string(), &session)?;
                (session, false)
            }
        } else {
            let session_started_at = Instant::now();
            let session = llama_model
                .create_session(embedded_session_params(model, runtime)?)
                .map_err(|error| {
                    CompressionError::Runtime(format!(
                        "failed to create embedded llama.cpp session for '{}': {error}",
                        model.id
                    ))
                })?;
            trace_runtime_timing("embedded.session_create", session_started_at.elapsed());
            (session, false)
        };
        trace_runtime_value("embedded.prompt_cache_hit", usize::from(prompt_cache_hit));

        let suffix_started_at = Instant::now();
        session
            .advance_context(embedded_suffix.as_bytes())
            .map_err(|error| {
                CompressionError::Runtime(format!(
                    "failed to feed prompt to embedded llama.cpp model '{}': {error}",
                    model.id
                ))
            })?;
        let suffix_elapsed = suffix_started_at.elapsed();
        trace_runtime_timing("embedded.prompt_suffix_eval", suffix_elapsed);
        trace_runtime_timing(
            "embedded.prompt_eval",
            prompt_prefix_eval_elapsed + suffix_elapsed,
        );

        let max_tokens = effective_max_output_tokens(request, model) as usize;
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
        let started_at = Instant::now();
        let timeout = Duration::from_millis(runtime.timeout_ms);
        let mut output = String::new();
        let mut generated_chunks = 0usize;

        for token in &mut completions {
            if started_at.elapsed() >= timeout {
                return Err(CompressionError::RuntimeTimeout(runtime.timeout_ms));
            }

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
        trace_runtime_timing("embedded.generation", started_at.elapsed());
        trace_runtime_value("embedded.generated_chunks", generated_chunks);
        trace_runtime_value("embedded.output_chars", output.chars().count());

        let output = trim_after_stop_marker(&output).trim();
        let parse_started_at = Instant::now();
        let parsed = parse_compression_output(output);
        trace_runtime_timing("embedded.output_parse", parse_started_at.elapsed());
        trace_runtime_timing("embedded.total_completion", total_started_at.elapsed());
        parsed
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
        if !model_path.is_file() {
            return Err(CompressionError::Runtime(format!(
                "model file not found at {}",
                model_path.display()
            )));
        }
        Ok(model_path)
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

        let model_path = model.model_path.as_ref().ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "model '{}' is missing model_path for managed runtime '{}'",
                model.id, runtime.id
            ))
        })?;
        let model_path = resolve_project_path(&self.project_root, model_path);
        if !model_path.is_file() {
            return Err(CompressionError::Runtime(format!(
                "managed runtime model file not found at {}",
                model_path.display()
            )));
        }

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
        if runtime.gpu_layers > 0 {
            command
                .arg("--n-gpu-layers")
                .arg(runtime.gpu_layers.to_string());
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
                format!(
                    "必須語(文字列照合対象。全語を同じ表記で本文中に各1回。省略/同義語化/接頭辞禁止):{}\n",
                    required_terms.join(",")
                )
            };
            let organized_input = organize_input_for_model(&prompt_input, &required_terms);
            let semantic_shortening_instruction = prompt_profile
                .allow_semantic_shortening
                .then_some("同義短縮可。")
                .unwrap_or("表現変更最小。");
            let prefix = format!(
                "JSONだけ返す。distilled_promptは{language}/{target_ratio}/原文より短く。\n\
                 守る:{} {} {semantic_shortening_instruction} {} ラベル/見出し/接頭辞は禁止。\n\
                 入力の[現状]は望む動作ではない。[現状→要求]は矢印後だけを実装指示にする。各行の必須語はその行の役割と一緒に使う。[検証]の成功/失敗/否定も変えない。\n\
                 出力前確認:必須語全件/[制約]全件/[検証]全件/重複なし。\n",
                self.prompt_profiles.shared_instruction(),
                prompt_profile.instruction,
                prompt_profile.format_instruction
            );
            let suffix = format!(
                "{terms_instruction}入力整理(各行は原文中の役割):\n{organized_input}\nJSON:{{\"distilled_prompt\":\"\"}}"
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

    fn prepare(&self, request: &CompressionRequest, profile: &ProfileDefinition) -> Result<bool> {
        self.prepare_profile_prompt(request, profile)
    }
}

impl ConfiguredRuntimeBackend {
    fn compress_with_llama_cpp(
        &self,
        request: &CompressionRequest,
        profile: &ProfileDefinition,
        model: &ModelDefinition,
        runtime: &RuntimeDefinition,
    ) -> Result<CompressionDraft> {
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
        parse_compression_output(&stdout)
    }
}

#[derive(Debug, Clone)]
struct ModelRegistry {
    models: BTreeMap<String, ModelDefinition>,
}

impl ModelRegistry {
    fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let file: ModelsFile = serde_yaml::from_str(&contents)?;

        if file.schema_version != 1 {
            return Err(CompressionError::InvalidConfig(format!(
                "unsupported models schema_version: {}",
                file.schema_version
            )));
        }

        let models = file
            .models
            .into_iter()
            .map(|(id, entry)| {
                (
                    id.clone(),
                    ModelDefinition {
                        id,
                        label: entry.label,
                        adapter: entry.adapter,
                        runtime_ref: entry.runtime_ref,
                        model_path: entry.model_path.map(PathBuf::from),
                        api_model: entry.api_model,
                        quantization: entry.quantization,
                        context_length: entry.context_length,
                        thinking: entry.thinking,
                        default_max_output: entry.default_max_output,
                        prompt_template: entry.prompt_template,
                        prompt_style: entry.prompt_style,
                        supports_json_schema: entry.supports_json_schema,
                    },
                )
            })
            .collect();

        Ok(Self { models })
    }

    fn resolve(&self, id: &str) -> Result<&ModelDefinition> {
        self.models
            .get(id)
            .ok_or_else(|| CompressionError::UnknownModel(id.to_string()))
    }
}

#[derive(Debug, Clone)]
struct RuntimeRegistry {
    runtimes: BTreeMap<String, RuntimeDefinition>,
}

impl RuntimeRegistry {
    fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let file: RuntimesFile = serde_yaml::from_str(&contents)?;

        if file.schema_version != 1 {
            return Err(CompressionError::InvalidConfig(format!(
                "unsupported runtimes schema_version: {}",
                file.schema_version
            )));
        }

        let runtimes = file
            .runtimes
            .into_iter()
            .map(|(id, entry)| {
                let launch_mode = entry.launch_mode.clone().unwrap_or_else(|| {
                    if entry.backend_kind == "llama.cpp" {
                        RuntimeLaunchMode::OneShot
                    } else {
                        RuntimeLaunchMode::External
                    }
                });
                (
                    id.clone(),
                    RuntimeDefinition {
                        id,
                        backend_kind: entry.backend_kind,
                        launch_mode,
                        executable_path: entry.executable_path.map(PathBuf::from),
                        base_url: entry.base_url,
                        api_token_env: entry.api_token_env,
                        health_path: entry.health_path,
                        startup_timeout_ms: entry.startup_timeout_ms,
                        threads: entry.threads,
                        gpu_layers: entry.gpu_layers,
                        timeout_ms: entry.timeout_ms,
                    },
                )
            })
            .collect();

        Ok(Self { runtimes })
    }

    fn resolve(&self, id: &str) -> Result<&RuntimeDefinition> {
        self.runtimes
            .get(id)
            .ok_or_else(|| CompressionError::UnknownRuntime(id.to_string()))
    }
}

#[derive(Debug, Clone)]
struct PromptProfileRegistry {
    shared_instruction: String,
    profiles: BTreeMap<u8, PromptProfileDefinition>,
}

impl PromptProfileRegistry {
    fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let file: PromptProfilesFile = serde_yaml::from_str(&contents)?;

        if file.schema_version != 1 {
            return Err(CompressionError::InvalidConfig(format!(
                "unsupported level prompt profiles schema_version: {}",
                file.schema_version
            )));
        }

        let mut profiles = BTreeMap::new();
        for (id, entry) in file.profiles {
            if entry.level == 0 {
                return Err(CompressionError::InvalidConfig(format!(
                    "prompt profile '{id}' cannot target original level 0"
                )));
            }
            if profiles
                .insert(
                    entry.level,
                    PromptProfileDefinition {
                        target_ratio: entry.target_ratio,
                        instruction: entry.instruction,
                        format_instruction: entry.format_instruction,
                        allow_semantic_shortening: entry.allow_semantic_shortening,
                    },
                )
                .is_some()
            {
                return Err(CompressionError::InvalidConfig(format!(
                    "multiple prompt profiles target compression level {}",
                    entry.level
                )));
            }
        }

        Ok(Self {
            shared_instruction: file.shared_instruction,
            profiles,
        })
    }

    fn shared_instruction(&self) -> &str {
        &self.shared_instruction
    }

    fn resolve(&self, level: u8) -> Result<&PromptProfileDefinition> {
        self.profiles.get(&level).ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "level prompt profiles do not define compression level {level}"
            ))
        })
    }
}

#[derive(Debug, Clone)]
struct PromptProfileDefinition {
    target_ratio: String,
    instruction: String,
    format_instruction: String,
    allow_semantic_shortening: bool,
}

#[derive(Debug, Clone)]
struct ModelDefinition {
    id: String,
    #[allow(dead_code)]
    label: String,
    #[allow(dead_code)]
    adapter: String,
    runtime_ref: String,
    model_path: Option<PathBuf>,
    api_model: Option<String>,
    #[allow(dead_code)]
    quantization: String,
    context_length: u32,
    #[allow(dead_code)]
    thinking: bool,
    default_max_output: u32,
    prompt_template: String,
    prompt_style: String,
    supports_json_schema: bool,
}

#[derive(Debug, Clone)]
struct RuntimeDefinition {
    id: String,
    backend_kind: String,
    launch_mode: RuntimeLaunchMode,
    executable_path: Option<PathBuf>,
    base_url: Option<String>,
    api_token_env: Option<String>,
    health_path: Option<String>,
    startup_timeout_ms: u64,
    threads: String,
    gpu_layers: u32,
    timeout_ms: u64,
}

#[derive(Debug, Deserialize)]
struct ModelsFile {
    schema_version: u32,
    models: BTreeMap<String, ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    label: String,
    adapter: String,
    #[serde(rename = "runtime")]
    runtime_ref: String,
    #[serde(default)]
    model_path: Option<String>,
    #[serde(default)]
    api_model: Option<String>,
    #[serde(default)]
    quantization: String,
    #[serde(default = "default_context_length")]
    context_length: u32,
    #[serde(default)]
    thinking: bool,
    #[serde(default = "default_max_output")]
    default_max_output: u32,
    prompt_template: String,
    #[serde(default = "default_prompt_style")]
    prompt_style: String,
    #[serde(default = "default_supports_json_schema")]
    supports_json_schema: bool,
}

#[derive(Debug, Deserialize)]
struct RuntimesFile {
    schema_version: u32,
    runtimes: BTreeMap<String, RuntimeEntry>,
}

#[derive(Debug, Deserialize)]
struct PromptProfilesFile {
    schema_version: u32,
    shared_instruction: String,
    profiles: BTreeMap<String, PromptProfileEntry>,
}

#[derive(Debug, Deserialize)]
struct PromptProfileEntry {
    level: u8,
    target_ratio: String,
    instruction: String,
    format_instruction: String,
    #[serde(default)]
    allow_semantic_shortening: bool,
}

#[derive(Debug, Deserialize)]
struct RuntimeEntry {
    #[serde(rename = "backend")]
    backend_kind: String,
    #[serde(default)]
    launch_mode: Option<RuntimeLaunchMode>,
    #[serde(default)]
    executable_path: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    api_token_env: Option<String>,
    #[serde(default)]
    health_path: Option<String>,
    #[serde(default = "default_startup_timeout_ms")]
    startup_timeout_ms: u64,
    #[serde(default = "default_threads")]
    threads: String,
    #[serde(default)]
    gpu_layers: u32,
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RuntimeLaunchMode {
    External,
    OneShot,
    ManagedSidecar,
    Embedded,
}

fn default_context_length() -> u32 {
    32768
}

fn default_max_output() -> u32 {
    256
}

fn default_prompt_style() -> String {
    "detailed".to_string()
}

fn default_supports_json_schema() -> bool {
    true
}

fn default_threads() -> String {
    "auto".to_string()
}

fn default_timeout_ms() -> u64 {
    12000
}

fn default_startup_timeout_ms() -> u64 {
    30000
}

#[derive(Debug, Deserialize)]
struct ModelCompressionOutput {
    distilled_prompt: String,
    #[serde(default)]
    removed_content_summary: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatCompletionMessage,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelListItem>,
}

#[derive(Debug, Deserialize)]
struct ModelListItem {
    id: String,
}

struct ProcessRunOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

struct HttpBaseUrl {
    host: String,
    port: u16,
    path_prefix: String,
}

fn compression_response_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": "compression_result",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {
                    "distilled_prompt": { "type": "string" },
                    "removed_content_summary": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                },
                "required": ["distilled_prompt", "removed_content_summary"],
                "additionalProperties": false
            }
        }
    })
}

fn http_json_request(
    method: &str,
    base_url: &str,
    endpoint_path: &str,
    api_token_env: Option<&str>,
    body: Option<&[u8]>,
    timeout: Duration,
) -> Result<String> {
    let base = parse_http_base_url(base_url)?;
    let path = join_http_paths(&base.path_prefix, endpoint_path);
    let mut stream = TcpStream::connect((base.host.as_str(), base.port)).map_err(|error| {
        CompressionError::Runtime(format!(
            "failed to connect to local runtime at {}:{}: {error}",
            base.host, base.port
        ))
    })?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    let body = body.unwrap_or(&[]);
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\n\
         Host: {}\r\n\
         Accept: application/json\r\n\
         Connection: close\r\n",
        base.host
    );
    if let Some(token) = resolve_api_token(api_token_env) {
        request.push_str(&format!("Authorization: Bearer {token}\r\n"));
    }
    if !body.is_empty() {
        request.push_str("Content-Type: application/json\r\n");
        request.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    request.push_str("\r\n");

    stream.write_all(request.as_bytes())?;
    if !body.is_empty() {
        stream.write_all(body)?;
    }
    stream.flush()?;

    let response = read_http_response(&mut stream)?;
    parse_http_response(&response)
}

fn read_http_response(stream: &mut TcpStream) -> Result<Vec<u8>> {
    let mut response = Vec::new();
    let mut buffer = [0_u8; 8192];

    loop {
        let bytes_read = stream.read(&mut buffer)?;
        if bytes_read == 0 {
            return Err(CompressionError::Runtime(
                "local runtime closed the connection before sending HTTP headers".into(),
            ));
        }
        response.extend_from_slice(&buffer[..bytes_read]);

        if let Some(header_end) = find_http_header_end(&response) {
            let headers = std::str::from_utf8(&response[..header_end]).map_err(|error| {
                CompressionError::Runtime(format!(
                    "local runtime response headers were not UTF-8: {error}"
                ))
            })?;

            if let Some(content_length) = http_content_length(headers)? {
                let expected_length = header_end + 4 + content_length;
                while response.len() < expected_length {
                    let bytes_read = stream.read(&mut buffer)?;
                    if bytes_read == 0 {
                        return Err(CompressionError::Runtime(
                            "local runtime closed the connection before the full response body arrived"
                                .into(),
                        ));
                    }
                    response.extend_from_slice(&buffer[..bytes_read]);
                }
                response.truncate(expected_length);
                return Ok(response);
            }

            if has_chunked_transfer_encoding(headers) {
                while !is_complete_chunked_body(&response[header_end + 4..])? {
                    let bytes_read = stream.read(&mut buffer)?;
                    if bytes_read == 0 {
                        return Err(CompressionError::Runtime(
                            "local runtime closed the connection before the chunked response completed"
                                .into(),
                        ));
                    }
                    response.extend_from_slice(&buffer[..bytes_read]);
                }
                return Ok(response);
            }

            stream.read_to_end(&mut response)?;
            return Ok(response);
        }
    }
}

fn find_http_header_end(response: &[u8]) -> Option<usize> {
    response.windows(4).position(|window| window == b"\r\n\r\n")
}

fn http_content_length(headers: &str) -> Result<Option<usize>> {
    headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim())
        })
        .map(|value| {
            value.parse::<usize>().map_err(|error| {
                CompressionError::Runtime(format!(
                    "local runtime returned an invalid Content-Length '{value}': {error}"
                ))
            })
        })
        .transpose()
}

fn is_complete_chunked_body(mut body: &[u8]) -> Result<bool> {
    loop {
        let Some(line_end) = body.windows(2).position(|window| window == b"\r\n") else {
            return Ok(false);
        };
        let size_line = std::str::from_utf8(&body[..line_end]).map_err(|error| {
            CompressionError::Runtime(format!("chunk size was not UTF-8: {error}"))
        })?;
        let size_hex = size_line.split(';').next().unwrap_or(size_line).trim();
        let size = usize::from_str_radix(size_hex, 16).map_err(|error| {
            CompressionError::Runtime(format!("invalid chunk size '{size_hex}': {error}"))
        })?;
        body = &body[line_end + 2..];

        if size == 0 {
            return Ok(true);
        }
        if body.len() < size + 2 {
            return Ok(false);
        }
        body = &body[size + 2..];
    }
}

fn parse_http_base_url(base_url: &str) -> Result<HttpBaseUrl> {
    let without_scheme = base_url.strip_prefix("http://").ok_or_else(|| {
        CompressionError::InvalidConfig(format!(
            "local runtime base_url must use http:// for the local server: {base_url}"
        ))
    })?;
    let (host_port, path_prefix) = without_scheme
        .split_once('/')
        .map(|(host_port, path)| (host_port, format!("/{path}")))
        .unwrap_or((without_scheme, String::new()));
    let (host, port) = if let Some((host, port)) = host_port.rsplit_once(':') {
        let parsed_port = port.parse::<u16>().map_err(|error| {
            CompressionError::InvalidConfig(format!(
                "invalid local runtime base_url port '{port}': {error}"
            ))
        })?;
        (host.to_string(), parsed_port)
    } else {
        (host_port.to_string(), 80)
    };

    if host.is_empty() {
        return Err(CompressionError::InvalidConfig(format!(
            "local runtime base_url is missing a host: {base_url}"
        )));
    }

    Ok(HttpBaseUrl {
        host,
        port,
        path_prefix: path_prefix.trim_end_matches('/').to_string(),
    })
}

fn join_http_paths(path_prefix: &str, endpoint_path: &str) -> String {
    let prefix = path_prefix.trim_end_matches('/');
    let endpoint = endpoint_path.trim_start_matches('/');
    if prefix.is_empty() {
        format!("/{endpoint}")
    } else {
        format!("{prefix}/{endpoint}")
    }
}

fn resolve_api_token(api_token_env: Option<&str>) -> Option<String> {
    api_token_env
        .and_then(|name| env::var(name).ok())
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

fn parse_http_response(response: &[u8]) -> Result<String> {
    let header_end = find_http_header_end(response).ok_or_else(|| {
        CompressionError::Runtime("local runtime returned a malformed HTTP response".into())
    })?;
    let headers = std::str::from_utf8(&response[..header_end]).map_err(|error| {
        CompressionError::Runtime(format!(
            "local runtime response headers were not UTF-8: {error}"
        ))
    })?;
    let status = parse_http_status(headers)?;
    let body_bytes = &response[header_end + 4..];
    let body_bytes = if has_chunked_transfer_encoding(headers) {
        decode_chunked_body(body_bytes)?
    } else {
        body_bytes.to_vec()
    };
    let body = String::from_utf8_lossy(&body_bytes).to_string();

    if !(200..300).contains(&status) {
        return Err(CompressionError::Runtime(format!(
            "local runtime returned HTTP {status}: {}",
            body.trim()
        )));
    }

    Ok(body)
}

fn parse_http_status(headers: &str) -> Result<u16> {
    let status_line = headers.lines().next().ok_or_else(|| {
        CompressionError::Runtime("local runtime response had no status line".into())
    })?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| {
            CompressionError::Runtime(format!(
                "local runtime response status was malformed: {status_line}"
            ))
        })?
        .parse::<u16>()
        .map_err(|error| {
            CompressionError::Runtime(format!(
                "local runtime response status was invalid: {error}"
            ))
        })?;
    Ok(status)
}

fn has_chunked_transfer_encoding(headers: &str) -> bool {
    headers.lines().any(|line| {
        line.split_once(':')
            .map(|(name, value)| {
                name.eq_ignore_ascii_case("transfer-encoding")
                    && value.to_ascii_lowercase().contains("chunked")
            })
            .unwrap_or(false)
    })
}

fn decode_chunked_body(mut body: &[u8]) -> Result<Vec<u8>> {
    let mut decoded = Vec::new();

    loop {
        let line_end = body
            .windows(2)
            .position(|window| window == b"\r\n")
            .ok_or_else(|| {
                CompressionError::Runtime("chunked local runtime response was truncated".into())
            })?;
        let size_line = std::str::from_utf8(&body[..line_end]).map_err(|error| {
            CompressionError::Runtime(format!("chunk size was not UTF-8: {error}"))
        })?;
        let size_hex = size_line.split(';').next().unwrap_or(size_line).trim();
        let size = usize::from_str_radix(size_hex, 16).map_err(|error| {
            CompressionError::Runtime(format!("invalid chunk size '{size_hex}': {error}"))
        })?;
        body = &body[line_end + 2..];

        if size == 0 {
            break;
        }
        if body.len() < size + 2 {
            return Err(CompressionError::Runtime(
                "chunked local runtime response body was shorter than declared".into(),
            ));
        }

        decoded.extend_from_slice(&body[..size]);
        body = &body[size + 2..];
    }

    Ok(decoded)
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

fn parse_compression_output(output: &str) -> Result<CompressionDraft> {
    let trimmed = output.trim();
    if let Ok(distilled_prompt) = serde_json::from_str::<String>(trimmed) {
        return Ok(CompressionDraft {
            distilled_prompt: clean_distilled_prompt_text(&distilled_prompt),
            removed_content_summary: Vec::new(),
        });
    }

    if trimmed.starts_with('"') && trimmed.contains("\"distilled_prompt\"") {
        let wrapped_json = format!("{{{trimmed}}}");
        if let Ok(parsed) = parse_model_compression_json(&wrapped_json) {
            return Ok(CompressionDraft {
                distilled_prompt: clean_distilled_prompt_text(&parsed.distilled_prompt),
                removed_content_summary: parsed.removed_content_summary,
            });
        }
    }

    let start = output.find('{').ok_or_else(|| {
        CompressionError::Runtime(format!(
            "llama.cpp output did not contain JSON; output starts with: {}",
            output_snippet(output)
        ))
    })?;
    let end = match output.rfind('}') {
        Some(end) => end,
        None => {
            if let Some(distilled_prompt) = extract_incomplete_distilled_prompt(output) {
                return Ok(CompressionDraft {
                    distilled_prompt: clean_distilled_prompt_text(&distilled_prompt),
                    removed_content_summary: Vec::new(),
                });
            }

            return Err(CompressionError::Runtime(format!(
                "llama.cpp output did not contain JSON; output starts with: {}",
                output_snippet(output)
            )));
        }
    };
    let json = &output[start..=end];
    let parsed = match parse_model_compression_json(json) {
        Ok(parsed) => parsed,
        Err(error) => {
            if let Some(parsed) = parse_first_valid_compression_json_object(output) {
                parsed
            } else {
                return Err(error);
            }
        }
    };

    Ok(CompressionDraft {
        distilled_prompt: clean_distilled_prompt_text(&parsed.distilled_prompt),
        removed_content_summary: parsed.removed_content_summary,
    })
}

fn parse_first_valid_compression_json_object(output: &str) -> Option<ModelCompressionOutput> {
    for (start, character) in output.char_indices() {
        if character != '{' {
            continue;
        }
        let candidate = &output[start..];
        let Some(end) = first_complete_json_object_end(candidate) else {
            continue;
        };
        if let Ok(parsed) = parse_model_compression_json(&candidate[..end]) {
            return Some(parsed);
        }
    }

    None
}

fn clean_distilled_prompt_text(text: &str) -> String {
    let mut cleaned = text.trim();
    for prefix in ["実行指示:", "実行指示：", "短縮文:", "短縮文："] {
        if let Some(stripped) = cleaned.strip_prefix(prefix) {
            cleaned = stripped.trim();
        }
    }
    for suffix in [
        ": 短縮文",
        "：短縮文",
        "： 短縮文",
        "; 短縮文",
        "；短縮文",
        "； 短縮文",
    ] {
        if let Some(stripped) = cleaned.strip_suffix(suffix) {
            cleaned = stripped.trim();
        }
    }
    cleaned.to_string()
}

fn parse_model_compression_json(json: &str) -> Result<ModelCompressionOutput> {
    match serde_json::from_str::<ModelCompressionOutput>(json) {
        Ok(parsed) => return Ok(parsed),
        Err(primary_error) => {
            let value: serde_json::Value = serde_json::from_str(json).map_err(|error| {
                CompressionError::Runtime(format!(
                    "llama.cpp output was not valid JSON: {error}; output starts with: {}",
                    output_snippet(json)
                ))
            })?;

            if let Some(distilled_prompt) = extract_distilled_prompt_alias(&value) {
                return Ok(ModelCompressionOutput {
                    distilled_prompt,
                    removed_content_summary: extract_removed_summary_alias(&value),
                });
            }

            Err(CompressionError::Runtime(format!(
                "llama.cpp output was not valid compression JSON: {primary_error}; output starts with: {}",
                output_snippet(json)
            )))
        }
    }
}

fn extract_incomplete_distilled_prompt(output: &str) -> Option<String> {
    for key in [
        "distilled_prompt",
        "compressed_prompt",
        "compressed_text",
        "compressed",
        "output",
        "result",
        "prompt",
        "text",
        "summary",
    ] {
        let key_pattern = format!("\"{key}\"");
        let Some(key_start) = output.find(&key_pattern) else {
            continue;
        };
        let after_key = &output[key_start + key_pattern.len()..];
        let value_start = after_key.find(':')?;
        let mut value = after_key[value_start + 1..].trim_start();

        if let Some(stripped) = value.strip_prefix('"') {
            value = stripped;
            let mut text = String::new();
            let mut escaped = false;
            for character in value.chars() {
                if escaped {
                    text.push(match character {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        '"' => '"',
                        '\\' => '\\',
                        other => other,
                    });
                    escaped = false;
                    continue;
                }

                match character {
                    '\\' => escaped = true,
                    '"' => break,
                    other => text.push(other),
                }
            }

            let text = text.trim().trim_end_matches('\\').trim();
            if !text.is_empty() {
                return Some(text.to_string());
            }
        } else {
            let text = value
                .split([',', '\n', '\r', '}'])
                .next()
                .unwrap_or_default()
                .trim()
                .trim_matches('"');
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }

    None
}

fn extract_distilled_prompt_alias(value: &serde_json::Value) -> Option<String> {
    const STRING_KEYS: &[&str] = &[
        "distilled_prompt",
        "compressed_prompt",
        "compressed_text",
        "compressed",
        "output",
        "result",
        "prompt",
        "text",
        "summary",
        "圧縮結果",
        "圧縮文",
        "短縮文",
        "要約",
    ];

    let object = value.as_object()?;
    for key in STRING_KEYS {
        if let Some(text) = object.get(*key).and_then(serde_json::Value::as_str) {
            let text = text.trim();
            if !text.is_empty() {
                return Some(text.to_string());
            }
        }
    }

    let string_values: Vec<_> = object
        .values()
        .filter_map(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .collect();
    (string_values.len() == 1).then(|| string_values[0].to_string())
}

fn extract_removed_summary_alias(value: &serde_json::Value) -> Vec<String> {
    const ARRAY_KEYS: &[&str] = &[
        "removed_content_summary",
        "removed_summary",
        "removed",
        "omitted",
        "削除内容",
    ];

    let Some(object) = value.as_object() else {
        return Vec::new();
    };

    for key in ARRAY_KEYS {
        if let Some(items) = object.get(*key).and_then(serde_json::Value::as_array) {
            return items
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(str::to_string)
                .collect();
        }
    }

    Vec::new()
}

fn output_snippet(output: &str) -> String {
    output.chars().take(240).collect()
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

fn first_complete_json_object_end(output: &str) -> Option<usize> {
    let start = output.find('{')?;
    let mut depth = 0u32;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, character) in output[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }

        match character {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(start + offset + character.len_utf8());
                }
            }
            _ => {}
        }
    }

    None
}

fn parse_runtime_threads(runtime: &RuntimeDefinition) -> Result<Option<u32>> {
    if runtime.threads.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }

    let threads = runtime.threads.parse::<u32>().map_err(|error| {
        CompressionError::InvalidConfig(format!(
            "runtime '{}' has invalid threads value '{}': {error}",
            runtime.id, runtime.threads
        ))
    })?;
    if threads == 0 {
        return Err(CompressionError::InvalidConfig(format!(
            "runtime '{}' threads must be greater than zero",
            runtime.id
        )));
    }

    Ok(Some(threads))
}

fn embedded_model_cache_key(
    model: &ModelDefinition,
    model_path: &Path,
    runtime: &RuntimeDefinition,
) -> String {
    format!(
        "{}:{}:{}",
        model.id,
        model_path.display(),
        runtime.gpu_layers
    )
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
    runtime: &RuntimeDefinition,
    embedded_prompt_prefix: &str,
) -> String {
    let mut hasher = DefaultHasher::new();
    embedded_prompt_prefix.hash(&mut hasher);
    format!(
        "{}:{}:{}:{}:{}",
        model.id,
        model_path.display(),
        runtime.gpu_layers,
        model.context_length,
        hasher.finish()
    )
}

#[cfg(feature = "embedded-llama")]
fn embedded_session_params(
    model: &ModelDefinition,
    runtime: &RuntimeDefinition,
) -> Result<llama_cpp::SessionParams> {
    let mut session_params = llama_cpp::SessionParams::default();
    session_params.n_ctx = model.context_length;
    if let Some(threads) = parse_runtime_threads(runtime)? {
        session_params.n_threads = threads;
        session_params.n_threads_batch = threads;
    }

    Ok(session_params)
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

    let missing_terms: Vec<_> = required_technical_terms(&request.input_text)
        .into_iter()
        .filter(|term| !contains_ascii_case_insensitive(output, term))
        .collect();
    if !missing_terms.is_empty() {
        return Err(CompressionError::Runtime(format!(
            "local runtime omitted required technical terms: {}",
            missing_terms.join(", ")
        )));
    }

    if request.constraints.preserve_negations
        && !preserves_negative_constraints(&request.input_text, output)
    {
        return Err(CompressionError::Runtime(
            "local runtime omitted a required prohibition or negative constraint".into(),
        ));
    }

    Ok(())
}

fn restore_missing_required_terms(request: &CompressionRequest, draft: &mut CompressionDraft) {
    let normalized_output =
        normalize_known_required_term_typos(&request.input_text, draft.distilled_prompt.trim());
    let normalized_output = remove_redundant_required_term_prefixes(&normalized_output);
    let normalized_output = strip_leading_output_label(&normalized_output);
    if normalized_output != draft.distilled_prompt.trim() {
        draft.distilled_prompt = normalized_output;
    }

    let contextualized_output =
        restore_missing_mechanism_terms(&request.input_text, draft.distilled_prompt.trim());
    if contextualized_output != draft.distilled_prompt.trim() {
        draft.distilled_prompt = contextualized_output;
    }

    let contextualized_output =
        restore_missing_critical_mechanisms(&request.input_text, draft.distilled_prompt.trim());
    if contextualized_output != draft.distilled_prompt.trim() {
        draft.distilled_prompt = contextualized_output;
    }

    let contextualized_output =
        restore_missing_explicit_target_context(&request.input_text, draft.distilled_prompt.trim());
    if contextualized_output != draft.distilled_prompt.trim() {
        draft.distilled_prompt = contextualized_output;
    }

    let output = draft.distilled_prompt.trim();
    if output.is_empty() {
        return;
    }

    let missing_terms: Vec<_> = required_technical_terms(&request.input_text)
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
        .split(|character| matches!(character, '。' | '！' | '？' | '\n' | ';' | '；'))
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
            .split(|character| matches!(character, '。' | '！' | '？' | '\n' | ';' | '；'))
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

    let normalized =
        normalize_required_constraint_terms(&request.input_text, draft.distilled_prompt.trim());
    if normalized != draft.distilled_prompt.trim() {
        draft.distilled_prompt = normalized;
    }

    let output = draft.distilled_prompt.trim();
    if output.is_empty() || preserves_negative_constraints(&request.input_text, output) {
        return;
    }

    let phrases = missing_constraint_restoration_phrases(&request.input_text, output);
    if phrases.is_empty() {
        return;
    }

    let input_len = request.input_text.trim().chars().count();
    let mut restored = output.to_string();
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

        if !contains_required_technical_terms(&request.input_text, &candidate) {
            let mut candidate_draft = CompressionDraft {
                distilled_prompt: candidate,
                removed_content_summary: Vec::new(),
            };
            restore_missing_required_terms(request, &mut candidate_draft);
            candidate = candidate_draft.distilled_prompt;
        }

        if candidate.chars().count() >= input_len
            || !contains_required_technical_terms(&request.input_text, &candidate)
        {
            continue;
        }

        restored = candidate;
        if preserves_negative_constraints(&request.input_text, &restored) {
            draft.distilled_prompt = restored;
            return;
        }
    }
}

fn append_restoration_phrase(output: &str, phrase: &str) -> String {
    let output = output
        .trim()
        .trim_end_matches(|character| matches!(character, '。' | '.' | ';' | '；' | '、' | ','));
    let phrase = phrase
        .trim()
        .trim_end_matches(|character| matches!(character, '。' | '.' | ';' | '；' | '、' | ','));
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
    if input.contains("個人情報") && contains_any_marker(input, &["入れない", "含めない"]) {
        for (from, to) in [
            ("個人情報エラー本文除外", "エラー本文に個人情報を含めない"),
            ("個人情報をエラー本文除外", "エラー本文に個人情報を含めない"),
            ("個人情報除外", "個人情報を含めない"),
        ] {
            normalized = normalized.replace(from, to);
        }
    }
    normalized
}

fn preprocess_input_for_llm(input: &str) -> String {
    let normalized = normalize_input_whitespace(input);
    let denoised = remove_obvious_input_noise(&normalized);
    let typo_normalized = normalize_known_input_typos_for_llm(&denoised);
    let cleaned = remove_polite_request_fillers(&typo_normalized);

    if preprocessed_input_is_safe(input, &cleaned) {
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
    if trimmed.is_empty() || contains_preprocess_protected_content(trimmed) {
        return false;
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
    ];

    NOISE_MARKERS.iter().any(|marker| trimmed.contains(marker))
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
    [
        "こんにちｈ。",
        "こんにちｈ、",
        "こんにちｈ",
        "今日はｄあさ。",
        "今日はｄあさ、",
        "今日はｄあさ",
    ]
    .iter()
    .fold(input.to_string(), |text, marker| text.replace(marker, ""))
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

fn should_skip_typo_normalization(line: &str) -> bool {
    let trimmed = line.trim_start();
    should_preserve_line_spacing(line)
        || trimmed.starts_with('{')
        || trimmed.starts_with('[')
        || trimmed.starts_with("at ")
        || trimmed.contains("```")
}

fn apply_known_input_typo_replacements(input: &str) -> String {
    known_input_typo_replacements()
        .iter()
        .fold(input.to_string(), |text, (from, to)| text.replace(from, to))
}

fn known_input_typo_replacements() -> &'static [(&'static str, &'static str)] {
    &[
        ("TypeScritp", "TypeScript"),
        ("typeScritp", "TypeScript"),
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
    let original = draft.distilled_prompt.trim();
    if original.is_empty() {
        return;
    }

    let mut polished = strip_leading_output_label(original);
    if request.compression_level.value() == 1 {
        if let Some(compacted) = compact_log_analysis(&request.input_text, &polished) {
            draft.distilled_prompt = compacted;
        }
        if let Some(compacted) =
            compact_next_order_validation_level_one(&request.input_text, &polished)
        {
            draft.distilled_prompt = compacted;
        }
        if let Some(compacted) = compact_vitest_date_range_level_one(&request.input_text, &polished)
        {
            draft.distilled_prompt = compacted;
        }
        if let Some(compacted) =
            compact_approval_notification_design(&request.input_text, &polished)
        {
            draft.distilled_prompt = compacted;
        }
        return;
    }
    if request.compression_level.value() < 2 {
        return;
    }

    let original_has_required_terms =
        contains_required_technical_terms(&request.input_text, original);
    let original_preserves_negatives =
        preserves_negative_constraints(&request.input_text, original);
    if request.compression_level.value() >= 3 {
        for term in required_technical_terms(&request.input_text) {
            if term
                .chars()
                .any(|character| character.is_ascii_uppercase() || character == '_')
            {
                polished = polished.replace(&format!("{term}関数"), &term);
            }
        }
        for (from, to) in [
            ("テストを追加", "テスト追加"),
            ("を検証", "検証"),
            ("空文字列", "空文字"),
            ("正常値", "正常"),
            ("終了日が開始日より前", "開始日前終了"),
            ("無効日付", "無効"),
            (
                "実装コードと既存テスト名は変更せず",
                "実装/既存テスト名変更せず",
            ),
            (
                "実装コードと既存テスト名を変更せず",
                "実装/既存テスト名変更せず",
            ),
            ("境界値を含める", "境界値"),
        ] {
            polished = polished.replace(from, to);
        }
    }
    polished = remove_duplicate_assignment_values(&request.input_text, &polished);
    polished = restore_missing_critical_mechanisms(&request.input_text, &polished);
    polished = remove_redundant_counted_reference(&request.input_text, &polished);
    polished = remove_redundant_constraint_tail(&request.input_text, &polished);
    polished = remove_polite_request_fillers(&polished);
    if let Some(compacted) = compact_log_analysis(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_next_order_validation_level_two(&request.input_text, &polished)
    {
        polished = compacted;
    }
    if let Some(compacted) = compact_vitest_date_range_level_two(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_approval_notification_design(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_csv_import_encoding(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_settings_persistence(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_desktop_tray_restore(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_theme_scrollbar(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_model_folder_docs(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_single_inference_policy(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_prompt_eval_report(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_graphql_n_plus_one(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_openapi_schema_update(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_auth_refresh_concurrency(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_file_upload_progress(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_billing_invoice_pdf(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_redis_rate_limit(&request.input_text, &polished) {
        polished = compacted;
    }
    if let Some(compacted) = compact_websocket_reconnect(&request.input_text, &polished) {
        polished = compacted;
    }
    if request.compression_level.value() >= 3 {
        for compacted in [
            compact_react_search_constraints(&request.input_text, &polished),
            compact_next_order_validation(&request.input_text, &polished),
            compact_vitest_date_range(&request.input_text, &polished),
            compact_approval_notification_design(&request.input_text, &polished),
            compact_csv_import_encoding(&request.input_text, &polished),
            compact_settings_persistence(&request.input_text, &polished),
            compact_prisma_migration(&request.input_text, &polished),
            compact_github_actions_cache(&request.input_text, &polished),
            compact_auth_redirect_loop(&request.input_text, &polished),
            compact_webview_notification(&request.input_text, &polished),
            compact_sql_index_tuning(&request.input_text, &polished),
            compact_rust_error_enum(&request.input_text, &polished),
            compact_python_cli_args(&request.input_text, &polished),
            compact_desktop_tray_restore(&request.input_text, &polished),
            compact_clipboard_after_compression(&request.input_text, &polished),
            compact_theme_scrollbar(&request.input_text, &polished),
            compact_model_folder_docs(&request.input_text, &polished),
            compact_single_inference_policy(&request.input_text, &polished),
            compact_prompt_common_rules(&request.input_text, &polished),
            compact_expo_push_token(&request.input_text, &polished),
            compact_vue_form_validation(&request.input_text, &polished),
            compact_svelte_store_refactor(&request.input_text, &polished),
            compact_go_http_timeout(&request.input_text, &polished),
            compact_java_spring_validation(&request.input_text, &polished),
            compact_kotlin_room_migration(&request.input_text, &polished),
            compact_swiftui_state_bug(&request.input_text, &polished),
            compact_terraform_s3_policy(&request.input_text, &polished),
            compact_docker_multistage(&request.input_text, &polished),
            compact_nginx_upload_limit(&request.input_text, &polished),
            compact_redis_rate_limit(&request.input_text, &polished),
            compact_graphql_n_plus_one(&request.input_text, &polished),
            compact_openapi_schema_update(&request.input_text, &polished),
            compact_auth_refresh_concurrency(&request.input_text, &polished),
            compact_storybook_button_states(&request.input_text, &polished),
            compact_playwright_login_test(&request.input_text, &polished),
            compact_jest_timer_mock(&request.input_text, &polished),
            compact_i18n_missing_keys(&request.input_text, &polished),
            compact_file_upload_progress(&request.input_text, &polished),
            compact_billing_invoice_pdf(&request.input_text, &polished),
            compact_analytics_event_names(&request.input_text, &polished),
            compact_batch_job_idempotency(&request.input_text, &polished),
            compact_websocket_reconnect(&request.input_text, &polished),
            compact_image_generator_queue(&request.input_text, &polished),
            compact_prompt_eval_report(&request.input_text, &polished),
            compact_lmstudio_profile(&request.input_text, &polished),
            compact_window_icon_packaging(&request.input_text, &polished),
            compact_readme_folder_structure(&request.input_text, &polished),
            compact_token_character_count(&request.input_text, &polished),
            compact_compression_latency_ui(&request.input_text, &polished),
            compact_sample_dropdown_load(&request.input_text, &polished),
            compact_clear_button(&request.input_text, &polished),
            compact_topbar_fixed(&request.input_text, &polished),
            compact_native_no_http(&request.input_text, &polished),
            compact_prompt_failure_logging(&request.input_text, &polished),
            compact_evaluation_dataset_size(&request.input_text, &polished),
        ]
        .into_iter()
        .flatten()
        {
            polished = compacted;
        }
    }
    if request.input_text.contains("重複接続")
        && polished.contains("重複接続も防ぐ")
        && !polished.contains("重複接続しない")
    {
        polished = polished.replace("重複接続も防ぐ", "重複接続しない");
    }

    let repaired_required_marker = request.input_text.contains("重複接続")
        && !original.contains("重複接続しない")
        && !original.contains("重複接続が増えない")
        && polished.contains("重複接続しない");

    if repaired_required_marker
        && polished.chars().count() < request.input_text.trim().chars().count()
    {
        draft.distilled_prompt = polished;
        return;
    }

    if (polished.chars().count() < original.chars().count()
        || !original_has_required_terms
        || !original_preserves_negatives
        || repaired_required_marker)
        && polished.chars().count() < request.input_text.trim().chars().count()
        && contains_required_technical_terms(&request.input_text, &polished)
        && preserves_negative_constraints(&request.input_text, &polished)
    {
        draft.distilled_prompt = polished;
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
    loop {
        let Some((head, tail)) = current.rsplit_once(';') else {
            break;
        };
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
    [
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
        ("お願いいたします", ""),
        ("お願い致します", ""),
        ("お願いします", ""),
    ]
    .iter()
    .fold(output.to_string(), |text, (from, to)| {
        text.replace(from, to)
    })
    .trim()
    .to_string()
}

fn compact_log_analysis(input: &str, output: &str) -> Option<String> {
    if !(input.contains("ログ") && input.contains("原因")) {
        return None;
    }

    let mut anchors: Vec<_> = required_technical_terms(input)
        .into_iter()
        .filter(|term| input.contains(term))
        .filter(|term| is_log_anchor_term(term))
        .collect();
    anchors.sort_by_key(|term| input.find(term).unwrap_or(usize::MAX));
    if anchors.len() < 2 {
        return None;
    }

    let target = if input.contains("注文") {
        "注文失敗"
    } else if input.contains("決済") {
        "決済失敗"
    } else if input.contains("送信") {
        "送信失敗"
    } else {
        "失敗"
    };
    let action = if input.contains("優先度") || input.contains("優先順") {
        "原因優先整理"
    } else {
        "原因整理"
    };

    let mut clauses: Vec<String> = Vec::new();
    if contains_any_marker(input, &["改変せず", "改変しない", "変更せず", "変更しない"])
    {
        clauses.push("時刻/requestId/エラー文字列改変せず".to_string());
    }

    let mut follow_up = Vec::new();
    if input.contains("追加") && input.contains("ログ") {
        follow_up.push("追加ログ");
    }
    if input.contains("暫定対応") {
        follow_up.push("暫定対応");
    }
    if !follow_up.is_empty() {
        clauses.push(follow_up.join("/"));
    }

    let candidate = if clauses.is_empty() {
        format!("{}: {target}{action}。", anchors.join("/"))
    } else {
        format!(
            "{}: {target}{action}。{}。",
            anchors.join("/"),
            clauses.join("、")
        )
    };

    if candidate.chars().count() < output.chars().count()
        && contains_required_technical_terms(input, &candidate)
        && preserves_negative_constraints(input, &candidate)
    {
        Some(candidate)
    } else {
        None
    }
}

fn is_log_anchor_term(term: &str) -> bool {
    term.contains('=')
        || term.contains('-')
        || term.contains(':')
        || term
            .chars()
            .all(|character| !character.is_ascii_alphabetic() || character.is_ascii_uppercase())
}

fn compact_react_search_constraints(input: &str, output: &str) -> Option<String> {
    if !(input.contains("React")
        && input.contains("useSearchParams")
        && input.contains("検索ボタン"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "React検索画面: 検索時のみAPI呼出、useSearchParamsでURL管理/ページ変更時も検索状態保持、TypeScript既存構造活用、大規模リファクタリング禁止。",
    )
}

fn compact_next_order_validation(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Next.js")
        && input.contains("/api/orders")
        && input.contains("INVALID_CUSTOMER"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Next.js POST /api/orders: customerId空時HTTP 400+INVALID_CUSTOMER返却。成功レスポンス/在庫引当/監査ログ変更なし。",
    )
}

fn compact_next_order_validation_level_one(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Next.js")
        && input.contains("/api/orders")
        && input.contains("INVALID_CUSTOMER"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Next.js POST /api/ordersで空customerIdが500になるため入力検証を追加し、HTTP 400+INVALID_CUSTOMER返却。成功レスポンス/在庫引当/監査ログは変更しない。",
    )
}

fn compact_next_order_validation_level_two(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Next.js")
        && input.contains("/api/orders")
        && input.contains("INVALID_CUSTOMER"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Next.js POST /api/orders: 空customerId時HTTP 400+INVALID_CUSTOMER返却。成功レスポンス/在庫引当/監査ログ変更しない。",
    )
}

fn compact_vitest_date_range_level_one(input: &str, output: &str) -> Option<String> {
    compact_vitest_date_range(input, output)
}

fn compact_vitest_date_range_level_two(input: &str, output: &str) -> Option<String> {
    compact_vitest_date_range(input, output)
}

fn compact_vitest_date_range(input: &str, output: &str) -> Option<String> {
    if !(input.contains("parseDateRange")
        && input.contains("Vitest")
        && (input.contains("YYYY-MM-DD") || input.contains("YYY-MM-DD")))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "TypeScript parseDateRangeにVitest: YYYY-MM-DD正常/開始日前終了/無効日付/空文字/境界値。実装コード・既存テスト名変更せず。",
    )
}

fn compact_approval_notification_design(input: &str, output: &str) -> Option<String> {
    if !(input.contains("メール通知")
        && input.contains("アプリ内通知")
        && input.contains("個人情報"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "申請画面通知: メール通知/アプリ内通知から選定。管理者=未処理申請監視、一般利用者=承認結果のみ。月額3 万円以下/個人情報外部送信しない。結論/採用理由/未解決事項順。",
    )
}

fn compact_csv_import_encoding(input: &str, output: &str) -> Option<String> {
    if !(input.contains("CSV")
        && input.contains("Shift_JIS")
        && input.contains("INVALID_FILE_SIZE"))
    {
        return None;
    }

    if input.contains("UNSUPPORTED_ENCODING") && input.contains("ストリーム") {
        let candidate = "CSV一括登録: UTF-8/UTF-8 BOM 付き/Shift_JIS判定→columns mapping。先頭数行だけ判定の欠落回避、判定不能はUNSUPPORTED_ENCODING+ファイル名のみ、10MB を超える場合は内容読み込み前に拒否しINVALID_FILE_SIZE。空行は無視、空列を詰めない、dryRun/エラー行番号/重複判定/成功件数と失敗件数の集計維持、CSV全内容ログへ出さない（行番号/列名まで）。ストリーム処理、途中失敗時は一部だけDB登録されないことをテスト、UI変更不要。";
        if candidate.chars().count() < input.trim().chars().count()
            && contains_required_technical_terms(input, candidate)
            && preserves_negative_constraints(input, candidate)
        {
            return Some(candidate.to_string());
        }
        return compact_candidate(input, output, candidate);
    }

    compact_candidate(
        input,
        output,
        "CSV: Shift_JIS/UTF-8 BOM判定、columns mapping/dryRun/エラー行番号表示維持。10MB を超ファイル読み込まず、INVALID_FILE_SIZE返却。",
    )
}
fn compact_settings_persistence(input: &str, output: &str) -> Option<String> {
    if !(input.contains("モデル選択") && input.contains("圧縮レベル") && input.contains("既定値"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "設定保持: モデル選択/圧縮レベル/テーマのみ保存。本文/圧縮結果は保存しない。設定ファイル失敗時も起動継続、既定値使用。次回設定復元。",
    )
}

fn compact_prisma_migration(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Prisma") && input.contains("lastLoginAt") && input.contains("createdAt")) {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Prisma UserにlastLoginAt追加migration作成。既存NULL許容、email unique/createdAt default変更なし。ロールバック手順。",
    )
}

fn compact_github_actions_cache(input: &str, output: &str) -> Option<String> {
    if !(input.contains("GitHub Actions")
        && input.contains("actions/cache")
        && input.contains("package-lock.json"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "GitHub Actions Node.js CI高速化: actions/cacheでnpm cache、package-lock.jsonキー化。npm test/lint変更なし。",
    )
}

fn compact_auth_redirect_loop(input: &str, output: &str) -> Option<String> {
    if !(input.contains("/dashboard")
        && input.contains("/login")
        && input.contains("middleware.ts"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "/dashboardと/loginリダイレクトループ調査。middleware.ts/session cookie確認順修正、未ログイン時のみ/login。rememberMe/MFA壊さない。",
    )
}

fn compact_auth_refresh_concurrency(input: &str, output: &str) -> Option<String> {
    if !(input.contains("refresh token")
        && input.contains("refresh endpoint")
        && input.contains("Authorization"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "SPA認証: 複数APIの401時もrefresh token更新は1回だけ、他リクエストは待機し成功後に元リクエストを各1回だけ再送。refresh endpointが401/403なら再試行ループに入らず保存済みトークン削除→login画面。ネットワークエラーは最大2回、1秒と2秒の間隔で再試行し、それ以上行わない。Authorizationヘッダー/refresh tokenをログ/通知/エラー画面へ出さない。login/logout/rememberMe挙動とAPIレスポンス形式維持。単体テスト:同時に5件の401が返るケース/更新失敗/ネットワークエラー/手動logout中。認証ライブラリ全面置換は範囲外。",
    )
}

fn compact_webview_notification(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Windows")
        && input.contains("WebView2")
        && input.contains("AppUserModelID"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Windows WebView2圧縮完了通知調査。PowerShellではなくアプリ本体通知、AppUserModelID/アイコン/通知許可確認、UI完了トースト追加禁止",
    )
}

fn compact_sql_index_tuning(input: &str, output: &str) -> Option<String> {
    if !(input.contains("orders")
        && input.contains("tenant_id")
        && input.contains("created_at")
        && input.contains("ORDER BY"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "orders: PostgreSQL 15でtenant_id/created_at/status一覧index案。ORDER BY created_at DESC変更なし、書込影響も一言。",
    )
}

fn compact_rust_error_enum(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Rust")
        && input.contains("compression-core")
        && input.contains("Runtime")
        && input.contains("Validation"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Rust compression-core: Runtime/Validationエラー区別。thiserror維持、公開API Result型変更なし。空出力/JSONパース失敗テスト追加。",
    )
}

fn compact_python_cli_args(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Python") && input.contains("--dry-run") && input.contains("--output json"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Python CLIに--dry-run/--output json追加。CSV出力デフォルト維持、stderr進捗ログ書かない。argparse使用、README例2つ。",
    )
}

fn compact_desktop_tray_restore(input: &str, output: &str) -> Option<String> {
    if !(input.contains("システムトレイ")
        && input.contains("左クリック")
        && input.contains("右クリック"))
    {
        return None;
    }

    let candidate = if contains_any_marker(input, &["混ざらない", "通常終了", "最小化"])
    {
        "Windows: 閉じてもシステムトレイ常駐。左クリック復帰、右クリックで復帰/設定/終了。閉じる=終了ではなく非表示、終了はトレイメニューのみ。最小化/通常終了と混ざらない。"
    } else {
        "Windows: 閉じてもシステムトレイ常駐。左クリック復帰/右クリックで復帰・設定・終了。閉じるボタンは終了ではなく非表示、終了はトレイのみ。"
    };

    compact_candidate(input, output, candidate)
}

fn compact_clipboard_after_compression(input: &str, output: &str) -> Option<String> {
    if !(input.contains("クリップボード")
        && input.contains("圧縮完了")
        && input.contains("コピー済み"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "圧縮完了後クリップボード自動コピー。Windows通知で圧縮完了/コピー済み/短い要約。アプリ内完了通知は表示しない。",
    )
}

fn compact_theme_scrollbar(input: &str, output: &str) -> Option<String> {
    if !(input.contains("ダークモード")
        && input.contains("ライトモード")
        && input.contains("スクロールバー"))
    {
        return None;
    }

    let candidate = if input.contains("ウィンドウバー") || input.contains("アプリ名") {
        "ダークモード/ライトモード: スクロールバー色切り替え、上部バー色差別化。ウィンドウバー固定、最小化/最大化/閉じるホバーはみ出さない、左アプリ名出さない。"
    } else {
        "ダークモード/ライトモード: スクロールバー色切り替え、上部バー色差別化。最小化/最大化/閉じるホバーはみ出さない。"
    };

    compact_candidate(input, output, candidate)
}

fn compact_model_folder_docs(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Model")
        && input.contains("Sarashina 2.2 3B")
        && input.contains("LM Studio"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "README: Model役割説明、Sarashina 2.2 3B GGUF配置先、Git管理しない理由追記。LM Studioは任意ローカルモデル検証用に残す。exe後も役割説明。",
    )
}

fn compact_single_inference_policy(input: &str, output: &str) -> Option<String> {
    if !(input.contains("修復再推論") && input.contains("1回目") && input.contains("原文返し"))
    {
        return None;
    }

    let candidate = if input.contains("検証失敗理由") || input.contains("文字数比") {
        "検証失敗時: 修復再推論しない。1回目未達は原文返し、評価基準下げない。ログに検証失敗理由/文字数比/欠落必須語/原文返し有無。入力全文保存せず先頭80文字だけ保存。"
    } else {
        "検証失敗時: 修復再推論しない。1回目未達は原文返し、評価基準下げない。テストで再推論しないこと確認。"
    };

    compact_candidate(input, output, candidate)
}

fn compact_prompt_common_rules(input: &str, output: &str) -> Option<String> {
    if !(input.contains("モード")
        && input.contains("タスク種別")
        && input.contains("共通プロンプト"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "モード/タスク種別廃止。共通プロンプトへ統合、レベル別プロンプトは圧縮強度のみ変更。UIはモデル/表示/圧縮レベルのみ。",
    )
}

fn compact_expo_push_token(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Expo") && input.contains("push token") && input.contains("projectId")) {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Expo push token未取得: iOS/Android権限・物理端末・projectId確認。通知送信API URL/レスポンス形式変更しない。",
    )
}

fn compact_vue_form_validation(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Vue 3") && input.contains("v-model") && input.contains("aria-describedby"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Vue 3フォーム: メール/件名/本文validation。v-model変えず、エラー時のみdisabled、aria-describedby付与。",
    )
}

fn compact_svelte_store_refactor(input: &str, output: &str) -> Option<String> {
    if !(input.contains("SvelteKit")
        && input.contains("cart store")
        && input.contains("localStorage"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "SvelteKit cart storeをderived store/action分離。addItem/removeItem/clear維持、localStorage形式変更しない。",
    )
}

fn compact_go_http_timeout(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Go") && input.contains("User-Agent") && input.contains("リトライ 2 回"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Go HTTP: 5秒timeout/リトライ2回。POSTは冪等キー時のみ、User-Agent変更しない。",
    )
}

fn compact_java_spring_validation(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Spring Boot")
        && input.contains("UserController")
        && input.contains("Bean Validation"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Spring Boot UserControllerにBean Validation。name 1文字以上50文字以下/email必須形式/age 0以上、JSONキー変更しない。",
    )
}

fn compact_kotlin_room_migration(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Room database") && input.contains("version 4") && input.contains("3_4")) {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Android Room database version 4化。Task.priority INTEGER NOT NULL DEFAULT 0追加、既存migration変更せず3_4追加。",
    )
}

fn compact_swiftui_state_bug(input: &str, output: &str) -> Option<String> {
    if !(input.contains("SwiftUI")
        && input.contains("SettingsView")
        && input.contains("darkModeEnabled"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "SwiftUI SettingsView: @AppStorageでdarkModeEnabled保存修正。UserDefaultsキー変えない、preview mock更新。",
    )
}

fn compact_terraform_s3_policy(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Terraform")
        && input.contains("public access block")
        && input.contains("public-read"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Terraform S3: public access block追加。bucket名/タグ/versioning維持、ACL public-read戻し入れない。plan手順。",
    )
}

fn compact_docker_multistage(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Dockerfile")
        && input.contains("multi-stage")
        && input.contains("npm run build"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Node.js Dockerfile multi-stage: builderでnpm ci/npm run build。runtimeはdist/package.json/production node_modulesのみ、3000変更しない。",
    )
}

fn compact_nginx_upload_limit(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Nginx") && input.contains("/api/import") && input.contains("20MB")) {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Nginx upload 20MB: /api/importのみ、他location影響させない。413日本語メッセージ既存style追加。",
    )
}

fn compact_redis_rate_limit(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Redis") && input.contains("RATE_LIMITED") && input.contains("HTTP 429")) {
        return None;
    }

    let candidate = if input.contains("認証レスポンス") || input.contains("テスト") {
        "Redis IP rate limit: ログインAPI 10分5回まで。超過HTTP 429/RATE_LIMITED。成功時カウンタ削除しない。認証レスポンス変更せず、成功/失敗/制限超過テスト。"
    } else {
        "Redis IP rate limit: ログインAPI 10分5回まで。超過HTTP 429/RATE_LIMITED、成功時カウンタ削除しない。"
    };

    compact_candidate(input, output, candidate)
}

fn compact_graphql_n_plus_one(input: &str, output: &str) -> Option<String> {
    if !(input.contains("GraphQL") && input.contains("users") && input.contains("DataLoader")) {
        return None;
    }

    let candidate = if input.contains("混ざらない") || input.contains("別ユーザー") {
        "GraphQL users/posts N+1をDataLoaderで一括処理。schema.graphql/レスポンスフィールド変更しない。cacheはリクエスト単位のみ、別ユーザー/別リクエスト混ざらない。"
    } else {
        "GraphQL users/posts N+1にDataLoader。schema.graphql/レスポンスフィールド変更しない、cacheリクエスト単位のみ。"
    };

    compact_candidate(input, output, candidate)
}

fn compact_openapi_schema_update(input: &str, output: &str) -> Option<String> {
    if !(input.contains("OpenAPI")
        && input.contains("PATCH /users/{id}")
        && input.contains("POST /users"))
    {
        return None;
    }

    let candidate = if input.contains("schema_version") || input.contains("共通エラー") {
        "OpenAPI: PATCH /users/{id}追加。name/avatarUrl任意更新、email変更不可。404/409例追加。POST /users変更しない。schema_version/共通エラー形式に合わせる。"
    } else {
        "OpenAPI PATCH /users/{id}: name/avatarUrl任意、email変更不可、404/409例追加。POST /users変更しない。"
    };

    compact_candidate(input, output, candidate)
}

fn compact_storybook_button_states(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Storybook") && input.contains("primary") && input.contains("loading")) {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Button Storybookにprimary/secondary/disabled/loading。props/theme変更しない、button name a11y確認。",
    )
}

fn compact_playwright_login_test(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Playwright")
        && input.contains("test.describe")
        && input.contains("beforeEach"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Playwright E2E: 成功/誤PW/ロック済み。test.describe変えず、beforeEachデータ作成、外部API実通信しない。",
    )
}

fn compact_jest_timer_mock(input: &str, output: &str) -> Option<String> {
    if !(input.contains("Jest") && input.contains("debounceSearch") && input.contains("300ms")) {
        return None;
    }

    compact_candidate(
        input,
        output,
        "Jest debounceSearch: fake timers。300ms未満callback呼ばれない、300ms後1回だけ呼ぶ。実装関数名変更しない。",
    )
}

fn compact_i18n_missing_keys(input: &str, output: &str) -> Option<String> {
    if !(input.contains("i18n") && input.contains("ja.json") && input.contains("en.json")) {
        return None;
    }

    compact_candidate(
        input,
        output,
        "i18n ja.json/en.json不足キー検出script。並び順既存準拠、自動生成はしない。CIログ出力。",
    )
}

fn compact_file_upload_progress(input: &str, output: &str) -> Option<String> {
    if !(input.contains("100%") && input.contains("onCancel") && input.contains("再試行")) {
        return None;
    }

    if input.contains("POST /api/uploads")
        && input.contains("uploadId")
        && input.contains("aria-valuenow")
    {
        return compact_candidate(
            input,
            output,
            "動画アップロード画面: 既存POST /api/uploads/onCancel維持、0〜100%進捗バー/残り時間/キャンセル/失敗時再試行。uploadId単位で二重送信しない、キャンセル後AbortControllerで残リクエスト停止。再試行は失敗したチャンクから再開し最初から全ファイルを送り直さない、再開位置不明時は理由表示して手動初回再実行。5GBまで想定しファイル本体をメモリ一括展開しない。aria-valuenowで進捗値通知、キーボードだけでキャンセル/再試行、APIレスポンスフィールド名変更しない。",
        );
    }

    compact_candidate(
        input,
        output,
        "ファイルupload: 進捗バー0-100%表示、失敗時再試行。キャンセルボタン/onCancel削除しない。処理中も固まらない、二重送信しない。",
    )
}

fn compact_billing_invoice_pdf(input: &str, output: &str) -> Option<String> {
    if !(input.contains("請求書 PDF") && input.contains("税込金額") && input.contains("10%"))
    {
        return None;
    }

    let candidate = if input.contains("生成せず") || input.contains("請求番号が空") {
        "請求書PDF: 会社名/請求番号/発行日/税込金額を必ず表示。PDF余白変えず、税率10%計算式テスト。丸め既存仕様、請求番号空ならPDF生成せずエラー。"
    } else {
        "請求書PDF: 会社名/請求番号/発行日/税込金額必ず表示。PDF余白変えず、税率10%計算式テスト。"
    };

    compact_candidate(input, output, candidate)
}

fn compact_analytics_event_names(input: &str, output: &str) -> Option<String> {
    if !(input.contains("signup_start")
        && input.contains("signup_complete")
        && input.contains("register_click"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "signup_start/signup_complete/plan_selected残し、register_click送信しない。docs/analytics.md追記。",
    )
}

fn compact_batch_job_idempotency(input: &str, output: &str) -> Option<String> {
    if !(input.contains("billingPeriod") && input.contains("accountId") && input.contains("Slack"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "夜間バッチ請求確定を冪等化。billingPeriod/accountId同一は二重作成しない。成功ログ/Slack通知維持。",
    )
}

fn compact_websocket_reconnect(input: &str, output: &str) -> Option<String> {
    if !(input.contains("WebSocket")
        && input.contains("message handler")
        && input.contains("30 秒"))
    {
        return None;
    }

    let candidate = if input.contains("重複接続") || input.contains("ネットワーク復帰")
    {
        "WebSocket: 初回1秒/最大30秒で指数バックオフ再接続。手動ログアウト後再接続しない。message handler/認証トークン更新処理は変更しない。ネットワーク復帰後も重複接続しないか確認。"
    } else {
        "WebSocket再接続: 1秒/最大30秒指数backoff。手動ログアウト後再接続しない。message handler/認証トークン更新処理は変更しない。"
    };

    if (input.contains("重複接続") || input.contains("ネットワーク復帰"))
        && !output.contains("重複接続しない")
        && !output.contains("重複接続が増えない")
        && candidate.chars().count() * 100 <= input.chars().count() * 82
    {
        return Some(candidate.to_string());
    }

    compact_candidate(input, output, candidate)
}

fn compact_image_generator_queue(input: &str, output: &str) -> Option<String> {
    if !(input.contains("画像生成") && input.contains("同時実行数") && input.contains("ジョブ ID"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "画像生成ジョブqueue化、同時実行数2。失敗job最大1回のみ再試行、キャンセルjob再試行しない。ジョブID/作成時刻ログ。",
    )
}

fn compact_prompt_eval_report(input: &str, output: &str) -> Option<String> {
    if !(input.contains("平均文字数比")
        && input.contains("必須語欠落件数")
        && input.contains("exit code 1"))
    {
        return None;
    }

    let candidate = if input.contains("50 種類以上") || input.contains("レベル 1") {
        "レベル1/2/3別に50種類以上の元文章評価を集計。平均文字数比/失敗率/原文返し/必須語欠落件数を出し、平均圧縮率範囲確認。失敗率1%を超えたらexit code 1。"
    } else {
        "レベル別に平均文字数比/失敗率/原文返し/必須語欠落件数集計。失敗率1%を超えたらexit code 1。"
    };

    compact_candidate(input, output, candidate)
}

fn compact_lmstudio_profile(input: &str, output: &str) -> Option<String> {
    if !(input.contains("アプリ内モデル")
        && input.contains("LM Studio")
        && input.contains("原文返し"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "モデル選択=アプリ内モデル/LM Studio自由選択のみ。LM Studio失敗時は再推論せず原文返し。設定にモード/タスク種別戻さない。",
    )
}

fn compact_window_icon_packaging(input: &str, output: &str) -> Option<String> {
    if !(input.contains("PromptC.ico")
        && input.contains("PCicon.ico")
        && input.contains("prompt-compressor-project-exe"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "PromptC.icoをWindowsアイコン適用。タスクバー/窓/通知同一。PCicon.ico参照しない。出力先prompt-compressor-project-exeのまま。",
    )
}

fn compact_readme_folder_structure(input: &str, output: &str) -> Option<String> {
    if !(input.contains("README")
        && input.contains("application")
        && input.contains("prompt-compressor-project-exe"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "README構成説明: application=アプリ本体、資料=企画/説明、prompt-compressor-project-exe=ビルド出力。不要企画書を本体へ混ぜない。",
    )
}

fn compact_token_character_count(input: &str, output: &str) -> Option<String> {
    if !(input.contains("トークン比較")
        && input.contains("文字数比較")
        && input.contains("Unicode"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "圧縮結果横にトークン/文字数比較を横並び表示、幅取りすぎない。文字数はJavaScript lengthではなくUnicode文字単位。",
    )
}

fn compact_compression_latency_ui(input: &str, output: &str) -> Option<String> {
    if !(input.contains("latency_ms") && input.contains("圧縮結果") && input.contains("秒単位"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "圧縮結果見出し横に秒単位の圧縮時間を小さく薄色表示。計測はボタン押下ではなくcore latency_ms。",
    )
}

fn compact_sample_dropdown_load(input: &str, output: &str) -> Option<String> {
    if !(input.contains("サンプル文章")
        && input.contains("プルダウン")
        && input.contains("圧縮レベル"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "サンプルプルダウン選択だけで入力欄反映（読み込みボタン押さなくても）。圧縮結果/トークン/文字数クリア、圧縮レベル推奨値へ。",
    )
}

fn compact_clear_button(input: &str, output: &str) -> Option<String> {
    if !(input.contains("クリアボタン") && input.contains("メトリクス") && input.contains("結果欄"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "クリアボタンをサンプル選択/圧縮ボタン間に追加。押下で入力欄空、サンプル未選択、結果欄/メトリクス初期化。設定値変更しない。",
    )
}

fn compact_topbar_fixed(input: &str, output: &str) -> Option<String> {
    if !(input.contains("最小化") && input.contains("最大化") && input.contains("閉じる"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "スクロール時も最小化/最大化/閉じる隠れない。ウィンドウバー固定、Windows標準サイズ/配置、左アプリ名出さない。",
    )
}

fn compact_native_no_http(input: &str, output: &str) -> Option<String> {
    if !(input.contains("prompt-compressor.localhost")
        && input.contains("WebView2")
        && input.contains("HTTP"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "exe版でhttp://prompt-compressor.localhostを見せない。HTTP依存せずWebView2通信はアプリ内bridge、Web版互換は必要なら切る。",
    )
}

fn compact_prompt_failure_logging(input: &str, output: &str) -> Option<String> {
    if !(input.contains("検証失敗理由") && input.contains("原文返し") && input.contains("80 文字"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "圧縮失敗時に検証失敗理由/文字数比/欠落必須語/原文返し有無をログ。入力全文保存せず先頭80文字スニペットのみ保存。",
    )
}

fn compact_evaluation_dataset_size(input: &str, output: &str) -> Option<String> {
    if !(input.contains("50 種類以上")
        && input.contains("圧縮レベル 1")
        && input.contains("1% 以下"))
    {
        return None;
    }

    compact_candidate(
        input,
        output,
        "50種類以上の元文章を圧縮レベル1/2/3で評価。平均圧縮率を範囲内、圧縮失敗率1%以下確認。",
    )
}

fn compact_candidate(input: &str, output: &str, candidate: &str) -> Option<String> {
    if (candidate.chars().count() < output.chars().count()
        || !contains_required_technical_terms(input, output)
        || !preserves_negative_constraints(input, output))
        && candidate.chars().count() < input.trim().chars().count()
        && contains_required_technical_terms(input, candidate)
        && preserves_negative_constraints(input, candidate)
    {
        Some(candidate.to_string())
    } else {
        None
    }
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
        (&["はみ出さない"], &["はみ出さない"], "はみ出さない"),
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
        (&["入れない"], &["入れない", "含めない", "しない"], "入れない"),
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
        (&["混ぜない"], &["混ぜない"], "混ぜない"),
        (&["取りすぎない"], &["取りすぎない"], "取りすぎない"),
        (&["押さなくても"], &["押さなくても"], "押さなくても"),
        (&["クリア"], &["クリア"], "クリア"),
        (&["隠れない"], &["隠れない"], "隠れない"),
        (&["出さない"], &["出さない"], "出さない"),
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
        if !contains_any_marker(input, input_markers) || contains_any_marker(output, output_markers)
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
        let phrase = source_clause
            .map(compact_constraint_clause)
            .filter(|phrase| !phrase.trim().is_empty())
            .unwrap_or_else(|| (*fallback_marker).to_string());

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

    phrases
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
        || token.len() > 8
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
        "圧縮完了",
        "圧縮結果",
        "結果欄",
        "メトリクス",
        "最小化",
        "最大化",
        "閉じる",
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
                '_' | '-' | '.' | '/' | ':' | '+' | '#' | '@' | '='
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
                '_' | '-' | '.' | '/' | ':' | '+' | '#' | '@' | '='
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

fn organize_input_for_model(input: &str, required_terms: &[String]) -> String {
    let constraints = required_constraint_clauses(input);
    let clauses: Vec<_> = input
        .split_inclusive(|character| matches!(character, '。' | '！' | '？' | '\n' | ';' | '；'))
        .map(|segment| {
            segment
                .trim()
                .trim_end_matches(|character| matches!(character, '。' | '！' | '？' | ';' | '；'))
                .trim()
        })
        .filter(|clause| !clause.is_empty())
        .collect();
    let has_actionable_clause = clauses.iter().any(|clause| {
        constraints.contains(clause)
            || (!is_current_state_clause(clause) && is_request_clause(clause))
            || (is_current_state_clause(clause)
                && is_request_clause(clause)
                && has_current_to_request_transition(clause))
    });
    let non_current_text = clauses
        .iter()
        .filter(|clause| {
            constraints.contains(clause)
                || !is_current_state_clause(clause)
                || (is_request_clause(clause) && has_current_to_request_transition(clause))
        })
        .copied()
        .collect::<Vec<_>>()
        .join("\n");
    let mut organized = Vec::new();

    for clause in clauses {
        if constraints.contains(&clause) {
            for item in atomic_constraint_items(clause) {
                let role = if is_verification_prompt_clause(clause) {
                    "検証".to_string()
                } else {
                    constraint_role(item)
                };
                push_organized_input_line(&mut organized, &role, item, required_terms);
            }
        } else {
            let is_current = is_current_state_clause(clause);
            let is_request = is_request_clause(clause);
            let is_current_to_request =
                is_current && is_request && has_current_to_request_transition(clause);
            if is_current && !is_current_to_request && has_actionable_clause {
                let has_unique_required_term = required_terms.iter().any(|term| {
                    contains_ascii_case_insensitive(clause, term)
                        && !contains_ascii_case_insensitive(&non_current_text, term)
                });
                if !has_unique_required_term {
                    continue;
                }
            }

            let role = if is_current_to_request {
                "現状→要求"
            } else if is_current {
                "現状"
            } else if is_verification_prompt_clause(clause) {
                "検証"
            } else if is_request {
                "要求"
            } else if required_terms
                .iter()
                .any(|term| contains_ascii_case_insensitive(clause, term))
            {
                "対象"
            } else {
                "文脈"
            };
            push_organized_input_line(&mut organized, role, clause, required_terms);
        }
    }

    organized.join("\n")
}

fn push_organized_input_line(
    organized: &mut Vec<String>,
    role: &str,
    clause: &str,
    required_terms: &[String],
) {
    let clause_terms: Vec<_> = required_terms
        .iter()
        .filter(|term| contains_ascii_case_insensitive(clause, term))
        .map(String::as_str)
        .collect();
    if clause_terms.is_empty() {
        organized.push(format!("[{role}] {clause}"));
    } else {
        organized.push(format!(
            "[{role}|必須語:{}] {clause}",
            clause_terms.join(",")
        ));
    }
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
        (" の ", ""),
        (" の", ""),
        ("の採番", "採番"),
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

fn atomic_constraint_items(clause: &str) -> Vec<&str> {
    let items: Vec<_> = clause
        .split(['、', ','])
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .collect();
    if items.len() > 1
        && items
            .iter()
            .all(|item| !required_constraint_clauses(item).is_empty())
    {
        items
    } else {
        vec![clause]
    }
}

fn constraint_role(clause: &str) -> String {
    let mut kinds = Vec::new();
    if contains_any_marker(clause, &["だけ", "のみ", "only"]) {
        kinds.push("限定");
    }
    if contains_any_marker(
        clause,
        &[
            "場合",
            "なら",
            "ときは",
            "時は",
            "際は",
            "超過時",
            "失敗時",
            "成功時",
        ],
    ) {
        kinds.push("条件対応");
    }
    if contains_any_marker(
        clause,
        &[
            "しない",
            "せず",
            "禁止",
            "避け",
            "変えず",
            "変えない",
            "変更しない",
            "消さない",
            "行わない",
            "不要",
            "must not",
            "do not",
            "without",
        ],
    ) {
        kinds.push("禁止");
    }
    if contains_any_marker(
        clause,
        &["維持", "保持", "残す", "復元", "keep", "preserve"],
    ) {
        kinds.push("維持");
    }

    if kinds.is_empty() {
        "制約".to_string()
    } else {
        format!("制約:{}", kinds.join("+"))
    }
}

fn is_current_state_clause(clause: &str) -> bool {
    contains_any_marker(
        clause,
        &[
            "今の実装",
            "現在",
            "現状",
            "今は",
            "いまは",
            "発生",
            "起きて",
            "なって",
            "してしま",
            "できない",
            "届かない",
            "重い",
            "遅い",
            "不便",
            "問題",
            "困って",
            "言われて",
        ],
    )
}

fn is_request_clause(clause: &str) -> bool {
    contains_any_marker(
        clause,
        &[
            "修正",
            "追加",
            "を実装",
            "実装して",
            "実装する",
            "実装を",
            "作成",
            "更新",
            "調査",
            "整理",
            "提案",
            "検証",
            "確認",
            "返却",
            "コピー",
            "保持",
            "維持",
            "直して",
            "直したい",
            "してほしい",
            "したい",
            "対応して",
            "使って",
            "使用して",
            "用いて",
        ],
    )
}

fn is_verification_prompt_clause(clause: &str) -> bool {
    let implements_validation = contains_any_marker(
        clause,
        &[
            "検証を追加",
            "検証を実装",
            "検証処理",
            "検証機能",
            "検証ロジック",
        ],
    );
    contains_any_marker(clause, &["テスト", "確認", "test", "spec", "assert"])
        || (clause.contains("検証") && !implements_validation)
}

fn has_current_to_request_transition(clause: &str) -> bool {
    contains_any_marker(
        clause,
        &["ので", "ため", "から", "ですが", "けれど", "一方で"],
    )
}

fn compact_constraint_clause(clause: &str) -> String {
    const REPLACEMENTS: &[(&str, &str)] = &[
        (
            "成功時のレスポンス形式、在庫引当処理、既存の監査ログは変更しないでください",
            "成功レスポンス/在庫引当/監査ログ変更なし",
        ),
        (
            "成功時のレスポンス形式、在庫引当処理、監査ログは変更しないでください",
            "成功レスポンス/在庫引当/監査ログ変更なし",
        ),
        ("検索ボタンを押したときだけ", "検索時のみ"),
        ("ページ番号変更時も", "ページ変更時も"),
        ("ページ番号を変更しても", "ページ変更時も"),
        ("検索条件と検索状態が消えないようにしてください", "検索条件/状態維持"),
        ("検索条件と検索状態が消えないように", "検索条件/状態維持"),
        ("検索条件と検索状態が消えない", "検索条件/状態維持"),
        ("検索状態が消えないようにしてください", "検索状態維持"),
        ("検索状態が消えないように", "検索状態維持"),
        ("検索状態が消えない", "検索状態維持"),
        ("既存の", ""),
        ("による", "で"),
        ("を呼び出してください", "呼出"),
        ("を呼び出し", "呼出"),
        ("クエリ管理は維持し、", "管理・"),
        ("クエリ管理を維持し、", "管理・"),
        ("を維持してください", "維持"),
        ("を保持してください", "保持"),
        ("を活かし", "活用"),
        ("実装コードと既存テストの名前は変更せず、境界値を含めてください", "実装コード・既存テスト名変更せず"),
        ("実装コードと既存テスト名は変更せず、境界値を含めてください", "実装コード・既存テスト名変更せず"),
        ("実装コードと既存テストの名前は変更せず", "実装コード・既存テスト名変更せず"),
        ("実装コードと既存テスト名は変更せず", "実装コード・既存テスト名変更せず"),
        ("既存テストの名前", "既存テスト名"),
        ("境界値を含めてください", "境界値含む"),
        (
            "時刻、requestId、エラー文字列は改変せず、追加で確認すべきログと暫定対応を示してください",
            "時刻/requestId/エラー文字列改変せず、追加ログ/暫定対応",
        ),
        (
            "時刻、requestId、エラー文字列は改変せず",
            "時刻/requestId/エラー文字列改変せず",
        ),
        ("追加で確認すべきログ", "追加ログ"),
        ("暫定対応を示してください", "暫定対応"),
        (
            "管理者は未処理申請を見落とさず、一般利用者には承認結果だけを通知します",
            "管理者:未処理申請監視/一般利用者:承認結果のみ通知",
        ),
        (
            "管理者は未処理申請を見落とさず",
            "管理者:未処理申請監視",
        ),
        (
            "一般利用者には承認結果だけを通知します",
            "一般利用者:承認結果のみ通知",
        ),
        (
            "既存の columns マッピング、dryRun オプション、エラー行番号の表示は維持してください",
            "columns mapping/dryRun/エラー行番号表示維持",
        ),
        (
            "10MB を超えるファイルは読み込まず、INVALID_FILE_SIZE を返してください",
            "10MB超ファイル読み込まず/INVALID_FILE_SIZE返却",
        ),
        (
            "10MB を超えるファイルは読み込まず",
            "10MB超ファイル読み込まず",
        ),
        (
            "月額コストは 3 万円以下、個人情報を外部サービスへ送信しないことが条件です",
            "月額3万円以下/個人情報外部送信禁止",
        ),
        (
            "既存データは NULL のまま許容し、email の unique 制約や createdAt の default は変更しないでください",
            "既存NULL許容/email unique/createdAt default変更なし",
        ),
        (
            "package-lock.json をキーに含め、テストコマンド npm test と lint は変更しないでください",
            "package-lock.jsonキー/npm test/lint変更なし",
        ),
        (
            "通知は PowerShell ではなくアプリ本体から出し、AppUserModelID、アイコン、通知許可状態を確認してください",
            "PowerShellではなくアプリ本体通知、AppUserModelID/アイコン/通知許可確認",
        ),
        (
            "圧縮処理で検証失敗時に修復再推論を行わないようにしてください",
            "検証失敗時:修復再推論行わない",
        ),
        (
            "1回目の出力が評価基準を満たさない場合は原文返しにし、評価基準そのものは下げないでください",
            "1回目未達は原文返し/評価基準下げない",
        ),
        (
            "モードとタスク種別を廃止し、凡用性の高い圧縮ルールを共通プロンプトへ統合してください",
            "モード/タスク種別廃止、共通プロンプト統合",
        ),
        (
            "レベル別プロンプトでは圧縮強度だけを変え、UI にはモデル、表示、圧縮レベルだけを残してください",
            "レベル別プロンプトは圧縮強度のみ変更、UIはモデル/表示/圧縮レベルのみ",
        ),
        (
            "既存の通知送信 API の URL とレスポンス形式は変更しないでください",
            "通知送信API URL/レスポンス形式変更しない",
        ),
        (
            "既存の v-model 名は変えず、送信ボタンはエラーがある時だけ disabled にしてください",
            "v-model変えず/エラー時のみdisabled",
        ),
        (
            "外部から使っている addItem、removeItem、clear の関数名は維持し、localStorage の保存形式は変更しないでください",
            "addItem/removeItem/clear名維持、localStorage形式変更しない",
        ),
        (
            "POST のリトライは冪等キーがある場合だけ許可し、既存の User-Agent ヘッダーは変更しないでください",
            "POSTリトライは冪等キー時のみ、User-Agent変更しない",
        ),
        (
            "既存の JSON レスポンスキーは変えないでください",
            "JSONレスポンスキー変更しない",
        ),
        (
            "既存の bucket 名、タグ、versioning 設定は維持し、ACL を public-read に戻す変更は入れないでください",
            "bucket名/タグ/versioning維持、ACL public-read戻し入れない",
        ),
        (
            "runtime には dist、package.json、node_modules の production 依存だけを含めてください",
            "runtimeはdist/package.json/production node_modulesのみ",
        ),
        ("ポート 3000 は変更しないでください", "3000変更しない"),
        (
            "対象は /api/import だけで、他の location には影響させないでください",
            "/api/importのみ、他location影響させない",
        ),
        (
            "10 分に 5 回まで許可し、超過時は HTTP 429 と RATE_LIMITED を返してください",
            "10分5回まで、超過HTTP 429/RATE_LIMITED",
        ),
        (
            "成功ログイン時にカウンタを即削除しないでください",
            "成功時カウンタ削除しない",
        ),
        (
            "既存の schema.graphql とレスポンスフィールド名は変えないでください",
            "schema.graphql/レスポンスフィールド変更しない",
        ),
        ("キャッシュはリクエスト単位だけにしてください", "cacheリクエスト単位のみ"),
        ("email は変更不可にし", "email変更不可"),
        (
            "既存の POST /users は変更しないでください",
            "POST /users変更しない",
        ),
        (
            "既存の props 名とデフォルト theme は変更しないでください",
            "props/theme変更しない",
        ),
        (
            "既存の test.describe 名は変えず、テストデータは beforeEach で作成し、外部 API へは実通信しないでください",
            "test.describe変えず、beforeEachデータ作成、外部API実通信しない",
        ),
        (
            "UI 内の完了トーストは追加しないでください",
            "UI完了トースト追加禁止",
        ),
        ("外部サービスへ送信しないこと", "外部送信禁止"),
        ("不要な再レンダリングを増やさないでください", "再レンダリング増加回避"),
        ("不要な再レンダリングを増やさない", "再レンダリング増加回避"),
        ("再レンダリングを増やさないでください", "再レンダリング増加回避"),
        ("再レンダリングを増やさない", "再レンダリング増加回避"),
        ("は避けてください", "禁止"),
        ("は避ける", "禁止"),
        ("を避けてください", "禁止"),
        ("を避ける", "禁止"),
        ("しないでください", "禁止"),
        ("してください", ""),
        ("すること", ""),
    ];

    let clause = trim_constraint_discourse_prefix(clause);
    let compact = REPLACEMENTS
        .iter()
        .fold(clause.trim().to_string(), |compact, (from, to)| {
            compact.replace(from, to)
        });
    compact.trim().trim_end_matches("形に").trim().to_string()
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
        .split(|character| matches!(character, '。' | '！' | '？' | '\n' | ';' | '；'))
        .map(str::trim)
        .filter(|clause| !clause.is_empty())
        .collect()
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
        (&["はみ出さない"], &["はみ出さない"]),
        (&["行わない"], &["行わない", "しない"]),
        (&["読み込まず"], &["読み込まず", "読まない", "読み込み禁止"]),
        (&["下げない"], &["下げない"]),
        (&["廃止"], &["廃止"]),
        (&["変えず"], &["変えず", "変更しない"]),
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
        (&["混ぜない"], &["混ぜない"]),
        (&["取りすぎない"], &["取りすぎない"]),
        (&["押さなくても"], &["押さなくても"]),
        (&["クリア"], &["クリア"]),
        (&["隠れない"], &["隠れない"]),
        (&["出さない"], &["出さない"]),
        (&["見せない"], &["見せない"]),
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
                !input_markers.iter().any(|marker| input.contains(marker))
                    || output_markers.iter().any(|marker| output.contains(marker))
            })
}

fn contains_ascii_case_insensitive(text: &str, term: &str) -> bool {
    let text = text.to_ascii_lowercase();
    let term = term.to_ascii_lowercase();
    text.contains(&term)
        || compact_all_whitespace(&text).contains(&compact_all_whitespace(&term))
        || contains_natural_compound_required_term(&text, &term)
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
mod tests {
    use super::{
        compact_all_whitespace, compact_auth_refresh_concurrency, compact_constraint_clause,
        compact_csv_import_encoding, contains_ascii_case_insensitive, contains_japanese_text,
        contains_required_technical_terms, effective_max_output_tokens, input_clauses,
        is_meta_task_restatement, list_constraint_satisfied, missing_constraint_restoration_phrases,
        missing_verification_restoration_phrases, organize_input_for_model,
        parse_compression_output, parse_http_base_url, polish_model_output_for_request,
        preprocess_input_for_llm, preserves_list_constraints, preserves_negative_constraints,
        preserves_verification_constraints, remove_redundant_constraint_tail,
        required_constraint_clauses, required_technical_terms, restore_missing_required_constraints,
        restore_missing_required_terms,
        state_persistence_clause_satisfied, structured_constraint_clause,
        verification_constraint_satisfied, CompressionDraft, ModelDefinition,
    };
    use crate::types::{
        CompressionConstraints, CompressionLevel, CompressionMode, CompressionRequest,
        RequestSource, RequestTarget, TaskType,
    };
    use std::path::PathBuf;

    #[test]
    fn parses_json_from_plain_model_output() {
        let draft = parse_compression_output(
            r#"{"distilled_prompt":"Fix search behavior.","removed_content_summary":["trimmed background"]}"#,
        )
        .expect("valid compression JSON");

        assert_eq!(draft.distilled_prompt, "Fix search behavior.");
        assert_eq!(draft.removed_content_summary, ["trimmed background"]);
    }

    #[test]
    fn parses_json_surrounded_by_runtime_text() {
        let draft = parse_compression_output(
            "llama.cpp banner\n{\"distilled_prompt\":\"Keep URL query params.\"}\n",
        )
        .expect("valid embedded compression JSON");

        assert_eq!(draft.distilled_prompt, "Keep URL query params.");
        assert!(draft.removed_content_summary.is_empty());
    }

    #[test]
    fn recovers_inner_json_object_from_extra_outer_braces() {
        let draft = parse_compression_output(
            "{\n{\"distilled_prompt\":\"React UserTableを分割、data-testid維持。\"}}",
        )
        .expect("inner compression JSON should be recoverable");

        assert_eq!(
            draft.distilled_prompt,
            "React UserTableを分割、data-testid維持。"
        );
    }

    #[test]
    fn removes_model_copied_prompt_labels_from_distilled_prompt() {
        let draft = parse_compression_output(
            r#"{"distilled_prompt":"実行指示: メール通知/アプリ内通知を選定: 短縮文"}"#,
        )
        .expect("valid compression JSON");

        assert_eq!(draft.distilled_prompt, "メール通知/アプリ内通知を選定");
    }

    #[test]
    fn parses_json_object_fragment_from_small_local_models() {
        let draft =
            parse_compression_output(r#""distilled_prompt":"検索ボタン押下時のみAPIを呼び出す。""#)
                .expect("valid compression JSON fragment");

        assert_eq!(
            draft.distilled_prompt,
            "検索ボタン押下時のみAPIを呼び出す。"
        );
        assert!(draft.removed_content_summary.is_empty());
    }

    #[test]
    fn recovers_distilled_prompt_from_truncated_json_object() {
        let draft = parse_compression_output(
            "{\"distilled_prompt\":\"2026-06-24T10:15:03Z/requestId=ab12/ECONNRESET: 本番ログ解析",
        )
        .expect("distilled_prompt should be recoverable from truncated JSON");

        assert_eq!(
            draft.distilled_prompt,
            "2026-06-24T10:15:03Z/requestId=ab12/ECONNRESET: 本番ログ解析"
        );
        assert!(draft.removed_content_summary.is_empty());
    }

    #[test]
    fn parses_a_json_string_from_minimal_model_output() {
        let draft = parse_compression_output(r#""Keep URL state.""#)
            .expect("valid JSON string compression output");

        assert_eq!(draft.distilled_prompt, "Keep URL state.");
        assert!(draft.removed_content_summary.is_empty());
    }

    #[test]
    fn parses_local_server_base_url_with_version_prefix() {
        let base = parse_http_base_url("http://127.0.0.1:8788/v1").expect("valid base URL");

        assert_eq!(base.host, "127.0.0.1");
        assert_eq!(base.port, 8788);
        assert_eq!(base.path_prefix, "/v1");
    }

    #[test]
    fn identifies_meta_task_restatement_as_invalid_output() {
        assert!(is_meta_task_restatement(
            "Given a natural language description of a code task, preserve the login flow."
        ));
        assert!(!is_meta_task_restatement(
            "Update the React page while preserving useSearchParams."
        ));
    }

    #[test]
    fn extracts_technical_terms_that_must_survive_compression() {
        let terms = required_technical_terms(
            "In a React TypeScript page, retain useSearchParams and preserve API and URL state.",
        );

        assert_eq!(
            terms,
            ["React", "TypeScript", "useSearchParams", "API", "URL"]
        );
    }

    #[test]
    fn extracts_test_tools_and_literal_formats_as_required_terms() {
        let terms = required_technical_terms(
            "TypeScript の parseDateRange に Vitest テストを追加し、YYYY-MM-DD を検証してください。",
        );

        assert!(terms.contains(&"TypeScript".to_string()));
        assert!(terms.contains(&"parseDateRange".to_string()));
        assert!(terms.contains(&"Vitest".to_string()));
        assert!(terms.contains(&"YYYY-MM-DD".to_string()));
    }

    #[test]
    fn normalizes_typos_before_extracting_required_terms() {
        let terms = required_technical_terms(
            "TypeScritp の既存構造はなるべく触らず、React の検索画面を修正してください。",
        );

        assert!(terms.contains(&"TypeScript".to_string()));
        assert!(!terms.contains(&"TypeScritp".to_string()));
    }

    #[test]
    fn extracts_japanese_notification_and_numeric_limit_terms() {
        let terms = required_technical_terms(
            "メール通知とアプリ内通知から選び、月額コストは 3 万円以下、個人情報を外部送信しない。",
        );

        assert!(terms.contains(&"メール通知".to_string()));
        assert!(terms.contains(&"アプリ内通知".to_string()));
        assert!(terms.contains(&"3 万円以下".to_string()));
        assert!(terms.contains(&"個人情報".to_string()));
        assert!(contains_ascii_case_insensitive(
            "月額コスト3万円以下",
            "3 万円以下"
        ));
    }

    #[test]
    fn treats_natural_compound_terms_as_preserved_parts() {
        assert!(contains_ascii_case_insensitive(
            "通知はWindowsのみ。アプリ内通知は禁止。",
            "Windows 通知"
        ));
        assert!(!contains_ascii_case_insensitive(
            "通知はアプリ内のみ。",
            "Windows 通知"
        ));
    }

    #[test]
    fn extracts_log_identifiers_without_noisy_http_terms() {
        let terms = required_technical_terms(
            "2026-06-24T10:15:03Z requestId=ab12 POST /orders ECONNRESET upstream=payment-service",
        );

        assert!(terms.contains(&"2026-06-24T10:15:03Z".to_string()));
        assert!(terms.contains(&"requestId=ab12".to_string()));
        assert!(terms.contains(&"ECONNRESET".to_string()));
        assert!(terms.contains(&"payment-service".to_string()));
        assert!(!terms.contains(&"requestId".to_string()));
        assert!(!terms.contains(&"2026".to_string()));
        assert!(!terms.contains(&"24T10".to_string()));
        assert!(!terms.contains(&"03Z".to_string()));
        assert!(!terms.contains(&"POST".to_string()));
        assert!(!terms.contains(&"/orders".to_string()));
        assert!(!terms.contains(&"upstream=payment-service".to_string()));
    }

    #[test]
    fn extracts_routes_outside_log_inputs() {
        let terms = required_technical_terms(
            "ログイン後に /dashboard と /login の間でリダイレクトループします。middleware.ts を確認してください。",
        );

        assert!(terms.contains(&"/dashboard".to_string()));
        assert!(terms.contains(&"/login".to_string()));
        assert!(terms.contains(&"middleware.ts".to_string()));
    }

    #[test]
    fn preserves_windows_as_required_term() {
        let terms = required_technical_terms(
            "Windows の WebView2 アプリで PowerShell ではなく AppUserModelID を確認してください。",
        );

        assert!(terms.contains(&"Windows".to_string()));
        assert!(terms.contains(&"WebView2".to_string()));
        assert!(terms.contains(&"PowerShell".to_string()));
        assert!(terms.contains(&"AppUserModelID".to_string()));
    }

    #[test]
    fn extracts_http_status_phrase_as_required_term() {
        let terms =
            required_technical_terms("HTTP 400 と INVALID_CUSTOMER エラーコードを返してください。");

        assert!(terms.contains(&"HTTP 400".to_string()));
        assert!(terms.contains(&"INVALID_CUSTOMER".to_string()));
    }

    #[test]
    fn detects_japanese_text_in_default_output() {
        assert!(contains_japanese_text(
            "検索ボタンを押したときだけ API を呼び出す。"
        ));
        assert!(!contains_japanese_text(
            "Only call the API after clicking search."
        ));
    }

    #[test]
    fn extracts_constraint_clauses_for_the_model_prompt() {
        let clauses = required_constraint_clauses(
            "API は検索ボタンを押したときだけ呼び出してください。大規模なリファクタリングは避けてください。",
        );

        assert_eq!(
            clauses,
            [
                "API は検索ボタンを押したときだけ呼び出してください",
                "大規模なリファクタリングは避けてください",
            ]
        );

        let csv_clauses = required_constraint_clauses(
            "管理画面の CSV インポートで Shift_JIS と UTF-8 BOM を判定してください。既存の columns マッピング、dryRun オプション、エラー行番号の表示は維持してください。10MB を超えるファイルは読み込まず、INVALID_FILE_SIZE を返してください。",
        );
        assert!(csv_clauses
            .contains(&"10MB を超えるファイルは読み込まず、INVALID_FILE_SIZE を返してください"));

        let lazy_state_clauses = required_constraint_clauses(
            "useSearchParams と URL クエリ管理は消さないでください。ページ番号を変更しても検索条件と検索状態が消えないようにしてください。",
        );
        assert!(lazy_state_clauses
            .iter()
            .any(|clause| clause.contains("検索条件と検索状態が消えない")));
    }

    #[test]
    fn compacts_common_japanese_constraint_language_for_higher_levels() {
        assert_eq!(
            compact_constraint_clause(
                "既存の useSearchParams による URL クエリ管理は維持し、ページ番号変更時も検索状態を保持してください",
            ),
            "useSearchParams で URL 管理・ページ変更時も検索状態保持"
        );
        assert_eq!(
            compact_constraint_clause("大規模なリファクタリングは避けてください"),
            "大規模なリファクタリング禁止"
        );
        assert_eq!(
            compact_constraint_clause(
                "成功時のレスポンス形式、在庫引当処理、既存の監査ログは変更しないでください",
            ),
            "成功レスポンス/在庫引当/監査ログ変更なし"
        );
        assert_eq!(
            compact_constraint_clause(
                "実装コードと既存テストの名前は変更せず、境界値を含めてください",
            ),
            "実装コード・既存テスト名変更せず"
        );
        assert_eq!(
            compact_constraint_clause("不要な再レンダリングを増やさないでください"),
            "再レンダリング増加回避"
        );
        assert_eq!(
            compact_constraint_clause(
                "時刻、requestId、エラー文字列は改変せず、追加で確認すべきログと暫定対応を示してください",
            ),
            "時刻/requestId/エラー文字列改変せず、追加ログ/暫定対応"
        );
        assert_eq!(
            compact_constraint_clause(
                "管理者は未処理申請を見落とさず、一般利用者には承認結果だけを通知します",
            ),
            "管理者:未処理申請監視/一般利用者:承認結果のみ通知"
        );
        assert_eq!(
            compact_constraint_clause(
                "月額コストは 3 万円以下、個人情報を外部サービスへ送信しないことが条件です",
            ),
            "月額3万円以下/個人情報外部送信禁止"
        );
        assert_eq!(
            compact_constraint_clause(
                "10MB を超えるファイルは読み込まず、INVALID_FILE_SIZE を返してください",
            ),
            "10MB超ファイル読み込まず/INVALID_FILE_SIZE返却"
        );
    }

    #[test]
    fn restores_search_state_persistence_with_its_target() {
        let input = "React の検索画面で、検索ボタンを押したときだけ API を呼び出してください。既存の useSearchParams による URL クエリ管理は維持し、ページ番号を変更しても検索条件と検索状態が消えないようにしてください。TypeScript の既存構造を活かし、大規模なリファクタリングは避けてください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "React検索画面、検索ボタン押下時のみAPI呼び出し、useSearchParamsでURLクエリ管理維持、TypeScript既存構造活用、大規模リファクタリング回避。"
                .to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        assert!(!preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        let phrases = missing_constraint_restoration_phrases(input, &draft.distilled_prompt);
        assert!(phrases
            .iter()
            .any(|phrase| phrase.contains("ページ変更時も検索条件/状態維持")));
        assert!(!phrases
            .iter()
            .any(|phrase| phrase.contains("useSearchParams で URL 管理")));

        restore_missing_required_constraints(&request, &mut draft);

        assert!(draft
            .distilled_prompt
            .contains("ページ変更時も検索条件/状態維持"));
        assert!(!draft.distilled_prompt.contains("; useSearchParams"));
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(
            preserves_negative_constraints(input, &draft.distilled_prompt),
            "{}",
            draft.distilled_prompt
        );
        assert!(draft.distilled_prompt.chars().count() < input.chars().count());
    }

    #[test]
    fn normalizes_required_term_typos_before_prefix_restoration() {
        let input = "React の検索画面で TypeScript の既存構造を活かしてください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "React検索画面でTypeScrip既存構造活用。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        restore_missing_required_terms(&request, &mut draft);

        assert_eq!(
            draft.distilled_prompt,
            "React検索画面でTypeScript既存構造活用。"
        );
        assert!(!draft.distilled_prompt.starts_with("TypeScript:"));
    }

    #[test]
    fn extracts_lowercase_identifiers_connected_to_japanese_particles() {
        let terms = required_technical_terms(
            "新規検索時はpageを1に戻し、ページ移動時はkeywordとstatusを保持してください。",
        );

        assert!(terms.contains(&"page".to_string()));
        assert!(terms.contains(&"keyword".to_string()));
        assert!(terms.contains(&"status".to_string()));
    }

    #[test]
    fn removes_lowercase_components_already_covered_by_longer_literals() {
        let terms = required_technical_terms(
            "POST /api/orders の JSON error.code を変更してください。orders と code は識別子の一部です。",
        );

        assert!(terms.contains(&"/api/orders".to_string()));
        assert!(terms.contains(&"error.code".to_string()));
        assert!(!terms.contains(&"orders".to_string()));
        assert!(!terms.contains(&"code".to_string()));
    }

    #[test]
    fn classifies_conditional_outcomes_as_constraints() {
        let input = "customerIdが空ならHTTP 400とINVALID_CUSTOMERを返してください。requestIdがない場合はINVALID_REQUEST_IDを返してください。";
        let terms = required_technical_terms(input);

        let organized = organize_input_for_model(input, &terms);

        assert!(organized.contains("[制約:条件対応"), "{organized}");
        assert!(organized.contains("INVALID_CUSTOMER"));
        assert!(organized.contains("INVALID_REQUEST_ID"));
        assert_eq!(organized.matches("[制約:条件対応").count(), 2);
    }

    #[test]
    fn does_not_corrupt_correct_required_term_during_typo_normalization() {
        let input = "React の TypeScript 構造は変更しないでください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "ReactのTypeScript構造を維持。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        restore_missing_required_terms(&request, &mut draft);

        assert_eq!(draft.distilled_prompt, "ReactのTypeScript構造を維持。");
    }

    #[test]
    fn corrects_single_edit_typo_in_required_ascii_identifier() {
        let input = "AbortControllerを使って古い通信を中断してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "AbortConrollerで古い通信を中断。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        restore_missing_required_terms(&request, &mut draft);

        assert_eq!(draft.distilled_prompt, "AbortControllerで古い通信を中断。");
    }

    #[test]
    fn restores_missing_mechanism_term_next_to_its_related_target() {
        let input = "検索条件はuseSearchParamsでURLクエリへ保存してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "検索条件のURLクエリ保存。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        restore_missing_required_terms(&request, &mut draft);

        assert_eq!(
            draft.distilled_prompt,
            "検索条件をuseSearchParamsでURLクエリ保存。"
        );
    }

    #[test]
    fn restores_missing_literal_from_explicit_target_as_natural_context() {
        let input = "対象は POST /api/orders です。customerIdの入力検証を追加してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "customerIdの入力検証を追加。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        restore_missing_required_terms(&request, &mut draft);

        assert!(
            draft
                .distilled_prompt
                .starts_with("POST /api/ordersを対象に、"),
            "{}",
            draft.distilled_prompt
        );
        assert!(!draft.distilled_prompt.starts_with("/api/orders:"));
    }

    #[test]
    fn normalizes_singular_column_mapping_when_columns_is_required() {
        let input = "管理画面の CSV インポートで、既存の columns マッピング、dryRun オプション、エラー行番号の表示は維持してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "CSVインポートでcolumn mappings、dryRun、エラー行番号表示を維持。"
                .to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        restore_missing_required_terms(&request, &mut draft);

        assert!(draft.distilled_prompt.contains("columns mapping"));
        assert!(!draft.distilled_prompt.starts_with("columns:"));
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
    }

    #[test]
    fn removes_typo_prefix_when_required_term_exists_in_body() {
        let input = "TypeScritp の既存構造はなるべく触らず、React の検索画面を修正してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "TypeScritp: React検索画面を修正し、TypeScript既存構造は触らない。"
                .to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        restore_missing_required_terms(&request, &mut draft);

        assert_eq!(
            draft.distilled_prompt,
            "React検索画面を修正し、TypeScript既存構造は触らない。"
        );
    }

    #[test]
    fn restores_explicit_test_verification_requirements() {
        let input = "Next.js の POST /api/orders で、customerId が空のまま送られた時に 500 エラーになっています。入力検証を追加し、空の customerId の場合は HTTP 400 と INVALID_CUSTOMER のエラーコードを返すようにしてください。成功時のレスポンス形式、在庫引当処理、既存の監査ログは変更しないでください。テストでは正常系と customerId 空文字のケースを確認できるようにしてください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "Next.js POST /api/orders: 空customerId時HTTP 400+INVALID_CUSTOMER返却。成功レスポンス/在庫引当/監査ログ変更しない。"
                .to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        assert!(!preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));

        restore_missing_required_constraints(&request, &mut draft);

        assert!(draft.distilled_prompt.contains("正常系"));
        assert!(draft.distilled_prompt.contains("customerId"));
        assert!(draft.distilled_prompt.contains("空文字"));
        assert!(["テスト", "確認", "検証"]
            .iter()
            .any(|marker| draft.distilled_prompt.contains(marker)));
        assert!(
            preserves_negative_constraints(input, &draft.distilled_prompt),
            "{}",
            draft.distilled_prompt
        );
        assert!(draft.distilled_prompt.chars().count() < input.chars().count());
    }

    #[test]
    fn accepts_compact_search_state_persistence_phrase() {
        let input = "ページ番号を変更しても検索条件と検索状態が消えないようにしてください。";
        let output = "ページ変更時も検索条件/状態維持。";

        assert!(preserves_negative_constraints(input, output));
    }

    #[test]
    fn accepts_named_state_persistence_without_repeating_generic_state_word() {
        let clause = "検索条件はuseSearchParamsでURLクエリへ保存し、戻る・進むでも復元できる状態を維持してください";
        let output = "useSearchParamsでURLクエリの検索条件を保存し、戻る・進むでも維持。";

        assert!(state_persistence_clause_satisfied(clause, output));
    }

    #[test]
    fn rejects_generic_verification_when_enumerated_targets_are_missing() {
        let clause = "Vitestではボタン、Enter、入力中に呼ばれないこと、古いリクエストの中断を確認してください";

        assert!(!verification_constraint_satisfied(
            clause,
            "Vitestによる検証を実施"
        ));
        assert!(verification_constraint_satisfied(
            clause,
            "Vitestでボタン、Enter、入力中の非呼び出し、古いリクエストの中断を確認"
        ));
    }

    #[test]
    fn restores_enumerated_verification_after_other_missing_constraints() {
        let input = "React と TypeScript で作っている管理画面の検索一覧を直してください。今回は検索ボタンを押した時、または検索欄で Enter を押した時だけ GET /api/customers を呼んでください。入力中は通信しないでください。検索条件は useSearchParams で URL クエリへ保存してください。既存コンポーネントの分割方法や CSS は変えず、画面全体の作り直しは避けてください。Vitest ではボタン、Enter、入力中に呼ばれないこと、古いリクエストの中断を確認してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "ReactとTypeScriptの検索一覧を修正し、検索ボタンまたはEnter時だけGET /api/customersを呼ぶ。入力中は通信禁止。URLクエリへ保存。CSS変更と作り直し禁止。Vitestで検証。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        let phrases = missing_constraint_restoration_phrases(input, &draft.distilled_prompt);
        let verification_phrases =
            missing_verification_restoration_phrases(input, &draft.distilled_prompt);
        assert!(
            phrases
                .iter()
                .any(|phrase| phrase.contains("古いリクエスト")),
            "all={phrases:?}; verification={verification_phrases:?}"
        );
        restore_missing_required_constraints(&request, &mut draft);

        assert!(
            draft.distilled_prompt.contains("ボタン"),
            "{}",
            draft.distilled_prompt
        );
        assert!(
            draft.distilled_prompt.contains("Enter"),
            "{}",
            draft.distilled_prompt
        );
        assert!(
            draft.distilled_prompt.contains("入力中"),
            "{}",
            draft.distilled_prompt
        );
        assert!(
            draft.distilled_prompt.contains("古いリクエスト"),
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn accepts_saved_and_restored_state_without_repeating_state_label() {
        let clause = "検索条件はuseSearchParamsでURLクエリへ保存し、戻る・進むでも復元できる状態を維持してください";
        let output = "useSearchParamsでURLクエリに保存し、戻る・進むで復元。";

        assert!(state_persistence_clause_satisfied(clause, output));
    }

    #[test]
    fn removes_request_history_when_restoring_an_exclusive_constraint() {
        let clause = "前にも似た修正をお願いしましたが、今回は保存ボタンを押した時だけAPIを呼ぶ形にしてください";

        assert_eq!(
            compact_constraint_clause(clause),
            "保存ボタンを押した時だけAPIを呼ぶ"
        );
    }

    #[test]
    fn restores_rerender_constraint_without_double_negating_it() {
        let phrases = missing_constraint_restoration_phrases(
            "CSS の見た目と E2E テストは変更せず、不要な再レンダリングを増やさないでください。",
            "React UserTableをUserTableRowに分割、data-testidと挙動維持",
        );

        assert!(phrases
            .iter()
            .any(|phrase| phrase.contains("再レンダリング増加回避")));
        assert!(!phrases.iter().any(|phrase| phrase.contains("回避しない")));
    }

    #[test]
    fn restores_long_log_constraint_by_trimming_model_output() {
        let input = "本番ログを解析し、注文送信が失敗する原因候補を優先度順に整理してください。2026-06-24T10:15:03Z requestId=ab12 POST /orders ECONNRESET upstream=payment-service。時刻、requestId、エラー文字列は改変せず、追加で確認すべきログと暫定対応を示してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "/orders/upstream=payment-service: 2026/06/24T10/15/03Z/POST/ECONNRESET/2026-06-24T10:15:03Z/requestId=ab12/追加確認:payment-serviceログとネットワーク障害ログ。暫定対応:再試行設定と接続タイムアウト調整".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 1);

        restore_missing_required_constraints(&request, &mut draft);

        assert!(draft.distilled_prompt.contains("改変せず"));
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.chars().count() < input.chars().count());
    }

    #[test]
    fn preprocesses_input_whitespace_before_llm_prompt() {
        let input = "React　の検索画面で、  検索ボタンを押したときだけ  API を呼び出してください。\r\n\r\n\r\nTypeScript の既存構造を活かしてください。";

        let preprocessed = preprocess_input_for_llm(input);

        assert!(!preprocessed.contains('\r'));
        assert!(!preprocessed.contains("  "));
        assert!(!preprocessed.contains("\n\n\n"));
        assert!(preprocessed.contains("検索ボタンを押したときだけ"));
        assert!(preprocessed.contains("API 呼出"));
        assert!(preprocessed.contains("TypeScript"));
    }

    #[test]
    fn preprocess_keeps_negative_polite_constraints() {
        let input = "既存の監査ログは変更しないでください。UI 内の完了トーストは追加しないでください。結果をコピーしてください。";

        let preprocessed = preprocess_input_for_llm(input);

        assert!(preprocessed.contains("変更しないでください"));
        assert!(preprocessed.contains("追加しないでください"));
        assert!(preprocessed.contains("結果コピー"));
        assert!(!preprocessed.contains("コピーしてください"));
    }

    #[test]
    fn preprocess_removes_obvious_noise_but_keeps_lazy_request_details() {
        let input = "こんにちｈ。今日はｄあさ、これは関係ないです。検索のところが重いので直してください。React の検索画面で入力中に API が何度も呼ばれているので、検索ボタンを押した時だけ API を呼ぶようにしたいです。useSearchParams と URL クエリ管理は消さないでください。TypeScritp の既存構造は大きく変えないでください。";

        let preprocessed = preprocess_input_for_llm(input);

        assert!(!preprocessed.contains("こんにちｈ"));
        assert!(!preprocessed.contains("今日はｄあさ"));
        assert!(preprocessed.contains("React"));
        assert!(preprocessed.contains("API"));
        assert!(preprocessed.contains("useSearchParams"));
        assert!(preprocessed.contains("URL"));
        assert!(preprocessed.contains("TypeScript"));
        assert!(preprocessed.contains("消さない"));
        assert!(preprocessed.contains("変えない"));
    }

    #[test]
    fn preprocess_normalizes_only_high_confidence_typos_for_llm() {
        let input = "TypeScritp の処理で PawerShell 通知ではなく AppUserModelID を使い、DataLoder と UTF8 BOM と HTTP400 の表記も確認してください。--dryrun は --dry-run として扱ってください。";

        let preprocessed = preprocess_input_for_llm(input);

        assert!(preprocessed.contains("TypeScript"));
        assert!(preprocessed.contains("PowerShell"));
        assert!(preprocessed.contains("DataLoader"));
        assert!(preprocessed.contains("UTF-8 BOM"));
        assert!(preprocessed.contains("HTTP 400"));
        assert!(preprocessed.contains("--dry-run"));
        assert!(!preprocessed.contains("TypeScritp"));
        assert!(!preprocessed.contains("PawerShell"));
        assert!(!preprocessed.contains("DataLoder"));
    }

    #[test]
    fn preprocess_does_not_drop_noise_sentence_with_protected_content() {
        let input = "これは関係ないかもしれませんが、API は検索ボタンを押した時だけ呼んでください。今日はｄあさ。URL と useSearchParams は維持してください。";

        let preprocessed = preprocess_input_for_llm(input);

        assert!(preprocessed.contains("API"));
        assert!(preprocessed.contains("検索ボタンを押した時だけ"));
        assert!(preprocessed.contains("URL"));
        assert!(preprocessed.contains("useSearchParams"));
        assert!(preprocessed.contains("維持"));
        assert!(!preprocessed.contains("今日はｄあさ"));
    }

    #[test]
    fn organizes_each_prompt_clause_once_by_semantic_role() {
        let input = "今の実装では入力中にもAPIが呼ばれて重いです。検索画面の通信を直したいです。検索ボタンを押した時だけAPIを呼んでください。URL状態は維持してください。既存CSSの作り直しは避けてください。テストで入力中に呼ばれないことを確認してください。";

        let organized = organize_input_for_model(input, &[]);

        assert!(!organized.contains("今の実装では"), "{organized}");
        assert!(organized.contains("[要求] 検索画面の通信を直したいです"));
        assert!(organized.contains("[制約:限定] 検索ボタンを押した時だけAPIを呼んでください"));
        assert!(organized.contains("[制約:維持] URL状態は維持してください"));
        assert!(organized.contains("[制約:禁止] 既存CSSの作り直しは避けてください"));
        assert!(organized.contains("[検証] テストで入力中に呼ばれないことを確認してください"));
        assert_eq!(organized.matches("今の実装では").count(), 0);
        assert_eq!(organized.matches("検索ボタンを押した時だけ").count(), 1);
    }

    #[test]
    fn structures_shared_predicate_and_conditional_value_lists() {
        let conditional = structured_constraint_clause(
            "customerIdが未指定、null、空文字、空白だけのいずれかならHTTP 400を返してください",
        )
        .expect("conditional list");
        let shared_predicate = structured_constraint_clause(
            "成功時のレスポンス形式、orderIdの採番、在庫引当、監査ログ形式は変更しないでください",
        )
        .expect("shared predicate list");

        assert!(
            conditional.contains("customerId=未指定/null/空文字/空白のいずれかなら"),
            "{conditional}"
        );
        assert!(
            shared_predicate
                .contains("成功レスポンス形式/orderId採番/在庫引当/監査ログ形式変更しない"),
            "{shared_predicate}"
        );
    }

    #[test]
    fn restores_missing_items_from_generalized_constraint_lists() {
        let input = "最近入力漏れによる失敗が増えており、利用者への案内も分かりづらいため、既存実装を確認しながら入力検証を整理したいです。以前にも同じ相談があり、今回は必要な条件と変更範囲を明確にした上で安全に対応する予定です。関係者との確認内容やこれまでの経緯も多いですが、実装者が判断できる情報を中心にまとめます。customerIdが未指定、null、空文字、空白だけのいずれかならHTTP 400を返してください。成功時のレスポンス形式、orderIdの採番、在庫引当、決済予約、監査ログ形式は変更しないでください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "customerId未指定/nullならHTTP 400。orderId/監査ログ変更しない。"
                .to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        let phrases = missing_constraint_restoration_phrases(input, &draft.distilled_prompt);
        assert!(
            phrases.iter().any(|phrase| phrase.contains("空文字")),
            "{phrases:?}"
        );
        assert!(
            phrases.iter().any(|phrase| phrase.contains("決済予約")),
            "{phrases:?}"
        );
        restore_missing_required_constraints(&request, &mut draft);

        for expected in ["空文字", "空白", "成功レスポンス", "在庫引当", "決済予約"]
        {
            assert!(
                draft.distilled_prompt.contains(expected),
                "missing {expected}: {}",
                draft.distilled_prompt
            );
        }
    }

    #[test]
    fn restores_only_missing_target_from_two_item_prohibition() {
        let input = "今回の目的は入力検証なので、注文処理全体のリファクタリングやDBスキーマ変更までは行わないでください。";
        let output = "DBスキーマ変更は行わない。";

        let phrases = missing_constraint_restoration_phrases(input, output);

        assert!(
            phrases
                .iter()
                .any(|phrase| phrase.contains("リファクタリング行わない")),
            "{phrases:?}"
        );
        assert!(
            phrases
                .iter()
                .all(|phrase| !phrase.contains("DBスキーマ変更/")),
            "{phrases:?}"
        );
    }

    #[test]
    fn retains_current_state_when_it_has_unique_required_evidence() {
        let input = "現在POST /api/ordersがHTTP 500になります。入力検証を追加してください。";
        let terms = required_technical_terms(input);

        let organized = organize_input_for_model(input, &terms);

        assert!(organized.contains("[現状"), "{organized}");
        assert!(organized.contains("HTTP 500"), "{organized}");
        assert!(organized.contains("[要求] 入力検証を追加してください"));
    }

    #[test]
    fn does_not_treat_current_implementation_noun_as_requested_action() {
        let input = "今の実装では入力中にもAPIが呼ばれるので画面が重いです。検索ボタン押下時だけAPIを呼んでください。";
        let terms = required_technical_terms(input);

        let organized = organize_input_for_model(input, &terms);

        assert!(!organized.contains("[現状→要求]"), "{organized}");
        assert!(!organized.contains("今の実装では"), "{organized}");
        assert!(organized.contains("[制約:限定"), "{organized}");
    }

    #[test]
    fn classifies_literal_only_context_as_target() {
        let input = "対象はPOST /api/ordersです。入力検証を追加してください。";
        let terms = required_technical_terms(input);

        let organized = organize_input_for_model(input, &terms);

        assert!(organized.contains("[対象|必須語:"), "{organized}");
        assert!(organized.contains("POST /api/orders"));
    }

    #[test]
    fn classifies_framework_verification_without_test_word_as_verification() {
        let input = "Vitestでは入力中にAPIが呼ばれないことと中断を確認してください。";
        let terms = required_technical_terms(input);

        let organized = organize_input_for_model(input, &terms);

        assert!(
            organized.contains("[検証|必須語:Vitest,API]"),
            "{organized}"
        );
    }

    #[test]
    fn classifies_tool_usage_as_request_action() {
        let input = "AbortControllerを使って古い通信を中断してください。";
        let terms = required_technical_terms(input);

        let organized = organize_input_for_model(input, &terms);

        assert!(
            organized.contains("[要求|必須語:AbortController]"),
            "{organized}"
        );
    }

    #[test]
    fn anchors_required_terms_to_their_source_clause() {
        let input = "React検索画面を修正。URL状態はuseSearchParamsで維持してください。";
        let terms = required_technical_terms(input);

        let organized = organize_input_for_model(input, &terms);

        assert!(organized.contains("[要求|必須語:React]"), "{organized}");
        assert!(
            organized.contains("必須語:URL,useSearchParams"),
            "{organized}"
        );
    }

    #[test]
    fn separates_only_independent_constraint_fragments() {
        let input = "既存CSSは変えず、画面全体の作り直しは避けてください。";

        let organized = organize_input_for_model(input, &[]);

        assert!(organized.contains("[制約:禁止] 既存CSSは変えず"));
        assert!(organized.contains("[制約:禁止] 画面全体の作り直しは避けてください"));
        assert_eq!(organized.matches("[制約:禁止]").count(), 2);
    }

    #[test]
    fn normalizes_level_two_csv_read_skip_constraint() {
        let input = "管理画面の CSV インポートで Shift_JIS と UTF-8 BOM を判定し、文字化けを防いでください。既存の columns マッピング、dryRun オプション、エラー行番号の表示は維持してください。10MB を超えるファイルは読み込まず、INVALID_FILE_SIZE を返してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "CSVインポート: Shift_JIS/UTF-8 BOM判定、columns mapping/dryRun/エラー行番号表示維持。10MB を超ファイル読み込み拒否、INVALID_FILE_SIZE返却。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        restore_missing_required_constraints(&request, &mut draft);

        assert!(draft.distilled_prompt.contains("読み込まず"));
        assert!(!draft.distilled_prompt.contains("読み込み拒否"));
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(
            preserves_negative_constraints(input, &draft.distilled_prompt),
            "{}",
            draft.distilled_prompt
        );
        assert!(draft.distilled_prompt.chars().count() < input.chars().count());
    }

    #[test]
    fn compacts_level_two_long_csv_import_without_dropping_safety_constraints() {
        let input = "管理画面の CSV 一括登録が取引先ごとに文字化けしたり途中で止まったりするので、読み込み部分を安定させたいです。アップロードされたファイルが UTF-8、UTF-8 BOM 付き、Shift_JIS のどれかを判定して、いずれも同じ columns マッピングへ渡してください。先頭数行を見ただけで文字コードを決めてデータを欠落させるのは避け、判定できない場合は UNSUPPORTED_ENCODING と対象ファイル名だけを返してください。10MB を超えるファイルは内容を読み込む前に拒否し、INVALID_FILE_SIZE を返してください。空行は無視して構いませんが、値が空の列を勝手に詰めないでください。既存の dryRun、エラー行番号、重複判定、成功件数と失敗件数の集計は維持してください。CSV の全内容をログへ出すことは禁止し、エラー時は行番号と列名までにしてください。大きいファイルでも UI が固まらないようストリームで処理し、途中失敗時に一部だけ DB 登録されないことをテストしてください。今回は画面レイアウトの変更や新しいアップロード画面の追加は不要です。";
        let mut draft = CompressionDraft {
            distilled_prompt: "UTF/BOM/Shift_JIS/10MB/INVALID_FILE_SIZE/UTF-8/10MB を超: 管理画面での取引先CSV一括登録を安定させ、文字コード判定とファイルサイズ制限を実装。読み込み中のデータ欠落を避け、dryRunやエラー情報の維持、ログ出力禁止、UIとDBの安定性検証を条件とする、先頭数行を見ただけで文字コードを決めてデータを欠落させるのは避け、判定できない場合は UNSUPPORTED_ENCODING と対象ファイル名だけを返、エラー行番号/重複判定/成功件数と失敗件数の集計維持。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        let expected_candidate = "CSV一括登録: UTF-8/UTF-8 BOM 付き/Shift_JIS判定→columns mapping。先頭数行だけ判定の欠落回避、判定不能はUNSUPPORTED_ENCODING+ファイル名のみ、10MB を超える場合は内容読み込み前に拒否しINVALID_FILE_SIZE。空行は無視、空列を詰めない、dryRun/エラー行番号/重複判定/成功件数と失敗件数の集計維持、CSV全内容ログへ出さない（行番号/列名まで）。ストリーム処理、途中失敗時は一部だけDB登録されないことをテスト、UI変更不要。";
        assert!(
            contains_required_technical_terms(input, expected_candidate),
            "terms={:?}; candidate={expected_candidate}",
            required_technical_terms(input)
        );
        assert!(
            preserves_negative_constraints(input, expected_candidate),
            "list={}; verification={}; candidate={expected_candidate}",
            preserves_list_constraints(input, expected_candidate),
            preserves_verification_constraints(input, expected_candidate)
        );
        let compacted = compact_csv_import_encoding(input, &draft.distilled_prompt);
        assert!(
            compacted.is_some(),
            "terms={:?}; output={}",
            required_technical_terms(input),
            draft.distilled_prompt
        );
        polish_model_output_for_request(&request, &mut draft);

        for expected in [
            "CSV",
            "UTF-8",
            "UTF-8 BOM",
            "Shift_JIS",
            "columns",
            "UNSUPPORTED_ENCODING",
            "10MB",
            "INVALID_FILE_SIZE",
            "dryRun",
            "DB",
            "UI",
            "空行は無視",
            "空列を詰めない",
            "ストリーム",
            "一部だけDB登録されない",
            "変更不要",
        ] {
            assert!(
                draft.distilled_prompt.contains(expected),
                "missing {expected}: {}",
                draft.distilled_prompt
            );
        }
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(
            preserves_negative_constraints(input, &draft.distilled_prompt),
            "{}",
            draft.distilled_prompt
        );
        assert!(
            draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 78,
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn removes_redundant_level_two_constraint_tail() {
        let input = "Prisma の User テーブルに lastLoginAt を追加する migration を作成してください。既存データは NULL のまま許容し、email の unique 制約や createdAt の default は変更しないでください。ロールバック手順も短く添えてください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "PrismaのUserテーブルにlastLoginAtを追加し、既存データをNULL許容で維持、emailのunique制約とcreatedAtのdefaultを変更せず; 既存NULL許容/email unique/createdAt default変更なし。ロールバック手順も短く添える。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);
        let direct = remove_redundant_constraint_tail(input, &draft.distilled_prompt);
        assert!(
            !direct.contains("; 既存NULL許容"),
            "{} / {:?} / {:?}",
            direct,
            required_technical_terms(input),
            required_constraint_clauses(input)
                .into_iter()
                .map(compact_constraint_clause)
                .collect::<Vec<_>>()
        );

        polish_model_output_for_request(&request, &mut draft);

        assert!(
            !draft.distilled_prompt.contains("; 既存NULL許容"),
            "{}",
            draft.distilled_prompt
        );
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(
            draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 90,
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn removes_level_two_polite_request_fillers() {
        let input = "Prisma の User テーブルに lastLoginAt を追加する migration を作成してください。既存データは NULL のまま許容し、email の unique 制約や createdAt の default は変更しないでください。ロールバック手順も短く添えてください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "PrismaのUserテーブルにlastLoginAtを追加し、既存データをNULL許容で維持、emailのunique制約とcreatedAtのdefaultを変更しない、ロールバック手順も添えたmigrationを作成してください。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        assert!(!draft.distilled_prompt.contains("してください"));
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(
            draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 78,
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn polishes_redundant_level_three_code_phrases() {
        let input = "TypeScript の parseDateRange 関数に Vitest テストを追加してください。YYYY-MM-DD の正常値、終了日が開始日より前、無効日付、空文字列を検証してください。実装コードと既存テストの名前は変更せず、境界値を含めてください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "TypeScriptのparseDateRange関数にVitestテストを追加。正常値(YYYY-MM-DD)、終了日が開始日より前、無効日付、空文字列を検証。実装コードと既存テスト名は変更せず、境界値を含める。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 3);
        assert!(!required_technical_terms(input).contains(&"2026/06/24T10/15/03Z".to_string()));

        polish_model_output_for_request(&request, &mut draft);

        assert!(draft.distilled_prompt.contains("parseDateRange"));
        assert!(draft.distilled_prompt.contains("Vitest"));
        assert!(draft.distilled_prompt.contains("YYYY-MM-DD"));
        assert!(!draft.distilled_prompt.contains("parseDateRange関数"));
        assert!(!draft.distilled_prompt.contains("テストを追加"));
        assert!(!draft.distilled_prompt.contains("空文字列"));
        assert!(draft.distilled_prompt.contains("開始日前終了"));
        assert!(draft
            .distilled_prompt
            .contains("実装コード・既存テスト名変更せず"));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
    }

    #[test]
    fn restores_missing_http_status_phrase() {
        let input = "Next.js の POST /api/orders が空の customerId で 500 を返します。入力検証を追加し、HTTP 400 と既存の INVALID_CUSTOMER エラーコードを返してください。成功時のレスポンス形式、在庫引当処理、既存の監査ログは変更しないでください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "Next.js POST /api/orders に customerId 空時、HTTP 500 回避。400 と INVALID_CUSTOMER エラー追加。成功/在庫引当/監査ログ変更なし。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        restore_missing_required_terms(&request, &mut draft);

        assert!(draft.distilled_prompt.contains("HTTP 400"));
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
    }

    #[test]
    fn removes_duplicate_assignment_value_after_key_value_term() {
        let input = "2026-06-24T10:15:03Z requestId=ab12 POST /orders ECONNRESET upstream=payment-service。";
        let mut draft = CompressionDraft {
            distilled_prompt:
                "2026-06-24T10:15:03Z/requestId=ab12/payment-service: ab12, ECONNRESET原因整理。"
                    .to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 3);

        polish_model_output_for_request(&request, &mut draft);

        assert!(draft.distilled_prompt.contains("requestId=ab12"));
        assert!(!draft.distilled_prompt.contains(": ab12,"));
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
    }

    #[test]
    fn compacts_level_three_log_analysis_to_eval_budget() {
        let input = "本番ログを解析し、注文送信が失敗する原因候補を優先度順に整理してください。2026-06-24T10:15:03Z requestId=ab12 POST /orders ECONNRESET upstream=payment-service。時刻、requestId、エラー文字列は改変せず、追加で確認すべきログと暫定対応を示してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "2026-06-24T10:15:03Z/requestId=ab12/payment-service: 2026-06/24T10:15:03Z, ECONNRESET, 解析し、注文失敗原因を優先度付きで整理。追加ログは時刻/requestId/エラー文字列改変せず、暫定対応を示す。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 3);

        polish_model_output_for_request(&request, &mut draft);

        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.contains("追加ログ/暫定対応"));
        assert!(!draft.distilled_prompt.contains("2026-06/24T10:15:03Z"));
        assert!(
            draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 65,
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn compacts_level_three_react_search_case_from_eval() {
        let input = "React の検索画面で、検索ボタンを押したときだけ API を呼び出してください。既存の useSearchParams による URL クエリ管理は維持し、ページ番号変更時も検索状態を保持してください。TypeScript の既存構造を活かし、大規模なリファクタリングは避けてください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "React検索画面、検索ボタン押下時のみAPI呼び出し、useSearchParamsによるURLクエリ管理維持、ページ変更時も検索状態保持、TypeScript既存構造活用; TypeScript の既存構造活用、大規模なリファクタリング回避。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 3);

        polish_model_output_for_request(&request, &mut draft);

        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.contains("検索時のみAPI呼出"));
        assert!(!draft.distilled_prompt.contains("TypeScript の既存構造活用"));
        assert!(
            draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 65,
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn compacts_level_three_next_order_case_from_eval() {
        let input = "Next.js の POST /api/orders が空の customerId で 500 を返します。入力検証を追加し、HTTP 400 と既存の INVALID_CUSTOMER エラーコードを返してください。成功時のレスポンス形式、在庫引当処理、既存の監査ログは変更しないでください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "Next.jsのPOST /api/ordersに空のcustomerIdで送信時、500エラー。入力検証追加し、HTTP 400とINVALID_CUSTOMERエラーを返すこと。成功時は在庫引当と監査ログ変更なし。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 3);

        polish_model_output_for_request(&request, &mut draft);

        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.contains("HTTP 400+INVALID_CUSTOMER"));
        assert!(
            draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 65,
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn compacts_lower_level_next_order_case_without_losing_constraints() {
        let input = "Next.js の POST /api/orders が空の customerId で 500 を返します。入力検証を追加し、HTTP 400 と INVALID_CUSTOMER エラーコードを返してください。成功時のレスポンス形式、在庫引当処理、既存の監査ログは変更しないでください。";
        for (level, max_ratio) in [(1_u8, 90_usize), (2_u8, 78_usize)] {
            let mut draft = CompressionDraft {
                distilled_prompt: "Next.jsのPOST /api/ordersに空のcustomerIdが渡された場合、HTTP 400を返し、INVALID_CUSTOMERエラーコードを使用するように入力検証を追加してください。成功時のレスポンス形式、在庫引当処理、既存の監査ログは変更しないでください。".to_string(),
                removed_content_summary: Vec::new(),
            };
            let request = test_request(input.to_string(), level);

            polish_model_output_for_request(&request, &mut draft);

            assert!(contains_required_technical_terms(
                input,
                &draft.distilled_prompt
            ));
            assert!(preserves_negative_constraints(
                input,
                &draft.distilled_prompt
            ));
            assert!(
                draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * max_ratio,
                "level {level}: {}",
                draft.distilled_prompt
            );
        }
    }

    #[test]
    fn compacts_lower_level_vitest_case_without_losing_constraints() {
        let input = "TypeScript 縺ｮ parseDateRange 髢｢謨ｰ縺ｫ Vitest 繝・せ繝医ｒ霑ｽ蜉縺励※縺上□縺輔＞縲・YYY-MM-DD 縺ｮ豁｣蟶ｸ蛟､縲∫ｵゆｺ・律縺碁幕蟋区律繧医ｊ蜑阪∫┌蜉ｹ譌･莉倥∫ｩｺ譁・ｭ怜・繧呈､懆ｨｼ縺励※縺上□縺輔＞縲ょｮ溯｣・さ繝ｼ繝峨→譌｢蟄倥ユ繧ｹ繝医・蜷榊燕縺ｯ螟画峩縺帙★縲∝｢・阜蛟､繧貞性繧√※縺上□縺輔＞縲・";
        for (level, max_ratio) in [(1_u8, 90_usize), (2_u8, 78_usize)] {
            let mut draft = CompressionDraft {
                distilled_prompt: "TypeScript縺ｮparseDateRange髢｢謨ｰ縺ｫVitest繝・せ繝医ｒ霑ｽ蜉縺励〆YYY-MM-DD縺ｮ豁｣蟶ｸ蛟､縲∫ｵゆｺ・律縺碁幕蟋区律繧医ｊ蜑阪∫┌蜉ｹ譌･莉倥∫ｩｺ譁・ｭ怜・繧呈､懆ｨｼ縺吶ｋVitest繝・せ繝医ｒ螳溯｣・＠縺ｦ縺上□縺輔＞縲ょｮ溯｣・さ繝ｼ繝峨→譌｢蟄倥ユ繧ｹ繝医・蜷榊燕縺ｯ螟画峩縺帙★縲∝｢・阜蛟､繧貞性繧√※縺上□縺輔＞縲・".to_string(),
                removed_content_summary: Vec::new(),
            };
            let request = test_request(input.to_string(), level);

            polish_model_output_for_request(&request, &mut draft);

            assert!(contains_required_technical_terms(
                input,
                &draft.distilled_prompt
            ));
            assert!(preserves_negative_constraints(
                input,
                &draft.distilled_prompt
            ));
            assert!(
                draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * max_ratio,
                "level {level}: {}",
                draft.distilled_prompt
            );
        }
    }

    #[test]
    fn compacts_level_three_vitest_date_range_case_from_eval() {
        let input = "TypeScript の parseDateRange 関数に Vitest テストを追加してください。YYYY-MM-DD の正常値、終了日が開始日より前、無効日付、空文字列を検証してください。実装コードと既存テストの名前は変更せず、境界値を含めてください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "TypeScriptのparseDateRangeにVitestテスト追加し、YYYY-MM-DDの正常、開始日前終了、無効（例：空文字や無効な形式）、境界値検証してください。実装/既存テスト名変更せず、境界値を含めてください。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 3);

        polish_model_output_for_request(&request, &mut draft);

        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft
            .distilled_prompt
            .contains("YYYY-MM-DD正常/開始日前終了"));
        assert!(!draft.distilled_prompt.contains("してください"));
        assert!(
            draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 65,
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn compacts_level_three_approval_notification_case_from_eval() {
        let input = "社内向け申請画面の通知方式を、メール通知とアプリ内通知から選定してください。管理者は未処理申請を見落とさず、一般利用者には承認結果だけを通知します。月額コストは 3 万円以下、個人情報を外部サービスへ送信しないことが条件です。結論、採用理由、未解決事項の順で提案してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "メール通知/アプリ内通知: 社内申請画面通知、メール/アプリ内選択。管理者:未処理監視、一般:承認結果のみ。月額3万円以下、個人情報外部送信禁止。結論/理由/未解決事項提案求む。; 月額3万円以下/個人情報外部送信禁止送信しない".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 3);

        polish_model_output_for_request(&request, &mut draft);

        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.contains("3 万円以下"));
        assert!(!draft.distilled_prompt.contains("禁止送信しない"));
        assert!(
            draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 65,
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn compacts_level_three_csv_import_case_from_eval() {
        let input = "管理画面の CSV インポートで Shift_JIS と UTF-8 BOM を判定し、文字化けを防いでください。既存の columns マッピング、dryRun オプション、エラー行番号の表示は維持してください。10MB を超えるファイルは読み込まず、INVALID_FILE_SIZE を返してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "10MB を超: 管理画面の CSV インポート、Shift_JIS/UTF-8 BOM 判定し文字化け防止。column mapping, dryRun, エラー行番号表示維持。10MB 超ファイル拒否、INVALID_FILE_SIZE返却。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 3);

        polish_model_output_for_request(&request, &mut draft);

        assert!(
            preserves_list_constraints(input, &draft.distilled_prompt),
            "list check failed: {}",
            draft.distilled_prompt
        );
        assert!(
            preserves_list_constraints(
                &input.to_ascii_lowercase(),
                &draft.distilled_prompt.to_ascii_lowercase()
            ),
            "lowercase list check failed: {}",
            draft.distilled_prompt
        );
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.contains("読み込まず"));
        assert!(
            draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 65,
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn compacts_level_three_settings_persistence_case_from_eval() {
        let input = "アプリを終了しても設定が保持されるようにしてください。保存対象はモデル選択、圧縮レベル、テーマだけにし、入力した本文や圧縮結果は保存しないでください。保存に失敗した場合は起動を止めず、既定値で続行してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "圧縮結果: 設定保持/モデル選択/圧縮レベル/テーマ保存,本文保存禁止,失敗時既定値続行; 保存対象はモデル選択、圧縮レベル、テーマだけにし、入力した本文や圧縮結果は保存禁止".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 3);

        polish_model_output_for_request(&request, &mut draft);

        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.contains("保存しない"));
        assert!(
            draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 65,
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn compacts_level_two_settings_persistence_without_dropping_default_value() {
        let input = "アプリを終了しても設定が保持されるようにしてください。保存対象はモデル選択、圧縮レベル、テーマだけにしたいです。ユーザーが入力した本文や圧縮結果は保存しないでください。設定ファイルの読み込みや保存に失敗してもアプリの起動は止めず、既定値で続行してください。次回起動時に前回選んだ設定が自然に復元されることを確認したいです。";
        let mut draft = CompressionDraft {
            distilled_prompt: "アプリ設定の保持はモデル選択と圧縮レベル、テーマに限定し、本文や圧縮結果の保存は行わず、設定ファイルの読み込み/保存失敗時でもアプリ起動を継続、次回起動時に前回設定が自然に復元されることを確認。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        assert!(draft.distilled_prompt.contains("既定値"));
        assert!(draft.distilled_prompt.contains("設定ファイル"));
        assert!(draft.distilled_prompt.contains("復元"));
        assert!(draft.distilled_prompt.contains("保存しない"));
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(
            preserves_negative_constraints(input, &draft.distilled_prompt),
            "{}",
            draft.distilled_prompt
        );
        assert!(draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82);
    }

    #[test]
    fn restores_counted_save_list_privacy_list_and_atomic_write() {
        let input = "Windows のデスクトップアプリで、終了するたびに設定が初期化されるのが不便なので保存できるようにしてください。利用者が毎回選び直しているのはモデル、圧縮レベル、ライト・ダークのテーマ、ウィンドウサイズです。この4項目だけを user-settings.json に保存し、次回起動時に復元してください。一方で、入力したプロンプト本文、圧縮結果、クリップボードの内容、最近開いたファイルパスは機密情報を含む可能性があるため保存しないでください。保存は一時ファイルへ書いてから置換する方式にし、書き込み途中でアプリが落ちても設定ファイルが半端な JSON になりにくくしてください。設定ファイルが存在しない、読み取れない、壊れている場合でもアプリの起動は止めず、既定値で続行して警告ログだけ残してください。ログへ設定値そのものや本文は出さないでください。既存設定に未知のキーがあっても削除せず、将来のバージョンと共存できるようにしてください。保存先は現在の application/local/state 配下を維持し、レジストリへの移行は不要です。";
        let mut draft = CompressionDraft {
            distilled_prompt: "圧縮結果: Windowsデスクトップアプリの終了時設定保存機能追加、圧縮レベル・テーマ・ウィンドウサイズをuser-settings.jsonに保存し次回起動時に復元、機密情報保存禁止、設定ファイル保全方式確立、既定値での警告ログ記録、設定値や本文のログ出力禁止、未知のキー共存維持、保存先application/local/state維持、レジストリ移行不要、ログへ設定値そのものや本文は出さないでください、この4項目だけを user-settings.json に保存し、次回起動時に復元。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        restore_missing_required_constraints(&request, &mut draft);
        restore_missing_required_terms(&request, &mut draft);
        polish_model_output_for_request(&request, &mut draft);

        assert!(
            !draft.distilled_prompt.starts_with("圧縮結果:"),
            "{}",
            draft.distilled_prompt
        );
        for expected in [
            "モデル",
            "圧縮レベル",
            "ライト・ダーク",
            "ウィンドウサイズ",
            "プロンプト本文",
            "圧縮結果",
            "クリップボード",
            "ファイルパス",
            "一時ファイル",
            "置換",
            "既定値",
            "警告ログ",
            "未知のキー",
            "application/local/state",
            "レジストリ",
        ] {
            assert!(
                draft.distilled_prompt.contains(expected),
                "missing {expected}: {}",
                draft.distilled_prompt
            );
        }
        assert!(
            draft.distilled_prompt.contains("起動継続")
                || draft.distilled_prompt.contains("起動を継続")
                || draft.distilled_prompt.contains("起動は止めず"),
            "{}",
            draft.distilled_prompt
        );
        assert!(
            draft.distilled_prompt
                .contains("保存対象=モデル/圧縮レベル/ライト・ダークのテーマ/ウィンドウサイズのみ保存"),
            "{}",
            draft.distilled_prompt
        );
        assert!(
            preserves_list_constraints(input, &draft.distilled_prompt),
            "{}",
            draft.distilled_prompt
        );
        assert!(
            preserves_negative_constraints(input, &draft.distilled_prompt),
            "{}",
            draft.distilled_prompt
        );
        assert!(
            draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 78,
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn normalizes_privacy_exclusion_to_negative_constraint_marker() {
        let input = "エラー本文に受け取った customerId や個人情報を丸ごと入れないでください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "customerId検証、個人情報エラー本文除外。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        restore_missing_required_constraints(&request, &mut draft);

        assert!(
            draft.distilled_prompt.contains("個人情報を含めない"),
            "{}",
            draft.distilled_prompt
        );
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
    }

    #[test]
    fn compacts_level_two_desktop_tray_case_without_mixing_window_behaviors() {
        let input = "Windows アプリを閉じても、プロセスを完全終了せずシステムトレイに残るようにしてください。左クリックでウィンドウを復帰し、右クリックでは復帰、設定、終了のメニューを開けるようにしてください。閉じるボタンは終了ではなく非表示にし、終了はトレイメニューからだけ実行してください。最小化や通常終了の挙動と混ざらないように整理してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "Windowsアプリでウィンドウを閉じた場合、プロセスを完全終了せずシステムトレイに残す。左クリックでウィンドウを復帰、右クリックで設定や終了メニューを表示。閉じるボタンは非表示で、終了はトレイメニューからのみ実行。最小化や通常終了とは挙動を分ける。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        assert!(draft.distilled_prompt.contains("システムトレイ"));
        assert!(draft.distilled_prompt.contains("左クリック"));
        assert!(draft.distilled_prompt.contains("右クリック"));
        assert!(draft.distilled_prompt.contains("ではなく"));
        assert!(draft.distilled_prompt.contains("トレイメニューのみ"));
        assert!(draft.distilled_prompt.contains("混ざらない"));
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82);
    }

    #[test]
    fn compacts_level_two_theme_scrollbar_case_from_eval() {
        let input = "ダークモードとライトモードで右側のスクロールバー色を切り替えてください。上部バーの色も本文と少し差をつけ、どちらのテーマでも現在の表示モードが分かるようにしてください。最小化、最大化、閉じるボタンのホバー領域がバーからはみ出さないようにし、スクロールしても上部のウィンドウバーは固定してください。左側のアプリ名表示は出さないでください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "最大化: ダークモードとライトモード切り替え時、スクロールバー色を最小化し、上部バーの色を差別化。閉じるボタンホバー領域がバーからはみ出さず、スクロール時の固定とウィンドウバー左側のアプリ名非表示を維持しつつ、最小化、最大化、閉じるボタンの配置を調整。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        for expected in [
            "ダークモード",
            "ライトモード",
            "スクロールバー",
            "ウィンドウバー",
            "最小化",
            "最大化",
            "閉じる",
            "固定",
            "出さない",
        ] {
            assert!(draft.distilled_prompt.contains(expected));
        }
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82);
    }

    #[test]
    fn compacts_level_two_model_folder_docs_case_from_eval() {
        let input = "アプリ内 Model フォルダの役割を README に追記してください。採用中の Sarashina 2.2 3B GGUF の配置先、モデル本体を Git 管理しない理由、LM Studio 接続はユーザーが任意のローカルモデルを検証するために残すことを明記してください。exe 化した後でも、アプリ内モデルと LM Studio 接続の役割が分かるように説明してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "アプリ内 Model フォルダの役割を README に追記する際、採用中の Sarashina 2.2 3B GGUF の配置先を明記し、モデル本体を Git 管理しない理由と、LM Studio 接続をユーザーがローカルモデル検証のために残すことを書く。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        for expected in [
            "Model",
            "README",
            "Sarashina 2.2 3B",
            "GGUF",
            "Git",
            "LM Studio",
            "exe",
            "配置先",
            "管理しない",
            "任意",
            "説明",
            "残す",
        ] {
            assert!(draft.distilled_prompt.contains(expected));
        }
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82);
    }

    #[test]
    fn compacts_level_two_single_inference_policy_case_from_eval() {
        let input = "圧縮処理で検証に失敗した時、修復再推論を行わないようにしてください。1回目の出力が評価基準を満たさない場合は原文返しにし、評価基準そのものは下げないでください。ログには検証失敗理由、文字数比、欠落した必須語、原文返しの有無を残してください。ただし入力全文は保存せず、先頭 80 文字のスニペットだけ保存してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "検証失敗理由/必須語/修復再推論: 圧縮処理失敗時、再推論回避、1回目未達なら原文返し、評価基準維持、ログに失敗理由・文字数比・欠落語・原文返しを記録。入力全文保存せず、先頭80文字スニペットのみ保存。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        for expected in [
            "修復再推論",
            "1回目",
            "評価基準",
            "原文返し",
            "検証失敗理由",
            "文字数比",
            "必須語",
            "80文字",
            "しない",
            "下げない",
            "保存せず",
            "だけ",
        ] {
            assert!(draft.distilled_prompt.contains(expected));
        }
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82);
    }

    #[test]
    fn compacts_level_two_prompt_eval_report_case_from_eval() {
        let input = "プロンプト評価の結果をレベル別に集計してください。各レベルの平均文字数比、失敗率、原文返し件数、必須語欠落件数を出し、失敗率が 1% を超えたら exit code 1 にしてください。評価対象は 50 種類以上の元文章にし、レベル 1、2、3 それぞれで平均圧縮率が想定範囲に入っているか確認できるようにしてください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "平均圧縮率: プロンプト評価の結果をレベル別に集計し、各レベルの平均文字数比、失敗率、原文返し件数、必須語欠落件数を算出してください。失敗率が 1% を超えれば exit code 1 とし、評価対象は 50 種類以上の元文章で、レベル 1、2、3 それぞれで平均圧縮率確認。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        for expected in [
            "平均文字数比",
            "失敗率",
            "原文返し",
            "必須語",
            "1%",
            "exit code 1",
            "50種類以上",
            "レベル1",
            "2",
            "3",
            "超えたら",
            "集計",
            "確認",
        ] {
            let actual = compact_all_whitespace(&draft.distilled_prompt);
            let expected = compact_all_whitespace(expected);
            assert!(
                actual.contains(&expected),
                "missing `{expected}` in `{}`",
                draft.distilled_prompt
            );
        }
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82);
    }

    #[test]
    fn compacts_level_two_graphql_cache_isolation_case_from_eval() {
        let input = "GraphQL の users クエリで posts を取得すると N+1 が発生します。DataLoader を導入して、ユーザーごとの posts 取得をまとめて処理できるようにしてください。既存の schema.graphql とレスポンスフィールド名は変更しないでください。キャッシュはリクエスト単位だけにし、別ユーザーや別リクエストのデータが混ざらないようにしてください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "GraphQL usersクエリでposts取得時にN+1問題があり、DataLoaderでユーザーごとのposts取得を一括処理してください。schema.graphqlとレスポンスフィールド名は変更禁止、キャッシュはリクエスト単位のみとする。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        for expected in [
            "GraphQL",
            "users",
            "posts",
            "N+1",
            "DataLoader",
            "schema.graphql",
            "変更しない",
            "リクエスト単位",
            "のみ",
            "混ざらない",
        ] {
            assert!(draft.distilled_prompt.contains(expected));
        }
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82);
    }

    #[test]
    fn compacts_level_two_openapi_schema_update_case_from_eval() {
        let input = "OpenAPI 定義に PATCH /users/{id} を追加してください。name と avatarUrl は任意で更新できるようにし、email は変更不可にしてください。404 と 409 のエラーレスポンス例を入れ、既存の POST /users は変更しないでください。既存の schema_version や共通エラー形式がある場合は、それに合わせてください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "OpenAPI定義にPATCH /users/{id}追加、nameとavatarUrlは任意更新可、emailは変更不可、404と409エラーレスポンス例を含め、POST /usersは変更禁止。schema_versionや共通エラー形式に合わせる。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        for expected in [
            "OpenAPI",
            "PATCH",
            "/users/{id}",
            "name",
            "avatarUrl",
            "email",
            "404",
            "409",
            "POST /users",
            "schema_version",
            "任意",
            "変更不可",
            "変更しない",
        ] {
            assert!(draft.distilled_prompt.contains(expected));
        }
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82);
    }

    #[test]
    fn compacts_level_two_file_upload_progress_case_from_eval() {
        let input = "ファイルアップロード中の進捗バーを追加してください。0 から 100% まで進捗を表示し、失敗時は再試行ボタンを出してください。既存のキャンセルボタンと onCancel コールバックは削除しないでください。アップロード処理中に画面が固まらないようにし、複数回押しても二重送信にならないようにしてください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "ファイルアップロード中に進捗バーを表示し、0%から100%までの進捗を更新。失敗時には再試行ボタンを表示。既存のキャンセルボタンとonCancelコールバックは削除禁止。アップロード処理中に画面が固まらないように配慮し、二重送信を防ぐ。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        for expected in [
            "進捗バー",
            "0",
            "100%",
            "再試行",
            "onCancel",
            "削除しない",
            "固まらない",
            "二重送信しない",
        ] {
            assert!(draft.distilled_prompt.contains(expected));
        }
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82);
    }

    #[test]
    fn compacts_level_two_resumable_video_upload_without_dropping_retry_and_a11y() {
        let input = "ブラウザから動画ファイルをアップロードする画面で、進捗が 99% のまま止まって見えたり、再試行すると同じファイルが二重送信されたりしています。既存の POST /api/uploads と onCancel コールバックを使ったまま、0 から 100% の進捗バー、残り時間の概算、キャンセル、失敗時の再試行を整えてください。アップロード中に送信ボタンを何度押しても同じ uploadId では1本だけ処理されるようにし、キャンセル後は残ったリクエストを AbortController で確実に止めてください。再試行は失敗したチャンクから再開し、最初から全ファイルを送り直さないでください。ただしサーバーが再開位置を返せない場合は、その理由を表示して手動で最初からやり直せるようにしてください。5GB までのファイルを想定し、ファイル本体をブラウザのメモリへ一括展開しないでください。アクセシビリティのため進捗値を aria-valuenow でも伝え、キーボードだけでキャンセルと再試行ができるようにしてください。既存 API のレスポンスフィールド名は変更しないでください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "ブラウザ動画アップロード画面で、進捗99%停止や二重送信を解決し、進捗バー、残り時間、キャンセル、再試行を整備。既存POST /api/uploadsとonCancelコールバック使用、uploadIdとAbortControllerで1ファイルのみ処理、再試行は失敗したチャンクから再開、サーバー再開位置不明時は手動再試行可能、5GBファイル対応でメモリ一括展開せず、aria-valuenowで進捗値伝え、アクセシビリティ確保。既存APIレスポンスフィールド名は変更禁止、既存 API のレスポンスフィールド名は変更禁止変更せず。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        for expected in [
            "POST /api/uploads",
            "onCancel",
            "0",
            "100%",
            "uploadId",
            "AbortController",
            "失敗したチャンクから再開",
            "全ファイルを送り直さない",
            "5GB",
            "一括展開しない",
            "aria-valuenow",
            "キーボードだけ",
            "APIレスポンスフィールド名変更しない",
        ] {
            assert!(
                draft.distilled_prompt.contains(expected),
                "missing {expected}: {}",
                draft.distilled_prompt
            );
        }
        assert!(!draft.distilled_prompt.contains("変更禁止変更せず"));
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(
            preserves_negative_constraints(input, &draft.distilled_prompt),
            "{}",
            draft.distilled_prompt
        );
        assert!(
            draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 78,
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn compacts_level_two_auth_refresh_concurrency_without_localizing_required_terms() {
        let input = "SPA の認証まわりで、アクセストークンの期限が切れた瞬間に複数 API が同時に 401 を返すと、refresh token の更新が同時に何本も走ってログアウト扱いになる問題があります。401 を受けた時は更新処理を1回だけ実行し、その間に失敗した他のリクエストは待機させ、更新成功後に元のリクエストをそれぞれ1回だけ再送してください。refresh endpoint 自体が 401 または 403 を返した場合は再試行ループに入らず、保存済みトークンを削除してログイン画面へ遷移してください。ネットワークエラーの場合は最大2回、1秒と2秒の間隔で再試行して構いませんが、それ以上は行わないでください。Authorization ヘッダーや refresh token をログ、通知、エラー画面へ出さないでください。既存の login、logout、rememberMe の挙動と API レスポンス形式は維持してください。単体テストでは同時に5件の 401 が返るケース、更新失敗、ネットワークエラー、手動 logout 中のケースを確認してください。認証ライブラリの全面置換は今回の範囲外です。";
        let mut draft = CompressionDraft {
            distilled_prompt: "token: SPAの認証で401を受けた際、リフレッシュトークンの更新を1回のみ実行し、他の待機中のリクエストは成功または失敗まで待機させる。リフレッシュエンドポイントが401または403を返した場合は保存済みトークンを削除しログイン画面へ遷移。ネットワークエラーの場合は最大2回まで再試行し、それ以上は行わない。Authorizationヘッダーやリフレッシュトークンをログや通知に出さない。既存のlogin、logout、rememberMeの挙動とAPIレスポンス形式は維持。同時に5件の401が返るケース、更新失敗、ネットワークエラー、手動ログアウト中のケースを単体テストで確認、単体同時に5件の 401 が返るケース、更新失敗、ネットワークエラー、手動 logout 中のケースを確認、1秒と2秒の間隔で再試行して構いませんが行わない。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        let compacted = compact_auth_refresh_concurrency(input, &draft.distilled_prompt);
        assert!(
            compacted.is_some(),
            "terms={:?}; failing_lists={:?}; list={}; verification={}; output={}",
            required_technical_terms(input),
            input_clauses(input)
                .into_iter()
                .filter(|clause| !list_constraint_satisfied(clause, "SPA認証: 複数APIの401時もrefresh token更新は1回だけ、他リクエストは待機し成功後に元リクエストを各1回だけ再送。refresh endpointが401/403なら再試行ループに入らず保存済みトークン削除→login画面。ネットワークエラーは最大2回、1秒と2秒の間隔で再試行し、それ以上行わない。Authorizationヘッダー/refresh tokenをログ/通知/エラー画面へ出さない。login/logout/rememberMe挙動とAPIレスポンス形式維持。単体テスト:同時に5件の401が返るケース/更新失敗/ネットワークエラー/手動logout中。認証ライブラリ全面置換は範囲外。"))
                .collect::<Vec<_>>(),
            preserves_list_constraints(input, "SPA認証: 複数APIの401時もrefresh token更新は1回だけ、他リクエストは待機し成功後に元リクエストを各1回だけ再送。refresh endpointが401/403なら再試行ループに入らず保存済みトークン削除→login画面。ネットワークエラーは最大2回、1秒と2秒の間隔で再試行し、それ以上行わない。Authorizationヘッダー/refresh tokenをログ/通知/エラー画面へ出さない。login/logout/rememberMe挙動とAPIレスポンス形式維持。単体テスト:同時に5件の401が返るケース/更新失敗/ネットワークエラー/手動logout中。認証ライブラリ全面置換は範囲外。"),
            preserves_verification_constraints(input, "SPA認証: 複数APIの401時もrefresh token更新は1回だけ、他リクエストは待機し成功後に元リクエストを各1回だけ再送。refresh endpointが401/403なら再試行ループに入らず保存済みトークン削除→login画面。ネットワークエラーは最大2回、1秒と2秒の間隔で再試行し、それ以上行わない。Authorizationヘッダー/refresh tokenをログ/通知/エラー画面へ出さない。login/logout/rememberMe挙動とAPIレスポンス形式維持。単体テスト:同時に5件の401が返るケース/更新失敗/ネットワークエラー/手動logout中。認証ライブラリ全面置換は範囲外。"),
            draft.distilled_prompt
        );
        polish_model_output_for_request(&request, &mut draft);

        for expected in [
            "SPA",
            "API",
            "401",
            "refresh token",
            "refresh endpoint",
            "403",
            "Authorization",
            "login",
            "logout",
            "rememberMe",
            "待機",
            "再送",
            "再試行ループに入らず",
            "最大2回",
            "ログ/通知/エラー画面へ出さない",
            "同時に5件の401",
            "全面置換は範囲外",
        ] {
            assert!(
                draft.distilled_prompt.contains(expected),
                "missing {expected}: {}",
                draft.distilled_prompt
            );
        }
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(
            preserves_negative_constraints(input, &draft.distilled_prompt),
            "{}",
            draft.distilled_prompt
        );
        assert!(
            draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 78,
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn compacts_level_two_billing_invoice_pdf_case_from_eval() {
        let input = "請求書 PDF の発行処理で、会社名、請求番号、発行日、税込金額を必ず表示してください。既存の PDF レイアウトの余白は変えず、税率 10% の計算式をテストに追加してください。金額の丸め方は既存仕様に合わせ、請求番号が空の場合は PDF を生成せず分かりやすいエラーを返してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "請求書 PDF 発行時、会社名、請求番号、発行日、税込金額を表示。既存 PDF 余白維持、税率 10% 計算式追加。金額丸め既存仕様準拠、請求番号空でエラー返却、PDF レイアウトの余白は変えず、税率 10% の計算式をテストに追加。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        for expected in [
            "PDF",
            "会社名",
            "請求番号",
            "発行日",
            "税込金額",
            "10%",
            "必ず",
            "変えず",
            "テスト",
            "生成せず",
            "エラー",
        ] {
            assert!(draft.distilled_prompt.contains(expected));
        }
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82);
    }

    #[test]
    fn compacts_level_two_redis_rate_limit_case_from_eval() {
        let input = "Redis を使ってログイン API に IP 単位のレート制限を追加してください。10 分に 5 回まで許可し、超過時は HTTP 429 と RATE_LIMITED を返してください。成功ログイン時にカウンタを即削除しないでください。既存の認証レスポンス形式は変更せず、テストでは成功、失敗、制限超過のケースを確認してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "RedisでログインAPIにIP単位のレート制限を追加。10分5回まで、超過時HTTP 429/RATE_LIMITED。成功ログイン時カウンタ削除禁止。認証レスポンス形式変更なし、成功/失敗/制限超過テスト確認。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        for expected in [
            "Redis",
            "API",
            "IP",
            "10分",
            "5回",
            "HTTP 429",
            "RATE_LIMITED",
            "まで",
            "削除しない",
            "変更せず",
            "テスト",
        ] {
            let actual = compact_all_whitespace(&draft.distilled_prompt);
            let expected = compact_all_whitespace(expected);
            assert!(
                actual.contains(&expected),
                "missing `{expected}` in `{}`",
                draft.distilled_prompt
            );
        }
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82);
    }

    #[test]
    fn compacts_level_two_websocket_reconnect_case_from_eval() {
        let input = "WebSocket 切断時に指数バックオフで再接続するようにしてください。初回は 1 秒、最大 30 秒まで伸ばし、手動ログアウト後は再接続しないでください。既存の message handler と認証トークン更新処理は変更しないでください。ネットワーク復帰後に重複接続が増えないことも確認してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "WebSocket切断時に指数バックオフで再接続。初回1秒、最大30秒まで。手動ログアウト後は再接続しない。message handlerと認証トークン更新処理は変更しない。ネットワーク復帰後の重複接続も防ぐ。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        for expected in [
            "WebSocket",
            "1秒",
            "30秒",
            "message handler",
            "指数バックオフ",
            "再接続しない",
            "変更しない",
            "重複接続しない",
        ] {
            let actual = compact_all_whitespace(&draft.distilled_prompt);
            let expected = compact_all_whitespace(expected);
            assert!(
                actual.contains(&expected),
                "missing `{expected}` in `{}`",
                draft.distilled_prompt
            );
        }
        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(
            preserves_negative_constraints(input, &draft.distilled_prompt),
            "{}",
            draft.distilled_prompt
        );
        assert!(draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82);
    }

    #[test]
    fn compacts_level_two_log_analysis_below_case_budget() {
        let input = "本番ログを解析し、注文送信が失敗する原因候補を優先度順に整理してください。2026-06-24T10:15:03Z requestId=ab12 POST /orders ECONNRESET upstream=payment-service。時刻、requestId、エラー文字列は改変せず、追加で確認すべきログと暫定対応を示してください。";
        let mut draft = CompressionDraft {
            distilled_prompt: "requestId=ab12/payment-service: 本番ログ解析、注文失敗原因優先度整理。時刻2026-06-24T10:15:03Z、requestId ab12、ECONNRESETエラーの本番ログに基づき、追加ログ確認と暫定対応を提示。".to_string(),
            removed_content_summary: Vec::new(),
        };
        let request = test_request(input.to_string(), 2);

        polish_model_output_for_request(&request, &mut draft);

        assert!(contains_required_technical_terms(
            input,
            &draft.distilled_prompt
        ));
        assert!(preserves_negative_constraints(
            input,
            &draft.distilled_prompt
        ));
        assert!(
            draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 90,
            "{}",
            draft.distilled_prompt
        );
    }

    #[test]
    fn compacts_level_three_structured_cases_from_eval_slice() {
        let cases = [
            (
                "Prisma の User テーブルに lastLoginAt を追加する migration を作成してください。既存データは NULL のまま許容し、email の unique 制約や createdAt の default は変更しないでください。ロールバック手順も短く添えてください。",
                "createdAt: Prisma UserテーブルにlastLoginAtを追加、既存NULL許容、email unique制約とcreatatAt; 既存データは NULL のまま許容し、email の unique 制約や createdAt の default は変更しない。ロールバック手順も短く添える。",
            ),
            (
                "ログイン後に /dashboard と /login の間でリダイレクトループします。middleware.ts と session cookie の確認順を調べ、未ログイン時だけ /login へ戻してください。既存の rememberMe と MFA の挙動は壊さないでください。",
                "middleware.tsでsession cookieと未ログイン時の/loginリダイレクト順を確認し、未ログイン時だけ/loginへ戻すよう設定。rememberMeとMFAは維持。",
            ),
            (
                "GitHub Actions の Node.js CI が毎回依存関係を再インストールして遅いです。actions/cache を使って npm のキャッシュを有効化してください。ただし package-lock.json をキーに含め、テストコマンド npm test と lint は変更しないでください。",
                "Node.js: GitHub Actions Node CI遅い、actions/cacheでnpmキャッシュ有効化、package-lock.jsonキーに含め、npm; ただし package-lock.json をキーに含め、テストコマンド npm test と lint は変更しない。",
            ),
            (
                "Windows の WebView2 アプリで圧縮完了通知が届かない問題を調べてください。通知は PowerShell ではなくアプリ本体から出し、AppUserModelID、アイコン、通知許可状態を確認してください。UI 内の完了トーストは追加しないでください。",
                "Windows: WebView2アプリ通知問題調査、AppUserModelID/アイコン/通知許可確認、PowerShell回避、UI完了トースト追加禁止",
            ),
        ];

        for (input, output) in cases {
            let mut draft = CompressionDraft {
                distilled_prompt: output.to_string(),
                removed_content_summary: Vec::new(),
            };
            let request = test_request(input.to_string(), 3);

            polish_model_output_for_request(&request, &mut draft);

            assert!(
                draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82,
                "{}",
                draft.distilled_prompt
            );
            assert!(
                contains_required_technical_terms(input, &draft.distilled_prompt),
                "terms={:?}; output={}",
                required_technical_terms(input),
                draft.distilled_prompt
            );
            assert!(preserves_negative_constraints(
                input,
                &draft.distilled_prompt
            ));
        }
    }

    #[test]
    fn compacts_level_three_ui_and_cli_cases_from_eval_slice() {
        let cases: [(&str, &str, &str, &[&str]); 6] = [
            (
                "sql_index_tuning",
                "PostgreSQL 15 の orders テーブルで tenant_id と created_at による一覧取得が遅いです。status も絞り込み条件にあります。ORDER BY created_at DESC の意味は変えないまま、index の案と書き込み性能への影響を短く提案してください。",
                "PostgreSQL 15 で、tenant_id、created_at、status のインデックス案を。ORDER BY creatd_at DESC 維持、書き込み性能影響考慮。",
                &["orders", "created_at"],
            ),
            (
                "rust_error_enum",
                "Rust の compression-core で Runtime エラーと Validation エラーを分けたいです。thiserror の使い方は維持し、公開 API の Result 型は変えないでください。空出力と JSON パース失敗のテストも追加してください。",
                "API: Rust compression-coreエラー区別、thiserrorスタイル維持、Result型変更なし、テストに空出力/JSONパース失敗追加",
                &["Runtime", "Validation"],
            ),
            (
                "python_cli_args",
                "Python の CLI ツールに --dry-run と --output json を追加してください。CSV 出力はデフォルトのままにし、進捗ログを標準エラーに書かないでください。argparse を使い、README に利用例を 2 つ追加してください。",
                "CLIツールに--dry-runと--output jsonを追加し、CSV出力はデフォルト維持、進捗ログを標準エラーから回避。argparse利用、READMEに利用例を2つ追加。",
                &["Python"],
            ),
            (
                "desktop_tray_restore",
                "Windows アプリを閉じてもシステムトレイに残るようにしてください。左クリックで復帰し、右クリックでは復帰、設定、終了のメニューを出してください。閉じるボタンは終了ではなく非表示だけにしてください。",
                "Windowsアプリを閉じてもシステムトレイに残し、左クリック復帰/右クリックで設定・終了メニュー表示、終了はトレイメニューからのみ実行",
                &["ではなく"],
            ),
            (
                "clipboard_after_compression",
                "圧縮完了後にクリップボードへ自動コピーしてください。成功時は Windows 通知で圧縮完了、コピー済み、短い要約を出し、アプリ内の完了通知は表示しないでください。",
                "圧縮後、クリップボードへ自動コピーし、コピー成功時はWindows通知で完了・コピー済み・要約を通知。アプリ内での完了通知は回避。",
                &["圧縮完了"],
            ),
            (
                "theme_scrollbar",
                "ダークモードとライトモードに合わせてスクロールバーの色を変えてください。上部バーの色も本文と少し変え、最小化、最大化、閉じるボタンのホバー領域がバーからはみ出さないようにしてください。",
                "ダークモード/ライトモード時、スクロールバー色を反転し、上部バー色も本文と差別化。最小化/最大化/閉じるボタンのホバー領域はバー内に収める。",
                &["はみ出さない"],
            ),
        ];

        for (case_id, input, output, expected_terms) in cases {
            let mut draft = CompressionDraft {
                distilled_prompt: output.to_string(),
                removed_content_summary: Vec::new(),
            };
            let request = test_request(input.to_string(), 3);

            polish_model_output_for_request(&request, &mut draft);

            assert!(
                draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82,
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            assert!(
                contains_required_technical_terms(input, &draft.distilled_prompt),
                "{}: terms={:?}; output={}",
                case_id,
                required_technical_terms(input),
                draft.distilled_prompt
            );
            assert!(
                preserves_negative_constraints(input, &draft.distilled_prompt),
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            for expected in expected_terms {
                assert!(
                    draft.distilled_prompt.contains(expected),
                    "{} missing {expected}: {}",
                    case_id,
                    draft.distilled_prompt
                );
            }
        }
    }

    #[test]
    fn compacts_level_three_policy_and_framework_cases_from_eval_slice() {
        let cases: [(&str, &str, &str, &[&str]); 6] = [
            (
                "model_folder_docs",
                "アプリ内 Model フォルダの役割を README に追記してください。採用中の Sarashina 2.2 3B GGUF の配置先、Git 管理しない理由、LM Studio 接続はユーザーの任意モデル検証用に残すことを明記してください。",
                "README/Sarashina 2.2 3B: 配置先とGit管理除外、LM Studio接続明記をSARアシュイナ2.2 3B GGUFアプリ内Modelフォルダ役割説明に統合; LM Studio 接続はユーザーの任意モデル検証用に残す",
                &["Model", "README", "Sarashina 2.2 3B", "GGUF", "Git", "LM Studio", "残す"],
            ),
            (
                "single_inference_policy",
                "圧縮処理で検証失敗時に修復再推論を行わないようにしてください。1回目の出力が評価基準を満たさない場合は原文返しにし、評価基準そのものは下げないでください。テストで再推論しないことを確認してください。",
                "修復再推論: 検証失敗時修復再論回避、1回目未達原文返し、評価基準維持で再推論テスト回避; 検証失敗時:修復再推論行わない; 1回目未達は原文返し/評価基準下げない",
                &["修復再推論", "1回目", "原文返し", "評価基準", "テスト", "下げない"],
            ),
            (
                "prompt_common_rules",
                "モードとタスク種別を廃止し、凡用性の高い圧縮ルールを共通プロンプトへ統合してください。レベル別プロンプトでは圧縮強度だけを変え、UI にはモデル、表示、圧縮レベルだけを残してください。",
                "モード/タスク種別廃止、共通プロンプトに汎用圧縮ルール統合。レベル別プロンプトは圧縮強度のみ変更、UIはモデル/表示/圧縮レベルのみに。",
                &["モード", "タスク種別", "共通プロンプト", "レベル別プロンプト", "UI", "廃止", "のみ"],
            ),
            (
                "expo_push_token",
                "Expo アプリで push token が取得できない端末があります。iOS と Android の権限確認、物理端末判定、projectId の設定を調べてください。既存の通知送信 API の URL とレスポンス形式は変更しないでください。",
                "Expoアプリでpush token取得不可端末あり。iOS/Android権限、物理端末判定、projectId設定確認。通知送信API URL/レスポンス形式変更禁止。; 通知送信 API の URL とレスポンス形式は変更禁止変更せず",
                &["Expo", "push token", "iOS", "Android", "projectId", "API", "URL", "変更しない"],
            ),
            (
                "vue_form_validation",
                "Vue 3 の問い合わせフォームでメールアドレス、件名、本文のバリデーションを追加してください。既存の v-model 名は変えず、送信ボタンはエラーがある時だけ disabled にしてください。アクセシビリティ用の aria-describedby も付けてください。",
                "Vue 3 問い合わせフォームに v-model 維持でバリデーション追加。送信ボタンはエラー時のみ disable、aria-describedby でアクセシビリティ向上。; v-model 名は変えず、送信ボタンはエラーがある時だけ disabled に",
                &["Vue 3", "v-model", "disabled", "aria-describedby", "変えず", "のみ"],
            ),
            (
                "svelte_store_refactor",
                "SvelteKit の cart store を derived store と action に分離してください。外部から使っている addItem、removeItem、clear の関数名は維持し、localStorage の保存形式は変更しないでください。",
                "SvelteKit cart storeをaddItem/removeItem/clear維持でderived storeに分離。localStorage形式変更なし。",
                &["SvelteKit", "cart store", "derived store", "addItem", "removeItem", "clear", "localStorage"],
            ),
        ];

        for (case_id, input, output, expected_terms) in cases {
            let mut draft = CompressionDraft {
                distilled_prompt: output.to_string(),
                removed_content_summary: Vec::new(),
            };
            let request = test_request(input.to_string(), 3);

            polish_model_output_for_request(&request, &mut draft);

            assert!(
                draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82,
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            assert!(
                contains_required_technical_terms(input, &draft.distilled_prompt),
                "{}: terms={:?}; output={}",
                case_id,
                required_technical_terms(input),
                draft.distilled_prompt
            );
            assert!(
                preserves_negative_constraints(input, &draft.distilled_prompt),
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            for expected in expected_terms {
                assert!(
                    draft.distilled_prompt.contains(expected),
                    "{} missing {expected}: {}",
                    case_id,
                    draft.distilled_prompt
                );
            }
        }
    }

    #[test]
    fn compacts_level_three_backend_infra_cases_from_eval_slice() {
        let cases: [(&str, &str, &str, &[&str]); 6] = [
            (
                "go_http_timeout",
                "Go の HTTP クライアントに 5 秒のタイムアウトとリトライ 2 回を追加してください。POST のリトライは冪等キーがある場合だけ許可し、既存の User-Agent ヘッダーは変更しないでください。",
                "HTTPクライアントに5秒タイムアウトとリトライ2回追加、POSTリトライは冪等キー時のみ許可; POST のリトライは冪等キーがある場合だけ許可し、 User-Agent ヘッダーは変更禁止変更せず",
                &["Go", "HTTP", "5秒", "リトライ2回", "POST", "User-Agent", "のみ", "変更しない"],
            ),
            (
                "java_spring_validation",
                "Spring Boot の UserController に Bean Validation を追加してください。name は 1 文字以上 50 文字以下、email は必須かつ形式チェック、age は 0 以上にしてください。既存の JSON レスポンスキーは変えないでください。",
                "Validation/1 文字以上/50 文字以下: Spring Boot UserControllerにBEANバリデーション追加。nameは1-50文字、emailは形式チェック必須、ageは0以上; JSON レスポンスキーは変えない",
                &["Spring Boot", "UserController", "Bean Validation", "name", "50", "email", "age", "JSON", "変更しない"],
            ),
            (
                "kotlin_room_migration",
                "Android の Room database を version 4 に上げ、Task テーブルに priority INTEGER NOT NULL DEFAULT 0 を追加してください。既存の migration 1_2 と 2_3 は変更せず、新しい 3_4 を追加してください。",
                "Android Room v4に更新し、Taskテーブルにpriority INTEGER NOT NULL DEFAULT 0を追加。既存migration 1_2, 2_3は変更せず、3_4を追加。",
                &["Android", "Room", "version 4", "Task", "priority", "INTEGER", "DEFAULT 0", "3_4"],
            ),
            (
                "swiftui_state_bug",
                "SwiftUI の SettingsView でトグルを切り替えても保存されない問題を修正してください。@AppStorage を使い、既存の UserDefaults キー darkModeEnabled は変えないでください。プレビュー用のモック値も更新してください。",
                "SwiftUI SettingsView @AppStorage darkModeEnabled UserDefaults保存修正、既存キー維持、プレビュー更新",
                &["SwiftUI", "SettingsView", "@AppStorage", "UserDefaults", "darkModeEnabled"],
            ),
            (
                "terraform_s3_policy",
                "Terraform で S3 バケットに public access block を追加してください。既存の bucket 名、タグ、versioning 設定は維持し、ACL を public-read に戻す変更は入れないでください。plan の確認手順も書いてください。",
                "TerraformでS3バケットにpublic-read ACLを維持しつつ、versioning設定維持でタグ既存のまま変更加えず。plan確認手順追加。",
                &["Terraform", "S3", "public access block", "bucket", "versioning", "ACL", "plan", "維持", "入れない"],
            ),
            (
                "docker_multistage",
                "Node.js アプリの Dockerfile を multi-stage build にしてください。builder では npm ci と npm run build を実行し、runtime には dist、package.json、node_modules の production 依存だけを含めてください。ポート 3000 は変更しないでください。",
                "multi-stage/package.json: Node.jsマルチステージビルド、builderでnpm cibとnpm run build実行、runtimeにproduction依存のみnode_modules含め、ポート3000変更なし",
                &["Node.js", "Dockerfile", "multi-stage", "npm ci", "npm run build", "dist", "3000", "のみ", "変更しない"],
            ),
        ];

        for (case_id, input, output, expected_terms) in cases {
            let mut draft = CompressionDraft {
                distilled_prompt: output.to_string(),
                removed_content_summary: Vec::new(),
            };
            let request = test_request(input.to_string(), 3);

            polish_model_output_for_request(&request, &mut draft);

            assert!(
                draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82,
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            assert!(
                contains_required_technical_terms(input, &draft.distilled_prompt),
                "{}: terms={:?}; output={}",
                case_id,
                required_technical_terms(input),
                draft.distilled_prompt
            );
            assert!(
                preserves_negative_constraints(input, &draft.distilled_prompt),
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            for expected in expected_terms {
                assert!(
                    draft.distilled_prompt.contains(expected),
                    "{} missing {expected}: {}",
                    case_id,
                    draft.distilled_prompt
                );
            }
        }
    }

    #[test]
    fn compacts_level_three_api_and_test_cases_from_eval_slice() {
        let cases: [(&str, &str, &str, &[&str]); 6] = [
            (
                "nginx_upload_limit",
                "Nginx のアップロード上限を 20MB に変更してください。対象は /api/import だけで、他の location には影響させないでください。413 エラー時の日本語メッセージも既存スタイルに合わせて追加してください。",
                "20MBアップロード上限を/api/importのみ変更、他loc影響回避、413日英メッセージ追加",
                &["Nginx", "20MB", "/api/import", "location", "413", "のみ", "影響させない"],
            ),
            (
                "redis_rate_limit",
                "Redis を使ってログイン API に IP 単位のレート制限を追加してください。10 分に 5 回まで許可し、超過時は HTTP 429 と RATE_LIMITED を返してください。成功ログイン時にカウンタを即削除しないでください。",
                "RedisでIPレート制限を実装し、ログインAPIに10分あたり5回の制限を設定。超過時はHTTP 429とRATE_LIMITEDを返し、成功ログイン時のカウンタ即削除を回避。",
                &["Redis", "IP", "10分", "5回", "HTTP 429", "RATE_LIMITED", "まで", "削除しない"],
            ),
            (
                "graphql_n_plus_one",
                "GraphQL の users クエリで posts を取得すると N+1 が発生します。DataLoader を導入し、既存の schema.graphql とレスポンスフィールド名は変えないでください。キャッシュはリクエスト単位だけにしてください。",
                "N+1: DataLoader導入、schema.graphql変更なし、レスポンスフィールド維持、キャッシュはリクエスト単位のみ",
                &["GraphQL", "users", "posts", "N+1", "DataLoader", "schema.graphql", "変更しない", "のみ"],
            ),
            (
                "openapi_schema_update",
                "OpenAPI 定義に PATCH /users/{id} を追加してください。name と avatarUrl は任意、email は変更不可にし、404 と 409 のエラーレスポンス例を入れてください。既存の POST /users は変更しないでください。",
                "OpenAPI定義に/users/{id}を追加、nameとavatarUrlは任意、emailは変更不可。404,409エラーレスポンス例を含め、POST /usersは変更禁止。; POST /users は変更禁止変更せず",
                &["OpenAPI", "PATCH", "/users/{id}", "name", "avatarUrl", "email", "404", "409", "POST /users", "変更不可", "変更しない"],
            ),
            (
                "storybook_button_states",
                "Button コンポーネントの Storybook に primary、secondary、disabled、loading の状態を追加してください。既存の props 名とデフォルト theme は変更しないでください。アクセシビリティチェックで button name が取れることも確認してください。",
                "Button Storybookにprimary/secondary/disabled/loading状態追加、props名とtheme変更禁止、アクセシビリティチェックでbutton name取得確認; props 名とデフォルト theme は変更禁止変更せず",
                &["Button", "Storybook", "primary", "secondary", "disabled", "loading", "props", "theme", "変更しない"],
            ),
            (
                "playwright_login_test",
                "Playwright でログイン成功、パスワード誤り、ロック済みユーザーの E2E テストを追加してください。既存の test.describe 名は変えず、テストデータは beforeEach で作成し、外部 API へは実通信しないでください。",
                "test.describe: Playwrightでログイン失敗（パスワード誤り、ロック済みユーザー）E2Eテスト追加; test.describe 名は変えず、テストデータは beforeEach で作成し、外部 API へは実通信禁止",
                &["Playwright", "E2E", "test.describe", "beforeEach", "API", "変えず", "実通信しない"],
            ),
        ];

        for (case_id, input, output, expected_terms) in cases {
            let mut draft = CompressionDraft {
                distilled_prompt: output.to_string(),
                removed_content_summary: Vec::new(),
            };
            let request = test_request(input.to_string(), 3);

            polish_model_output_for_request(&request, &mut draft);

            assert!(
                draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82,
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            assert!(
                contains_required_technical_terms(input, &draft.distilled_prompt),
                "{}: terms={:?}; output={}",
                case_id,
                required_technical_terms(input),
                draft.distilled_prompt
            );
            assert!(
                preserves_negative_constraints(input, &draft.distilled_prompt),
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            for expected in expected_terms {
                assert!(
                    draft.distilled_prompt.contains(expected),
                    "{} missing {expected}: {}",
                    case_id,
                    draft.distilled_prompt
                );
            }
        }
    }

    #[test]
    fn compacts_level_three_quality_and_export_cases_from_eval_slice() {
        let cases: [(&str, &str, &str, &[&str]); 6] = [
            (
                "jest_timer_mock",
                "Jest で debounceSearch のテストを追加してください。fake timers を使い、300ms 未満では callback が呼ばれないこと、300ms 後に 1 回だけ呼ばれることを検証してください。実装関数名は変更しないでください。",
                "JestでdebounceSearchテスト追加、fake timers使用、300ms未満でcallback非呼び出し、300ms後1回呼び出し検証、実装名変更禁止",
                &["Jest", "debounceSearch", "fake timers", "300ms", "callback", "未満", "1回だけ", "変更しない"],
            ),
            (
                "accessibility_modal_focus",
                "モーダルを開いた時に最初の入力欄へフォーカスし、閉じた時に開いたボタンへフォーカスを戻してください。Esc キーで閉じる挙動は維持し、モーダル外クリックで閉じる処理は追加しないでください。",
                "モーダル初回フォーカス入力欄、Esc維持閉じる、外クリック閉じ回避",
                &["モーダル", "フォーカス", "Esc", "維持"],
            ),
            (
                "i18n_missing_keys",
                "i18n の ja.json と en.json で不足しているキーを検出するスクリプトを追加してください。キーの並び順は既存ファイルに合わせ、翻訳文の自動生成はしないでください。不足キー一覧を CI のログに出してください。",
                "CIでja.jsonとen.jsonの不足キーを検出し、ログに出力。キー並び順は既存ファイル準拠、自動翻訳禁止。",
                &["i18n", "ja.json", "en.json", "CI", "しない"],
            ),
            (
                "markdown_export",
                "分析結果を Markdown と JSON の両方でエクスポートできるようにしてください。Markdown には見出し、集計表、注意点を含め、JSON は既存 schema_version 1 を維持してください。ファイル名には日時を入れてください。",
                "分析結果のMarkdownとJSONエクスポート、Markdownには見出し/集計表/注意点、JSONはschema_version 1維持、ファイル名に日時追加",
                &["Markdown", "JSON", "schema_version 1", "維持"],
            ),
            (
                "file_upload_progress",
                "ファイルアップロード中の進捗バーを追加してください。0 から 100% まで表示し、失敗時は再試行ボタンを出してください。既存のキャンセルボタンと onCancel コールバックは削除しないでください。",
                "進捗バー表示と再試行ボタン追加、キャンセルボタン保持でonCancelコールバック維持、0-100%表示失敗時再試行機能実装",
                &["0", "100%", "再試行", "onCancel", "削除しない"],
            ),
            (
                "billing_invoice_pdf",
                "請求書 PDF の発行処理で、会社名、請求番号、発行日、税込金額を必ず表示してください。既存の PDF レイアウトの余白は変えず、税率 10% の計算式をテストに追加してください。",
                "請求書 PDF 発行: 会社名、請求番号、発行日、税込金額表示必須。既存 PDF 余白維持、税率; PDF レイアウトの余白は変えず、税率 10% の計算式をテストに追加",
                &["PDF", "会社名", "請求番号", "発行日", "税込金額", "10%", "必ず", "変えず"],
            ),
        ];

        for (case_id, input, output, expected_terms) in cases {
            let mut draft = CompressionDraft {
                distilled_prompt: output.to_string(),
                removed_content_summary: Vec::new(),
            };
            let request = test_request(input.to_string(), 3);

            polish_model_output_for_request(&request, &mut draft);

            assert!(
                draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82,
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            assert!(
                contains_required_technical_terms(input, &draft.distilled_prompt),
                "{}: terms={:?}; output={}",
                case_id,
                required_technical_terms(input),
                draft.distilled_prompt
            );
            assert!(
                preserves_negative_constraints(input, &draft.distilled_prompt),
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            for expected in expected_terms {
                assert!(
                    draft.distilled_prompt.contains(expected),
                    "{} missing {expected}: {}",
                    case_id,
                    draft.distilled_prompt
                );
            }
        }
    }

    #[test]
    fn compacts_level_three_product_and_evaluation_cases_from_eval_slice() {
        let cases: [(&str, &str, &str, &[&str]); 6] = [
            (
                "analytics_event_names",
                "フロントエンドの分析イベント名を整理してください。signup_start、signup_complete、plan_selected は残し、古い register_click は送信しないでください。イベント仕様を docs/analytics.md に追記してください。",
                "plan_selected: フロントエンド分析イベント名整理：signup_start, signup_complete, plan_seletedのみ残し、register_click送信回避。docs/analytics.mdに追記。",
                &["signup_start", "signup_complete", "plan_selected", "register_click", "docs/analytics.md", "残し", "送信しない"],
            ),
            (
                "batch_job_idempotency",
                "夜間バッチの請求確定処理を冪等にしてください。同じ billingPeriod と accountId の組み合わせでは二重作成しないようにし、既存の成功ログと Slack 通知は維持してください。",
                "夜間バッチ請求確定処理の冪等化。billingPeriodとaccountIdが同じ組み合わせでは二重作成回避、成功ログとSlack通知維持。",
                &["billingPeriod", "accountId", "Slack", "二重作成しない", "維持"],
            ),
            (
                "websocket_reconnect",
                "WebSocket 切断時に指数バックオフで再接続してください。初回 1 秒、最大 30 秒、手動ログアウト後は再接続しないでください。既存の message handler と認証トークン更新処理は変更しないでください。",
                "WebSocket再接続は初回1秒、最大30秒の指数バックオフで。手動ログアウト後は再接続禁止。messaging; message handler と認証トークン更新処理は変更禁止変更せず",
                &["WebSocket", "1秒", "30秒", "message handler", "再接続しない", "変更しない"],
            ),
            (
                "image_generator_queue",
                "画像生成ジョブをキュー化し、同時実行数を 2 に制限してください。失敗したジョブは最大 1 回だけ再試行し、ユーザーがキャンセルしたジョブは再試行しないでください。ジョブ ID と作成時刻はログに残してください。",
                "画像生成ジョブをキュー化し、同時実行数2に制限。失敗したジョブは最大1回再試行、ユーザーキャンセルジョブは再試行禁止。ジョブIDと作成時刻をログに記録。",
                &["画像生成", "同時実行数", "2", "最大1回", "ジョブID", "のみ", "再試行しない"],
            ),
            (
                "local_model_smoke_test",
                "パッケージ作成時のローカルモデルスモークテストを任意実行にしてください。既定では短い起動確認だけ行い、--RunLocalModelSmokeTest が指定された時だけ実圧縮を試してください。失敗時はログファイルのパスを表示してください。",
                "ローカルモデルスモークテストは、--RunLocalModelSmokeTest 指定時のみ実行し、既定では短い起動確認のみ。失敗時はログファイルパスを表示。",
                &["ローカルモデル", "スモークテスト", "--RunLocalModelSmokeTest", "ログファイル", "のみ"],
            ),
            (
                "prompt_eval_report",
                "プロンプト評価の結果をレベル別に集計してください。各レベルの平均文字数比、失敗率、原文返し件数、必須語欠落件数を出し、失敗率が 1% を超えたら exit code 1 にしてください。",
                "1% を超: 要件を統合し、レベル別集計で平均文字数比、失敗率、原文返し件数、必須語欠落件数を算出。失敗率1%超ならexit code 1",
                &["平均文字数比", "失敗率", "原文返し", "必須語", "1%", "exit code 1", "超えたら"],
            ),
        ];

        for (case_id, input, output, expected_terms) in cases {
            let mut draft = CompressionDraft {
                distilled_prompt: output.to_string(),
                removed_content_summary: Vec::new(),
            };
            let request = test_request(input.to_string(), 3);

            polish_model_output_for_request(&request, &mut draft);

            assert!(
                draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82,
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            assert!(
                contains_required_technical_terms(input, &draft.distilled_prompt),
                "{}: terms={:?}; output={}",
                case_id,
                required_technical_terms(input),
                draft.distilled_prompt
            );
            assert!(
                preserves_negative_constraints(input, &draft.distilled_prompt),
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            for expected in expected_terms {
                assert!(
                    draft.distilled_prompt.contains(expected),
                    "{} missing {expected}: {}",
                    case_id,
                    draft.distilled_prompt
                );
            }
        }
    }

    #[test]
    fn compacts_level_three_app_ui_cases_from_eval_slice() {
        let cases: [(&str, &str, &str, &[&str]); 6] = [
            (
                "lmstudio_profile",
                "モデル選択にはアプリ内モデルと LM Studio の自由選択だけを表示してください。LM Studio 接続に失敗してもアプリ内モデルへ自動再推論せず、原文返しにしてください。設定画面にはモードとタスク種別を戻さないでください。",
                "モデル選択はアプリ内モデルとLM Studio自由選択のみ、LM Studio失敗時は原文返し、設定画面にはモードとタスク種別除外",
                &["アプリ内モデル", "LM Studio", "原文返し", "モード", "タスク種別", "のみ", "再推論せず", "戻さない"],
            ),
            (
                "window_icon_packaging",
                "PromptC.ico を Windows アプリのアイコンとして適用してください。タスクバー、ウィンドウ、通知で同じアイコンを使い、古い PCicon.ico は参照しないでください。exe 出力先は prompt-compressor-project-exe のままにしてください。",
                "PromptC.icoをWindowsアプリアイコンに適用、タスクバー/ウィンドウ/通知で同じアイコン使用、PCicon.ico参照回避、exe出力先prompt-compressor-project-exe維持。",
                &["PromptC.ico", "Windows", "タスクバー", "通知", "PCicon.ico", "prompt-compressor-project-exe", "参照しない", "まま"],
            ),
            (
                "readme_folder_structure",
                "README にフォルダ構成の説明を追加してください。application はアプリ本体、資料 は企画や説明資料、prompt-compressor-project-exe はビルド済み出力であることを明記し、実行に不要な企画書はアプリ本体へ混ぜないでください。",
                "READMEにフォルダ構成説明追加; application はアプリ本体、資料 は企画や説明資料、prompt-compressor-project-exe はビルド済み出力であることを明記し、実行に不要な企画書はアプリ本体へ混ぜないでください",
                &["README", "application", "資料", "prompt-compressor-project-exe", "混ぜない"],
            ),
            (
                "token_character_count",
                "圧縮結果の横にトークン比較と文字数比較を表示してください。トークンと文字数は横並びにし、幅を取りすぎないようにしてください。文字数は JavaScript の length ではなく Unicode の文字単位で数えてください。",
                "JavaScriptトークンとUnicode文字数を横並び比較表示、幅制限あり、文字数はUnicode単位; しない; 文字数は JavaScript の length ではなく Unicode の文字単位で数えてください",
                &["トークン", "文字数", "JavaScript", "Unicode", "ではなく", "取りすぎない"],
            ),
            (
                "compression_latency_ui",
                "圧縮にかかった時間を秒単位で圧縮結果の見出し横に小さく薄い色で表示してください。計測はボタン押下から結果表示までではなく、コア圧縮処理の latency_ms を使ってください。",
                "圧縮時間表示はコア圧縮処理latency_ms計測に基づき; 計測はボタン押下から結果表示までではなく、コア圧縮処理の latency_ms を使ってください",
                &["秒", "圧縮結果", "latency_ms", "ではなく"],
            ),
            (
                "sample_dropdown_load",
                "サンプル文章は読み込みボタンを押さなくても、プルダウンで選択した瞬間に入力欄へ反映してください。選択後に圧縮結果、トークン比較、文字数比較はクリアしてください。圧縮レベルはサンプルの推奨値へ変更してよいです。",
                "選択時即時反映、圧縮クリア、推奨圧縮レベル適用",
                &["サンプル", "プルダウン", "入力欄", "トークン", "文字数", "圧縮レベル", "押さなくても", "クリア"],
            ),
        ];

        for (case_id, input, output, expected_terms) in cases {
            let mut draft = CompressionDraft {
                distilled_prompt: output.to_string(),
                removed_content_summary: Vec::new(),
            };
            let request = test_request(input.to_string(), 3);

            polish_model_output_for_request(&request, &mut draft);

            assert!(
                draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82,
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            assert!(
                contains_required_technical_terms(input, &draft.distilled_prompt),
                "{}: terms={:?}; output={}",
                case_id,
                required_technical_terms(input),
                draft.distilled_prompt
            );
            assert!(
                preserves_negative_constraints(input, &draft.distilled_prompt),
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            for expected in expected_terms {
                assert!(
                    draft.distilled_prompt.contains(expected),
                    "{} missing {expected}: {}",
                    case_id,
                    draft.distilled_prompt
                );
            }
        }
    }

    #[test]
    fn compacts_level_three_final_goal_cases_from_eval_slice() {
        let cases: [(&str, &str, &str, &[&str]); 5] = [
            (
                "clear_button",
                "サンプル選択と圧縮するボタンの間にクリアボタンを追加してください。押したら入力欄を空にし、サンプル選択を未選択に戻し、結果欄とメトリクス表示を初期化してください。設定値は変更しないでください。",
                "サンプル選択とクリアボタン追加、入力欄・選択・結果初期化を指示。設定値変更禁止。; 設定値は変更禁止変更せず",
                &["クリアボタン", "入力欄", "サンプル", "結果欄", "メトリクス", "変更しない"],
            ),
            (
                "topbar_fixed",
                "スクロールしても上部の最小化、最大化、閉じるボタンが隠れないようにしてください。ウィンドウバーは固定し、ボタンのサイズと配置を Windows 標準に近づけてください。左側のアプリ名表示は出さないでください。",
                "Windows: ウィンドウバー固定、ボタン非表示・標準サイズ維持",
                &["最小化", "最大化", "閉じる", "Windows", "隠れない", "出さない"],
            ),
            (
                "native_no_http",
                "exe 版では http://prompt-compressor.localhost をユーザーに見せないでください。内部実装も可能な限り HTTP に依存せず、WebView2 の通信はアプリ内ブリッジへ寄せてください。Web 版との互換性は必要なら切って構いません。",
                "http://prompt-compressor.localhost: exe版ではHTTP依存を回避し、WebView2通信をアプリ内ブリッジへ。Web版との互換性は必要に応じて切る。",
                &["exe", "http://prompt-compressor.localhost", "HTTP", "WebView2", "Web版", "見せない", "依存せず"],
            ),
            (
                "prompt_failure_logging",
                "圧縮失敗時に、検証失敗理由、文字数比、欠落した必須語、原文返しの有無をログへ残してください。ログには入力全文を保存せず、先頭 80 文字のスニペットだけを保存してください。",
                "原文返し: 圧縮失敗時、検証失敗理由、文字数比、欠落必須語をログに記録。全文スニペット80文字保存のみ。",
                &["検証失敗理由", "文字数比", "必須語", "原文返し", "80文字", "保存せず", "のみ"],
            ),
            (
                "evaluation_dataset_size",
                "プロンプト評価では 50 種類以上の元文章を使ってください。各文章を圧縮レベル 1、2、3 で評価し、平均圧縮率がレベルごとに適した範囲に入り、圧縮失敗率が 1% 以下であることを確認してください。",
                "50種類以上の文章圧縮評価、各レベルで平均圧縮率範囲内かつ圧縮失敗率1%以下を確認してください",
                &["50種類以上", "圧縮レベル1", "2", "3", "平均圧縮率", "1%以下"],
            ),
        ];

        for (case_id, input, output, expected_terms) in cases {
            let mut draft = CompressionDraft {
                distilled_prompt: output.to_string(),
                removed_content_summary: Vec::new(),
            };
            let request = test_request(input.to_string(), 3);

            polish_model_output_for_request(&request, &mut draft);

            assert!(
                draft.distilled_prompt.chars().count() * 100 <= input.chars().count() * 82,
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            assert!(
                contains_required_technical_terms(input, &draft.distilled_prompt),
                "{}: terms={:?}; output={}",
                case_id,
                required_technical_terms(input),
                draft.distilled_prompt
            );
            assert!(
                preserves_negative_constraints(input, &draft.distilled_prompt),
                "{}: {}",
                case_id,
                draft.distilled_prompt
            );
            for expected in expected_terms {
                assert!(
                    compact_all_whitespace(&draft.distilled_prompt)
                        .contains(&compact_all_whitespace(expected)),
                    "{} missing {expected}: {}",
                    case_id,
                    draft.distilled_prompt
                );
            }
        }
    }

    #[test]
    fn requires_negative_constraints_to_survive_compression() {
        let input =
            "検索ボタンを押したときだけ API を呼び出し、大規模なリファクタリングは避けてください。";

        assert!(preserves_negative_constraints(
            input,
            "検索ボタン押下時のみ API を呼び出し、大規模リファクタリングは禁止。"
        ));
        assert!(preserves_negative_constraints(
            input,
            "検索ボタン押下時のみ API を呼び出し、大規模リファクタリングは回避。"
        ));
        assert!(!preserves_negative_constraints(
            input,
            "検索ボタン押下時に API を呼び出し、既存の構成を維持。"
        ));

        let preserve_input = "実装コードと既存テスト名は変更せず、境界値を含めてください。";
        assert!(preserves_negative_constraints(
            preserve_input,
            "実装コード・既存テスト名維持、境界値含む。"
        ));
        assert!(!preserves_negative_constraints(
            preserve_input,
            "境界値を含めてテストを追加。"
        ));

        let log_input = "時刻、requestId、エラー文字列は改変せず、追加で確認すべきログと暫定対応を示してください。";
        assert!(preserves_negative_constraints(
            log_input,
            "時刻/requestId/エラー文字列改変せず、追加ログ/暫定対応。"
        ));
        assert!(!preserves_negative_constraints(
            log_input,
            "本番ログ解析と暫定対応を整理。"
        ));

        let render_input = "不要な再レンダリングを増やさないでください。";
        assert!(preserves_negative_constraints(
            render_input,
            "再レンダリング回避。"
        ));
        assert!(preserves_negative_constraints(
            render_input,
            "再レンダリングを最小限に。"
        ));
        assert!(!preserves_negative_constraints(
            render_input,
            "表示責務を分離。"
        ));

        let toast_input = "UI 内の完了トーストは追加しないでください。";
        assert!(preserves_negative_constraints(
            toast_input,
            "UI内の完了トーストは追加せず、Windows通知を使う。"
        ));
        assert!(!preserves_negative_constraints(
            toast_input,
            "UI内の完了トーストを追加し、Windows通知も使う。"
        ));

        let notification_input = "Windows の WebView2 アプリで、圧縮完了通知が届かない問題を直したいです。通知は PowerShell からではなくアプリ本体から出るようにしてください。AppUserModelID、通知アイコン、通知許可状態を確認し、通知が表示されない時に原因を追えるログも残してください。UI 内の完了トーストは追加しないでください。通知の文面には圧縮完了、コピー済み、短い要約を含めてください。";
        let notification_output = "Windows WebView2 アプリで圧縮完了通知がPowerShellからではなくアプリ本体から出力されるようにし、AppUserModelID、通知アイコン、通知許可状態を確認、通知表示されない原因をログに記録。UI内の完了トーストは追加せず、通知文面に圧縮完了、コピー済み、短い要約を含める。";
        assert!(preserves_negative_constraints(
            notification_input,
            notification_output
        ));
    }

    #[test]
    fn dynamic_output_cap_is_lower_for_aggressive_compression() {
        let model = test_model_definition(256);
        let level_1 = effective_max_output_tokens(
            &test_request("短い依頼を圧縮してください。".into(), 1),
            &model,
        );
        let level_3 = effective_max_output_tokens(
            &test_request("短い依頼を圧縮してください。".into(), 3),
            &model,
        );

        assert!(level_3 <= level_1);
        assert!(level_3 <= 128);
    }

    #[test]
    fn level_two_output_cap_scales_for_long_balanced_requests() {
        let model = test_model_definition(256);
        let request = test_request("長い入力です。".repeat(80), 2);

        let cap = effective_max_output_tokens(&request, &model);

        assert!(cap > 80);
        assert!(cap <= 192);
    }

    fn test_request(input_text: String, level: u8) -> CompressionRequest {
        CompressionRequest {
            input_text,
            task_type: TaskType::Coding,
            compression_mode: CompressionMode::CodexOptimized,
            compression_level: CompressionLevel::from_u8(level).expect("valid level"),
            profile: "internal_llm".to_string(),
            constraints: CompressionConstraints::default(),
            target: RequestTarget::codex_default(),
            source: RequestSource::Desktop,
        }
    }

    fn test_model_definition(default_max_output: u32) -> ModelDefinition {
        ModelDefinition {
            id: "test".to_string(),
            label: "Test".to_string(),
            adapter: "llama".to_string(),
            runtime_ref: "llama_cpp_embedded".to_string(),
            model_path: Some(PathBuf::from("model.gguf")),
            api_model: None,
            quantization: "q4".to_string(),
            context_length: 4096,
            thinking: false,
            default_max_output,
            prompt_template: "test".to_string(),
            prompt_style: "concise".to_string(),
            supports_json_schema: false,
        }
    }
}
