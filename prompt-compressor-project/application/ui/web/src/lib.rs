use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, TryLockError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[cfg(target_os = "windows")]
use std::os::windows::ffi::OsStrExt;
#[cfg(target_os = "windows")]
use std::ptr::null_mut;

use anyhow::{Context, Result};
use prompt_compressor_core::{
    CompressionConstraints, CompressionError, CompressionLevel, CompressionRequest,
    CompressionService, ConfiguredRuntimeBackend, ModelDownloadCancellation, ModelDownloadProgress,
    ProfileDefinition, ProfileRegistry, RequestSource, RequestTarget,
};
use serde::{Deserialize, Serialize};

#[cfg(target_os = "windows")]
use windows_sys::Win32::{
    Foundation::GetLastError,
    Storage::FileSystem::{
        GetDiskFreeSpaceExW, MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    },
    System::{
        DataExchange::{CloseClipboard, EmptyClipboard, OpenClipboard, SetClipboardData},
        Memory::{GlobalAlloc, GlobalLock, GlobalUnlock, GMEM_MOVEABLE},
        Ole::CF_UNICODETEXT,
    },
};

const INDEX_HTML: &str = include_str!("../static/index.html");
const STYLES_CSS: &str = include_str!("../static/styles.css");
const APP_JS: &str = include_str!("../static/app.js");
const SAMPLE_PROMPTS_JS: &str = include_str!("../static/sample-prompts.js");

const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_REQUEST_PATH_BYTES: usize = 2 * 1024;
const MAX_REQUEST_BODY_BYTES: usize = 256 * 1024;
const MAX_SETTINGS_BODY_BYTES: usize = 16 * 1024;
const MAX_PROMPT_CHARS: usize = 100_000;
const MAX_PROFILE_CHARS: usize = 128;
const MAX_CLIPBOARD_CHARS: usize = 100_000;
const HTTP_WORKER_COUNT: usize = 4;
const HTTP_QUEUE_CAPACITY: usize = 16;
const INSTALL_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;
const CPU_ENGINE_SELECTION_SCHEMA_VERSION: u32 = 4;
const CPU_ENGINE_MINIMUM_GAIN_PERCENT: u64 = 3;
const CPU_ENGINE_SELECTION_FILE: &str = "cpu-engine-selection-v1.json";
const CPU_ENGINE_MODE_ENV: &str = "TRIMPROMPT_CPU_ENGINE_MODE";
const INFERENCE_COMPATIBILITY_ID_ENV: &str = "TRIMPROMPT_INFERENCE_COMPATIBILITY_ID";

#[derive(Debug, Clone, Copy)]
pub struct RuntimeQualityProbeCase {
    pub id: &'static str,
    pub input_text: &'static str,
    required_marker_groups: &'static [&'static [&'static str]],
    semantic_requirements: &'static [RuntimeQualitySemanticRequirement],
}

#[derive(Debug, Clone, Copy)]
struct RuntimeQualitySemanticRequirement {
    label: &'static str,
    alternatives: &'static [&'static [&'static str]],
}

const RUNTIME_QUALITY_PROBE_CASES: [RuntimeQualityProbeCase; 5] = [
    RuntimeQualityProbeCase {
        id: "search-form",
        input_text: "ReactとTypeScriptで検索フォームを修正してください。検索ボタンを押した時だけAPIを呼び、入力が空なら通信せずエラーを表示します。既存のキーボード操作とテストは維持し、変更したファイルと確認結果も示してください。",
        required_marker_groups: &[
            &["React"], &["TypeScript"], &["検索ボタン"], &["API"], &["空"],
            &["通信"], &["エラー"], &["キーボード"], &["テスト"], &["変更"], &["確認"],
        ],
        semantic_requirements: &[],
    },
    RuntimeQualityProbeCase {
        id: "registration-api",
        input_text: "ユーザー登録APIに入力検証を追加してください。メールアドレスの形式とパスワード12文字以上を確認し、不正ならHTTP 400を返します。失敗時はデータベースへ書き込まず、正常系と異常系のテストを追加してください。",
        required_marker_groups: &[
            &["ユーザー登録"], &["API"], &["メールアドレス"], &["パスワード"],
            &["12"], &["HTTP400"], &["データベース"], &["書き込", "保存"],
            &["正常"], &["異常"], &["テスト"],
        ],
        semantic_requirements: &[],
    },
    RuntimeQualityProbeCase {
        id: "csv-import",
        input_text: "CSV取込処理を改善してください。最大10MB、UTF-8 BOM付きにも対応し、不正な行は行番号と理由を返してください。一部だけ保存せずトランザクションで処理し、既存の列名と数値精度は変えないでください。",
        required_marker_groups: &[
            &["CSV"], &["10MB"], &["UTF-8BOM"], &["不正"], &["行番号"], &["理由"],
            &["一部"], &["保存"], &["トランザクション"], &["列名"], &["数値精度"],
        ],
        semantic_requirements: &[],
    },
    RuntimeQualityProbeCase {
        id: "ci-cache",
        input_text: "CIの依存関係キャッシュを最適化してください。OS、ロックファイル、Rustのバージョンが変わった時だけ無効化し、キャッシュがなくても通常ビルドへ進めるようにします。シークレットをログへ出さないでください。",
        required_marker_groups: &[
            &["CI"], &["依存関係"], &["キャッシュ"], &["OS"], &["ロックファイル"],
            &["Rust"], &["バージョン"], &["無効"], &["ビルド"], &["シークレット"], &["ログ"],
        ],
        semantic_requirements: &[],
    },
    RuntimeQualityProbeCase {
        id: "monitoring-log",
        input_text: "監視ログの集計を追加してください。5分単位でエラー件数と95パーセンタイルの応答時間を計算し、個人情報は保存前に除去します。処理が30秒を超えた場合は中断し、元データを削除しないでください。",
        required_marker_groups: &[
            &["5分"], &["エラー"], &["95", "P95"], &["応答時間"],
            &["個人情報"], &["除去", "匿名化", "マスキング"], &["30秒"], &["中断"],
            &["元データ"], &["削除しない", "残す", "維持"],
        ],
        semantic_requirements: &[
            RuntimeQualitySemanticRequirement {
                label: "監視ログ集計",
                alternatives: &[&["監視ログ"], &["エラー", "応答時間", "集計"]],
            },
            RuntimeQualitySemanticRequirement {
                label: "保存前の個人情報除去",
                alternatives: &[
                    &["保存", "除去"],
                    &["保存", "匿名化"],
                    &["保存", "マスキング"],
                    &["除去済み"],
                    &["匿名化済み"],
                    &["マスキング済み"],
                ],
            },
        ],
    },
];

pub fn runtime_quality_probe_cases() -> &'static [RuntimeQualityProbeCase] {
    &RUNTIME_QUALITY_PROBE_CASES
}

pub fn runtime_quality_missing_requirements(
    case: RuntimeQualityProbeCase,
    output: &str,
    should_send_original: bool,
) -> Vec<String> {
    if should_send_original || output.trim().is_empty() {
        return vec!["compressed output".to_string()];
    }
    let normalized_output = normalize_quality_text(output);
    let mut missing: Vec<String> = case
        .required_marker_groups
        .iter()
        .filter(|group| {
            !group.iter().any(|marker| {
                let normalized_marker = normalize_quality_text(marker);
                normalized_output.contains(&normalized_marker)
            })
        })
        .map(|group| group.join("/"))
        .collect();
    missing.extend(
        case.semantic_requirements
            .iter()
            .filter(|requirement| {
                !requirement.alternatives.iter().any(|alternative| {
                    alternative
                        .iter()
                        .all(|marker| normalized_output.contains(&normalize_quality_text(marker)))
                })
            })
            .map(|requirement| requirement.label.to_string()),
    );
    missing
}

fn normalize_quality_text(text: &str) -> String {
    text.chars()
        .filter(|character| !character.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}
static SETTINGS_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct ServerOptions {
    pub host: String,
    pub port: u16,
    pub settings_dir: Option<PathBuf>,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8787,
            settings_dir: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServerInfo {
    pub address: String,
    pub url: String,
    pub settings_dir: PathBuf,
}

pub struct ServerHandle {
    pub info: ServerInfo,
    shutdown_tx: Option<mpsc::Sender<()>>,
    join_handle: Option<JoinHandle<Result<()>>>,
}

impl ServerHandle {
    pub fn shutdown(mut self) -> Result<()> {
        self.shutdown_inner()
    }

    fn shutdown_inner(&mut self) -> Result<()> {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        if let Some(join_handle) = self.join_handle.take() {
            match join_handle.join() {
                Ok(result) => result?,
                Err(_) => {
                    return Err(anyhow::anyhow!("server thread panicked during shutdown"));
                }
            }
        }
        Ok(())
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        let _ = self.shutdown_inner();
    }
}

#[derive(Clone)]
pub struct EmbeddedWebApp {
    settings_dir: PathBuf,
    state: Arc<AppState>,
}

impl EmbeddedWebApp {
    pub fn settings_dir(&self) -> &Path {
        &self.settings_dir
    }

    pub fn start_default_profile_warmup(&self) {
        start_default_profile_warmup(self.state.clone());
    }

    pub fn tune_and_benchmark_default_cpu_engine(
        &self,
    ) -> Result<Option<CpuEngineBenchmarkResult>> {
        let profile_id = self
            .state
            .registry
            .default_profile_id()
            .context("default profile is not configured")?;
        let profile = self.state.registry.resolve(profile_id)?;
        let cancellation = AtomicBool::new(false);
        self.state
            .backend
            .tune_profile_threads(profile, &cancellation)?;
        anyhow::ensure!(
            !self.state.backend.profile_thread_tuning_required(profile)?,
            "CPU engine thread tuning did not produce a saved configuration"
        );
        benchmark_cpu_engine_pipeline(profile, &self.state, &cancellation)
    }

    pub fn handle_request(&self, method: &str, path: &str, body: &[u8]) -> LocalAppResponse {
        route_application_request(method, path, body, &self.state)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuEngineBenchmarkResult {
    pub elapsed_micros: u64,
    pub quality_passed: bool,
    pub quality_case_count: u32,
}

#[derive(Debug, Clone)]
pub struct LocalAppResponse {
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
}

#[derive(Clone)]
struct AppState {
    application_root: PathBuf,
    registry: ProfileRegistry,
    backend: ConfiguredRuntimeBackend,
    service: Arc<CompressionService<ConfiguredRuntimeBackend>>,
    warmup: RuntimeWarmupState,
    ui_settings: PersistedUiSettings,
    inference_gate: InferenceGate,
    model_downloads: ModelDownloadControl,
    runtime_tuning_cancellation: Arc<AtomicBool>,
}

#[derive(Clone, Default)]
struct InferenceGate {
    slot: Arc<Mutex<()>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InferenceGateError {
    Busy,
    Unavailable,
}

impl InferenceGate {
    fn try_enter(&self) -> std::result::Result<std::sync::MutexGuard<'_, ()>, InferenceGateError> {
        match self.slot.try_lock() {
            Ok(permit) => Ok(permit),
            Err(TryLockError::WouldBlock) => Err(InferenceGateError::Busy),
            Err(TryLockError::Poisoned(_)) => Err(InferenceGateError::Unavailable),
        }
    }

    fn enter(&self) -> std::result::Result<std::sync::MutexGuard<'_, ()>, InferenceGateError> {
        self.slot
            .lock()
            .map_err(|_| InferenceGateError::Unavailable)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CpuEngineProbeResult {
    schema_version: u32,
    build_id: String,
    inference_compatibility_id: String,
    cpu_engine: String,
    elapsed_micros: u64,
    quality_passed: bool,
    quality_case_count: u32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedCpuEngineSelection {
    schema_version: u32,
    build_id: String,
    inference_compatibility_id: String,
    cpu_key: String,
    cpu_engine: String,
}

#[derive(Debug, Clone, Copy, Default)]
struct CpuInstructionCapabilities {
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

#[derive(Clone, Default)]
struct ModelDownloadControl {
    next_id: Arc<AtomicU64>,
    active: Arc<Mutex<Option<ActiveModelDownload>>>,
}

#[derive(Clone)]
struct ActiveModelDownload {
    id: u64,
    profile: String,
    cancellation: ModelDownloadCancellation,
}

struct ModelDownloadPermit {
    control: ModelDownloadControl,
    active: ActiveModelDownload,
}

impl ModelDownloadControl {
    fn start(&self, profile: String) -> Result<ModelDownloadPermit> {
        let mut active = self
            .active
            .lock()
            .map_err(|_| anyhow::anyhow!("model download control is unavailable"))?;
        if active.is_some() {
            anyhow::bail!("a model download is already active");
        }
        let operation = ActiveModelDownload {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            profile,
            cancellation: ModelDownloadCancellation::default(),
        };
        *active = Some(operation.clone());
        Ok(ModelDownloadPermit {
            control: self.clone(),
            active: operation,
        })
    }

    fn cancel(&self, profile: &str) -> Result<bool> {
        let active = self
            .active
            .lock()
            .map_err(|_| anyhow::anyhow!("model download control is unavailable"))?;
        let Some(active) = active.as_ref() else {
            return Ok(false);
        };
        if active.profile != profile {
            anyhow::bail!("the requested profile is not being downloaded");
        }
        active.cancellation.cancel();
        Ok(true)
    }
}

impl ModelDownloadPermit {
    fn cancellation(&self) -> &ModelDownloadCancellation {
        &self.active.cancellation
    }
}

impl Drop for ModelDownloadPermit {
    fn drop(&mut self) {
        let Ok(mut active) = self.control.active.lock() else {
            return;
        };
        if active.as_ref().map(|value| value.id) == Some(self.active.id) {
            *active = None;
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct RuntimeWarmupSnapshot {
    phase: String,
    profile: Option<String>,
    message: String,
    error: Option<String>,
    downloaded_bytes: Option<u64>,
    total_bytes: Option<u64>,
}

#[derive(Clone)]
struct RuntimeWarmupState {
    snapshot: Arc<Mutex<RuntimeWarmupSnapshot>>,
}

impl RuntimeWarmupState {
    fn new() -> Self {
        Self {
            snapshot: Arc::new(Mutex::new(RuntimeWarmupSnapshot {
                phase: "idle".to_string(),
                profile: None,
                message: "モデル読み込み待機中".to_string(),
                error: None,
                downloaded_bytes: None,
                total_bytes: None,
            })),
        }
    }

    fn set(
        &self,
        phase: impl Into<String>,
        profile: Option<String>,
        message: impl Into<String>,
        error: Option<String>,
    ) {
        match self.snapshot.lock() {
            Ok(mut snapshot) => {
                snapshot.phase = phase.into();
                snapshot.profile = profile;
                snapshot.message = message.into();
                snapshot.error = error;
                snapshot.downloaded_bytes = None;
                snapshot.total_bytes = None;
            }
            Err(error) => {
                eprintln!("runtime warmup status lock failed: {error}");
            }
        }
    }

    fn set_download_progress(&self, profile: String, downloaded_bytes: u64, total_bytes: u64) {
        let percent = downloaded_bytes
            .saturating_mul(100)
            .checked_div(total_bytes)
            .unwrap_or(0);
        match self.snapshot.lock() {
            Ok(mut snapshot) => {
                snapshot.phase = "downloading".to_string();
                snapshot.profile = Some(profile);
                snapshot.message = format!("Hugging Faceからモデルを取得中 {percent}%");
                snapshot.error = None;
                snapshot.downloaded_bytes = Some(downloaded_bytes);
                snapshot.total_bytes = Some(total_bytes);
            }
            Err(error) => eprintln!("runtime warmup status lock failed: {error}"),
        }
    }

    fn json(&self) -> serde_json::Value {
        match self.snapshot.lock() {
            Ok(snapshot) => serde_json::to_value(snapshot.clone()).unwrap_or_else(|error| {
                serde_json::json!({
                    "phase": "error",
                    "profile": null,
                    "message": "モデル状態をJSON化できません",
                    "error": error.to_string()
                })
            }),
            Err(error) => serde_json::json!({
                "phase": "error",
                "profile": null,
                "message": "モデル状態を取得できません",
                "error": error.to_string()
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct UiSettings {
    #[serde(default = "default_ui_settings_schema_version")]
    schema_version: u32,
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    task: Option<String>,
    #[serde(default)]
    level: Option<u8>,
    #[serde(default)]
    theme: Option<String>,
    #[serde(default)]
    cpu_engine: Option<String>,
    #[serde(default)]
    thread_mode: Option<String>,
    #[serde(default)]
    generation_threads: Option<u32>,
    #[serde(default)]
    batch_threads: Option<u32>,
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            schema_version: default_ui_settings_schema_version(),
            profile: None,
            mode: None,
            task: None,
            level: None,
            theme: None,
            cpu_engine: None,
            thread_mode: None,
            generation_threads: None,
            batch_threads: None,
            extra: BTreeMap::new(),
        }
    }
}

impl UiSettings {
    fn normalized(mut self) -> Self {
        self.schema_version = default_ui_settings_schema_version();
        self.profile = normalize_optional_text(self.profile);
        self.mode = normalize_optional_text(self.mode);
        self.task = normalize_optional_text(self.task);
        self.level = self.level.map(|level| level.clamp(1, 3));
        self.theme = normalize_optional_text(self.theme)
            .filter(|theme| matches!(theme.as_str(), "light" | "dark"));
        self.cpu_engine = normalize_optional_text(self.cpu_engine)
            .filter(|engine| matches!(engine.as_str(), "auto" | "compatible" | "avx2" | "avx512"));
        self.thread_mode = normalize_optional_text(self.thread_mode)
            .filter(|mode| matches!(mode.as_str(), "auto" | "manual"));
        self.generation_threads = self.generation_threads.filter(|threads| *threads > 0);
        self.batch_threads = self.batch_threads.filter(|threads| *threads > 0);
        self
    }
}

#[derive(Clone)]
struct PersistedUiSettings {
    path: PathBuf,
    settings: Arc<Mutex<UiSettings>>,
}

impl PersistedUiSettings {
    fn load(path: PathBuf) -> Self {
        let settings = match fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str::<UiSettings>(&contents)
                .map(UiSettings::normalized)
                .unwrap_or_else(|error| {
                    eprintln!("failed to parse UI settings at {}: {error}", path.display());
                    UiSettings::default()
                }),
            Err(error) if error.kind() == ErrorKind::NotFound => UiSettings::default(),
            Err(error) => {
                eprintln!("failed to read UI settings at {}: {error}", path.display());
                UiSettings::default()
            }
        };

        Self {
            path,
            settings: Arc::new(Mutex::new(settings)),
        }
    }

    fn json(&self) -> serde_json::Value {
        match self.settings.lock() {
            Ok(settings) => serde_json::to_value(settings.clone()).unwrap_or_else(|error| {
                serde_json::json!({
                    "schema_version": default_ui_settings_schema_version(),
                    "error": format!("failed to serialize UI settings: {error}")
                })
            }),
            Err(error) => serde_json::json!({
                "schema_version": default_ui_settings_schema_version(),
                "error": format!("failed to read UI settings: {error}")
            }),
        }
    }

    fn snapshot(&self) -> Result<UiSettings> {
        self.settings
            .lock()
            .map(|settings| settings.clone())
            .map_err(|error| anyhow::anyhow!("failed to read UI settings: {error}"))
    }

    fn save(&self, mut settings: UiSettings) -> Result<serde_json::Value> {
        let mut guard = self
            .settings
            .lock()
            .map_err(|error| anyhow::anyhow!("failed to lock UI settings: {error}"))?;

        for (key, value) in &guard.extra {
            settings.extra.entry(key.clone()).or_insert(value.clone());
        }
        let settings = settings.normalized();
        if *guard == settings {
            return Ok(serde_json::to_value(settings)?);
        }

        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }

        let body =
            serde_json::to_vec_pretty(&settings).context("failed to serialize UI settings")?;
        atomic_write_file(&self.path, &body)?;

        *guard = settings.clone();
        Ok(serde_json::to_value(settings)?)
    }
}

fn atomic_write_file(path: &Path, body: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("settings path has no file name: {}", path.display()))?;

    let (temp_path, mut temp_file) = loop {
        let counter = SETTINGS_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut temp_name = file_name.to_os_string();
        temp_name.push(format!(".{}.{}.tmp", std::process::id(), counter));
        let temp_path = parent.join(temp_name);

        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => break (temp_path, file),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to create {}", temp_path.display()));
            }
        }
    };

    // 一時ファイルを完全に同期してから置換し、旧設定か新設定のどちらかだけが残るようにする。
    let write_result = temp_file
        .write_all(body)
        .with_context(|| format!("failed to write {}", temp_path.display()))
        .and_then(|()| {
            temp_file
                .sync_all()
                .with_context(|| format!("failed to sync {}", temp_path.display()))
        });
    drop(temp_file);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temp_path);
        return Err(error);
    }

    if let Err(error) = replace_file(&temp_path, path) {
        let _ = fs::remove_file(&temp_path);
        return Err(error).with_context(|| format!("failed to replace {}", path.display()));
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    let source = path_to_wide_null(source);
    let destination = path_to_wide_null(destination);
    let flags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH;

    if unsafe { MoveFileExW(source.as_ptr(), destination.as_ptr(), flags) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    fs::rename(source, destination)?;
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::File::open(parent)?.sync_all()
}

#[cfg(target_os = "windows")]
fn path_to_wide_null(path: &Path) -> Vec<u16> {
    path.as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

fn normalize_optional_text(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn default_ui_settings_schema_version() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
struct CompressPayload {
    input_text: String,
    profile: String,
    compression_level: u8,
    constraints: Option<CompressionConstraints>,
}

#[derive(Debug, Deserialize)]
struct PrepareCompressionPayload {
    profile: String,
    compression_level: u8,
    constraints: Option<CompressionConstraints>,
}

#[derive(Debug, Deserialize)]
struct ModelProfilePayload {
    profile: String,
}

#[derive(Debug, Deserialize)]
struct ClipboardPayload {
    text: String,
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

#[derive(Debug)]
struct HttpReadError {
    status: u16,
    message: &'static str,
}

impl HttpReadError {
    fn new(status: u16, message: &'static str) -> Self {
        Self { status, message }
    }
}

pub fn run_server(options: ServerOptions) -> Result<()> {
    run_server_with_ready(options, |info| {
        println!("TrimPrompt local UI: {}", info.url);
    })
}

pub fn prepare_embedded_web_app(settings_dir: Option<PathBuf>) -> Result<EmbeddedWebApp> {
    let settings_dir = resolve_settings_dir(settings_dir.as_deref())?;
    let state = prepare_app_state(&settings_dir)?;

    Ok(EmbeddedWebApp {
        settings_dir,
        state,
    })
}

pub fn start_server_in_background(options: ServerOptions) -> Result<ServerHandle> {
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let (shutdown_tx, shutdown_rx) = mpsc::channel();
    let join_handle = thread::spawn(move || match prepare_server(options) {
        Ok((info, listener, state)) => {
            let _ = ready_tx.send(Ok(info));
            start_default_profile_warmup(state.clone());
            serve(listener, state, Some(shutdown_rx))
        }
        Err(error) => {
            let _ = ready_tx.send(Err(error));
            Ok(())
        }
    });

    let info = ready_rx
        .recv()
        .context("server thread exited before startup")??;

    Ok(ServerHandle {
        info,
        shutdown_tx: Some(shutdown_tx),
        join_handle: Some(join_handle),
    })
}

fn run_server_with_ready(options: ServerOptions, on_ready: impl FnOnce(ServerInfo)) -> Result<()> {
    let (info, listener, state) = prepare_server(options)?;
    on_ready(info);
    start_default_profile_warmup(state.clone());
    serve(listener, state, None)
}

fn prepare_server(options: ServerOptions) -> Result<(ServerInfo, TcpListener, Arc<AppState>)> {
    if !is_loopback_host(&options.host) {
        anyhow::bail!("local UI host must resolve to a loopback address");
    }
    let settings_dir = resolve_settings_dir(options.settings_dir.as_deref())?;
    let state = prepare_app_state(&settings_dir)?;

    let address = format!("{}:{}", options.host, options.port);
    let listener =
        TcpListener::bind(&address).with_context(|| format!("failed to bind {address}"))?;
    let address = listener
        .local_addr()
        .context("failed to determine local server address")?;
    if !address.ip().is_loopback() {
        anyhow::bail!("local UI listener must use a loopback address");
    }
    let address = address.to_string();
    let url = format!("http://{address}/");

    Ok((
        ServerInfo {
            address,
            url,
            settings_dir,
        },
        listener,
        state,
    ))
}

fn prepare_app_state(settings_dir: &Path) -> Result<Arc<AppState>> {
    let application_root = settings_dir
        .parent()
        .context("settings directory must be inside the application directory")?;
    let profiles_path = settings_dir.join("compression-profiles.yaml");
    let registry = ProfileRegistry::from_path(&profiles_path)
        .with_context(|| format!("failed to load profiles from {}", profiles_path.display()))?;
    let backend = ConfiguredRuntimeBackend::from_settings_dir(settings_dir)
        .context("failed to initialize configured runtime backend")?;
    let service = Arc::new(CompressionService::new(registry.clone(), backend.clone()));
    let ui_settings = PersistedUiSettings::load(
        application_root
            .join("local")
            .join("state")
            .join("ui-settings.json"),
    );
    let state = Arc::new(AppState {
        application_root: application_root.to_path_buf(),
        registry,
        backend,
        service,
        warmup: RuntimeWarmupState::new(),
        ui_settings,
        inference_gate: InferenceGate::default(),
        model_downloads: ModelDownloadControl::default(),
        runtime_tuning_cancellation: Arc::new(AtomicBool::new(false)),
    });
    Ok(state)
}

fn start_default_profile_warmup(state: Arc<AppState>) {
    let request = match startup_compression_request(&state.registry, &state.ui_settings) {
        Ok(request) => request,
        Err(error) => {
            state.warmup.set(
                "error",
                None,
                "起動時の圧縮設定を解決できません",
                Some(error.to_string()),
            );
            return;
        }
    };
    let profile_id = request.profile.clone();
    let profile = match state.registry.resolve(&profile_id) {
        Ok(profile) => profile.clone(),
        Err(error) => {
            state.warmup.set(
                "error",
                Some(profile_id),
                "先読み対象のプロファイルを解決できません",
                Some(error.to_string()),
            );
            return;
        }
    };

    state.warmup.set(
        "checking",
        Some(profile.id.clone()),
        "アプリ内モデルを確認中",
        None,
    );

    thread::spawn(move || {
        let profile_id = profile.id.clone();
        let _inference_permit = match state.inference_gate.slot.lock() {
            Ok(permit) => permit,
            Err(error) => {
                state.warmup.set(
                    "error",
                    Some(profile_id),
                    "起動時準備を開始できません",
                    Some(error.to_string()),
                );
                return;
            }
        };
        let status = match state.backend.profile_model_status(&profile) {
            Ok(status) => status,
            Err(error) => {
                state.warmup.set(
                    "error",
                    Some(profile_id),
                    "モデル状態の確認に失敗しました",
                    Some(error.to_string()),
                );
                return;
            }
        };
        if status.requires_install && !status.installed {
            state
                .warmup
                .set("missing", Some(profile_id), "モデルの取得が必要です", None);
            return;
        }

        state.warmup.set(
            "loading",
            Some(profile_id.clone()),
            "インストール済みモデルを読み込み中",
            None,
        );
        match state.backend.warm_profile(&profile) {
            Ok(true) => {
                state.warmup.set(
                    "loading",
                    Some(profile_id.clone()),
                    "保存済みの圧縮レベルを準備中",
                    None,
                );
                match state.service.prepare(request) {
                    Ok(true) => state.warmup.set(
                        "ready",
                        Some(profile_id),
                        "アプリ内モデルは準備完了です",
                        None,
                    ),
                    Ok(false) => state.warmup.set(
                        "ready",
                        Some(profile_id),
                        "アプリ内モデルは準備完了です",
                        None,
                    ),
                    Err(error) => state.warmup.set(
                        "error",
                        Some(profile_id),
                        "圧縮プロンプトの準備に失敗しました",
                        Some(error.to_string()),
                    ),
                }
            }
            Ok(false) => state.warmup.set(
                "skipped",
                Some(profile_id),
                "このプロファイルは先読み不要です",
                None,
            ),
            Err(error) => state.warmup.set(
                "error",
                Some(profile_id),
                "アプリ内モデルの読み込みに失敗しました",
                Some(error.to_string()),
            ),
        }
    });
}

fn startup_compression_request(
    registry: &ProfileRegistry,
    ui_settings: &PersistedUiSettings,
) -> Result<CompressionRequest> {
    let settings = ui_settings.snapshot()?;
    let profile = settings
        .profile
        .filter(|profile| {
            registry
                .resolve(profile)
                .is_ok_and(|definition| definition.selectable)
        })
        .or_else(|| registry.default_profile_id().map(str::to_owned))
        .context("no selectable startup profile is available")?;
    let level = settings.level.unwrap_or(2).clamp(2, 3);

    Ok(CompressionRequest {
        input_text: String::new(),
        compression_level: CompressionLevel::from_u8(level)?,
        profile,
        constraints: CompressionConstraints::default(),
        target: RequestTarget::codex_default(),
        source: RequestSource::Desktop,
    })
}

fn serve(
    listener: TcpListener,
    state: Arc<AppState>,
    shutdown_rx: Option<mpsc::Receiver<()>>,
) -> Result<()> {
    let server_address = listener
        .local_addr()
        .context("failed to determine local server address")?;
    listener
        .set_nonblocking(shutdown_rx.is_some())
        .context("failed to configure local server listener")?;
    let (job_tx, workers) = start_http_workers(state, server_address);

    let serve_result = loop {
        if let Some(shutdown_rx) = &shutdown_rx {
            match shutdown_rx.try_recv() {
                Ok(()) | Err(mpsc::TryRecvError::Disconnected) => break Ok(()),
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }

        match listener.accept() {
            Ok((stream, peer)) => {
                if !peer.ip().is_loopback() {
                    eprintln!("rejected non-loopback client");
                    continue;
                }
                match job_tx.try_send(stream) {
                    Ok(()) => {}
                    Err(mpsc::TrySendError::Full(stream)) => {
                        reject_overloaded_client(stream);
                    }
                    Err(mpsc::TrySendError::Disconnected(_)) => {
                        break Err(anyhow::anyhow!("HTTP worker pool stopped unexpectedly"));
                    }
                }
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(50));
            }
            Err(error) => eprintln!("connection failed: {error}"),
        }
    };

    drop(job_tx);
    for worker in workers {
        if worker.join().is_err() {
            return Err(anyhow::anyhow!("HTTP worker panicked during shutdown"));
        }
    }
    serve_result
}

fn start_http_workers(
    state: Arc<AppState>,
    server_address: SocketAddr,
) -> (mpsc::SyncSender<TcpStream>, Vec<JoinHandle<()>>) {
    let (job_tx, job_rx) = mpsc::sync_channel::<TcpStream>(HTTP_QUEUE_CAPACITY);
    let job_rx = Arc::new(Mutex::new(job_rx));
    let mut workers = Vec::with_capacity(HTTP_WORKER_COUNT);

    for _ in 0..HTTP_WORKER_COUNT {
        let state = state.clone();
        let job_rx = job_rx.clone();
        workers.push(thread::spawn(move || loop {
            let stream = match job_rx.lock() {
                Ok(receiver) => receiver.recv(),
                Err(error) => {
                    eprintln!("HTTP worker queue lock failed: {error}");
                    return;
                }
            };
            let Ok(stream) = stream else {
                return;
            };
            if let Err(error) = handle_client(stream, &state, server_address) {
                eprintln!("request failed: {error:#}");
            }
        }));
    }

    (job_tx, workers)
}

fn reject_overloaded_client(mut stream: TcpStream) {
    let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
    let response = http_transport_response(json_response(
        503,
        &serde_json::json!({ "error": "local UI is busy" }),
    ));
    let _ = stream.write_all(&response);
    let _ = stream.flush();
}

fn handle_client(
    mut stream: TcpStream,
    state: &AppState,
    server_address: SocketAddr,
) -> Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    let response = match read_request(&mut stream) {
        Ok(request) => route_request(request, state, server_address),
        Err(error) => http_transport_response(json_response(
            error.status,
            &serde_json::json!({ "error": error.message }),
        )),
    };
    stream.write_all(&response)?;
    stream.flush()?;
    Ok(())
}

fn route_request(request: HttpRequest, state: &AppState, server_address: SocketAddr) -> Vec<u8> {
    if let Err(error) = validate_network_request(&request, server_address) {
        return http_transport_response(json_response(
            error.status,
            &serde_json::json!({ "error": error.message }),
        ));
    }
    http_transport_response(route_application_request(
        &request.method,
        &request.path,
        &request.body,
        state,
    ))
}

fn route_application_request(
    method: &str,
    raw_path: &str,
    body: &[u8],
    state: &AppState,
) -> LocalAppResponse {
    let path = raw_path.split('?').next().unwrap_or(raw_path);
    if body.len() > request_body_limit(path) {
        return json_response(
            413,
            &serde_json::json!({ "error": "request body is too large" }),
        );
    }
    match (method, path) {
        ("GET", "/") => http_response(200, "text/html; charset=utf-8", INDEX_HTML.as_bytes()),
        ("GET", "/styles.css") => {
            http_response(200, "text/css; charset=utf-8", STYLES_CSS.as_bytes())
        }
        ("GET", "/app.js") => http_response(
            200,
            "application/javascript; charset=utf-8",
            APP_JS.as_bytes(),
        ),
        ("GET", "/sample-prompts.js") => http_response(
            200,
            "application/javascript; charset=utf-8",
            SAMPLE_PROMPTS_JS.as_bytes(),
        ),
        ("GET", "/api/profiles") => json_response(200, &profiles_json(&state.registry)),
        ("GET", "/api/runtime-status") => json_response(200, &state.warmup.json()),
        ("POST", "/api/model-status") => match model_status_from_body(body, state) {
            Ok(value) => json_response(200, &value),
            Err(error) => json_response(400, &serde_json::json!({ "error": error.to_string() })),
        },
        ("POST", "/api/model-install") => {
            run_inference_route(state, || install_model_from_body(body, state))
        }
        ("POST", "/api/model-cancel") => match cancel_model_download_from_body(body, state) {
            Ok(value) => json_response(200, &value),
            Err(error) => json_response(400, &serde_json::json!({ "error": error.to_string() })),
        },
        ("GET", "/api/settings") => json_response(200, &state.ui_settings.json()),
        ("PUT" | "POST", "/api/settings") => match save_settings_from_body(body, state) {
            Ok(value) => json_response(200, &value),
            Err(error) => json_response(
                400,
                &serde_json::json!({
                    "error": error.to_string()
                }),
            ),
        },
        ("POST", "/api/runtime-configuration") => {
            match runtime_configuration_from_body(body, state) {
                Ok(value) => json_response(200, &value),
                Err(error) => {
                    json_response(400, &serde_json::json!({ "error": error.to_string() }))
                }
            }
        }
        ("POST", "/api/clipboard") => match copy_to_clipboard_from_body(body) {
            Ok(value) => json_response(200, &value),
            Err(error) => json_response(
                400,
                &serde_json::json!({
                    "copied": false,
                    "error": error.to_string()
                }),
            ),
        },
        ("POST", "/api/prepare-compression") => {
            run_inference_route(state, || prepare_compression_from_body(body, state))
        }
        ("POST", "/api/prepare-input") => {
            run_inference_route(state, || prepare_input_from_body(body, state))
        }
        ("POST", "/api/tune-runtime") => {
            run_inference_route(state, || tune_runtime_from_body(body, state))
        }
        ("POST", "/api/runtime-setup-status") => {
            match runtime_setup_status_from_body(body, state) {
                Ok(value) => json_response(200, &value),
                Err(error) => {
                    json_response(400, &serde_json::json!({ "error": error.to_string() }))
                }
            }
        }
        ("POST", "/api/tune-runtime-reset") => {
            state
                .runtime_tuning_cancellation
                .store(true, Ordering::Relaxed);
            run_waiting_inference_route(state, || reset_runtime_tuning_from_body(body, state))
        }
        ("POST", "/api/compress") => {
            run_waiting_inference_route(state, || compress_from_body(body, state))
        }
        _ => json_response(
            404,
            &serde_json::json!({
                "error": "not found"
            }),
        ),
    }
}

fn request_body_limit(path: &str) -> usize {
    match path {
        "/api/settings"
        | "/api/prepare-compression"
        | "/api/model-status"
        | "/api/model-install"
        | "/api/model-cancel"
        | "/api/runtime-configuration"
        | "/api/runtime-setup-status"
        | "/api/tune-runtime"
        | "/api/tune-runtime-reset" => MAX_SETTINGS_BODY_BYTES,
        "/api/clipboard" | "/api/compress" | "/api/prepare-input" => MAX_REQUEST_BODY_BYTES,
        _ => MAX_SETTINGS_BODY_BYTES,
    }
}

fn run_inference_route(
    state: &AppState,
    operation: impl FnOnce() -> Result<serde_json::Value>,
) -> LocalAppResponse {
    let _permit = match state.inference_gate.try_enter() {
        Ok(permit) => permit,
        Err(InferenceGateError::Busy) => {
            return json_response(
                429,
                &serde_json::json!({ "error": "compression is already running" }),
            );
        }
        Err(InferenceGateError::Unavailable) => {
            return json_response(
                500,
                &serde_json::json!({ "error": "compression gate is unavailable" }),
            );
        }
    };

    match operation() {
        Ok(value) => json_response(200, &value),
        Err(error) => json_response(400, &serde_json::json!({ "error": error.to_string() })),
    }
}

fn run_waiting_inference_route(
    state: &AppState,
    operation: impl FnOnce() -> Result<serde_json::Value>,
) -> LocalAppResponse {
    let _permit = match state.inference_gate.enter() {
        Ok(permit) => permit,
        Err(InferenceGateError::Unavailable) => {
            return json_response(
                500,
                &serde_json::json!({ "error": "compression gate is unavailable" }),
            );
        }
        Err(InferenceGateError::Busy) => unreachable!("blocking inference entry cannot be busy"),
    };

    match operation() {
        Ok(value) => json_response(200, &value),
        Err(error) => json_response(400, &serde_json::json!({ "error": error.to_string() })),
    }
}

fn save_settings_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let settings: UiSettings = serde_json::from_slice(body).context("invalid JSON settings")?;
    for (field, value) in [
        ("profile", settings.profile.as_deref()),
        ("mode", settings.mode.as_deref()),
        ("task", settings.task.as_deref()),
        ("theme", settings.theme.as_deref()),
        ("cpu_engine", settings.cpu_engine.as_deref()),
        ("thread_mode", settings.thread_mode.as_deref()),
    ] {
        if let Some(value) = value {
            validate_text_field(field, value, MAX_PROFILE_CHARS, true)?;
        }
    }
    validate_runtime_preferences(&settings)?;
    state.ui_settings.save(settings)
}

fn validate_runtime_preferences(settings: &UiSettings) -> Result<()> {
    let capabilities = detect_cpu_instruction_capabilities();
    let cpu_engine = settings.cpu_engine.as_deref().unwrap_or("auto");
    anyhow::ensure!(
        matches!(cpu_engine, "auto" | "compatible" | "avx2" | "avx512"),
        "unsupported CPU engine setting"
    );
    anyhow::ensure!(
        cpu_engine_supported(capabilities, cpu_engine),
        "selected CPU engine is not supported by this processor"
    );

    let thread_mode = settings.thread_mode.as_deref().unwrap_or("auto");
    anyhow::ensure!(
        matches!(thread_mode, "auto" | "manual"),
        "unsupported thread setting"
    );
    if thread_mode == "manual" {
        let maximum = u32::try_from(
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
        )
        .unwrap_or(u32::MAX)
        .max(1);
        for (name, value) in [
            ("generation", settings.generation_threads),
            ("batch", settings.batch_threads),
        ] {
            let value = value.with_context(|| format!("manual {name} thread count is missing"))?;
            anyhow::ensure!(
                (1..=maximum).contains(&value),
                "manual {name} thread count must be between 1 and {maximum}"
            );
        }
    }
    Ok(())
}

fn runtime_configuration_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let payload: ModelProfilePayload =
        serde_json::from_slice(body).context("invalid JSON runtime configuration request")?;
    validate_text_field("profile", &payload.profile, MAX_PROFILE_CHARS, false)?;
    let profile = state.registry.resolve(&payload.profile)?;
    let threads = state.backend.profile_thread_status(profile)?;
    let capabilities = detect_cpu_instruction_capabilities();
    let current_cpu_engine = current_cpu_engine();
    let current_cpu_mode = if std::env::var(CPU_ENGINE_MODE_ENV).as_deref() == Ok("manual") {
        "manual"
    } else {
        "auto"
    };

    Ok(serde_json::json!({
        "current_cpu_engine": current_cpu_engine,
        "current_cpu_mode": current_cpu_mode,
        "current_thread_mode": threads.mode,
        "generation_threads": threads.generation_threads,
        "batch_threads": threads.batch_threads,
        "logical_batch_size": threads.logical_batch_size,
        "physical_batch_size": threads.physical_batch_size,
        "available_threads": threads.available_threads,
        "cpu_engines": [
            { "id": "auto", "supported": true },
            { "id": "compatible", "supported": cpu_engine_supported(capabilities, "compatible") },
            { "id": "avx2", "supported": cpu_engine_supported(capabilities, "avx2") },
            { "id": "avx512", "supported": cpu_engine_supported(capabilities, "avx512") }
        ]
    }))
}

fn copy_to_clipboard_from_body(body: &[u8]) -> Result<serde_json::Value> {
    let payload: ClipboardPayload =
        serde_json::from_slice(body).context("invalid JSON clipboard request")?;
    validate_text_field("clipboard text", &payload.text, MAX_CLIPBOARD_CHARS, false)?;

    let copied = write_text_to_clipboard(&payload.text)?;
    Ok(serde_json::json!({
        "copied": copied
    }))
}

fn model_status_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let payload: ModelProfilePayload =
        serde_json::from_slice(body).context("invalid JSON model status request")?;
    validate_text_field("profile", &payload.profile, MAX_PROFILE_CHARS, false)?;
    let profile = state.registry.resolve(&payload.profile)?;
    let status = state.backend.profile_model_status(profile)?;
    let available_bytes = status
        .destination
        .as_deref()
        .and_then(available_disk_space_bytes);
    let mut value = serde_json::to_value(status)?;
    value["available_bytes"] = available_bytes
        .map(serde_json::Value::from)
        .unwrap_or(serde_json::Value::Null);
    Ok(value)
}

fn install_model_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let payload: ModelProfilePayload =
        serde_json::from_slice(body).context("invalid JSON model install request")?;
    validate_text_field("profile", &payload.profile, MAX_PROFILE_CHARS, false)?;
    let profile = state.registry.resolve(&payload.profile)?.clone();
    let status = state.backend.profile_model_status(&profile)?;
    if !status.requires_install {
        return Ok(serde_json::json!({
            "installed": true,
            "message": "このプロファイルはモデル取得不要です"
        }));
    }
    ensure_model_disk_capacity(&status)?;
    let download = state.model_downloads.start(profile.id.clone())?;

    let profile_id = profile.id.clone();
    state.warmup.set(
        "downloading",
        Some(profile_id.clone()),
        "Hugging Faceからモデルを取得しています",
        None,
    );
    let warmup = state.warmup.clone();
    let result = state
        .backend
        .install_profile_with_progress_and_cancellation(
            &profile,
            download.cancellation(),
            move |ModelDownloadProgress {
                      downloaded_bytes,
                      total_bytes,
                  }| {
                warmup.set_download_progress(profile_id.clone(), downloaded_bytes, total_bytes);
            },
        );
    match result {
        Ok(true) => {
            state.warmup.set(
                "ready",
                Some(profile.id),
                "モデル取得と読み込みが完了しました",
                None,
            );
            Ok(serde_json::json!({
                "installed": true,
                "message": "モデル取得と読み込みが完了しました"
            }))
        }
        Ok(false) => Ok(serde_json::json!({
            "installed": true,
            "message": "このプロファイルはモデル取得不要です"
        })),
        Err(CompressionError::Cancelled(_)) => {
            state.warmup.set(
                "cancelled",
                Some(profile.id),
                "モデル取得を中止しました。次回は続きから再開できます",
                None,
            );
            Ok(serde_json::json!({
                "installed": false,
                "cancelled": true,
                "message": "モデル取得を中止しました"
            }))
        }
        Err(error) => {
            state.warmup.set(
                "error",
                Some(profile.id),
                "モデル取得に失敗しました",
                Some(error.to_string()),
            );
            Err(error.into())
        }
    }
}

fn cancel_model_download_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let payload: ModelProfilePayload =
        serde_json::from_slice(body).context("invalid JSON model cancel request")?;
    validate_text_field("profile", &payload.profile, MAX_PROFILE_CHARS, false)?;
    let cancelled = state.model_downloads.cancel(&payload.profile)?;
    if cancelled {
        state.warmup.set(
            "cancelling",
            Some(payload.profile),
            "モデル取得を中止しています",
            None,
        );
    }
    Ok(serde_json::json!({
        "cancelled": cancelled,
        "message": if cancelled {
            "モデル取得の中止を受け付けました"
        } else {
            "実行中のモデル取得はありません"
        }
    }))
}

fn ensure_model_disk_capacity(status: &prompt_compressor_core::ProfileModelStatus) -> Result<()> {
    let Some(destination) = status.destination.as_deref() else {
        return Ok(());
    };
    let Some(available_bytes) = available_disk_space_bytes(destination) else {
        return Ok(());
    };
    let Some(required_bytes) = required_model_install_bytes(status) else {
        return Ok(());
    };
    if available_bytes < required_bytes {
        anyhow::bail!(
            "insufficient disk space for model installation: required {required_bytes} bytes, available {available_bytes} bytes"
        );
    }
    Ok(())
}

fn required_model_install_bytes(
    status: &prompt_compressor_core::ProfileModelStatus,
) -> Option<u64> {
    if status.installed {
        return None;
    }
    let size_bytes = status.size_bytes?;
    let downloaded_bytes = status.partial_downloaded_bytes.unwrap_or(0).min(size_bytes);
    Some(
        size_bytes
            .saturating_sub(downloaded_bytes)
            .saturating_add(INSTALL_HEADROOM_BYTES),
    )
}

#[cfg(target_os = "windows")]
fn available_disk_space_bytes(path: &Path) -> Option<u64> {
    let existing = path.ancestors().find(|candidate| candidate.exists())?;
    let volume_path = if existing.is_file() {
        existing.parent()?
    } else {
        existing
    };
    let absolute = fs::canonicalize(volume_path).ok()?;
    let wide = path_to_wide_null(&absolute);
    let mut available = 0u64;
    let success = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    (success != 0).then_some(available)
}

#[cfg(not(target_os = "windows"))]
fn available_disk_space_bytes(_path: &Path) -> Option<u64> {
    None
}

fn prepare_compression_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let payload: PrepareCompressionPayload =
        serde_json::from_slice(body).context("invalid JSON prepare request")?;
    validate_text_field("profile", &payload.profile, MAX_PROFILE_CHARS, false)?;
    let request = CompressionRequest {
        input_text: String::new(),
        compression_level: CompressionLevel::from_u8(payload.compression_level)
            .context("invalid compression level")?,
        profile: payload.profile,
        constraints: payload.constraints.unwrap_or_default(),
        target: RequestTarget::codex_default(),
        source: RequestSource::Desktop,
    };

    state.warmup.set(
        "loading",
        Some(request.profile.clone()),
        "圧縮プロンプトを準備中",
        None,
    );

    let started_at = Instant::now();
    match state.service.prepare(request.clone()) {
        Ok(prepared) => {
            let message = if prepared {
                "圧縮プロンプト準備完了"
            } else {
                "このモデルは事前準備対象外です"
            };
            state.warmup.set(
                if prepared { "ready" } else { "skipped" },
                Some(request.profile),
                message,
                None,
            );
            Ok(serde_json::json!({
                "prepared": prepared,
                "elapsed_ms": started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
                "message": message
            }))
        }
        Err(error) => {
            state.warmup.set(
                "error",
                Some(request.profile),
                "圧縮プロンプト準備でエラーが発生しました",
                Some(error.to_string()),
            );
            Err(error.into())
        }
    }
}

fn prepare_input_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let payload: CompressPayload =
        serde_json::from_slice(body).context("invalid JSON input prepare request")?;
    validate_text_field("input_text", &payload.input_text, MAX_PROMPT_CHARS, false)?;
    validate_text_field("profile", &payload.profile, MAX_PROFILE_CHARS, false)?;
    let request = CompressionRequest {
        input_text: payload.input_text,
        compression_level: CompressionLevel::from_u8(payload.compression_level)
            .context("invalid compression level")?,
        profile: payload.profile,
        constraints: payload.constraints.unwrap_or_default(),
        target: RequestTarget::codex_default(),
        source: RequestSource::Desktop,
    };

    let started_at = Instant::now();
    let prepared = state.service.prepare(request)?;
    Ok(serde_json::json!({
        "prepared": prepared,
        "elapsed_ms": started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
    }))
}

fn tune_runtime_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let payload: PrepareCompressionPayload =
        serde_json::from_slice(body).context("invalid JSON runtime tuning request")?;
    validate_text_field("profile", &payload.profile, MAX_PROFILE_CHARS, false)?;
    CompressionLevel::from_u8(payload.compression_level).context("invalid compression level")?;
    let profile = state.registry.resolve(&payload.profile)?;

    state
        .runtime_tuning_cancellation
        .store(false, Ordering::Relaxed);
    let started_at = Instant::now();
    let tuned = state
        .backend
        .tune_profile_threads(profile, &state.runtime_tuning_cancellation)?;
    let cpu_engine_tuned = if state.runtime_tuning_cancellation.load(Ordering::Relaxed)
        || !state
            .backend
            .profile_supports_embedded_cpu_tuning(profile)?
    {
        false
    } else {
        match tune_cpu_engine_for_next_launch(profile, state) {
            Ok(tuned) => tuned,
            Err(error) => {
                eprintln!("CPU engine tuning is unavailable: {error:#}");
                false
            }
        }
    };
    let cancelled = state.runtime_tuning_cancellation.load(Ordering::Relaxed);
    Ok(serde_json::json!({
        "completed": !cancelled,
        "tuned": tuned,
        "cpu_engine_tuned": cpu_engine_tuned,
        "restart_required": !cancelled && (tuned || cpu_engine_tuned),
        "elapsed_ms": started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
    }))
}

fn runtime_setup_status_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let payload: PrepareCompressionPayload =
        serde_json::from_slice(body).context("invalid JSON runtime setup status request")?;
    validate_text_field("profile", &payload.profile, MAX_PROFILE_CHARS, false)?;
    CompressionLevel::from_u8(payload.compression_level).context("invalid compression level")?;
    let profile = state.registry.resolve(&payload.profile)?;
    let model_status = state.backend.profile_model_status(profile)?;
    let model_ready = !model_status.requires_install || model_status.installed;
    if !model_ready {
        return Ok(serde_json::json!({
            "required": false,
            "model_ready": false,
            "thread_tuning_required": false,
            "cpu_engine_tuning_required": false
        }));
    }

    let thread_tuning_required = state.backend.profile_thread_tuning_required(profile)?;
    let cpu_engine_tuning_required = state
        .backend
        .profile_supports_embedded_cpu_tuning(profile)?
        && cpu_engine_tuning_required(&state.application_root);
    Ok(serde_json::json!({
        "required": thread_tuning_required || cpu_engine_tuning_required,
        "model_ready": true,
        "thread_tuning_required": thread_tuning_required,
        "cpu_engine_tuning_required": cpu_engine_tuning_required
    }))
}

fn tune_cpu_engine_for_next_launch(
    profile: &prompt_compressor_core::ProfileDefinition,
    state: &AppState,
) -> Result<bool> {
    if std::env::var(CPU_ENGINE_MODE_ENV).as_deref() == Ok("manual") {
        return Ok(false);
    }
    let build_id = match std::env::var("TRIMPROMPT_EXPECTED_BUILD_ID") {
        Ok(value) if !value.trim().is_empty() => value,
        _ => return Ok(false),
    };
    let inference_compatibility_id = match std::env::var(INFERENCE_COMPATIBILITY_ID_ENV) {
        Ok(value) if !value.trim().is_empty() => value,
        _ => return Ok(false),
    };
    if !cpu_supports_avx512_engine() {
        return Ok(false);
    }
    let current_engine = match std::env::var("TRIMPROMPT_CPU_ENGINE").as_deref() {
        Ok("avx2") => "avx2",
        Ok("avx512") => "avx512",
        _ => return Ok(false),
    };
    let selection_path = cpu_engine_selection_path(&state.application_root);
    let cpu_key = current_cpu_key();
    if valid_cpu_engine_selection(&selection_path, &inference_compatibility_id, &cpu_key) {
        return Ok(false);
    }

    let selected = match (|| -> Result<&'static str> {
        let current_elapsed =
            benchmark_cpu_engine_pipeline(profile, state, &state.runtime_tuning_cancellation)?
                .context("current CPU engine benchmark was cancelled")?;
        if state.runtime_tuning_cancellation.load(Ordering::Relaxed) {
            anyhow::bail!("CPU engine benchmark was cancelled");
        }

        let alternate_engine = if current_engine == "avx2" {
            "avx512"
        } else {
            "avx2"
        };
        let alternate_elapsed = run_external_cpu_engine_probe(
            &state.application_root,
            alternate_engine,
            &build_id,
            &inference_compatibility_id,
            &state.runtime_tuning_cancellation,
        )?
        .context("alternate CPU engine benchmark was cancelled")?;
        if state.runtime_tuning_cancellation.load(Ordering::Relaxed) {
            anyhow::bail!("CPU engine benchmark was cancelled");
        }

        let (avx2_elapsed, avx512_elapsed) = if current_engine == "avx2" {
            (current_elapsed, alternate_elapsed)
        } else {
            (alternate_elapsed, current_elapsed)
        };
        Ok(select_benchmarked_cpu_engine(avx2_elapsed, avx512_elapsed))
    })() {
        Ok(selected) => selected,
        Err(error) => {
            if state.runtime_tuning_cancellation.load(Ordering::Relaxed) {
                return Ok(false);
            }
            // 診断失敗を次回起動へ持ち越さず、今回安全に動いた版を固定する。
            eprintln!("CPU engine comparison failed; keeping {current_engine}: {error:#}");
            current_engine
        }
    };
    persist_cpu_engine_selection(
        &selection_path,
        &build_id,
        &inference_compatibility_id,
        &cpu_key,
        selected,
    )?;
    Ok(true)
}

fn persist_cpu_engine_selection(
    selection_path: &Path,
    build_id: &str,
    inference_compatibility_id: &str,
    cpu_key: &str,
    selected: &str,
) -> Result<()> {
    let record = PersistedCpuEngineSelection {
        schema_version: CPU_ENGINE_SELECTION_SCHEMA_VERSION,
        build_id: build_id.to_string(),
        inference_compatibility_id: inference_compatibility_id.to_string(),
        cpu_key: cpu_key.to_string(),
        cpu_engine: selected.to_string(),
    };
    let bytes =
        serde_json::to_vec_pretty(&record).context("failed to serialize CPU engine selection")?;
    if let Some(parent) = selection_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    atomic_write_file(selection_path, &bytes)?;
    Ok(())
}

fn benchmark_cpu_engine_pipeline(
    profile: &ProfileDefinition,
    state: &AppState,
    cancellation: &AtomicBool,
) -> Result<Option<CpuEngineBenchmarkResult>> {
    anyhow::ensure!(
        state.backend.warm_profile(profile)?,
        "CPU engine benchmark requires an embedded model"
    );
    let warmup_request = CompressionRequest {
        input_text: String::new(),
        compression_level: CompressionLevel::from_u8(2)?,
        profile: profile.id.clone(),
        constraints: CompressionConstraints::default(),
        target: RequestTarget::codex_default(),
        source: RequestSource::Desktop,
    };
    state.service.prepare(warmup_request)?;

    let mut total_elapsed = Duration::ZERO;
    let mut quality_passed = true;
    for case in runtime_quality_probe_cases() {
        if cancellation.load(Ordering::Relaxed) {
            return Ok(None);
        }

        let request = CompressionRequest {
            input_text: case.input_text.to_string(),
            compression_level: CompressionLevel::from_u8(2)?,
            profile: profile.id.clone(),
            constraints: CompressionConstraints::default(),
            target: RequestTarget::codex_default(),
            source: RequestSource::Desktop,
        };
        let started_at = Instant::now();
        state.service.prepare(request.clone())?;
        let observed = state.service.compress_with_observation(request)?;
        anyhow::ensure!(
            observed.runtime_observation.is_some(),
            "CPU engine benchmark inference did not complete"
        );
        let missing = runtime_quality_missing_requirements(
            *case,
            &observed.result.distilled_prompt,
            observed.result.should_send_original,
        );
        if !missing.is_empty() {
            quality_passed = false;
            eprintln!(
                "CPU engine quality probe '{}' missed requirements: {}",
                case.id,
                missing.join(", ")
            );
        }
        total_elapsed += started_at.elapsed();
    }

    let average_micros = total_elapsed.as_micros()
        / u128::try_from(runtime_quality_probe_cases().len()).unwrap_or(1);
    Ok(Some(CpuEngineBenchmarkResult {
        elapsed_micros: average_micros.max(1).min(u128::from(u64::MAX)) as u64,
        quality_passed,
        quality_case_count: runtime_quality_probe_cases().len() as u32,
    }))
}

fn select_benchmarked_cpu_engine(
    avx2: CpuEngineBenchmarkResult,
    avx512: CpuEngineBenchmarkResult,
) -> &'static str {
    if avx512.quality_passed && !avx2.quality_passed {
        "avx512"
    } else if !avx512.quality_passed {
        "avx2"
    } else {
        let avx512_limit =
            u128::from(avx2.elapsed_micros) * u128::from(100 - CPU_ENGINE_MINIMUM_GAIN_PERCENT);
        if u128::from(avx512.elapsed_micros) * 100 <= avx512_limit {
            "avx512"
        } else {
            "avx2"
        }
    }
}

fn run_external_cpu_engine_probe(
    application_root: &Path,
    engine: &str,
    build_id: &str,
    inference_compatibility_id: &str,
    cancellation: &AtomicBool,
) -> Result<Option<CpuEngineBenchmarkResult>> {
    let engine_root = fs::canonicalize(application_root.join("runtime").join("cpu"))
        .context("CPU engine directory is unavailable")?;
    let executable = fs::canonicalize(engine_root.join(format!("TrimPrompt-{engine}.exe")))
        .with_context(|| format!("{engine} CPU engine is unavailable"))?;
    anyhow::ensure!(
        executable.starts_with(&engine_root) && executable.is_file(),
        "CPU engine path is outside the package"
    );

    let result_path = cpu_engine_probe_path(application_root, engine);
    match fs::remove_file(&result_path) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let package_root = application_root
        .parent()
        .context("application directory has no package root")?;
    let mut child = Command::new(&executable)
        .arg("--cpu-engine-speed-probe")
        .current_dir(package_root)
        .env("TRIMPROMPT_CPU_ENGINE", engine)
        .env("TRIMPROMPT_EXPECTED_BUILD_ID", build_id)
        .env(INFERENCE_COMPATIBILITY_ID_ENV, inference_compatibility_id)
        .spawn()
        .with_context(|| format!("failed to start {}", executable.display()))?;

    loop {
        if cancellation.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(None);
        }
        if let Some(status) = child
            .try_wait()
            .context("failed to poll CPU engine probe")?
        {
            anyhow::ensure!(status.success(), "{engine} CPU engine probe failed");
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }

    let bytes = fs::read(&result_path)
        .with_context(|| format!("failed to read {}", result_path.display()))?;
    let result: CpuEngineProbeResult =
        serde_json::from_slice(&bytes).context("invalid CPU engine probe result")?;
    let _ = fs::remove_file(result_path);
    anyhow::ensure!(
        result.schema_version == CPU_ENGINE_SELECTION_SCHEMA_VERSION
            && result.build_id == build_id
            && result.inference_compatibility_id == inference_compatibility_id
            && result.cpu_engine == engine
            && result.elapsed_micros > 0
            && result.quality_case_count == runtime_quality_probe_cases().len() as u32,
        "CPU engine probe result does not match the running package"
    );
    Ok(Some(CpuEngineBenchmarkResult {
        elapsed_micros: result.elapsed_micros,
        quality_passed: result.quality_passed,
        quality_case_count: result.quality_case_count,
    }))
}

fn valid_cpu_engine_selection(
    path: &Path,
    inference_compatibility_id: &str,
    cpu_key: &str,
) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    let Ok(record) = serde_json::from_slice::<PersistedCpuEngineSelection>(&bytes) else {
        return false;
    };
    record.schema_version == CPU_ENGINE_SELECTION_SCHEMA_VERSION
        && record.inference_compatibility_id == inference_compatibility_id
        && record.cpu_key == cpu_key
        && matches!(record.cpu_engine.as_str(), "avx2" | "avx512")
}

fn cpu_engine_tuning_required(application_root: &Path) -> bool {
    if std::env::var(CPU_ENGINE_MODE_ENV).as_deref() == Ok("manual") {
        return false;
    }
    let Ok(inference_compatibility_id) = std::env::var(INFERENCE_COMPATIBILITY_ID_ENV) else {
        return false;
    };
    if inference_compatibility_id.trim().is_empty() || !cpu_supports_avx512_engine() {
        return false;
    }
    if !matches!(
        std::env::var("TRIMPROMPT_CPU_ENGINE").as_deref(),
        Ok("avx2" | "avx512")
    ) {
        return false;
    }

    !valid_cpu_engine_selection(
        &cpu_engine_selection_path(application_root),
        &inference_compatibility_id,
        &current_cpu_key(),
    )
}

fn cpu_engine_selection_path(application_root: &Path) -> PathBuf {
    application_root
        .join("local")
        .join("state")
        .join(CPU_ENGINE_SELECTION_FILE)
}

fn cpu_engine_probe_path(application_root: &Path, engine: &str) -> PathBuf {
    application_root
        .join("local")
        .join("state")
        .join(format!("cpu-engine-probe-{engine}.json"))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpu_supports_avx512_engine() -> bool {
    cpu_engine_supported(detect_cpu_instruction_capabilities(), "avx512")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn cpu_supports_avx512_engine() -> bool {
    false
}

fn cpu_engine_supported(capabilities: CpuInstructionCapabilities, engine: &str) -> bool {
    let compatible = capabilities.sse42;
    let avx2 = compatible
        && capabilities.avx2
        && capabilities.fma
        && capabilities.f16c
        && capabilities.bmi2;
    match engine {
        "auto" => true,
        "compatible" => compatible,
        "avx2" => avx2,
        "avx512" => {
            avx2 && capabilities.avx512f
                && capabilities.avx512cd
                && capabilities.avx512bw
                && capabilities.avx512dq
                && capabilities.avx512vl
        }
        _ => false,
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_cpu_instruction_capabilities() -> CpuInstructionCapabilities {
    CpuInstructionCapabilities {
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
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_cpu_instruction_capabilities() -> CpuInstructionCapabilities {
    CpuInstructionCapabilities::default()
}

fn current_cpu_engine() -> &'static str {
    match std::env::var("TRIMPROMPT_CPU_ENGINE").as_deref() {
        Ok("compatible") => "compatible",
        Ok("avx2") => "avx2",
        Ok("avx512") => "avx512",
        _ if cfg!(feature = "embedded-llama-avx512") => "avx512",
        _ if cfg!(feature = "embedded-llama-avx2") => "avx2",
        _ => "compatible",
    }
}

fn current_cpu_key() -> String {
    let processor = std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".to_string());
    let capabilities = detect_cpu_instruction_capabilities();
    format!(
        "{processor}|sse42={}|avx2={}|fma={}|f16c={}|bmi2={}|avx512f={}|avx512cd={}|avx512bw={}|avx512dq={}|avx512vl={}",
        u8::from(capabilities.sse42),
        u8::from(capabilities.avx2),
        u8::from(capabilities.fma),
        u8::from(capabilities.f16c),
        u8::from(capabilities.bmi2),
        u8::from(capabilities.avx512f),
        u8::from(capabilities.avx512cd),
        u8::from(capabilities.avx512bw),
        u8::from(capabilities.avx512dq),
        u8::from(capabilities.avx512vl),
    )
}

fn reset_runtime_tuning_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let payload: PrepareCompressionPayload =
        serde_json::from_slice(body).context("invalid JSON runtime tuning reset request")?;
    validate_text_field("profile", &payload.profile, MAX_PROFILE_CHARS, false)?;
    CompressionLevel::from_u8(payload.compression_level).context("invalid compression level")?;
    let profile = state.registry.resolve(&payload.profile)?;
    let removed = state.backend.reset_profile_thread_tuning(profile)?;
    let engine_selection_path = cpu_engine_selection_path(&state.application_root);
    let engine_removed = match fs::remove_file(engine_selection_path) {
        Ok(()) => true,
        Err(error) if error.kind() == ErrorKind::NotFound => false,
        Err(error) => return Err(error.into()),
    };
    Ok(serde_json::json!({
        "reset": true,
        "removed": removed,
        "cpu_engine_removed": engine_removed
    }))
}

fn compress_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let payload: CompressPayload = serde_json::from_slice(body).context("invalid JSON request")?;
    validate_text_field("input_text", &payload.input_text, MAX_PROMPT_CHARS, false)?;
    validate_text_field("profile", &payload.profile, MAX_PROFILE_CHARS, false)?;
    let request = CompressionRequest {
        input_text: payload.input_text,
        compression_level: CompressionLevel::from_u8(payload.compression_level)
            .context("invalid compression level")?,
        profile: payload.profile,
        constraints: payload.constraints.unwrap_or_default(),
        target: RequestTarget::codex_default(),
        source: RequestSource::Desktop,
    };

    let result = state.service.compress(request)?;
    Ok(serde_json::to_value(result)?)
}

fn validate_text_field(
    field: &str,
    value: &str,
    max_chars: usize,
    allow_empty: bool,
) -> Result<()> {
    if !allow_empty && value.trim().is_empty() {
        anyhow::bail!("{field} is empty");
    }
    if value.chars().count() > max_chars {
        anyhow::bail!("{field} exceeds {max_chars} characters");
    }
    if value.contains('\0') {
        anyhow::bail!("{field} contains a null character");
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn write_text_to_clipboard(text: &str) -> Result<bool> {
    let mut wide = to_wide_null(text);
    let byte_len = wide.len() * std::mem::size_of::<u16>();

    unsafe {
        if OpenClipboard(null_mut()) == 0 {
            anyhow::bail!("OpenClipboard failed: {}", GetLastError());
        }

        let result = (|| -> Result<bool> {
            if EmptyClipboard() == 0 {
                anyhow::bail!("EmptyClipboard failed: {}", GetLastError());
            }

            let handle = GlobalAlloc(GMEM_MOVEABLE, byte_len);
            if handle.is_null() {
                anyhow::bail!("GlobalAlloc for clipboard failed: {}", GetLastError());
            }

            let locked = GlobalLock(handle) as *mut u16;
            if locked.is_null() {
                anyhow::bail!("GlobalLock for clipboard failed: {}", GetLastError());
            }

            std::ptr::copy_nonoverlapping(wide.as_mut_ptr(), locked, wide.len());
            GlobalUnlock(handle);

            if SetClipboardData(CF_UNICODETEXT as u32, handle).is_null() {
                anyhow::bail!("SetClipboardData failed: {}", GetLastError());
            }

            Ok(true)
        })();

        CloseClipboard();
        result
    }
}

#[cfg(not(target_os = "windows"))]
fn write_text_to_clipboard(_text: &str) -> Result<bool> {
    Ok(false)
}

fn profiles_json(registry: &ProfileRegistry) -> serde_json::Value {
    let profiles: Vec<_> = registry
        .list_selectable()
        .into_iter()
        .map(|profile| {
            serde_json::json!({
                "id": profile.id,
                "label": profile.label,
                "model_ref": profile.model_ref,
                "policy_ref": profile.policy_ref,
                "runtime_ref": profile.runtime_ref,
                "fallback_profile": profile.fallback_profile,
                "target_tokenizer_profile": profile.target_tokenizer_profile,
            })
        })
        .collect();

    serde_json::json!({
        "default_profile": registry.default_profile_id(),
        "profiles": profiles
    })
}

fn read_request(stream: &mut impl Read) -> std::result::Result<HttpRequest, HttpReadError> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 4096];

    loop {
        let bytes_read = stream.read(&mut chunk).map_err(http_read_io_error)?;
        if bytes_read == 0 {
            return Err(HttpReadError::new(400, "incomplete HTTP headers"));
        }
        buffer.extend_from_slice(&chunk[..bytes_read]);
        if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
        if buffer.len() > MAX_HEADER_BYTES {
            return Err(HttpReadError::new(431, "request headers are too large"));
        }
    }

    let header_end = find_header_end(&buffer)
        .ok_or_else(|| HttpReadError::new(400, "malformed HTTP request"))?;
    if header_end > MAX_HEADER_BYTES {
        return Err(HttpReadError::new(431, "request headers are too large"));
    }
    let header_text = std::str::from_utf8(&buffer[..header_end])
        .map_err(|_| HttpReadError::new(400, "request headers are not UTF-8"))?;
    let mut lines = header_text.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| HttpReadError::new(400, "missing request line"))?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| HttpReadError::new(400, "missing method"))?
        .to_string();
    let raw_path = request_parts
        .next()
        .ok_or_else(|| HttpReadError::new(400, "missing path"))?;
    let version = request_parts
        .next()
        .ok_or_else(|| HttpReadError::new(400, "missing HTTP version"))?;
    if request_parts.next().is_some() || version != "HTTP/1.1" {
        return Err(HttpReadError::new(400, "unsupported request line"));
    }
    if !method
        .chars()
        .all(|character| character.is_ascii_uppercase())
        || !raw_path.starts_with('/')
        || raw_path.len() > MAX_REQUEST_PATH_BYTES
    {
        return Err(HttpReadError::new(400, "invalid request target"));
    }
    let path = raw_path.split('?').next().unwrap_or(raw_path).to_string();
    let headers = parse_headers(lines)?;
    if headers.contains_key("transfer-encoding") {
        return Err(HttpReadError::new(
            400,
            "transfer encoding is not supported",
        ));
    }
    let content_length = match headers.get("content-length") {
        Some(value)
            if !value.is_empty() && value.chars().all(|character| character.is_ascii_digit()) =>
        {
            value
                .parse::<usize>()
                .map_err(|_| HttpReadError::new(400, "invalid content length"))?
        }
        Some(_) => return Err(HttpReadError::new(400, "invalid content length")),
        None if matches!(method.as_str(), "POST" | "PUT" | "PATCH") => {
            return Err(HttpReadError::new(411, "content length is required"));
        }
        None => 0,
    };
    if content_length > request_body_limit(&path).min(MAX_REQUEST_BODY_BYTES) {
        return Err(HttpReadError::new(413, "request body is too large"));
    }

    let body_start = header_end + 4;
    let mut body = buffer[body_start..].to_vec();
    if body.len() > content_length {
        return Err(HttpReadError::new(
            400,
            "multiple HTTP requests are not supported",
        ));
    }
    while body.len() < content_length {
        let bytes_read = stream.read(&mut chunk).map_err(http_read_io_error)?;
        if bytes_read == 0 {
            return Err(HttpReadError::new(400, "incomplete request body"));
        }
        if body.len() + bytes_read > content_length {
            return Err(HttpReadError::new(
                400,
                "request body exceeds content length",
            ));
        }
        body.extend_from_slice(&chunk[..bytes_read]);
    }

    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_headers<'a>(
    lines: impl Iterator<Item = &'a str>,
) -> std::result::Result<BTreeMap<String, String>, HttpReadError> {
    let mut headers = BTreeMap::new();
    for line in lines {
        if line.starts_with([' ', '\t']) {
            return Err(HttpReadError::new(400, "folded headers are not supported"));
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| HttpReadError::new(400, "malformed request header"))?;
        if name.is_empty()
            || !name
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || character == '-')
        {
            return Err(HttpReadError::new(400, "invalid request header name"));
        }
        let name = name.to_ascii_lowercase();
        if headers.insert(name, value.trim().to_string()).is_some() {
            return Err(HttpReadError::new(400, "duplicate request header"));
        }
    }
    Ok(headers)
}

fn http_read_io_error(error: std::io::Error) -> HttpReadError {
    if matches!(error.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock) {
        HttpReadError::new(408, "request timed out")
    } else {
        HttpReadError::new(400, "failed to read request")
    }
}

fn validate_network_request(
    request: &HttpRequest,
    server_address: SocketAddr,
) -> std::result::Result<(), HttpReadError> {
    let host = request
        .headers
        .get("host")
        .ok_or_else(|| HttpReadError::new(403, "local host header is required"))?;
    if !authority_matches_local_server(host, server_address.port()) {
        return Err(HttpReadError::new(403, "host is not allowed"));
    }

    if matches!(request.method.as_str(), "POST" | "PUT" | "PATCH" | "DELETE") {
        let origin = request
            .headers
            .get("origin")
            .ok_or_else(|| HttpReadError::new(403, "same-origin request is required"))?;
        let authority = origin
            .strip_prefix("http://")
            .filter(|authority| !authority.contains('/'))
            .ok_or_else(|| HttpReadError::new(403, "origin is not allowed"))?;
        if !authority_matches_local_server(authority, server_address.port()) {
            return Err(HttpReadError::new(403, "origin is not allowed"));
        }
    }
    Ok(())
}

fn authority_matches_local_server(authority: &str, expected_port: u16) -> bool {
    let Some((host, port)) = split_authority(authority) else {
        return false;
    };
    port == expected_port && is_loopback_host(host)
}

fn split_authority(authority: &str) -> Option<(&str, u16)> {
    if let Some(rest) = authority.strip_prefix('[') {
        let closing = rest.find(']')?;
        let host = &rest[..closing];
        let port = rest[closing + 1..].strip_prefix(':')?.parse().ok()?;
        return Some((host, port));
    }
    let (host, port) = authority.rsplit_once(':')?;
    Some((host, port.parse().ok()?))
}

fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn json_response(status: u16, value: &serde_json::Value) -> LocalAppResponse {
    match serde_json::to_vec_pretty(value) {
        Ok(body) => http_response(status, "application/json; charset=utf-8", &body),
        Err(error) => http_response(
            500,
            "application/json; charset=utf-8",
            format!(r#"{{"error":"failed to serialize response: {error}"}}"#).as_bytes(),
        ),
    }
}

#[cfg(target_os = "windows")]
fn to_wide_null(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

fn http_response(status: u16, content_type: &str, body: &[u8]) -> LocalAppResponse {
    LocalAppResponse {
        status,
        content_type: content_type.to_string(),
        body: body.to_vec(),
    }
}

fn http_transport_response(response: LocalAppResponse) -> Vec<u8> {
    let status = response.status;
    let content_type = response.content_type;
    let body = response.body;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        408 => "Request Timeout",
        411 => "Length Required",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let mut response = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
         Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'\r\n\
         Referrer-Policy: no-referrer\r\n\
         X-Content-Type-Options: nosniff\r\n\
         X-Frame-Options: DENY\r\n\
         Connection: close\r\n\r\n",
        body.len(),
        status = status,
        reason = reason,
        content_type = content_type
    )
    .into_bytes();
    response.extend_from_slice(&body);
    response
}

pub fn resolve_settings_dir(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }

    let cwd = std::env::current_dir().context("failed to determine current directory")?;
    if let Some(settings_dir) = find_upward_settings_dir(&cwd) {
        return Ok(settings_dir);
    }

    let exe = std::env::current_exe().context("failed to determine current executable path")?;
    if let Some(parent) = exe.parent() {
        if let Some(settings_dir) = find_upward_settings_dir(parent) {
            return Ok(settings_dir);
        }
    }

    Err(anyhow::anyhow!(
        "could not find ./application/config directory from current directory {} or executable {}",
        cwd.display(),
        exe.display()
    ))
}

fn find_upward_settings_dir(start: &Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        for candidate in [
            ancestor.join("config"),
            ancestor.join("application").join("config"),
        ] {
            if candidate.is_dir() {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::sync::Barrier;

    use super::*;

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let counter = SETTINGS_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "prompt-compressor-{label}-{}-{counter}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create test directory");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn settings_temp_files(directory: &Path) -> Vec<PathBuf> {
        fs::read_dir(directory)
            .expect("read settings directory")
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.ends_with(".tmp"))
            })
            .collect()
    }

    fn benchmark_result(elapsed_micros: u64, quality_passed: bool) -> CpuEngineBenchmarkResult {
        CpuEngineBenchmarkResult {
            elapsed_micros,
            quality_passed,
            quality_case_count: runtime_quality_probe_cases().len() as u32,
        }
    }

    #[test]
    fn accepts_only_loopback_bind_hosts() {
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("127.42.0.1"));
        assert!(is_loopback_host("::1"));
        assert!(!is_loopback_host("0.0.0.0"));
        assert!(!is_loopback_host("192.168.1.10"));
        assert!(!is_loopback_host("example.com"));
    }

    #[test]
    fn parses_bounded_http_request_with_unambiguous_length() {
        let body = br#"{"input_text":"hello"}"#;
        let raw = format!(
            "POST /api/compress HTTP/1.1\r\nHost: 127.0.0.1:8787\r\nOrigin: http://127.0.0.1:8787\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            String::from_utf8_lossy(body)
        );

        let request = read_request(&mut Cursor::new(raw.into_bytes())).expect("valid request");

        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/api/compress");
        assert_eq!(request.body, body);
        assert_eq!(request.headers["host"], "127.0.0.1:8787");
    }

    #[test]
    fn rejects_oversized_and_ambiguous_http_requests_before_parsing_json() {
        let oversized = format!(
            "POST /api/compress HTTP/1.1\r\nHost: 127.0.0.1:8787\r\nContent-Length: {}\r\n\r\n",
            MAX_REQUEST_BODY_BYTES + 1
        );
        let duplicate = "POST /api/compress HTTP/1.1\r\nHost: 127.0.0.1:8787\r\nContent-Length: 0\r\nContent-Length: 1\r\n\r\n";
        let chunked = "POST /api/compress HTTP/1.1\r\nHost: 127.0.0.1:8787\r\nTransfer-Encoding: chunked\r\nContent-Length: 0\r\n\r\n";

        let oversized_error =
            read_request(&mut Cursor::new(oversized.into_bytes())).expect_err("oversized request");
        let duplicate_error =
            read_request(&mut Cursor::new(duplicate.as_bytes())).expect_err("duplicate length");
        let chunked_error =
            read_request(&mut Cursor::new(chunked.as_bytes())).expect_err("chunked request");

        assert_eq!(oversized_error.status, 413);
        assert_eq!(duplicate_error.status, 400);
        assert_eq!(chunked_error.status, 400);
    }

    #[test]
    fn enforces_local_host_and_same_origin_for_mutating_requests() {
        let server = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8787);
        let mut request = HttpRequest {
            method: "POST".to_string(),
            path: "/api/compress".to_string(),
            headers: BTreeMap::from([
                ("host".to_string(), "localhost:8787".to_string()),
                ("origin".to_string(), "http://127.0.0.1:8787".to_string()),
            ]),
            body: Vec::new(),
        };

        assert!(validate_network_request(&request, server).is_ok());

        request
            .headers
            .insert("origin".to_string(), "https://attacker.example".to_string());
        assert_eq!(
            validate_network_request(&request, server)
                .expect_err("foreign origin")
                .status,
            403
        );

        request
            .headers
            .insert("origin".to_string(), "http://localhost:8787".to_string());
        request
            .headers
            .insert("host".to_string(), "attacker.example:8787".to_string());
        assert_eq!(
            validate_network_request(&request, server)
                .expect_err("foreign host")
                .status,
            403
        );
    }

    #[test]
    fn rejects_oversized_fields_and_null_characters() {
        assert!(validate_text_field("profile", "internal_llm", MAX_PROFILE_CHARS, false).is_ok());
        assert!(validate_text_field("profile", "", MAX_PROFILE_CHARS, false).is_err());
        assert!(validate_text_field("profile", "bad\0value", MAX_PROFILE_CHARS, false).is_err());
        assert!(validate_text_field(
            "profile",
            &"x".repeat(MAX_PROFILE_CHARS + 1),
            MAX_PROFILE_CHARS,
            false
        )
        .is_err());
    }

    #[test]
    fn inference_gate_is_shared_and_non_blocking() {
        let gate = InferenceGate::default();
        let cloned = gate.clone();
        let permit = gate.try_enter().expect("first permit");

        assert_eq!(
            cloned.try_enter().expect_err("busy gate"),
            InferenceGateError::Busy
        );

        drop(permit);
        assert!(cloned.try_enter().is_ok());
    }

    #[test]
    fn model_download_control_cancels_only_the_active_profile() {
        let control = ModelDownloadControl::default();
        let permit = control
            .start("internal_llm".to_string())
            .expect("start model download");

        assert!(control.cancel("other_profile").is_err());
        assert!(control
            .cancel("internal_llm")
            .expect("cancel active download"));
        assert!(permit.cancellation().is_cancelled());

        drop(permit);
        assert!(!control
            .cancel("internal_llm")
            .expect("no active download after permit drop"));
    }

    #[test]
    fn install_capacity_counts_only_bytes_missing_from_a_partial_download() {
        let status = prompt_compressor_core::ProfileModelStatus {
            profile: "internal_llm".into(),
            model_id: "model".into(),
            label: "Model".into(),
            requires_install: true,
            installed: false,
            repository: Some("owner/model".into()),
            revision: Some("a".repeat(40)),
            filename: Some("model.gguf".into()),
            size_bytes: Some(2 * 1024 * 1024 * 1024),
            partial_downloaded_bytes: Some(768 * 1024 * 1024),
            destination: Some(PathBuf::from("model.gguf")),
        };

        assert_eq!(
            required_model_install_bytes(&status),
            Some(1792 * 1024 * 1024)
        );

        let installed_status = prompt_compressor_core::ProfileModelStatus {
            installed: true,
            ..status
        };
        assert_eq!(required_model_install_bytes(&installed_status), None);
    }

    #[test]
    fn transport_response_sets_browser_security_headers() {
        let response = String::from_utf8(http_transport_response(http_response(
            200,
            "text/html; charset=utf-8",
            b"ok",
        )))
        .expect("HTTP response");

        assert!(response.contains("Content-Security-Policy:"));
        assert!(response.contains("X-Content-Type-Options: nosniff"));
        assert!(response.contains("X-Frame-Options: DENY"));
        assert!(response.contains("Referrer-Policy: no-referrer"));
    }

    #[test]
    fn embedded_ui_requires_explicit_model_install_and_preserves_fallback_copy_choice() {
        assert!(INDEX_HTML.contains("id=\"modelInstallButton\""));
        assert!(APP_JS.contains("fetch(\"/api/model-install\""));
        assert!(INDEX_HTML.contains("id=\"modelCancelButton\""));
        assert!(APP_JS.contains("fetch(\"/api/model-cancel\""));
        assert!(APP_JS.contains("if (payload.should_send_original)"));
        assert!(APP_JS.contains("自動コピーは行っていません"));
        assert!(!APP_JS.contains("navigator.clipboard.writeText(promptOutput.value)"));
        let native_copy = APP_JS
            .find("if (await copyTextWithNativeApi(text))")
            .expect("native clipboard fallback");
        let browser_copy = APP_JS
            .find("await navigator.clipboard.writeText(text)")
            .expect("browser clipboard fallback");
        assert!(native_copy < browser_copy);
        assert!(APP_JS.contains("let modelRuntimeReady = false"));
        assert!(APP_JS.contains(
            "compressButton.disabled = isCompressing || !modelInstalled || !modelRuntimeReady"
        ));
        assert!(!APP_JS.contains("/api/windows-notification"));
        assert!(APP_JS.contains("settingsSaveDelayMs = 250"));
        assert!(APP_JS.contains("scheduleSettingsSave(settings)"));
    }

    #[test]
    fn input_pause_prepares_model_state_without_reusing_compression_results() {
        assert_eq!(
            request_body_limit("/api/prepare-input"),
            MAX_REQUEST_BODY_BYTES
        );
        assert!(APP_JS.contains("inputPrepareDelayMs = 450"));
        assert!(APP_JS.contains("pastedInputPrepareDelayMs = 50"));
        assert!(APP_JS.contains("promptInput.addEventListener(\"input\""));
        assert!(APP_JS.contains("fetch(\"/api/prepare-input\""));
        assert!(APP_JS.contains("await inputPreparePromise"));
        assert!(APP_JS.contains("readyInputPrepareKey = \"\";"));
        assert!(APP_JS.contains("fetch(\"/api/compress\""));
    }

    #[test]
    fn runtime_tuning_uses_an_explicit_startup_gate_and_restart() {
        assert_eq!(
            request_body_limit("/api/tune-runtime"),
            MAX_SETTINGS_BODY_BYTES
        );
        assert_eq!(
            request_body_limit("/api/runtime-setup-status"),
            MAX_SETTINGS_BODY_BYTES
        );
        assert!(APP_JS.contains("fetch(\"/api/tune-runtime\""));
        assert!(APP_JS.contains("fetch(\"/api/runtime-setup-status\""));
        assert!(APP_JS.contains("fetch(\"/api/tune-runtime-reset\""));
        assert!(APP_JS.contains("postDesktopMessage(\"app:restart\")"));
        assert!(!APP_JS.contains("runtimeTuningIdleDelayMs"));
        assert!(!APP_JS.contains("tune-runtime-cancel"));
        assert!(!APP_JS.contains("document.addEventListener(\"pointerdown\", deferRuntimeTuning"));
        assert!(INDEX_HTML.contains("id=\"runtimeTuningResetButton\""));
        assert!(INDEX_HTML.contains("id=\"runtimeSetupScreen\""));
        assert!(INDEX_HTML.contains("初期設定中") || APP_JS.contains("初期設定中"));
        assert!(APP_JS.contains("完了まで少し時間がかかります"));
    }

    #[test]
    fn advanced_runtime_settings_expose_safe_manual_controls_and_current_values() {
        assert_eq!(
            request_body_limit("/api/runtime-configuration"),
            MAX_SETTINGS_BODY_BYTES
        );
        for control in [
            "runtimeDetails",
            "cpuEngineSelect",
            "threadModeSelect",
            "generationThreadsInput",
            "batchThreadsInput",
            "runtimeCurrentValue",
            "runtimeApplyButton",
        ] {
            assert!(
                INDEX_HTML.contains(&format!("id=\"{control}\"")),
                "missing advanced runtime control {control}"
            );
        }
        assert!(APP_JS.contains("fetch(\"/api/runtime-configuration\""));
        assert!(APP_JS.contains("option.disabled = availability ? !availability.supported"));
        assert!(APP_JS.contains("postDesktopMessage(\"app:restart\")"));
        assert!(APP_JS.contains("runtimeConfiguration?.available_threads"));
        assert!(APP_JS.contains("runtimeConfiguration.logical_batch_size"));
        assert!(APP_JS.contains("runtimeConfiguration.physical_batch_size"));
        assert!(STYLES_CSS.contains(".runtime-details"));
        assert!(STYLES_CSS.contains(".thread-inputs[hidden]"));
    }

    #[test]
    fn cpu_engine_support_requires_every_instruction_used_by_each_binary() {
        let avx2 = CpuInstructionCapabilities {
            sse42: true,
            avx2: true,
            fma: true,
            f16c: true,
            bmi2: true,
            ..Default::default()
        };
        assert!(cpu_engine_supported(avx2, "compatible"));
        assert!(cpu_engine_supported(avx2, "avx2"));
        assert!(!cpu_engine_supported(avx2, "avx512"));

        let avx512 = CpuInstructionCapabilities {
            avx512f: true,
            avx512cd: true,
            avx512bw: true,
            avx512dq: true,
            avx512vl: true,
            ..avx2
        };
        assert!(cpu_engine_supported(avx512, "avx512"));
        assert!(!cpu_engine_supported(
            CpuInstructionCapabilities {
                bmi2: false,
                ..avx512
            },
            "avx2"
        ));
        assert!(!cpu_engine_supported(avx512, "unknown"));
    }

    #[test]
    fn runtime_settings_reject_incomplete_or_excessive_manual_threads() {
        assert!(validate_runtime_preferences(&UiSettings {
            cpu_engine: Some("auto".into()),
            thread_mode: Some("auto".into()),
            ..UiSettings::default()
        })
        .is_ok());
        assert!(validate_runtime_preferences(&UiSettings {
            cpu_engine: Some("auto".into()),
            thread_mode: Some("manual".into()),
            generation_threads: Some(1),
            batch_threads: None,
            ..UiSettings::default()
        })
        .is_err());
        assert!(validate_runtime_preferences(&UiSettings {
            cpu_engine: Some("auto".into()),
            thread_mode: Some("manual".into()),
            generation_threads: Some(u32::MAX),
            batch_threads: Some(1),
            ..UiSettings::default()
        })
        .is_err());
    }

    #[test]
    fn avx512_requires_a_measurable_gain_over_avx2() {
        assert_eq!(
            select_benchmarked_cpu_engine(
                benchmark_result(1_000, true),
                benchmark_result(970, true)
            ),
            "avx512"
        );
        assert_eq!(
            select_benchmarked_cpu_engine(
                benchmark_result(1_000, true),
                benchmark_result(971, true)
            ),
            "avx2"
        );
        assert_eq!(
            select_benchmarked_cpu_engine(
                benchmark_result(1_000, true),
                benchmark_result(900, false)
            ),
            "avx2"
        );
        assert_eq!(
            select_benchmarked_cpu_engine(
                benchmark_result(1_000, false),
                benchmark_result(1_100, true)
            ),
            "avx512"
        );
    }

    #[test]
    fn cpu_engine_probe_uses_five_representative_inputs() {
        assert_eq!(runtime_quality_probe_cases().len(), 5);
    }

    #[test]
    fn cpu_engine_quality_checks_requirements_instead_of_text_identity() {
        let case = runtime_quality_probe_cases()[1];
        let concise_output = "ユーザー登録API: メールアドレス形式とパスワード12文字以上を検証。不正時はHTTP 400とし、データベースへ書き込まない。正常・異常テストを追加。";
        assert_ne!(concise_output, case.input_text);
        assert!(runtime_quality_missing_requirements(case, concise_output, false).is_empty());

        let incomplete = concise_output.replace("HTTP 400", "エラー");
        assert!(!runtime_quality_missing_requirements(case, &incomplete, false).is_empty());
        assert!(!runtime_quality_missing_requirements(case, concise_output, true).is_empty());

        let monitoring_case = runtime_quality_probe_cases()[4];
        let paraphrased_monitoring = "5分単位の個人情報除去済みエラー件数と95パーセンタイル応答時間を集計。30秒超過時は中断し、元データを削除しない。";
        assert!(runtime_quality_missing_requirements(
            monitoring_case,
            paraphrased_monitoring,
            false
        )
        .is_empty());
        let unordered_privacy =
            paraphrased_monitoring.replace("個人情報除去済み", "個人情報を扱う");
        assert!(
            !runtime_quality_missing_requirements(monitoring_case, &unordered_privacy, false)
                .is_empty()
        );
    }

    #[test]
    fn persisted_cpu_engine_selection_completes_the_setup_gate() {
        let directory = TestDirectory::new("cpu-engine-selection");
        let selection_path = directory.path().join(CPU_ENGINE_SELECTION_FILE);

        persist_cpu_engine_selection(&selection_path, "build-1", "inference-1", "cpu-1", "avx2")
            .expect("persist CPU engine selection");

        assert!(valid_cpu_engine_selection(
            &selection_path,
            "inference-1",
            "cpu-1"
        ));
        assert!(!valid_cpu_engine_selection(
            &selection_path,
            "inference-2",
            "cpu-1"
        ));
    }

    #[test]
    fn sample_prompts_only_populate_the_input() {
        assert!(INDEX_HTML.contains("src=\"/sample-prompts.js\""));
        assert!(SAMPLE_PROMPTS_JS.contains("promptInput.value = sample;"));
        assert!(SAMPLE_PROMPTS_JS.contains("new Event(\"input\", { bubbles: true })"));
        assert!(!APP_JS.contains("sampleSelect"));
        for forbidden in [
            "levelInput",
            "profileSelect",
            "saveSettings",
            "scheduleCompressionPrepare",
            "clearResultLists",
            "fetch(",
        ] {
            assert!(
                !SAMPLE_PROMPTS_JS.contains(forbidden),
                "sample helper must not use {forbidden}"
            );
        }
    }

    #[test]
    fn current_model_and_level_are_shown_next_to_the_app_title() {
        let title = INDEX_HTML.find("<h1>TrimPrompt</h1>").expect("app title");
        let settings_summary = INDEX_HTML
            .find("id=\"settingsSummary\"")
            .expect("current settings summary");
        let workspace = INDEX_HTML
            .find("class=\"workspace-grid\"")
            .expect("workspace");

        assert!(INDEX_HTML.contains("class=\"topbar-brand\""));
        assert!(INDEX_HTML.contains("内部モデル / 標準"));
        assert!(title < settings_summary && settings_summary < workspace);
        assert!(STYLES_CSS.contains(".current-settings"));
        assert!(APP_JS.contains("? \"内部モデル\" : \"自由選択\""));
        assert!(APP_JS.contains("levelDetail?.name || \"標準\""));
        assert!(APP_JS.contains("settingsSummary.title = summary"));
    }

    #[test]
    fn renamed_product_preserves_legacy_browser_settings() {
        assert!(INDEX_HTML.contains("<title>TrimPrompt</title>"));
        assert!(INDEX_HTML.contains("<h1>TrimPrompt</h1>"));
        assert!(APP_JS.contains("trimPromptSettingsV1"));
        assert!(APP_JS.contains("trimPromptThemeV1"));
        assert!(APP_JS.contains("promptCompressorSettingsV3"));
        assert!(APP_JS.contains("promptCompressorSettingsV2"));
        assert!(APP_JS.contains("promptCompressorThemeV1"));
        assert!(APP_JS.contains("window.trimPromptOpenSettings"));
        assert!(!APP_JS.contains("window.promptCompressorOpenSettings"));
    }

    #[test]
    fn interface_uses_the_material_inspired_neutral_and_blue_palette() {
        assert!(STYLES_CSS.contains("--bg: #f8fafd"));
        assert!(STYLES_CSS.contains("--accent: #0b57d0"));
        assert!(STYLES_CSS.contains("--accent-soft: #e8f0fe"));
        assert!(STYLES_CSS.contains("--success: #188038"));
        assert!(STYLES_CSS.contains("--menu-shadow:"));
    }

    #[test]
    fn input_heading_and_actions_are_kept_on_one_line() {
        assert!(INDEX_HTML.contains("class=\"section-head input-section-head\""));
        assert!(STYLES_CSS.contains(".input-section-head h2"));
        assert!(STYLES_CSS.contains(".input-section-head .head-actions"));
        assert!(STYLES_CSS
            .contains("grid-template-columns: minmax(72px, 160px) max-content max-content"));
        assert!(STYLES_CSS.contains("max-width: 160px"));
        assert!(INDEX_HTML.contains("id=\"compressButton\" type=\"button\">圧縮</button>"));
    }

    #[test]
    fn compression_level_ui_only_offers_standard_and_high_compression() {
        assert!(INDEX_HTML.contains("data-compression-level=\"2\""));
        assert!(INDEX_HTML.contains("data-compression-level=\"3\""));
        assert!(!INDEX_HTML.contains("data-compression-level=\"1\""));
        assert!(INDEX_HTML.contains("標準"));
        assert!(INDEX_HTML.contains("高圧縮"));
        assert!(APP_JS.contains("compressionLevelMin = 2"));
        assert!(APP_JS.contains("name: \"高圧縮\""));
        assert!(APP_JS.contains("await refreshModelAvailability(false)"));
        assert!(STYLES_CSS.contains(".level-switch"));
        assert!(STYLES_CSS.contains(".level-option[aria-pressed=\"true\"]"));
    }

    #[test]
    fn desktop_ui_uses_compact_status_and_flat_metrics() {
        assert!(STYLES_CSS.contains(".status-pill::before"));
        assert!(STYLES_CSS.contains(".status-pill.running::before"));
        assert!(STYLES_CSS.contains("@keyframes status-ring-spin"));
        assert!(INDEX_HTML.contains("class=\"status-label\""));
        assert!(APP_JS.contains("element.replaceChildren(labelElement, textElement)"));
        assert!(APP_JS.contains("element.title = fullStatus"));
        assert!(APP_JS.contains("setModelStatus(\"準備完了\", \"\")"));
        assert!(APP_JS.contains("setWorkStatus(\"圧縮中\", \"running\")"));
        assert!(APP_JS.contains("setWorkStatus(\"圧縮完了\", \"\")"));
        assert!(!APP_JS.contains("圧縮完了・コピー済み"));
        assert!(!APP_JS.contains("setWorkStatus(copied ? \"コピー済み\""));
        assert!(STYLES_CSS.contains("#modelStatus"));
        assert!(STYLES_CSS.contains(".status-label"));
        assert!(STYLES_CSS.contains("width: min(440px, calc(100vw - 32px))"));
        assert!(STYLES_CSS.contains(".metric:last-child"));
        assert!(STYLES_CSS.contains("border-right: 1px solid var(--border)"));
        assert!(INDEX_HTML.contains("推定トークン"));
        assert!(!INDEX_HTML.contains("トークン（推定）"));
        assert!(!INDEX_HTML.contains("モデル、圧縮モード、表示"));
    }

    #[test]
    fn displayed_compression_time_covers_button_to_output_elapsed_time() {
        assert!(APP_JS.contains("const compressionStartedAt = performance.now();"));
        assert!(APP_JS.contains("renderResult(payload, compressionStartedAt);"));
        assert!(APP_JS.contains("promptOutput.value = result.distilled_prompt || \"\";"));
        assert!(APP_JS.contains("performance.now() - startedAt"));
        assert!(APP_JS.contains("formatLatencySeconds(displayedElapsedMs)"));
        assert!(APP_JS.contains("return `${seconds.toFixed(1)}s`;"));
    }

    #[test]
    fn compression_error_remains_visible_after_result_cleanup() {
        assert!(APP_JS.contains(
            "clearResultLists();\n    promptOutput.value = String(error.message || error);"
        ));
    }

    #[test]
    fn resolves_available_disk_space_for_existing_directory() {
        #[cfg(target_os = "windows")]
        assert!(available_disk_space_bytes(std::env::temp_dir().as_path()).is_some());
    }

    #[test]
    fn resolves_available_disk_space_for_existing_file() {
        #[cfg(target_os = "windows")]
        assert!(available_disk_space_bytes(
            std::env::current_exe().expect("test executable").as_path()
        )
        .is_some());
    }

    #[test]
    fn normalizes_known_settings_without_dropping_unknown_fields() {
        let settings: UiSettings = serde_json::from_value(serde_json::json!({
            "schema_version": 99,
            "profile": " internal_llm ",
            "level": 9,
            "theme": "future-theme",
            "future_panel": { "density": "compact" }
        }))
        .expect("deserialize settings");

        let value = serde_json::to_value(settings.normalized()).expect("serialize settings");

        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["profile"], "internal_llm");
        assert_eq!(value["level"], 3);
        assert!(value["theme"].is_null());
        assert_eq!(value["future_panel"]["density"], "compact");
    }

    #[test]
    fn startup_prepare_uses_saved_profile_and_supported_level() {
        let registry = ProfileRegistry::bootstrap();
        let settings = PersistedUiSettings {
            path: PathBuf::from("unused-startup-settings.json"),
            settings: Arc::new(Mutex::new(UiSettings {
                profile: Some("lmstudio_local".to_string()),
                level: Some(3),
                ..UiSettings::default()
            })),
        };

        let request =
            startup_compression_request(&registry, &settings).expect("resolve startup request");
        assert_eq!(request.profile, "lmstudio_local");
        assert_eq!(request.compression_level.value(), 3);

        *settings.settings.lock().expect("lock settings") = UiSettings {
            profile: Some("missing-profile".to_string()),
            level: Some(1),
            ..UiSettings::default()
        };
        let fallback = startup_compression_request(&registry, &settings)
            .expect("fallback to supported defaults");
        assert_eq!(fallback.profile, "internal_llm");
        assert_eq!(fallback.compression_level.value(), 2);
    }

    #[test]
    fn settings_save_preserves_existing_unknown_fields_and_cleans_temp_file() {
        let directory = TestDirectory::new("settings-preserve");
        let path = directory.path().join("ui-settings.json");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema_version": 1,
                "profile": "old",
                "future_panel": { "density": "compact" }
            }))
            .expect("serialize initial settings"),
        )
        .expect("write initial settings");
        let persisted = PersistedUiSettings::load(path.clone());
        let incoming = serde_json::from_value(serde_json::json!({
            "schema_version": 1,
            "profile": "new",
            "level": 2,
            "future_request": true
        }))
        .expect("deserialize incoming settings");

        persisted.save(incoming).expect("save settings");

        let disk: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).expect("read persisted settings"))
                .expect("parse persisted settings");
        assert_eq!(disk, persisted.json());
        assert_eq!(disk["profile"], "new");
        assert_eq!(disk["future_panel"]["density"], "compact");
        assert_eq!(disk["future_request"], true);
        assert!(settings_temp_files(directory.path()).is_empty());
    }

    #[test]
    fn unchanged_settings_skip_filesystem_writes() {
        let directory = TestDirectory::new("settings-unchanged");
        let invalid_parent = directory.path().join("not-a-directory");
        fs::write(&invalid_parent, b"sentinel").expect("write conflicting parent file");
        let current = UiSettings {
            profile: Some("internal_llm".to_string()),
            level: Some(2),
            theme: Some("dark".to_string()),
            ..UiSettings::default()
        };
        let persisted = PersistedUiSettings {
            path: invalid_parent.join("ui-settings.json"),
            settings: Arc::new(Mutex::new(current.clone())),
        };

        persisted
            .save(current)
            .expect("unchanged settings should not touch the invalid path");
        let changed = UiSettings {
            profile: Some("internal_llm".to_string()),
            level: Some(3),
            theme: Some("dark".to_string()),
            ..UiSettings::default()
        };
        persisted
            .save(changed)
            .expect_err("changed settings must attempt a filesystem write");
    }

    #[test]
    fn failed_settings_save_keeps_memory_and_removes_temp_file() {
        let directory = TestDirectory::new("settings-failure");
        let path = directory.path().join("ui-settings.json");
        fs::create_dir(&path).expect("create conflicting destination directory");
        fs::write(path.join("sentinel"), b"keep").expect("write sentinel");
        let previous = UiSettings {
            profile: Some("previous".to_string()),
            ..UiSettings::default()
        };
        let persisted = PersistedUiSettings {
            path,
            settings: Arc::new(Mutex::new(previous)),
        };
        let incoming = UiSettings {
            profile: Some("replacement".to_string()),
            ..UiSettings::default()
        };

        persisted.save(incoming).expect_err("save must fail");

        assert_eq!(persisted.json()["profile"], "previous");
        assert!(settings_temp_files(directory.path()).is_empty());
    }

    #[test]
    fn concurrent_settings_saves_keep_disk_and_memory_consistent() {
        let directory = TestDirectory::new("settings-concurrent");
        let path = directory.path().join("ui-settings.json");
        let persisted = PersistedUiSettings::load(path.clone());
        let barrier = Arc::new(Barrier::new(5));
        let mut handles = Vec::new();

        for index in 0..4 {
            let persisted = persisted.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                let settings = UiSettings {
                    profile: Some(format!("profile-{index}")),
                    ..UiSettings::default()
                };
                barrier.wait();
                persisted.save(settings).expect("concurrent save");
            }));
        }

        barrier.wait();
        for handle in handles {
            handle.join().expect("save thread");
        }

        let disk: serde_json::Value =
            serde_json::from_slice(&fs::read(path).expect("read persisted settings"))
                .expect("parse persisted settings");
        assert_eq!(disk, persisted.json());
        assert!(settings_temp_files(directory.path()).is_empty());
    }
}
