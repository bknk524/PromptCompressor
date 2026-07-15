use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
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
    ProfileRegistry, RequestSource, RequestTarget,
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

    pub fn handle_request(&self, method: &str, path: &str, body: &[u8]) -> LocalAppResponse {
        route_application_request(method, path, body, &self.state)
    }
}

#[derive(Debug, Clone)]
pub struct LocalAppResponse {
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
}

#[derive(Clone)]
struct AppState {
    registry: ProfileRegistry,
    backend: ConfiguredRuntimeBackend,
    service: Arc<CompressionService<ConfiguredRuntimeBackend>>,
    warmup: RuntimeWarmupState,
    ui_settings: PersistedUiSettings,
    inference_gate: InferenceGate,
    model_downloads: ModelDownloadControl,
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
        println!("Prompt Compressor local UI: {}", info.url);
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
        registry,
        backend,
        service,
        warmup: RuntimeWarmupState::new(),
        ui_settings,
        inference_gate: InferenceGate::default(),
        model_downloads: ModelDownloadControl::default(),
    });
    Ok(state)
}

fn start_default_profile_warmup(state: Arc<AppState>) {
    let Some(profile_id) = state.registry.default_profile_id().map(str::to_owned) else {
        state
            .warmup
            .set("skipped", None, "先読み対象の既定モデルがありません", None);
        return;
    };

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
            Ok(true) => state.warmup.set(
                "ready",
                Some(profile_id),
                "アプリ内モデルは準備完了です",
                None,
            ),
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
        ("POST", "/api/compress") => run_inference_route(state, || compress_from_body(body, state)),
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
        | "/api/model-cancel" => MAX_SETTINGS_BODY_BYTES,
        "/api/clipboard" | "/api/compress" => MAX_REQUEST_BODY_BYTES,
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

fn save_settings_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let settings: UiSettings = serde_json::from_slice(body).context("invalid JSON settings")?;
    for (field, value) in [
        ("profile", settings.profile.as_deref()),
        ("mode", settings.mode.as_deref()),
        ("task", settings.task.as_deref()),
        ("theme", settings.theme.as_deref()),
    ] {
        if let Some(value) = value {
            validate_text_field(field, value, MAX_PROFILE_CHARS, true)?;
        }
    }
    state.ui_settings.save(settings)
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
    fn sample_prompts_only_populate_the_input() {
        assert!(INDEX_HTML.contains("src=\"/sample-prompts.js\""));
        assert!(SAMPLE_PROMPTS_JS.contains("promptInput.value = sample;"));
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
        let title = INDEX_HTML
            .find("<h1>Prompt Compressor</h1>")
            .expect("app title");
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
    fn input_heading_and_actions_are_kept_on_one_line() {
        assert!(INDEX_HTML.contains("class=\"section-head input-section-head\""));
        assert!(STYLES_CSS.contains(".input-section-head h2"));
        assert!(STYLES_CSS.contains(".input-section-head .head-actions"));
        assert!(STYLES_CSS.contains("flex-wrap: nowrap"));
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
        assert!(STYLES_CSS.contains(".level-switch"));
        assert!(STYLES_CSS.contains(".level-option[aria-pressed=\"true\"]"));
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
