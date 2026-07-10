use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[cfg(target_os = "windows")]
use std::ptr::{null, null_mut};

use anyhow::{Context, Result};
use prompt_compressor_core::{
    CompressionConstraints, CompressionLevel, CompressionMode, CompressionRequest,
    CompressionService, ConfiguredRuntimeBackend, ProfileRegistry, RequestSource, RequestTarget,
    TaskType,
};
use serde::{Deserialize, Serialize};

#[cfg(target_os = "windows")]
use windows_sys::Win32::{
    Foundation::{GetLastError, HWND, LPARAM, LRESULT, WPARAM},
    System::{
        DataExchange::{CloseClipboard, EmptyClipboard, OpenClipboard, SetClipboardData},
        LibraryLoader::GetModuleHandleW,
        Memory::{GlobalAlloc, GlobalLock, GlobalUnlock, GMEM_MOVEABLE},
        Ole::CF_UNICODETEXT,
    },
    UI::{
        Shell::{
            Shell_NotifyIconW, NIF_ICON, NIF_INFO, NIF_MESSAGE, NIF_TIP, NIIF_INFO,
            NIIF_LARGE_ICON, NIIF_USER, NIM_ADD, NIM_DELETE, NIM_MODIFY, NIM_SETVERSION,
            NOTIFYICONDATAW, NOTIFYICON_VERSION_4,
        },
        WindowsAndMessaging::{
            CreateWindowExW, DefWindowProcW, DestroyIcon, DestroyWindow, LoadIconW, LoadImageW,
            RegisterClassW, UnregisterClassW, HICON, HWND_MESSAGE, IDI_INFORMATION, IMAGE_ICON,
            LR_DEFAULTSIZE, WM_USER, WNDCLASSW, WS_OVERLAPPED,
        },
    },
};

const INDEX_HTML: &str = include_str!("../static/index.html");
const STYLES_CSS: &str = include_str!("../static/styles.css");
const APP_JS: &str = include_str!("../static/app.js");

#[cfg(target_os = "windows")]
const APP_ICON_RESOURCE_ID: u16 = 1;

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
    warmup: RuntimeWarmupState,
    ui_settings: PersistedUiSettings,
}

#[derive(Debug, Clone, Serialize)]
struct RuntimeWarmupSnapshot {
    phase: String,
    profile: Option<String>,
    message: String,
    error: Option<String>,
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
            }
            Err(error) => {
                eprintln!("runtime warmup status lock failed: {error}");
            }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    fn save(&self, settings: UiSettings) -> Result<serde_json::Value> {
        let settings = settings.normalized();
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }

        let body =
            serde_json::to_vec_pretty(&settings).context("failed to serialize UI settings")?;
        fs::write(&self.path, body)
            .with_context(|| format!("failed to write {}", self.path.display()))?;

        let mut guard = self
            .settings
            .lock()
            .map_err(|error| anyhow::anyhow!("failed to lock UI settings: {error}"))?;
        *guard = settings.clone();
        Ok(serde_json::to_value(settings)?)
    }
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
    task_type: TaskType,
    compression_mode: CompressionMode,
    compression_level: u8,
    constraints: Option<CompressionConstraints>,
}

#[derive(Debug, Deserialize)]
struct PrepareCompressionPayload {
    profile: String,
    task_type: TaskType,
    compression_mode: CompressionMode,
    compression_level: u8,
    constraints: Option<CompressionConstraints>,
}

#[derive(Debug, Deserialize)]
struct NotificationPayload {
    #[serde(default)]
    title: Option<String>,
    body: String,
}

#[derive(Debug, Deserialize)]
struct ClipboardPayload {
    text: String,
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
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
    let settings_dir = resolve_settings_dir(options.settings_dir.as_deref())?;
    let state = prepare_app_state(&settings_dir)?;

    let address = format!("{}:{}", options.host, options.port);
    let listener =
        TcpListener::bind(&address).with_context(|| format!("failed to bind {address}"))?;
    let address = listener
        .local_addr()
        .context("failed to determine local server address")?
        .to_string();
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
    let backend = ConfiguredRuntimeBackend::from_settings_dir(&settings_dir)
        .context("failed to initialize configured runtime backend")?;
    let ui_settings = PersistedUiSettings::load(
        application_root
            .join("local")
            .join("state")
            .join("ui-settings.json"),
    );
    let state = Arc::new(AppState {
        registry,
        backend,
        warmup: RuntimeWarmupState::new(),
        ui_settings,
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
        "loading",
        Some(profile.id.clone()),
        "アプリ内モデルを読み込み中",
        None,
    );

    thread::spawn(move || {
        let profile_id = profile.id.clone();
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
                "アプリ内モデルの先読みでエラーが発生しました",
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
    listener
        .set_nonblocking(shutdown_rx.is_some())
        .context("failed to configure local server listener")?;

    loop {
        if let Some(shutdown_rx) = &shutdown_rx {
            match shutdown_rx.try_recv() {
                Ok(()) | Err(mpsc::TryRecvError::Disconnected) => break,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }

        match listener.accept() {
            Ok((stream, _peer)) => {
                let state = state.clone();
                if let Err(error) = handle_client(stream, &state) {
                    eprintln!("request failed: {error:#}");
                }
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(50));
            }
            Err(error) => eprintln!("connection failed: {error}"),
        }
    }

    Ok(())
}

fn handle_client(mut stream: TcpStream, state: &AppState) -> Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    let request = read_request(&mut stream)?;
    let response = route_request(request, state);
    stream.write_all(&response)?;
    stream.flush()?;
    Ok(())
}

fn route_request(request: HttpRequest, state: &AppState) -> Vec<u8> {
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
        ("GET", "/api/profiles") => json_response(200, &profiles_json(&state.registry)),
        ("GET", "/api/runtime-status") => json_response(200, &state.warmup.json()),
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
        ("POST", "/api/windows-notification") => match notify_from_body(body) {
            Ok(value) => json_response(200, &value),
            Err(error) => json_response(
                400,
                &serde_json::json!({
                    "notified": false,
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
        ("POST", "/api/prepare-compression") => match prepare_compression_from_body(body, state) {
            Ok(value) => json_response(200, &value),
            Err(error) => json_response(
                400,
                &serde_json::json!({
                    "prepared": false,
                    "error": error.to_string()
                }),
            ),
        },
        ("POST", "/api/compress") => match compress_from_body(body, state) {
            Ok(value) => json_response(200, &value),
            Err(error) => json_response(
                400,
                &serde_json::json!({
                    "error": error.to_string()
                }),
            ),
        },
        _ => json_response(
            404,
            &serde_json::json!({
                "error": "not found"
            }),
        ),
    }
}

fn save_settings_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let settings: UiSettings = serde_json::from_slice(body).context("invalid JSON settings")?;
    state.ui_settings.save(settings)
}

fn notify_from_body(body: &[u8]) -> Result<serde_json::Value> {
    let payload: NotificationPayload =
        serde_json::from_slice(body).context("invalid JSON notification request")?;
    let title = payload
        .title
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("圧縮完了");
    let body = payload.body.trim();
    if body.is_empty() {
        anyhow::bail!("notification body is empty");
    }

    let notified = show_windows_notification(title, body)?;
    Ok(serde_json::json!({
        "notified": notified
    }))
}

fn copy_to_clipboard_from_body(body: &[u8]) -> Result<serde_json::Value> {
    let payload: ClipboardPayload =
        serde_json::from_slice(body).context("invalid JSON clipboard request")?;
    if payload.text.trim().is_empty() {
        anyhow::bail!("clipboard text is empty");
    }

    let copied = write_text_to_clipboard(&payload.text)?;
    Ok(serde_json::json!({
        "copied": copied
    }))
}

fn prepare_compression_from_body(body: &[u8], state: &AppState) -> Result<serde_json::Value> {
    let payload: PrepareCompressionPayload =
        serde_json::from_slice(body).context("invalid JSON prepare request")?;
    let request = CompressionRequest {
        input_text: String::new(),
        task_type: payload.task_type,
        compression_mode: payload.compression_mode,
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
    let service = CompressionService::new(state.registry.clone(), state.backend.clone());
    match service.prepare(request.clone()) {
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
    let request = CompressionRequest {
        input_text: payload.input_text,
        task_type: payload.task_type,
        compression_mode: payload.compression_mode,
        compression_level: CompressionLevel::from_u8(payload.compression_level)
            .context("invalid compression level")?,
        profile: payload.profile,
        constraints: payload.constraints.unwrap_or_default(),
        target: RequestTarget::codex_default(),
        source: RequestSource::Desktop,
    };

    let service = CompressionService::new(state.registry.clone(), state.backend.clone());
    let result = service.compress(request)?;
    Ok(serde_json::to_value(result)?)
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

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 4096];

    loop {
        let bytes_read = stream.read(&mut chunk)?;
        if bytes_read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..bytes_read]);
        if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }

    let header_end = find_header_end(&buffer).context("malformed HTTP request")?;
    let headers = std::str::from_utf8(&buffer[..header_end]).context("request was not UTF-8")?;
    let mut lines = headers.lines();
    let request_line = lines.next().context("missing request line")?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().context("missing method")?.to_string();
    let raw_path = request_parts.next().context("missing path")?;
    let path = raw_path.split('?').next().unwrap_or(raw_path).to_string();
    let content_length = parse_content_length(headers);

    let body_start = header_end + 4;
    let mut body = buffer[body_start..].to_vec();
    while body.len() < content_length {
        let bytes_read = stream.read(&mut chunk)?;
        if bytes_read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..bytes_read]);
    }
    body.truncate(content_length);

    Ok(HttpRequest { method, path, body })
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_content_length(headers: &str) -> usize {
    headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0)
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
fn show_windows_notification(title: &str, body: &str) -> Result<bool> {
    unsafe {
        let class_name = to_wide_null(&format!(
            "PromptCompressorNotificationWindow{}",
            std::process::id()
        ));
        let hinstance = GetModuleHandleW(null());
        let window_class = WNDCLASSW {
            lpfnWndProc: Some(notification_window_proc),
            hInstance: hinstance,
            lpszClassName: class_name.as_ptr(),
            ..Default::default()
        };

        if RegisterClassW(&window_class) == 0 {
            anyhow::bail!("RegisterClassW for notification failed: {}", GetLastError());
        }

        let hwnd = CreateWindowExW(
            0,
            class_name.as_ptr(),
            class_name.as_ptr(),
            WS_OVERLAPPED,
            0,
            0,
            0,
            0,
            HWND_MESSAGE,
            null_mut(),
            hinstance,
            null(),
        );

        let result = if hwnd.is_null() {
            Err(anyhow::anyhow!(
                "CreateWindowExW for notification failed: {}",
                GetLastError()
            ))
        } else {
            show_shell_notification(hwnd, title, body)
        };

        if !hwnd.is_null() {
            DestroyWindow(hwnd);
        }
        UnregisterClassW(class_name.as_ptr(), hinstance);
        result
    }
}

#[cfg(not(target_os = "windows"))]
fn show_windows_notification(_title: &str, _body: &str) -> Result<bool> {
    Ok(false)
}

fn limit_notification_text(value: &str, max_chars: usize) -> String {
    let value = value.replace(['\r', '\n', '\t'], " ");
    let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if value.chars().count() <= max_chars {
        return value;
    }

    value
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>()
        + "..."
}

#[cfg(target_os = "windows")]
unsafe extern "system" fn notification_window_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}

#[cfg(target_os = "windows")]
fn show_shell_notification(hwnd: HWND, title: &str, body: &str) -> Result<bool> {
    unsafe {
        let hinstance = GetModuleHandleW(null());
        let (notification_icon, owns_icon) = load_notification_icon(hinstance);
        if notification_icon.is_null() {
            anyhow::bail!("notification icon could not be loaded: {}", GetLastError());
        }

        let mut data = NOTIFYICONDATAW {
            cbSize: std::mem::size_of::<NOTIFYICONDATAW>() as u32,
            hWnd: hwnd,
            uID: 1,
            uFlags: NIF_MESSAGE | NIF_ICON | NIF_TIP,
            uCallbackMessage: WM_USER + 1,
            hIcon: notification_icon,
            hBalloonIcon: notification_icon,
            ..Default::default()
        };

        copy_wide(&mut data.szTip, "Prompt Compressor");
        if Shell_NotifyIconW(NIM_ADD, &data) == 0 {
            if owns_icon {
                let _ = DestroyIcon(notification_icon);
            }
            anyhow::bail!("Shell_NotifyIconW NIM_ADD failed: {}", GetLastError());
        }

        data.Anonymous.uVersion = NOTIFYICON_VERSION_4;
        let _ = Shell_NotifyIconW(NIM_SETVERSION, &data);

        data.uFlags = NIF_INFO;
        data.dwInfoFlags = if owns_icon {
            NIIF_USER | NIIF_LARGE_ICON
        } else {
            NIIF_INFO
        };
        let title_text = limit_notification_text(title, data.szInfoTitle.len().saturating_sub(1));
        let body_text = limit_notification_text(body, data.szInfo.len().saturating_sub(1));
        copy_wide(&mut data.szInfoTitle, &title_text);
        copy_wide(&mut data.szInfo, &body_text);

        let modify_result = Shell_NotifyIconW(NIM_MODIFY, &data);
        thread::sleep(Duration::from_secs(6));
        let _ = Shell_NotifyIconW(NIM_DELETE, &data);
        if owns_icon {
            let _ = DestroyIcon(notification_icon);
        }

        if modify_result == 0 {
            anyhow::bail!("Shell_NotifyIconW NIM_MODIFY failed: {}", GetLastError());
        }

        Ok(true)
    }
}

#[cfg(target_os = "windows")]
unsafe fn load_notification_icon(
    hinstance: windows_sys::Win32::Foundation::HMODULE,
) -> (HICON, bool) {
    let app_icon = unsafe {
        LoadImageW(
            hinstance,
            resource_id(APP_ICON_RESOURCE_ID),
            IMAGE_ICON,
            32,
            32,
            LR_DEFAULTSIZE,
        ) as HICON
    };

    if !app_icon.is_null() {
        return (app_icon, true);
    }

    (unsafe { LoadIconW(null_mut(), IDI_INFORMATION) }, false)
}

#[cfg(target_os = "windows")]
fn resource_id(id: u16) -> windows_sys::core::PCWSTR {
    id as usize as windows_sys::core::PCWSTR
}

#[cfg(target_os = "windows")]
fn to_wide_null(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(target_os = "windows")]
fn copy_wide<const N: usize>(target: &mut [u16; N], value: &str) {
    target.fill(0);
    for (index, code_unit) in value.encode_utf16().take(N.saturating_sub(1)).enumerate() {
        target[index] = code_unit;
    }
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
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let mut response = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Cache-Control: no-store\r\n\
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
