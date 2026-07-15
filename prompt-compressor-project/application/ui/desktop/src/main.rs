#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::borrow::Cow;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use clap::Parser;
use prompt_compressor_local_ui::{prepare_embedded_web_app, EmbeddedWebApp, LocalAppResponse};
use tao::{
    dpi::{LogicalSize, PhysicalSize},
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy},
    window::WindowBuilder,
};
use wry::{
    http::{header::CONTENT_TYPE, Request, Response},
    RequestAsyncResponder, WebContext, WebViewBuilder,
};

#[cfg(windows)]
use tao::{
    platform::windows::{IconExtWindows, WindowExtWindows},
    window::{Icon, Theme},
};

const APP_USER_MODEL_ID: &str = "TrimPrompt.Desktop";
const APP_WINDOW_TITLE: &str = "TrimPrompt";
const SINGLE_INSTANCE_MUTEX_NAME: &str = "Local\\TrimPrompt.Desktop.SingleInstance";
const LEGACY_SINGLE_INSTANCE_MUTEX_NAME: &str = "Local\\PromptCompressor.Desktop.SingleInstance";
const ALREADY_RUNNING_MESSAGE: &str = "TrimPrompt はすでに起動しています。";
const PROTOCOL_WORKER_COUNT: usize = 4;
const PROTOCOL_QUEUE_CAPACITY: usize = 16;
const MAX_NOTIFICATION_TITLE_CHARS: usize = 128;
const MAX_NOTIFICATION_BODY_CHARS: usize = 1_024;

#[derive(Debug, Parser)]
#[command(name = "TrimPrompt")]
#[command(about = "Native Windows shell for TrimPrompt")]
struct Args {
    #[arg(long, value_name = "DIR")]
    settings_dir: Option<PathBuf>,

    #[arg(long, hide = true)]
    package_smoke_test: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppTheme {
    Light,
    Dark,
}

impl AppTheme {
    fn from_ipc_message(message: &str) -> Option<Self> {
        match message {
            "theme:light" => Some(Self::Light),
            "theme:dark" => Some(Self::Dark),
            _ => None,
        }
    }

    fn from_settings_file(application_root: &Path) -> Self {
        let settings_path = application_root
            .join("local")
            .join("state")
            .join("ui-settings.json");
        let Ok(settings_text) = fs::read_to_string(settings_path) else {
            return Self::Light;
        };
        let Ok(settings) = serde_json::from_str::<serde_json::Value>(&settings_text) else {
            return Self::Light;
        };

        match settings.get("theme").and_then(serde_json::Value::as_str) {
            Some("dark") => Self::Dark,
            _ => Self::Light,
        }
    }
}

#[derive(Debug)]
enum DesktopEvent {
    ApplyTheme(AppTheme),
    WindowControl(WindowControl),
    ShowNotification(DesktopNotification),
    RestoreFromTray,
    OpenSettingsFromTray,
    ExitRequested,
}

#[derive(Debug)]
struct DesktopNotification {
    title: String,
    body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowControl {
    Minimize,
    ToggleMaximize,
    Close,
    Drag,
}

fn main() {
    let args = Args::parse();
    let package_smoke_test = args.package_smoke_test;
    if let Err(error) = run(args) {
        if !package_smoke_test {
            show_startup_error(&error);
        }
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<()> {
    if args.package_smoke_test {
        return run_packaged_local_model_smoke_test(args.settings_dir);
    }

    // ウィンドウ作成前に判定し、モデルの二重読み込みも防ぐ。
    let Some(_single_instance) = acquire_single_instance()? else {
        return Ok(());
    };
    set_app_user_model_id();
    let app = prepare_embedded_web_app(args.settings_dir)?;
    app.start_default_profile_warmup();

    run_window(app)
}

#[cfg(windows)]
struct SingleInstanceGuard {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

#[cfg(windows)]
impl Drop for SingleInstanceGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = windows_sys::Win32::Foundation::CloseHandle(self.handle);
        }
    }
}

#[cfg(windows)]
fn acquire_single_instance() -> Result<Option<Vec<SingleInstanceGuard>>> {
    let legacy_instance = try_acquire_named_mutex(LEGACY_SINGLE_INSTANCE_MUTEX_NAME)?;
    let Some(legacy_instance) = legacy_instance else {
        show_already_running_notice();
        return Ok(None);
    };

    let instance = try_acquire_named_mutex(SINGLE_INSTANCE_MUTEX_NAME)?;
    let Some(instance) = instance else {
        show_already_running_notice();
        return Ok(None);
    };

    Ok(Some(vec![legacy_instance, instance]))
}

#[cfg(windows)]
fn try_acquire_named_mutex(name: &str) -> Result<Option<SingleInstanceGuard>> {
    use std::ptr;
    use windows_sys::Win32::{
        Foundation::{CloseHandle, GetLastError, ERROR_ALREADY_EXISTS},
        System::Threading::CreateMutexW,
    };

    let name = wide_null(name);
    let handle = unsafe { CreateMutexW(ptr::null(), 0, name.as_ptr()) };
    let last_error = unsafe { GetLastError() };
    if handle.is_null() {
        anyhow::bail!(
            "failed to create desktop single-instance mutex: {}",
            last_error
        );
    }

    if last_error == ERROR_ALREADY_EXISTS {
        unsafe {
            let _ = CloseHandle(handle);
        }
        return Ok(None);
    }

    Ok(Some(SingleInstanceGuard { handle }))
}

#[cfg(windows)]
fn show_already_running_notice() {
    use std::ptr;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        FindWindowW, MessageBoxW, SetForegroundWindow, ShowWindow, MB_ICONINFORMATION, MB_OK,
        MB_SETFOREGROUND, SW_RESTORE,
    };

    let title = wide_null(APP_WINDOW_TITLE);
    let window = unsafe { FindWindowW(ptr::null(), title.as_ptr()) };
    if !window.is_null() {
        unsafe {
            let _ = ShowWindow(window, SW_RESTORE);
            let _ = SetForegroundWindow(window);
        }
    }

    let message = wide_null(ALREADY_RUNNING_MESSAGE);
    unsafe {
        let _ = MessageBoxW(
            ptr::null_mut(),
            message.as_ptr(),
            title.as_ptr(),
            MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND,
        );
    }
}

#[cfg(not(windows))]
struct SingleInstanceGuard;

#[cfg(not(windows))]
fn acquire_single_instance() -> Result<Option<SingleInstanceGuard>> {
    Ok(Some(SingleInstanceGuard))
}

const PACKAGED_SMOKE_RESULT_FILE: &str = "package-smoke-result.json";
const PACKAGED_SMOKE_INPUT: &str = "React の検索画面で、検索ボタンを押したときだけ API を呼ぶようにしてください。既存の useSearchParams による URL クエリ管理は維持し、大規模なリファクタリングは避けてください。";

fn run_packaged_local_model_smoke_test(settings_dir: Option<PathBuf>) -> Result<()> {
    let app = prepare_embedded_web_app(settings_dir)?;
    let application_root = app
        .settings_dir()
        .parent()
        .context("settings directory must be inside the application directory")?;
    let result_path = application_root
        .join("local")
        .join("state")
        .join(PACKAGED_SMOKE_RESULT_FILE);
    if let Some(parent) = result_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let request_body = packaged_smoke_request_body()?;
    let response = app.handle_request("POST", "/api/compress", &request_body);
    fs::write(&result_path, &response.body)
        .with_context(|| format!("failed to write {}", result_path.display()))?;
    anyhow::ensure!(
        response.status == 200,
        "packaged local model smoke test returned HTTP {}",
        response.status
    );

    let result: serde_json::Value =
        serde_json::from_slice(&response.body).context("smoke test response was not valid JSON")?;
    validate_packaged_smoke_result(&result)
}

fn packaged_smoke_request_body() -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&serde_json::json!({
        "input_text": PACKAGED_SMOKE_INPUT,
        "profile": "internal_llm",
        "task_type": "coding",
        "compression_mode": "codex_optimized",
        "compression_level": 2,
        "constraints": {
            "preserve_code_blocks": true,
            "preserve_file_names": true,
            "preserve_error_messages": true,
            "preserve_numbers": true,
            "preserve_negations": true
        }
    }))?)
}

fn validate_packaged_smoke_result(result: &serde_json::Value) -> Result<()> {
    anyhow::ensure!(
        result.get("profile").and_then(serde_json::Value::as_str) == Some("internal_llm"),
        "packaged local model smoke test used an unexpected profile"
    );
    anyhow::ensure!(
        result.get("runtime").and_then(serde_json::Value::as_str) == Some("llama_cpp_embedded"),
        "packaged local model smoke test used an unexpected runtime"
    );
    anyhow::ensure!(
        result
            .get("should_send_original")
            .and_then(serde_json::Value::as_bool)
            == Some(false),
        "packaged local model smoke test returned the original prompt"
    );
    anyhow::ensure!(
        result
            .get("distilled_prompt")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|prompt| !prompt.trim().is_empty()),
        "packaged local model smoke test returned an empty prompt"
    );

    let input_characters = result
        .pointer("/metrics/input_characters")
        .and_then(serde_json::Value::as_u64)
        .context("smoke test result is missing metrics.input_characters")?;
    let output_characters = result
        .pointer("/metrics/output_characters")
        .and_then(serde_json::Value::as_u64)
        .context("smoke test result is missing metrics.output_characters")?;
    anyhow::ensure!(
        output_characters < input_characters,
        "packaged local model smoke test did not reduce character count"
    );
    Ok(())
}

fn show_startup_error(error: &anyhow::Error) {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(PathBuf::from))
        .unwrap_or_else(std::env::temp_dir);
    let message = format!(
        "TrimPrompt could not start.\n\n\
         This desktop package must keep TrimPrompt.exe next to the application folder.\n\
         Required shape:\n\n\
         TrimPrompt.exe\n\
         application\\config\\\n\
         application\\resources\\\n\n\
         The local model is downloaded from Hugging Face after startup.\n\n\
         Error details:\n{error:#}\n"
    );

    let report_path = exe_dir.join("TrimPrompt_STARTUP_ERROR.txt");
    let report_path = match fs::write(&report_path, &message) {
        Ok(()) => report_path,
        Err(_) => {
            let fallback_path = std::env::temp_dir().join("TrimPrompt_STARTUP_ERROR.txt");
            let _ = fs::write(&fallback_path, message);
            fallback_path
        }
    };
    let _ = Command::new("notepad.exe").arg(report_path).spawn();
}

fn run_window(app: EmbeddedWebApp) -> Result<()> {
    let application_root = app
        .settings_dir()
        .parent()
        .context("settings directory must be inside the application directory")?;
    let webview_data_dir = application_root
        .join("local")
        .join("state")
        .join("webview2");
    fs::create_dir_all(&webview_data_dir)
        .with_context(|| format!("failed to create {}", webview_data_dir.display()))?;
    let notification_log_path = application_root
        .join("local")
        .join("state")
        .join("notification.log");

    let initial_theme = AppTheme::from_settings_file(application_root);
    write_notification_log(
        &notification_log_path,
        "desktop startup reached notification setup",
    );
    let event_loop = EventLoopBuilder::<DesktopEvent>::with_user_event().build();

    let window = WindowBuilder::new()
        .with_title(APP_WINDOW_TITLE)
        .with_decorations(false)
        .with_inner_size(LogicalSize::new(980.0, 860.0))
        .with_min_inner_size(LogicalSize::new(520.0, 640.0))
        .build(&event_loop)?;
    configure_window_chrome(&window, initial_theme);
    let mut tray_message_window = install_tray_message_window(&window, event_loop.create_proxy());
    let mut tray_icon = install_tray_icon(&window, tray_message_window.as_ref());

    let mut web_context = WebContext::new(Some(webview_data_dir));
    let app = Arc::new(app);
    let protocol_app = app.clone();
    let protocol_executor = BoundedExecutor::new(
        PROTOCOL_WORKER_COUNT,
        PROTOCOL_QUEUE_CAPACITY,
        ProtocolJob::run,
    );
    let desktop_proxy = event_loop.create_proxy();
    let webview = WebViewBuilder::new_with_web_context(&mut web_context)
        .with_asynchronous_custom_protocol(
            "prompt-compressor".into(),
            move |_webview_id, request, responder| {
                let job = ProtocolJob {
                    app: Arc::clone(&protocol_app),
                    request,
                    responder,
                };
                if let Err(job) = protocol_executor.submit(job) {
                    job.reject_busy();
                }
            },
        )
        .with_ipc_handler(move |request: Request<String>| {
            let message = request.body();
            if let Some(theme) = AppTheme::from_ipc_message(message) {
                let _ = desktop_proxy.send_event(DesktopEvent::ApplyTheme(theme));
            } else if let Some(control) = WindowControl::from_ipc_message(message) {
                let _ = desktop_proxy.send_event(DesktopEvent::WindowControl(control));
            } else if let Some(notification) = DesktopNotification::from_ipc_message(message) {
                let _ = desktop_proxy.send_event(DesktopEvent::ShowNotification(notification));
            }
        })
        .with_url("prompt-compressor://localhost/")
        .build(&window)?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::UserEvent(DesktopEvent::ApplyTheme(theme)) => {
                apply_window_theme(&window, theme);
            }
            Event::UserEvent(DesktopEvent::WindowControl(control)) => {
                if control == WindowControl::Close {
                    drop(tray_icon.take());
                    drop(tray_message_window.take());
                    *control_flow = ControlFlow::Exit;
                } else {
                    handle_window_control(&window, control);
                }
            }
            Event::UserEvent(DesktopEvent::ShowNotification(notification)) => {
                show_desktop_notification(
                    tray_icon.as_ref(),
                    &notification,
                    &notification_log_path,
                );
            }
            Event::UserEvent(DesktopEvent::RestoreFromTray) => {
                restore_window_from_tray(&window);
            }
            Event::UserEvent(DesktopEvent::OpenSettingsFromTray) => {
                restore_window_from_tray(&window);
                let _ = webview.evaluate_script("window.trimPromptOpenSettings?.();");
            }
            Event::UserEvent(DesktopEvent::ExitRequested) => {
                drop(tray_icon.take());
                drop(tray_message_window.take());
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                drop(tray_icon.take());
                drop(tray_message_window.take());
                *control_flow = ControlFlow::Exit;
            }
            _ => {}
        }
    });
}

fn configure_window_chrome(window: &tao::window::Window, theme: AppTheme) {
    window.set_title(APP_WINDOW_TITLE);
    set_window_icons(window);
    set_window_shadow(window);
    apply_window_theme(window, theme);
}

#[cfg(windows)]
fn set_app_user_model_id() {
    let app_id = wide_null(APP_USER_MODEL_ID);
    unsafe {
        let _ =
            windows_sys::Win32::UI::Shell::SetCurrentProcessExplicitAppUserModelID(app_id.as_ptr());
    }
}

#[cfg(not(windows))]
fn set_app_user_model_id() {}

#[cfg(windows)]
fn set_window_icons(window: &tao::window::Window) {
    const APP_ICON_RESOURCE_ID: u16 = 1;

    if let Ok(icon) = Icon::from_resource(
        APP_ICON_RESOURCE_ID,
        Some(PhysicalSize::new(32_u32, 32_u32)),
    ) {
        window.set_window_icon(Some(icon));
    }

    let taskbar_icon = Icon::from_resource(
        APP_ICON_RESOURCE_ID,
        Some(PhysicalSize::new(256_u32, 256_u32)),
    )
    .or_else(|_| Icon::from_resource(APP_ICON_RESOURCE_ID, None));

    if let Ok(icon) = taskbar_icon {
        window.set_taskbar_icon(Some(icon));
    }
}

#[cfg(not(windows))]
fn set_window_icons(_window: &tao::window::Window) {}

#[cfg(windows)]
fn set_window_shadow(window: &tao::window::Window) {
    window.set_undecorated_shadow(true);
}

#[cfg(not(windows))]
fn set_window_shadow(_window: &tao::window::Window) {}

impl DesktopNotification {
    fn from_ipc_message(message: &str) -> Option<Self> {
        let payload = message.strip_prefix("notification:")?;
        let value = serde_json::from_str::<serde_json::Value>(payload).ok()?;
        let title = limit_notification_text(
            value
                .get("title")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or("圧縮完了"),
            MAX_NOTIFICATION_TITLE_CHARS,
        );
        let body = limit_notification_text(
            value
                .get("body")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())?,
            MAX_NOTIFICATION_BODY_CHARS,
        );
        if body.is_empty() {
            return None;
        }

        Some(Self { title, body })
    }
}

#[cfg(windows)]
struct TrayIcon {
    hwnd: windows_sys::Win32::Foundation::HWND,
    icon: windows_sys::Win32::UI::WindowsAndMessaging::HICON,
}

#[cfg(windows)]
struct TrayMessageWindow {
    hwnd: windows_sys::Win32::Foundation::HWND,
    state: *mut TrayMessageState,
}

#[cfg(windows)]
struct TrayMessageState {
    proxy: EventLoopProxy<DesktopEvent>,
}

#[cfg(windows)]
const APP_ICON_RESOURCE_ID: u16 = 1;

#[cfg(windows)]
const TRAY_ICON_ID: u32 = 1;

#[cfg(windows)]
impl TrayIcon {
    fn install(window: &tao::window::Window, message_window: &TrayMessageWindow) -> Option<Self> {
        use std::mem::size_of;
        use windows_sys::Win32::{
            Foundation::{HINSTANCE, HWND},
            UI::{
                Shell::{
                    Shell_NotifyIconW, NIF_ICON, NIF_MESSAGE, NIF_TIP, NIM_ADD, NIM_SETVERSION,
                    NOTIFYICONDATAW, NOTIFYICON_VERSION_4,
                },
                WindowsAndMessaging::{LoadImageW, HICON, IMAGE_ICON, LR_DEFAULTSIZE},
            },
        };

        let hwnd = message_window.hwnd as HWND;
        let icon = unsafe {
            LoadImageW(
                window.hinstance() as HINSTANCE,
                resource_id(APP_ICON_RESOURCE_ID),
                IMAGE_ICON,
                32,
                32,
                LR_DEFAULTSIZE,
            ) as HICON
        };

        if icon.is_null() {
            return None;
        }

        let mut data = NOTIFYICONDATAW {
            cbSize: size_of::<NOTIFYICONDATAW>() as u32,
            hWnd: hwnd,
            uID: TRAY_ICON_ID,
            uFlags: NIF_MESSAGE | NIF_ICON | NIF_TIP,
            uCallbackMessage: tray_callback_message(),
            hIcon: icon,
            ..Default::default()
        };
        copy_utf16_to_fixed(&mut data.szTip, "TrimPrompt");

        let installed = unsafe { Shell_NotifyIconW(NIM_ADD, &data) } != 0;
        if !installed {
            unsafe {
                windows_sys::Win32::UI::WindowsAndMessaging::DestroyIcon(icon);
            }
            return None;
        }

        unsafe {
            data.Anonymous.uVersion = NOTIFYICON_VERSION_4;
            let _ = Shell_NotifyIconW(NIM_SETVERSION, &data);
        }

        Some(Self { hwnd, icon })
    }

    fn show_notification(&self, title: &str, body: &str) -> bool {
        use windows_sys::Win32::UI::Shell::{
            Shell_NotifyIconW, NIF_INFO, NIIF_LARGE_ICON, NIIF_USER, NIM_MODIFY,
        };

        let mut data = self.data();
        data.uFlags = NIF_INFO;
        data.dwInfoFlags = NIIF_USER | NIIF_LARGE_ICON;
        data.hBalloonIcon = self.icon;
        data.Anonymous.uTimeout = 10_000;
        copy_utf16_to_fixed(&mut data.szInfoTitle, &limit_notification_text(title, 63));
        copy_utf16_to_fixed(&mut data.szInfo, &limit_notification_text(body, 255));

        unsafe { Shell_NotifyIconW(NIM_MODIFY, &data) != 0 }
    }

    fn data(&self) -> windows_sys::Win32::UI::Shell::NOTIFYICONDATAW {
        use std::mem::size_of;
        use windows_sys::Win32::UI::Shell::NOTIFYICONDATAW;

        NOTIFYICONDATAW {
            cbSize: size_of::<NOTIFYICONDATAW>() as u32,
            hWnd: self.hwnd,
            uID: TRAY_ICON_ID,
            ..Default::default()
        }
    }
}

#[cfg(windows)]
impl TrayMessageWindow {
    fn create(window: &tao::window::Window, proxy: EventLoopProxy<DesktopEvent>) -> Option<Self> {
        use std::ptr;
        use windows_sys::Win32::{
            Foundation::{HINSTANCE, HWND},
            UI::WindowsAndMessaging::{
                CreateWindowExW, RegisterClassW, SetWindowLongPtrW, GWLP_USERDATA, WNDCLASSW,
                WS_OVERLAPPED,
            },
        };

        let hinstance = window.hinstance() as HINSTANCE;
        let class_name = wide_null("TrimPromptTrayMessageWindow");
        let window_name = wide_null("TrimPromptTrayMessageWindow");
        let window_class = WNDCLASSW {
            lpfnWndProc: Some(tray_message_window_proc),
            hInstance: hinstance,
            lpszClassName: class_name.as_ptr(),
            ..Default::default()
        };

        unsafe {
            let _ = RegisterClassW(&window_class);
        }

        let state = Box::new(TrayMessageState { proxy });
        let state_ptr = Box::into_raw(state);
        let hwnd = unsafe {
            CreateWindowExW(
                0,
                class_name.as_ptr(),
                window_name.as_ptr(),
                WS_OVERLAPPED,
                0,
                0,
                0,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                hinstance,
                ptr::null(),
            )
        };

        if hwnd.is_null() {
            unsafe {
                drop(Box::from_raw(state_ptr));
            }
            return None;
        }

        unsafe {
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, state_ptr as isize);
        }

        Some(Self {
            hwnd: hwnd as HWND,
            state: state_ptr,
        })
    }
}

#[cfg(windows)]
impl Drop for TrayMessageWindow {
    fn drop(&mut self) {
        unsafe {
            let _ = windows_sys::Win32::UI::WindowsAndMessaging::SetWindowLongPtrW(
                self.hwnd,
                windows_sys::Win32::UI::WindowsAndMessaging::GWLP_USERDATA,
                0,
            );
            let _ = windows_sys::Win32::UI::WindowsAndMessaging::DestroyWindow(self.hwnd);
            drop(Box::from_raw(self.state));
        }
    }
}

#[cfg(windows)]
unsafe extern "system" fn tray_message_window_proc(
    hwnd: windows_sys::Win32::Foundation::HWND,
    message: u32,
    wparam: windows_sys::Win32::Foundation::WPARAM,
    lparam: windows_sys::Win32::Foundation::LPARAM,
) -> windows_sys::Win32::Foundation::LRESULT {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        DefWindowProcW, GetWindowLongPtrW, GWLP_USERDATA,
    };

    if message == tray_callback_message() {
        if let Some(event) = tray_event_from_parts(hwnd, wparam, lparam) {
            let state_ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut TrayMessageState;
            if !state_ptr.is_null() {
                let _ = (*state_ptr).proxy.send_event(event);
                return 0;
            }
        }
    }

    DefWindowProcW(hwnd, message, wparam, lparam)
}

#[cfg(windows)]
impl Drop for TrayIcon {
    fn drop(&mut self) {
        unsafe {
            let data = self.data();
            let _ = windows_sys::Win32::UI::Shell::Shell_NotifyIconW(
                windows_sys::Win32::UI::Shell::NIM_DELETE,
                &data,
            );
            let _ = windows_sys::Win32::UI::WindowsAndMessaging::DestroyIcon(self.icon);
        }
    }
}

#[cfg(windows)]
fn install_tray_message_window(
    window: &tao::window::Window,
    proxy: EventLoopProxy<DesktopEvent>,
) -> Option<TrayMessageWindow> {
    TrayMessageWindow::create(window, proxy)
}

#[cfg(not(windows))]
fn install_tray_message_window(
    _window: &tao::window::Window,
    _proxy: EventLoopProxy<DesktopEvent>,
) -> Option<()> {
    None
}

#[cfg(windows)]
fn install_tray_icon(
    window: &tao::window::Window,
    message_window: Option<&TrayMessageWindow>,
) -> Option<TrayIcon> {
    message_window.and_then(|message_window| TrayIcon::install(window, message_window))
}

#[cfg(windows)]
fn show_desktop_notification(
    tray_icon: Option<&TrayIcon>,
    notification: &DesktopNotification,
    log_path: &Path,
) {
    write_notification_log(
        log_path,
        &format!(
            "windows notification state: {}",
            query_user_notification_state_label()
        ),
    );
    write_notification_log(
        log_path,
        &format!(
            "notification requested: title={} body_chars={}",
            notification.title,
            notification.body.chars().count()
        ),
    );

    if let Some(tray_icon) = tray_icon {
        let sent = tray_icon.show_notification(&notification.title, &notification.body);
        write_notification_log(log_path, &format!("tray notification sent={sent}"));
    } else {
        write_notification_log(log_path, "tray notification unavailable");
    }
}

#[cfg(not(windows))]
fn install_tray_icon(_window: &tao::window::Window, _message_window: Option<&()>) -> Option<()> {
    None
}

#[cfg(not(windows))]
fn show_desktop_notification(
    _tray_icon: Option<&()>,
    _notification: &DesktopNotification,
    _log_path: &Path,
) {
}

#[cfg(windows)]
fn query_user_notification_state_label() -> &'static str {
    use windows_sys::Win32::UI::Shell::{
        SHQueryUserNotificationState, QUNS_ACCEPTS_NOTIFICATIONS, QUNS_APP, QUNS_BUSY,
        QUNS_NOT_PRESENT, QUNS_PRESENTATION_MODE, QUNS_QUIET_TIME, QUNS_RUNNING_D3D_FULL_SCREEN,
    };

    let mut state = 0;
    let ok = unsafe { SHQueryUserNotificationState(&mut state) } >= 0;
    if !ok {
        return "query_failed";
    }

    match state {
        QUNS_ACCEPTS_NOTIFICATIONS => "accepts_notifications",
        QUNS_APP => "app_mode",
        QUNS_BUSY => "busy",
        QUNS_NOT_PRESENT => "not_present",
        QUNS_PRESENTATION_MODE => "presentation_mode",
        QUNS_QUIET_TIME => "quiet_time",
        QUNS_RUNNING_D3D_FULL_SCREEN => "d3d_full_screen",
        _ => "unknown",
    }
}

#[cfg(windows)]
fn tray_event_from_parts(
    message_hwnd: windows_sys::Win32::Foundation::HWND,
    wparam: windows_sys::Win32::Foundation::WPARAM,
    lparam: windows_sys::Win32::Foundation::LPARAM,
) -> Option<DesktopEvent> {
    use windows_sys::Win32::UI::{
        Shell::{NINF_KEY, NIN_SELECT},
        WindowsAndMessaging::{
            WM_CONTEXTMENU, WM_LBUTTONDBLCLK, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_RBUTTONUP,
        },
    };

    let legacy_icon_id = wparam as u32;
    let lparam = lparam as u32;
    let version4_icon_id = (lparam >> 16) & 0xffff;
    let is_legacy_event = legacy_icon_id == TRAY_ICON_ID;
    let is_version4_event = version4_icon_id == TRAY_ICON_ID;

    if !is_legacy_event && !is_version4_event {
        return None;
    }

    let event_code = if is_legacy_event {
        lparam
    } else {
        lparam & 0xffff
    };
    let key_select = NIN_SELECT | NINF_KEY;

    if matches!(
        event_code,
        WM_LBUTTONDOWN | WM_LBUTTONUP | WM_LBUTTONDBLCLK | NIN_SELECT
    ) || event_code == key_select
    {
        return Some(DesktopEvent::RestoreFromTray);
    }

    if matches!(event_code, WM_RBUTTONUP | WM_CONTEXTMENU) {
        return show_tray_menu(message_hwnd);
    }

    None
}

#[cfg(windows)]
fn show_tray_menu(message_hwnd: windows_sys::Win32::Foundation::HWND) -> Option<DesktopEvent> {
    use windows_sys::Win32::{
        Foundation::POINT,
        UI::WindowsAndMessaging::{
            AppendMenuW, CreatePopupMenu, DestroyMenu, GetCursorPos, PostMessageW,
            SetForegroundWindow, TrackPopupMenu, MF_SEPARATOR, MF_STRING, TPM_RETURNCMD,
            TPM_RIGHTBUTTON, WM_NULL,
        },
    };

    const MENU_OPEN: usize = 1001;
    const MENU_SETTINGS: usize = 1002;
    const MENU_EXIT: usize = 1003;

    unsafe {
        let menu = CreatePopupMenu();
        if menu.is_null() {
            return None;
        }

        let open = wide_null("開く");
        let settings = wide_null("設定を開く");
        let exit = wide_null("終了");
        let _ = AppendMenuW(menu, MF_STRING, MENU_OPEN, open.as_ptr());
        let _ = AppendMenuW(menu, MF_STRING, MENU_SETTINGS, settings.as_ptr());
        let _ = AppendMenuW(menu, MF_SEPARATOR, 0, std::ptr::null());
        let _ = AppendMenuW(menu, MF_STRING, MENU_EXIT, exit.as_ptr());

        let mut point = POINT { x: 0, y: 0 };
        let _ = GetCursorPos(&mut point);
        let _ = SetForegroundWindow(message_hwnd);
        let command = TrackPopupMenu(
            menu,
            TPM_RETURNCMD | TPM_RIGHTBUTTON,
            point.x,
            point.y,
            0,
            message_hwnd,
            std::ptr::null(),
        );
        let _ = PostMessageW(message_hwnd, WM_NULL, 0, 0);
        let _ = DestroyMenu(menu);

        match command as usize {
            MENU_OPEN => Some(DesktopEvent::RestoreFromTray),
            MENU_SETTINGS => Some(DesktopEvent::OpenSettingsFromTray),
            MENU_EXIT => Some(DesktopEvent::ExitRequested),
            _ => None,
        }
    }
}

#[cfg(windows)]
fn tray_callback_message() -> u32 {
    windows_sys::Win32::UI::WindowsAndMessaging::WM_APP + 1
}

#[cfg(windows)]
fn resource_id(id: u16) -> windows_sys::core::PCWSTR {
    id as usize as windows_sys::core::PCWSTR
}

#[cfg(windows)]
fn copy_utf16_to_fixed(target: &mut [u16], value: &str) {
    if target.is_empty() {
        return;
    }

    let max_text_len = target.len() - 1;
    for (slot, code_unit) in target
        .iter_mut()
        .take(max_text_len)
        .zip(value.encode_utf16())
    {
        *slot = code_unit;
    }
    target[max_text_len] = 0;
}

fn write_notification_log(path: &Path, message: &str) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let timestamp = format!("{:?}", std::time::SystemTime::now());
    if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "[{timestamp}] {message}");
    }
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

#[cfg(windows)]
fn wide_null(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

impl WindowControl {
    fn from_ipc_message(message: &str) -> Option<Self> {
        match message {
            "window:minimize" => Some(Self::Minimize),
            "window:maximize" => Some(Self::ToggleMaximize),
            "window:close" => Some(Self::Close),
            "window:drag" => Some(Self::Drag),
            _ => None,
        }
    }
}

fn handle_window_control(window: &tao::window::Window, control: WindowControl) {
    match control {
        WindowControl::Minimize => window.set_minimized(true),
        WindowControl::ToggleMaximize => window.set_maximized(!window.is_maximized()),
        WindowControl::Close => {}
        WindowControl::Drag => {
            let _ = window.drag_window();
        }
    }
}

fn restore_window_from_tray(window: &tao::window::Window) {
    window.set_visible(true);
    window.set_minimized(false);
    restore_window_to_foreground(window);
    window.set_focus();
}

#[cfg(windows)]
fn restore_window_to_foreground(window: &tao::window::Window) {
    use windows_sys::Win32::{
        Foundation::HWND,
        UI::WindowsAndMessaging::{BringWindowToTop, SetForegroundWindow, ShowWindow, SW_RESTORE},
    };

    let hwnd = window.hwnd() as HWND;
    unsafe {
        let _ = ShowWindow(hwnd, SW_RESTORE);
        let _ = BringWindowToTop(hwnd);
        let _ = SetForegroundWindow(hwnd);
    }
}

#[cfg(not(windows))]
fn restore_window_to_foreground(_window: &tao::window::Window) {}

#[cfg(windows)]
fn apply_window_theme(window: &tao::window::Window, theme: AppTheme) {
    use std::mem::size_of;
    use windows_sys::Win32::{
        Foundation::HWND,
        Graphics::Dwm::{
            DwmSetWindowAttribute, DWMWA_BORDER_COLOR, DWMWA_CAPTION_COLOR, DWMWA_TEXT_COLOR,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
        },
    };

    let is_dark = matches!(theme, AppTheme::Dark);
    window.set_theme(Some(if is_dark { Theme::Dark } else { Theme::Light }));

    let hwnd = window.hwnd() as HWND;
    let immersive_dark_mode = if is_dark { 1i32 } else { 0i32 };
    let caption_color = titlebar_color(theme);
    let border_color = if is_dark {
        color_ref(58, 58, 58)
    } else {
        color_ref(201, 208, 216)
    };
    let text_color = if is_dark {
        color_ref(242, 242, 234)
    } else {
        color_ref(24, 33, 43)
    };

    unsafe {
        set_dwm_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &immersive_dark_mode);
        set_dwm_attribute(hwnd, DWMWA_CAPTION_COLOR, &caption_color);
        set_dwm_attribute(hwnd, DWMWA_BORDER_COLOR, &border_color);
        set_dwm_attribute(hwnd, DWMWA_TEXT_COLOR, &text_color);
    }

    unsafe fn set_dwm_attribute<T>(hwnd: HWND, attribute: i32, value: &T) {
        let _ = DwmSetWindowAttribute(
            hwnd,
            attribute as u32,
            value as *const T as *const std::ffi::c_void,
            size_of::<T>() as u32,
        );
    }
}

#[cfg(not(windows))]
fn apply_window_theme(_window: &tao::window::Window, _theme: AppTheme) {}

#[cfg(windows)]
fn titlebar_color(theme: AppTheme) -> u32 {
    let (red, green, blue) = titlebar_rgb(theme);
    color_ref(red, green, blue)
}

#[cfg(windows)]
fn titlebar_rgb(theme: AppTheme) -> (u8, u8, u8) {
    if matches!(theme, AppTheme::Dark) {
        (32, 32, 32)
    } else {
        (232, 237, 243)
    }
}

#[cfg(windows)]
fn color_ref(red: u8, green: u8, blue: u8) -> u32 {
    u32::from(red) | (u32::from(green) << 8) | (u32::from(blue) << 16)
}

fn handle_protocol_request(
    app: &EmbeddedWebApp,
    request: Request<Vec<u8>>,
) -> Response<Cow<'static, [u8]>> {
    let response = app.handle_request(
        request.method().as_str(),
        request.uri().path(),
        request.body(),
    );
    protocol_response(response)
}

struct ProtocolJob {
    app: Arc<EmbeddedWebApp>,
    request: Request<Vec<u8>>,
    responder: RequestAsyncResponder,
}

impl ProtocolJob {
    fn run(self) {
        let response = handle_protocol_request(&self.app, self.request);
        self.responder.respond(response);
    }

    fn reject_busy(self) {
        let response = LocalAppResponse {
            status: 503,
            content_type: "application/json; charset=utf-8".to_string(),
            body: serde_json::to_vec(&serde_json::json!({
                "error": "処理が混み合っています。少し待ってから再試行してください。"
            }))
            .expect("static busy response should serialize"),
        };
        self.responder.respond(protocol_response(response));
    }
}

struct BoundedExecutor<T> {
    sender: SyncSender<T>,
    _workers: Vec<JoinHandle<()>>,
}

impl<T: Send + 'static> BoundedExecutor<T> {
    fn new(
        worker_count: usize,
        queue_capacity: usize,
        process: impl Fn(T) + Send + Sync + 'static,
    ) -> Self {
        assert!(worker_count > 0, "worker_count must be greater than zero");
        let (sender, receiver) = mpsc::sync_channel(queue_capacity);
        let receiver = Arc::new(Mutex::new(receiver));
        let process = Arc::new(process);
        let mut workers = Vec::with_capacity(worker_count);

        for _ in 0..worker_count {
            let receiver = Arc::clone(&receiver);
            let process = Arc::clone(&process);
            workers.push(std::thread::spawn(move || loop {
                let item = {
                    let Ok(receiver) = receiver.lock() else {
                        return;
                    };
                    receiver.recv()
                };
                let Ok(item) = item else {
                    return;
                };
                process(item);
            }));
        }

        Self {
            sender,
            _workers: workers,
        }
    }

    fn submit(&self, item: T) -> std::result::Result<(), T> {
        match self.sender.try_send(item) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(item) | TrySendError::Disconnected(item)) => Err(item),
        }
    }
}

fn protocol_response(response: LocalAppResponse) -> Response<Cow<'static, [u8]>> {
    Response::builder()
        .status(response.status)
        .header(CONTENT_TYPE, response.content_type)
        .header("Cache-Control", "no-store")
        .body(Cow::Owned(response.body))
        .unwrap_or_else(|error| {
            Response::builder()
                .status(500)
                .header(CONTENT_TYPE, "text/plain; charset=utf-8")
                .body(Cow::Owned(
                    format!("failed to build protocol response: {error}").into_bytes(),
                ))
                .expect("fallback protocol response should be valid")
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_identity_uses_trim_prompt_with_legacy_single_instance_compatibility() {
        assert_eq!(APP_WINDOW_TITLE, "TrimPrompt");
        assert_eq!(APP_USER_MODEL_ID, "TrimPrompt.Desktop");
        assert_eq!(
            SINGLE_INSTANCE_MUTEX_NAME,
            "Local\\TrimPrompt.Desktop.SingleInstance"
        );
        assert_eq!(
            LEGACY_SINGLE_INSTANCE_MUTEX_NAME,
            "Local\\PromptCompressor.Desktop.SingleInstance"
        );
    }

    #[cfg(windows)]
    #[test]
    fn named_mutex_allows_only_one_desktop_instance() {
        let mutex_name = format!("Local\\TrimPrompt.Desktop.Test.{}", std::process::id());
        let first = try_acquire_named_mutex(&mutex_name)
            .expect("create first mutex")
            .expect("first instance should acquire mutex");

        assert!(try_acquire_named_mutex(&mutex_name)
            .expect("check duplicate mutex")
            .is_none());

        drop(first);
        assert!(try_acquire_named_mutex(&mutex_name)
            .expect("reacquire released mutex")
            .is_some());
    }

    fn valid_smoke_result() -> serde_json::Value {
        serde_json::json!({
            "profile": "internal_llm",
            "runtime": "llama_cpp_embedded",
            "should_send_original": false,
            "distilled_prompt": "検索ボタン押下時のみAPIを呼ぶ。",
            "metrics": {
                "input_characters": 100,
                "output_characters": 20
            }
        })
    }

    #[test]
    fn packaged_smoke_request_uses_the_embedded_profile_and_level_two() {
        let body: serde_json::Value = serde_json::from_slice(
            &packaged_smoke_request_body().expect("serialize smoke request"),
        )
        .expect("parse smoke request");

        assert_eq!(body["profile"], "internal_llm");
        assert_eq!(body["compression_level"], 2);
        assert_eq!(body["task_type"], "coding");
        assert_eq!(body["compression_mode"], "codex_optimized");
        assert_eq!(body["input_text"], PACKAGED_SMOKE_INPUT);
    }

    #[test]
    fn packaged_smoke_validation_requires_real_compression() {
        assert!(validate_packaged_smoke_result(&valid_smoke_result()).is_ok());

        let mut fallback = valid_smoke_result();
        fallback["should_send_original"] = serde_json::Value::Bool(true);
        assert!(validate_packaged_smoke_result(&fallback).is_err());

        let mut unchanged = valid_smoke_result();
        unchanged["metrics"]["output_characters"] = serde_json::json!(100);
        assert!(validate_packaged_smoke_result(&unchanged).is_err());
    }

    #[test]
    fn protocol_executor_runs_tasks_outside_the_calling_thread() {
        let caller = std::thread::current().id();
        let (tx, rx) = mpsc::channel();
        let executor = BoundedExecutor::new(1, 1, move |()| {
            tx.send(std::thread::current().id())
                .expect("send worker id");
        });

        executor.submit(()).expect("submit protocol task");
        let worker_thread = rx.recv().expect("receive worker id");
        assert_ne!(worker_thread, caller);
    }

    #[test]
    fn protocol_executor_rejects_work_when_workers_and_queue_are_full() {
        let gate = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
        let worker_gate = Arc::clone(&gate);
        let (started_tx, started_rx) = mpsc::channel();
        let (completed_tx, completed_rx) = mpsc::channel();
        let executor = BoundedExecutor::new(1, 1, move |value| {
            if value == 1 {
                started_tx.send(()).expect("signal worker start");
                let (lock, condition) = &*worker_gate;
                let mut released = lock.lock().expect("lock test gate");
                while !*released {
                    released = condition.wait(released).expect("wait for test gate");
                }
            }
            completed_tx.send(value).expect("signal completed task");
        });

        executor.submit(1).expect("start first task");
        started_rx.recv().expect("worker should start");
        executor.submit(2).expect("queue second task");
        assert_eq!(executor.submit(3), Err(3));

        let (lock, condition) = &*gate;
        *lock.lock().expect("lock test gate") = true;
        condition.notify_one();
        assert_eq!(completed_rx.recv().expect("first task completes"), 1);
        assert_eq!(completed_rx.recv().expect("second task completes"), 2);
    }

    #[test]
    fn desktop_notification_normalizes_and_limits_ipc_text() {
        let long_body = format!("line one\n{}", "x".repeat(MAX_NOTIFICATION_BODY_CHARS + 20));
        let message = format!(
            "notification:{}",
            serde_json::json!({ "title": "  title\ntext  ", "body": long_body })
        );

        let notification =
            DesktopNotification::from_ipc_message(&message).expect("valid notification");

        assert_eq!(notification.title, "title text");
        assert_eq!(
            notification.body.chars().count(),
            MAX_NOTIFICATION_BODY_CHARS
        );
        assert!(!notification.body.contains('\n'));
    }
}
