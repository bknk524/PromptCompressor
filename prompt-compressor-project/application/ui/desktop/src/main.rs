#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::borrow::Cow;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

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
    WebContext, WebViewBuilder,
};

#[cfg(windows)]
use tao::{
    platform::windows::{IconExtWindows, WindowExtWindows},
    window::{Icon, Theme},
};

const APP_USER_MODEL_ID: &str = "PromptCompressor.Desktop";

#[derive(Debug, Parser)]
#[command(name = "prompt-compressor-desktop")]
#[command(about = "Native Windows shell for Prompt Compressor")]
struct Args {
    #[arg(long, value_name = "DIR")]
    settings_dir: Option<PathBuf>,
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

#[derive(Debug)]
enum WindowControl {
    Minimize,
    ToggleMaximize,
    Close,
    Drag,
}

fn main() {
    if let Err(error) = run() {
        show_startup_error(&error);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    set_app_user_model_id();
    let app = prepare_embedded_web_app(args.settings_dir)?;
    app.start_default_profile_warmup();

    run_window(app)
}

fn show_startup_error(error: &anyhow::Error) {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(PathBuf::from))
        .unwrap_or_else(std::env::temp_dir);
    let message = format!(
        "Prompt Compressor could not start.\n\n\
         This desktop package must keep PromptCompressor.exe next to the application folder.\n\
         Required shape:\n\n\
         PromptCompressor.exe\n\
         application\\config\\\n\
         application\\resources\\\n\
         application\\local\\models\\...\n\n\
         Error details:\n{error:#}\n"
    );

    let report_path = exe_dir.join("PromptCompressor_STARTUP_ERROR.txt");
    let report_path = match fs::write(&report_path, &message) {
        Ok(()) => report_path,
        Err(_) => {
            let fallback_path = std::env::temp_dir().join("PromptCompressor_STARTUP_ERROR.txt");
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
        .with_title("Prompt Compressor")
        .with_decorations(false)
        .with_inner_size(LogicalSize::new(980.0, 860.0))
        .with_min_inner_size(LogicalSize::new(520.0, 640.0))
        .build(&event_loop)?;
    configure_window_chrome(&window, initial_theme);
    let mut tray_message_window = install_tray_message_window(&window, event_loop.create_proxy());
    let mut tray_icon = install_tray_icon(&window, tray_message_window.as_ref());
    let tray_available = tray_icon.is_some();

    let mut web_context = WebContext::new(Some(webview_data_dir));
    let app = Arc::new(app);
    let protocol_app = app.clone();
    let desktop_proxy = event_loop.create_proxy();
    let webview = WebViewBuilder::new_with_web_context(&mut web_context)
        .with_custom_protocol("prompt-compressor".into(), move |_webview_id, request| {
            handle_protocol_request(&protocol_app, request)
        })
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
                handle_window_control(&window, control, tray_available);
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
                let _ = webview.evaluate_script("window.promptCompressorOpenSettings?.();");
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
                hide_window_to_tray(&window, tray_available);
            }
            _ => {}
        }
    });
}

fn configure_window_chrome(window: &tao::window::Window, theme: AppTheme) {
    window.set_title("Prompt Compressor");
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
        let title = value
            .get("title")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("圧縮完了")
            .to_string();
        let body = value
            .get("body")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())?
            .to_string();

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
        copy_utf16_to_fixed(&mut data.szTip, "Prompt Compressor");

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
        let class_name = wide_null("PromptCompressorTrayMessageWindow");
        let window_name = wide_null("PromptCompressorTrayMessageWindow");
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

fn handle_window_control(
    window: &tao::window::Window,
    control: WindowControl,
    tray_available: bool,
) {
    match control {
        WindowControl::Minimize => hide_window_to_tray(window, tray_available),
        WindowControl::ToggleMaximize => window.set_maximized(!window.is_maximized()),
        WindowControl::Close => hide_window_to_tray(window, tray_available),
        WindowControl::Drag => {
            let _ = window.drag_window();
        }
    }
}

fn hide_window_to_tray(window: &tao::window::Window, tray_available: bool) {
    if tray_available {
        window.set_visible(false);
    } else {
        window.set_minimized(true);
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
