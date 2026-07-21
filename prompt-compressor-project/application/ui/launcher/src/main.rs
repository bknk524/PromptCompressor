#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use serde::Deserialize;

const APP_TITLE: &str = "TrimPrompt";
const CPU_ENGINE_DIRECTORY: &str = "application/runtime/cpu";
const CPU_ENGINE_ENV: &str = "TRIMPROMPT_CPU_ENGINE";
const CPU_ENGINE_MODE_ENV: &str = "TRIMPROMPT_CPU_ENGINE_MODE";
const THREAD_MODE_ENV: &str = "TRIMPROMPT_THREAD_MODE";
const GENERATION_THREADS_ENV: &str = "TRIMPROMPT_GENERATION_THREADS";
const BATCH_THREADS_ENV: &str = "TRIMPROMPT_BATCH_THREADS";
const EXPECTED_BUILD_ID_ENV: &str = "TRIMPROMPT_EXPECTED_BUILD_ID";
const INFERENCE_COMPATIBILITY_ID_ENV: &str = "TRIMPROMPT_INFERENCE_COMPATIBILITY_ID";
const BUILD_ID: &str = match option_env!("TRIMPROMPT_BUILD_ID") {
    Some(value) => value,
    None => "development",
};
const INFERENCE_COMPATIBILITY_ID: &str = match option_env!("TRIMPROMPT_INFERENCE_COMPATIBILITY_ID")
{
    Some(value) => value,
    None => "development",
};
const CPU_ENGINE_SELECTION_SCHEMA_VERSION: u32 = 4;
const CPU_ENGINE_SELECTION_FILE: &str = "application/local/state/cpu-engine-selection-v1.json";
const UI_SETTINGS_FILE: &str = "application/local/state/ui-settings.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpuEngine {
    Compatible,
    Avx2,
    Avx512,
}

impl CpuEngine {
    fn name(self) -> &'static str {
        match self {
            Self::Compatible => "compatible",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
        }
    }

    fn executable_name(self) -> &'static str {
        match self {
            Self::Compatible => "TrimPrompt-compatible.exe",
            Self::Avx2 => "TrimPrompt-avx2.exe",
            Self::Avx512 => "TrimPrompt-avx512.exe",
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct CpuCapabilities {
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedCpuEngineSelection {
    schema_version: u32,
    build_id: String,
    inference_compatibility_id: String,
    cpu_key: String,
    cpu_engine: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct PersistedRuntimePreferences {
    cpu_engine: Option<String>,
    thread_mode: Option<String>,
    generation_threads: Option<u32>,
    batch_threads: Option<u32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RuntimePreferences {
    cpu_engine: Option<CpuEngine>,
    manual_threads: Option<RuntimeThreadCounts>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RuntimeThreadCounts {
    generation: u32,
    batch: u32,
}

fn main() {
    match run(std::env::args_os().skip(1).collect()) {
        Ok(exit_code) => std::process::exit(exit_code),
        Err(error) => {
            show_startup_error(&format!("TrimPrompt を起動できませんでした。\n\n{error:#}"));
            std::process::exit(1);
        }
    }
}

fn run(mut arguments: Vec<OsString>) -> Result<i32> {
    if let Some(process_id) = take_restart_after_pid(&mut arguments)? {
        wait_for_process_exit(process_id)?;
    }
    let launcher_path = std::env::current_exe().context("起動ファイルの場所を取得できません")?;
    let package_root = launcher_path
        .parent()
        .context("起動ファイルの親フォルダを取得できません")?;
    let capabilities = detect_cpu_capabilities();
    let runtime_preferences = load_runtime_preferences(package_root, capabilities);
    let preferred = runtime_preferences
        .cpu_engine
        .or_else(|| load_cpu_engine_selection(package_root, capabilities))
        .or_else(|| select_safe_initial_cpu_engine(capabilities))
        .context("このCPUはTrimPromptの互換版に必要なSSE4.2命令セットへ対応していません")?;
    let (engine, engine_path) = resolve_engine_path(package_root, preferred)?;

    let mut command = Command::new(&engine_path);
    command
        .args(&arguments)
        .env(CPU_ENGINE_ENV, engine.name())
        .env(
            CPU_ENGINE_MODE_ENV,
            if runtime_preferences.cpu_engine.is_some() {
                "manual"
            } else {
                "auto"
            },
        )
        .env(EXPECTED_BUILD_ID_ENV, BUILD_ID)
        .env(INFERENCE_COMPATIBILITY_ID_ENV, INFERENCE_COMPATIBILITY_ID);
    if let Some(threads) = runtime_preferences.manual_threads {
        command
            .env(THREAD_MODE_ENV, "manual")
            .env(GENERATION_THREADS_ENV, threads.generation.to_string())
            .env(BATCH_THREADS_ENV, threads.batch.to_string());
    } else {
        command.env(THREAD_MODE_ENV, "auto");
    }

    if should_wait_for_child(&arguments) {
        let status = command
            .status()
            .with_context(|| format!("{} を開始できません", engine_path.display()))?;
        return Ok(status.code().unwrap_or(1));
    }

    command
        .spawn()
        .with_context(|| format!("{} を開始できません", engine_path.display()))?;
    Ok(0)
}

fn take_restart_after_pid(arguments: &mut Vec<OsString>) -> Result<Option<u32>> {
    let Some(position) = arguments
        .iter()
        .position(|argument| argument == "--restart-after-pid")
    else {
        return Ok(None);
    };
    arguments.remove(position);
    let value = arguments
        .get(position)
        .context("--restart-after-pid にプロセスIDが指定されていません")?
        .to_string_lossy()
        .parse::<u32>()
        .context("--restart-after-pid のプロセスIDが不正です")?;
    arguments.remove(position);
    anyhow::ensure!(
        !arguments
            .iter()
            .any(|argument| argument == "--restart-after-pid"),
        "--restart-after-pid が重複しています"
    );
    Ok(Some(value))
}

#[cfg(windows)]
fn wait_for_process_exit(process_id: u32) -> Result<()> {
    use windows_sys::Win32::{
        Foundation::CloseHandle,
        System::Threading::{OpenProcess, WaitForSingleObject},
    };

    const SYNCHRONIZE_ACCESS: u32 = 0x0010_0000;
    const WAIT_FOREVER: u32 = u32::MAX;
    const WAIT_FAILED_RESULT: u32 = u32::MAX;

    let handle = unsafe { OpenProcess(SYNCHRONIZE_ACCESS, 0, process_id) };
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(87) {
            return Ok(());
        }
        return Err(error).context("以前のTrimPromptプロセスを確認できません");
    }

    let wait_result = unsafe { WaitForSingleObject(handle, WAIT_FOREVER) };
    unsafe {
        let _ = CloseHandle(handle);
    }
    anyhow::ensure!(
        wait_result != WAIT_FAILED_RESULT,
        "以前のTrimPromptプロセスの終了待機に失敗しました"
    );
    Ok(())
}

#[cfg(not(windows))]
fn wait_for_process_exit(_process_id: u32) -> Result<()> {
    Ok(())
}

fn resolve_engine_path(package_root: &Path, preferred: CpuEngine) -> Result<(CpuEngine, PathBuf)> {
    let engine_root = package_root.join(CPU_ENGINE_DIRECTORY);
    let engine_root = fs::canonicalize(&engine_root).with_context(|| {
        format!(
            "推論エンジンのフォルダがありません: {}",
            engine_root.display()
        )
    })?;

    // 最適化版が欠けている場合だけ互換版へ戻し、未検証の場所は実行しない。
    for &engine in fallback_order(preferred) {
        let candidate = engine_root.join(engine.executable_name());
        let Ok(candidate) = fs::canonicalize(&candidate) else {
            continue;
        };
        if candidate.starts_with(&engine_root) && candidate.is_file() {
            return Ok((engine, candidate));
        }
    }

    anyhow::bail!(
        "利用できるCPU推論エンジンがありません: {}",
        engine_root.display()
    )
}

fn fallback_order(preferred: CpuEngine) -> &'static [CpuEngine] {
    match preferred {
        CpuEngine::Avx512 => &[CpuEngine::Avx512, CpuEngine::Avx2, CpuEngine::Compatible],
        CpuEngine::Avx2 => &[CpuEngine::Avx2, CpuEngine::Compatible],
        CpuEngine::Compatible => &[CpuEngine::Compatible],
    }
}

fn should_wait_for_child(arguments: &[OsString]) -> bool {
    arguments
        .iter()
        .any(|argument| argument == "--package-smoke-test")
}

fn select_cpu_engine(capabilities: CpuCapabilities) -> Option<CpuEngine> {
    if supports_avx512_engine(capabilities) {
        return Some(CpuEngine::Avx512);
    }
    if supports_avx2_engine(capabilities) {
        Some(CpuEngine::Avx2)
    } else if supports_compatible_engine(capabilities) {
        Some(CpuEngine::Compatible)
    } else {
        None
    }
}

fn select_safe_initial_cpu_engine(capabilities: CpuCapabilities) -> Option<CpuEngine> {
    match select_cpu_engine(capabilities) {
        Some(CpuEngine::Avx512) => Some(CpuEngine::Avx2),
        engine => engine,
    }
}

fn load_cpu_engine_selection(
    package_root: &Path,
    capabilities: CpuCapabilities,
) -> Option<CpuEngine> {
    let bytes = fs::read(package_root.join(CPU_ENGINE_SELECTION_FILE)).ok()?;
    let record = serde_json::from_slice::<PersistedCpuEngineSelection>(&bytes).ok()?;
    if record.schema_version != CPU_ENGINE_SELECTION_SCHEMA_VERSION
        || record.build_id.trim().is_empty()
        || record.inference_compatibility_id != INFERENCE_COMPATIBILITY_ID
        || record.cpu_key != cpu_key(capabilities)
    {
        return None;
    }
    let engine = match record.cpu_engine.as_str() {
        "avx512" => CpuEngine::Avx512,
        "avx2" => CpuEngine::Avx2,
        "compatible" => CpuEngine::Compatible,
        _ => return None,
    };
    supports_engine(capabilities, engine).then_some(engine)
}

fn load_runtime_preferences(
    package_root: &Path,
    capabilities: CpuCapabilities,
) -> RuntimePreferences {
    let Ok(bytes) = fs::read(package_root.join(UI_SETTINGS_FILE)) else {
        return RuntimePreferences::default();
    };
    let Ok(settings) = serde_json::from_slice::<PersistedRuntimePreferences>(&bytes) else {
        return RuntimePreferences::default();
    };
    resolve_runtime_preferences(
        settings,
        capabilities,
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
    )
}

fn resolve_runtime_preferences(
    settings: PersistedRuntimePreferences,
    capabilities: CpuCapabilities,
    available_threads: usize,
) -> RuntimePreferences {
    let cpu_engine = match settings.cpu_engine.as_deref() {
        Some("compatible") => Some(CpuEngine::Compatible),
        Some("avx2") => Some(CpuEngine::Avx2),
        Some("avx512") => Some(CpuEngine::Avx512),
        _ => None,
    }
    .filter(|engine| supports_engine(capabilities, *engine));

    let maximum = u32::try_from(available_threads).unwrap_or(u32::MAX).max(1);
    let manual_threads = if settings.thread_mode.as_deref() == Some("manual") {
        settings
            .generation_threads
            .zip(settings.batch_threads)
            .filter(|(generation, batch)| {
                (1..=maximum).contains(generation) && (1..=maximum).contains(batch)
            })
            .map(|(generation, batch)| RuntimeThreadCounts { generation, batch })
    } else {
        None
    };

    RuntimePreferences {
        cpu_engine,
        manual_threads,
    }
}

fn supports_engine(capabilities: CpuCapabilities, engine: CpuEngine) -> bool {
    match engine {
        CpuEngine::Avx512 => supports_avx512_engine(capabilities),
        CpuEngine::Avx2 => supports_avx2_engine(capabilities),
        CpuEngine::Compatible => supports_compatible_engine(capabilities),
    }
}

fn cpu_key(capabilities: CpuCapabilities) -> String {
    let processor = std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".into());
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

fn supports_compatible_engine(capabilities: CpuCapabilities) -> bool {
    capabilities.sse42
}

fn supports_avx2_engine(capabilities: CpuCapabilities) -> bool {
    supports_compatible_engine(capabilities)
        && capabilities.avx2
        && capabilities.fma
        && capabilities.f16c
        && capabilities.bmi2
}

fn supports_avx512_engine(capabilities: CpuCapabilities) -> bool {
    supports_avx2_engine(capabilities)
        && capabilities.avx512f
        && capabilities.avx512cd
        && capabilities.avx512bw
        && capabilities.avx512dq
        && capabilities.avx512vl
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_cpu_capabilities() -> CpuCapabilities {
    CpuCapabilities {
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
fn detect_cpu_capabilities() -> CpuCapabilities {
    CpuCapabilities::default()
}

#[cfg(windows)]
fn show_startup_error(message: &str) {
    use std::ptr;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        MessageBoxW, MB_ICONERROR, MB_OK, MB_SETFOREGROUND,
    };

    let message = wide_null(message);
    let title = wide_null(APP_TITLE);
    unsafe {
        let _ = MessageBoxW(
            ptr::null_mut(),
            message.as_ptr(),
            title.as_ptr(),
            MB_OK | MB_ICONERROR | MB_SETFOREGROUND,
        );
    }
}

#[cfg(windows)]
fn wide_null(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(not(windows))]
fn show_startup_error(message: &str) {
    eprintln!("{message}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx2_engine_requires_every_compiled_instruction() {
        assert_eq!(
            select_cpu_engine(CpuCapabilities {
                sse42: true,
                avx2: true,
                fma: true,
                f16c: true,
                bmi2: true,
                ..Default::default()
            }),
            Some(CpuEngine::Avx2)
        );
        assert_eq!(
            select_cpu_engine(CpuCapabilities {
                sse42: true,
                avx2: true,
                fma: false,
                f16c: true,
                bmi2: true,
                ..Default::default()
            }),
            Some(CpuEngine::Compatible)
        );
        assert_eq!(
            select_cpu_engine(CpuCapabilities {
                sse42: true,
                avx2: true,
                fma: true,
                f16c: true,
                bmi2: false,
                ..Default::default()
            }),
            Some(CpuEngine::Compatible)
        );
    }

    #[test]
    fn compatible_engine_requires_sse42() {
        assert_eq!(
            select_safe_initial_cpu_engine(CpuCapabilities::default()),
            None
        );
        assert_eq!(
            select_safe_initial_cpu_engine(CpuCapabilities {
                sse42: true,
                ..Default::default()
            }),
            Some(CpuEngine::Compatible)
        );
    }

    #[test]
    fn cpu_key_tracks_sse42_and_bmi2_support() {
        let complete = CpuCapabilities {
            sse42: true,
            bmi2: true,
            ..Default::default()
        };
        let without_sse42 = CpuCapabilities {
            sse42: false,
            ..complete
        };
        let without_bmi2 = CpuCapabilities {
            bmi2: false,
            ..complete
        };

        assert_ne!(cpu_key(complete), cpu_key(without_sse42));
        assert_ne!(cpu_key(complete), cpu_key(without_bmi2));
    }

    #[test]
    fn avx512_engine_requires_the_complete_compiler_feature_group() {
        let complete = CpuCapabilities {
            sse42: true,
            avx2: true,
            fma: true,
            f16c: true,
            bmi2: true,
            avx512f: true,
            avx512cd: true,
            avx512bw: true,
            avx512dq: true,
            avx512vl: true,
        };
        assert_eq!(select_cpu_engine(complete), Some(CpuEngine::Avx512));

        let without_vl = CpuCapabilities {
            avx512vl: false,
            ..complete
        };
        assert_eq!(select_cpu_engine(without_vl), Some(CpuEngine::Avx2));
    }

    #[test]
    fn first_launch_stays_on_avx2_until_benchmark_selection_exists() {
        let complete = CpuCapabilities {
            sse42: true,
            avx2: true,
            fma: true,
            f16c: true,
            bmi2: true,
            avx512f: true,
            avx512cd: true,
            avx512bw: true,
            avx512dq: true,
            avx512vl: true,
        };
        assert_eq!(
            select_safe_initial_cpu_engine(complete),
            Some(CpuEngine::Avx2)
        );
        assert_eq!(select_cpu_engine(complete), Some(CpuEngine::Avx512));
    }

    #[test]
    fn fallback_order_degrades_one_engine_at_a_time() {
        assert_eq!(
            fallback_order(CpuEngine::Avx512),
            [CpuEngine::Avx512, CpuEngine::Avx2, CpuEngine::Compatible]
        );
        assert_eq!(
            fallback_order(CpuEngine::Avx2),
            [CpuEngine::Avx2, CpuEngine::Compatible]
        );
    }

    #[test]
    fn only_smoke_tests_wait_for_the_engine_to_exit() {
        assert!(should_wait_for_child(&[OsString::from(
            "--package-smoke-test"
        )]));
        assert!(!should_wait_for_child(&[]));
    }

    #[test]
    fn restart_wait_argument_is_consumed_before_engine_dispatch() {
        let mut arguments = vec![
            OsString::from("--restart-after-pid"),
            OsString::from("1234"),
            OsString::from("--package-smoke-test"),
        ];

        assert_eq!(
            take_restart_after_pid(&mut arguments).expect("restart argument"),
            Some(1234)
        );
        assert_eq!(arguments, [OsString::from("--package-smoke-test")]);
    }

    #[test]
    fn restart_wait_argument_rejects_missing_or_duplicate_values() {
        let mut missing = vec![OsString::from("--restart-after-pid")];
        assert!(take_restart_after_pid(&mut missing).is_err());

        let mut duplicate = vec![
            OsString::from("--restart-after-pid"),
            OsString::from("1"),
            OsString::from("--restart-after-pid"),
            OsString::from("2"),
        ];
        assert!(take_restart_after_pid(&mut duplicate).is_err());
    }

    #[test]
    fn persisted_selection_reuses_compatible_inference_builds() {
        let root = std::env::temp_dir().join(format!(
            "trimprompt-launcher-selection-{}",
            std::process::id()
        ));
        let path = root.join(CPU_ENGINE_SELECTION_FILE);
        fs::create_dir_all(path.parent().expect("selection parent"))
            .expect("create selection directory");
        let capabilities = CpuCapabilities {
            sse42: true,
            avx2: true,
            fma: true,
            f16c: true,
            bmi2: true,
            ..Default::default()
        };
        let record = serde_json::json!({
            "schema_version": CPU_ENGINE_SELECTION_SCHEMA_VERSION,
            "build_id": BUILD_ID,
            "inference_compatibility_id": INFERENCE_COMPATIBILITY_ID,
            "cpu_key": cpu_key(capabilities),
            "cpu_engine": "avx2"
        });
        fs::write(
            &path,
            serde_json::to_vec(&record).expect("serialize selection"),
        )
        .expect("write selection");
        assert_eq!(
            load_cpu_engine_selection(&root, capabilities),
            Some(CpuEngine::Avx2)
        );

        let mut stale = record;
        stale["build_id"] = serde_json::Value::String("stale-build".to_string());
        fs::write(
            &path,
            serde_json::to_vec(&stale).expect("serialize stale selection"),
        )
        .expect("write stale selection");
        assert_eq!(
            load_cpu_engine_selection(&root, capabilities),
            Some(CpuEngine::Avx2)
        );

        stale["inference_compatibility_id"] =
            serde_json::Value::String("incompatible-inference".to_string());
        fs::write(
            &path,
            serde_json::to_vec(&stale).expect("serialize incompatible selection"),
        )
        .expect("write incompatible selection");
        assert_eq!(load_cpu_engine_selection(&root, capabilities), None);

        fs::remove_dir_all(root).expect("remove selection directory");
    }

    #[test]
    fn manual_runtime_preferences_are_filtered_by_cpu_and_thread_limits() {
        let capabilities = CpuCapabilities {
            sse42: true,
            avx2: true,
            fma: true,
            f16c: true,
            bmi2: true,
            ..Default::default()
        };
        let settings = PersistedRuntimePreferences {
            cpu_engine: Some("avx2".into()),
            thread_mode: Some("manual".into()),
            generation_threads: Some(3),
            batch_threads: Some(4),
        };
        assert_eq!(
            resolve_runtime_preferences(settings, capabilities, 4),
            RuntimePreferences {
                cpu_engine: Some(CpuEngine::Avx2),
                manual_threads: Some(RuntimeThreadCounts {
                    generation: 3,
                    batch: 4,
                }),
            }
        );

        let unsupported = PersistedRuntimePreferences {
            cpu_engine: Some("avx512".into()),
            thread_mode: Some("manual".into()),
            generation_threads: Some(3),
            batch_threads: Some(4),
        };
        assert_eq!(
            resolve_runtime_preferences(unsupported, capabilities, 4),
            RuntimePreferences {
                cpu_engine: None,
                manual_threads: Some(RuntimeThreadCounts {
                    generation: 3,
                    batch: 4,
                }),
            }
        );

        let excessive_threads = PersistedRuntimePreferences {
            cpu_engine: Some("auto".into()),
            thread_mode: Some("manual".into()),
            generation_threads: Some(5),
            batch_threads: Some(4),
        };
        assert_eq!(
            resolve_runtime_preferences(excessive_threads, capabilities, 4),
            RuntimePreferences::default()
        );
    }
}
