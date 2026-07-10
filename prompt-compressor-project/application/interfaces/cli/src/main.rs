use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use prompt_compressor_core::{
    CompressionConstraints, CompressionLevel, CompressionMode, CompressionRequest,
    CompressionService, ConfiguredRuntimeBackend, ProfileRegistry, RequestSource, RequestTarget,
    RuntimeBackend, TaskType,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Parser)]
#[command(name = "prompt-compressor")]
#[command(about = "Local-first prompt compression scaffold for Codex workflows")]
struct Cli {
    #[arg(long, value_name = "DIR")]
    settings_dir: Option<PathBuf>,

    #[arg(value_name = "PROMPT")]
    prompt: Option<String>,

    #[arg(long, value_name = "FILE")]
    file: Option<PathBuf>,

    #[arg(long)]
    stdin: bool,

    #[arg(long, default_value = "internal_llm")]
    profile: String,

    #[arg(long, value_enum, default_value_t = ModeArg::CodexOptimized)]
    mode: ModeArg,

    #[arg(long, value_enum, default_value_t = FormatArg::Text)]
    format: FormatArg,

    #[arg(long, default_value_t = 2)]
    level: u8,

    #[arg(long)]
    copy: bool,

    #[arg(long)]
    list_profiles: bool,

    #[arg(long, value_delimiter = ',', value_name = "PROFILE1,PROFILE2")]
    compare_profiles: Vec<String>,

    #[arg(long, value_name = "FILE")]
    eval_fixture: Option<PathBuf>,

    #[arg(long, value_delimiter = ',', value_name = "1,2,3")]
    eval_levels: Vec<u8>,

    #[arg(long, value_name = "N")]
    eval_case_limit: Option<usize>,

    #[arg(long, value_name = "N", default_value_t = 0)]
    eval_case_offset: usize,

    #[arg(long)]
    eval_progress: bool,
}

#[derive(Clone, Debug, ValueEnum)]
enum FormatArg {
    Text,
    Json,
}

#[derive(Clone, Debug, ValueEnum)]
enum ModeArg {
    Lossless,
    InstructionExtract,
    CodexOptimized,
    PrivacyRedaction,
    DeveloperMode,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let settings_dir = resolve_settings_dir(cli.settings_dir.as_deref())?;
    let profiles_path = settings_dir.join("compression-profiles.yaml");
    let registry = ProfileRegistry::from_path(&profiles_path)
        .with_context(|| format!("failed to load profiles from {}", profiles_path.display()))?;

    if cli.list_profiles {
        for profile in registry.list_selectable() {
            println!("{}\t{}", profile.id, profile.label);
        }
        return Ok(());
    }

    if let Some(eval_fixture) = &cli.eval_fixture {
        let backend = ConfiguredRuntimeBackend::from_settings_dir(&settings_dir)
            .context("failed to initialize configured runtime backend")?;
        let service = CompressionService::new(registry, backend);
        let report = run_evaluation(
            &service,
            eval_fixture,
            &cli.profile,
            &cli.eval_levels,
            cli.eval_case_limit,
            cli.eval_case_offset,
            cli.eval_progress,
        )?;
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .context("failed to serialize evaluation report")?
        );
        if !report.passed {
            bail!(
                "evaluation failed: {:.3}% failure rate across {} runs",
                report.failure_rate * 100.0,
                report.run_count
            );
        }
        return Ok(());
    }

    let input = read_input(&cli)?;
    let request = CompressionRequest {
        input_text: input,
        task_type: TaskType::Coding,
        compression_mode: map_mode(cli.mode),
        compression_level: CompressionLevel::from_u8(cli.level)?,
        profile: cli.profile,
        constraints: CompressionConstraints::default(),
        target: RequestTarget::codex_default(),
        source: RequestSource::Cli,
    };

    let backend = ConfiguredRuntimeBackend::from_settings_dir(&settings_dir)
        .context("failed to initialize configured runtime backend")?;
    let service = CompressionService::new(registry, backend);

    if !cli.compare_profiles.is_empty() {
        let results = service.compare_profiles(request, &cli.compare_profiles);
        for result in results {
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &result.context("profile comparison failed while serializing output")?
                )?
            );
        }
        return Ok(());
    }

    let result = service.compress(request)?;

    if cli.copy {
        eprintln!("copy support is not wired yet; use the displayed output for now.");
    }

    match cli.format {
        FormatArg::Text => {
            println!("{}", result.distilled_prompt);
        }
        FormatArg::Json => {
            println!(
                "{}",
                serde_json::to_string_pretty(&result)
                    .context("failed to serialize CompressionResult as JSON")?
            );
        }
    }

    Ok(())
}

#[derive(Debug, Deserialize)]
struct EvaluationFixture {
    schema_version: u32,
    id: String,
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    task_type: Option<TaskType>,
    #[serde(default)]
    compression_mode: Option<CompressionMode>,
    #[serde(default)]
    failure_policy: Option<EvaluationFailurePolicy>,
    levels: BTreeMap<String, EvaluationLevelConfig>,
    cases: Vec<EvaluationCase>,
}

#[derive(Debug, Deserialize)]
struct EvaluationFailurePolicy {
    max_failure_rate: f32,
}

#[derive(Debug, Deserialize)]
struct EvaluationLevelConfig {
    #[serde(default)]
    min_average_character_ratio: Option<f32>,
    #[serde(default)]
    max_average_character_ratio: Option<f32>,
    #[serde(default)]
    max_case_character_ratio: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct EvaluationCase {
    id: String,
    input_text: String,
    #[serde(default)]
    task_type: Option<TaskType>,
    #[serde(default)]
    compression_mode: Option<CompressionMode>,
    #[serde(default)]
    required_terms: Vec<String>,
    #[serde(default)]
    required_marker_groups: Vec<String>,
    #[serde(default)]
    levels: Option<Vec<u8>>,
}

#[derive(Debug, Serialize)]
struct EvaluationReport {
    fixture_id: String,
    profile: String,
    case_count: usize,
    run_count: usize,
    failure_count: usize,
    failure_rate: f32,
    max_failure_rate: f32,
    passed: bool,
    levels: Vec<EvaluationLevelReport>,
    samples: Vec<EvaluationSampleReport>,
    failures: Vec<EvaluationFailureReport>,
}

#[derive(Debug, Serialize)]
struct EvaluationLevelReport {
    level: u8,
    run_count: usize,
    failure_count: usize,
    failure_rate: f32,
    average_character_ratio: f32,
    min_average_character_ratio: Option<f32>,
    max_average_character_ratio: Option<f32>,
    max_case_character_ratio: Option<f32>,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct EvaluationFailureReport {
    case_id: String,
    level: u8,
    character_ratio: Option<f32>,
    should_send_original: Option<bool>,
    fallback_reason: Option<String>,
    reasons: Vec<String>,
    missing_terms: Vec<String>,
    missing_marker_groups: Vec<String>,
    output_preview: Option<String>,
}

#[derive(Debug, Serialize)]
struct EvaluationSampleReport {
    case_id: String,
    level: u8,
    character_ratio: Option<f32>,
    should_send_original: Option<bool>,
    fallback_reason: Option<String>,
    output_preview: Option<String>,
}

#[derive(Debug, Default)]
struct EvaluationLevelAccumulator {
    run_count: usize,
    failure_count: usize,
    ratio_sum: f32,
}

fn run_evaluation<B: RuntimeBackend>(
    service: &CompressionService<B>,
    fixture_path: &Path,
    fallback_profile: &str,
    requested_levels: &[u8],
    case_limit: Option<usize>,
    case_offset: usize,
    progress: bool,
) -> Result<EvaluationReport> {
    let fixture_text = fs::read_to_string(fixture_path)
        .with_context(|| format!("failed to read eval fixture: {}", fixture_path.display()))?;
    let fixture: EvaluationFixture =
        serde_json::from_str(&fixture_text).context("failed to parse eval fixture JSON")?;
    if fixture.schema_version != 1 {
        bail!(
            "unsupported evaluation schema_version: {}",
            fixture.schema_version
        );
    }

    let profile = fixture
        .profile
        .clone()
        .unwrap_or_else(|| fallback_profile.to_string());
    let levels = resolve_evaluation_levels(&fixture, requested_levels)?;
    let max_failure_rate = fixture
        .failure_policy
        .as_ref()
        .map(|policy| policy.max_failure_rate)
        .unwrap_or(0.01);
    let cases: Vec<_> = fixture
        .cases
        .iter()
        .skip(case_offset)
        .take(case_limit.unwrap_or(usize::MAX))
        .collect();
    let total_runs = cases
        .iter()
        .map(|case| {
            case.levels
                .as_deref()
                .unwrap_or(&levels)
                .iter()
                .filter(|level| levels.contains(level))
                .count()
        })
        .sum::<usize>();

    let mut level_accumulators = BTreeMap::<u8, EvaluationLevelAccumulator>::new();
    let mut failures = Vec::new();
    let mut samples = Vec::new();
    let mut run_count = 0usize;
    let mut failure_count = 0usize;

    for case in cases.iter() {
        let case_levels = case.levels.as_deref().unwrap_or(&levels);
        for level in case_levels.iter().copied() {
            if !levels.contains(&level) {
                continue;
            }

            let level_config = fixture
                .levels
                .get(&level.to_string())
                .with_context(|| format!("evaluation fixture does not define level {level}"))?;
            run_count += 1;
            let accumulator = level_accumulators.entry(level).or_default();
            accumulator.run_count += 1;
            if progress {
                eprintln!(
                    "[eval] {}/{} start case={} level={}",
                    run_count, total_runs, case.id, level
                );
            }
            let started_at = Instant::now();

            let request = CompressionRequest {
                input_text: case.input_text.clone(),
                task_type: case
                    .task_type
                    .clone()
                    .or_else(|| fixture.task_type.clone())
                    .unwrap_or(TaskType::Coding),
                compression_mode: case
                    .compression_mode
                    .clone()
                    .or_else(|| fixture.compression_mode.clone())
                    .unwrap_or(CompressionMode::CodexOptimized),
                compression_level: CompressionLevel::from_u8(level)?,
                profile: profile.clone(),
                constraints: CompressionConstraints::default(),
                target: RequestTarget::codex_default(),
                source: RequestSource::Cli,
            };

            let run = service.compress(request);
            let mut reasons = Vec::new();
            let mut missing_terms = Vec::new();
            let mut missing_marker_groups = Vec::new();
            let mut character_ratio = None;
            let mut should_send_original = None;
            let mut fallback_reason = None;
            let mut output_preview = None;

            match run {
                Ok(result) => {
                    let output = result.distilled_prompt.trim();
                    character_ratio = Some(result.metrics.character_ratio);
                    should_send_original = Some(result.should_send_original);
                    fallback_reason = result.fallback_reason.clone();
                    output_preview = Some(trim_preview(output, 120));
                    accumulator.ratio_sum += result.metrics.character_ratio;

                    if result.should_send_original {
                        reasons.push("should_send_original".to_string());
                    }
                    if let Some(max_case_ratio) = level_config.max_case_character_ratio {
                        if result.metrics.character_ratio > max_case_ratio {
                            reasons.push(format!(
                                "character_ratio {:.3} exceeds max_case {:.3}",
                                result.metrics.character_ratio, max_case_ratio
                            ));
                        }
                    }

                    missing_terms = missing_required_terms(output, &case.required_terms);
                    if !missing_terms.is_empty() {
                        reasons.push("missing_required_terms".to_string());
                    }
                    missing_marker_groups =
                        missing_required_marker_groups(output, &case.required_marker_groups);
                    if !missing_marker_groups.is_empty() {
                        reasons.push("missing_required_marker_groups".to_string());
                    }
                }
                Err(error) => {
                    reasons.push(format!("runtime_error: {error}"));
                    accumulator.ratio_sum += 1.0;
                }
            }

            if progress {
                let status = if reasons.is_empty() { "ok" } else { "fail" };
                let ratio = character_ratio
                    .map(|ratio| format!("{ratio:.3}"))
                    .unwrap_or_else(|| "n/a".to_string());
                eprintln!(
                    "[eval] {}/{} done case={} level={} status={} ratio={} elapsed={:.1}s",
                    run_count,
                    total_runs,
                    case.id,
                    level,
                    status,
                    ratio,
                    started_at.elapsed().as_secs_f32()
                );
            }

            if !reasons.is_empty() {
                failure_count += 1;
                accumulator.failure_count += 1;
                failures.push(EvaluationFailureReport {
                    case_id: case.id.clone(),
                    level,
                    character_ratio,
                    should_send_original,
                    fallback_reason: fallback_reason.clone(),
                    reasons,
                    missing_terms,
                    missing_marker_groups,
                    output_preview: output_preview.clone(),
                });
            }
            samples.push(EvaluationSampleReport {
                case_id: case.id.clone(),
                level,
                character_ratio,
                should_send_original,
                fallback_reason,
                output_preview,
            });
        }
    }

    let mut level_reports = Vec::new();
    let mut passed = run_count > 0;
    for level in levels {
        let config = fixture
            .levels
            .get(&level.to_string())
            .with_context(|| format!("evaluation fixture does not define level {level}"))?;
        let accumulator = level_accumulators.remove(&level).unwrap_or_default();
        let average_ratio = if accumulator.run_count == 0 {
            1.0
        } else {
            accumulator.ratio_sum / accumulator.run_count as f32
        };
        let failure_rate = if accumulator.run_count == 0 {
            1.0
        } else {
            accumulator.failure_count as f32 / accumulator.run_count as f32
        };
        let level_passed = accumulator.run_count > 0
            && failure_rate <= max_failure_rate
            && config
                .min_average_character_ratio
                .map_or(true, |minimum| average_ratio >= minimum)
            && config
                .max_average_character_ratio
                .map_or(true, |maximum| average_ratio <= maximum);
        passed &= level_passed;
        level_reports.push(EvaluationLevelReport {
            level,
            run_count: accumulator.run_count,
            failure_count: accumulator.failure_count,
            failure_rate,
            average_character_ratio: average_ratio,
            min_average_character_ratio: config.min_average_character_ratio,
            max_average_character_ratio: config.max_average_character_ratio,
            max_case_character_ratio: config.max_case_character_ratio,
            passed: level_passed,
        });
    }

    let failure_rate = if run_count == 0 {
        1.0
    } else {
        failure_count as f32 / run_count as f32
    };
    passed &= failure_rate <= max_failure_rate;

    Ok(EvaluationReport {
        fixture_id: fixture.id.clone(),
        profile,
        case_count: cases.len(),
        run_count,
        failure_count,
        failure_rate,
        max_failure_rate,
        passed,
        levels: level_reports,
        samples,
        failures,
    })
}

fn resolve_evaluation_levels(
    fixture: &EvaluationFixture,
    requested_levels: &[u8],
) -> Result<Vec<u8>> {
    let mut levels = if requested_levels.is_empty() {
        fixture
            .levels
            .keys()
            .map(|key| {
                key.parse::<u8>()
                    .with_context(|| format!("invalid evaluation level key: {key}"))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        requested_levels.to_vec()
    };
    levels.sort_unstable();
    levels.dedup();
    for level in &levels {
        if !fixture.levels.contains_key(&level.to_string()) {
            bail!("evaluation fixture does not define level {level}");
        }
    }
    Ok(levels)
}

fn missing_required_terms(output: &str, required_terms: &[String]) -> Vec<String> {
    required_terms
        .iter()
        .filter(|term| !contains_required_term(output, term))
        .cloned()
        .collect()
}

fn contains_required_term(haystack: &str, needle: &str) -> bool {
    if contains_text(haystack, needle) {
        return true;
    }

    let compact_needle = compact_all_whitespace(needle);
    if compact_needle != needle.trim().to_lowercase()
        && compact_all_whitespace(haystack).contains(&compact_needle)
    {
        return true;
    }

    let compact_needle = compact_ascii_whitespace_if_code_like(needle);
    if compact_needle == needle.to_lowercase() {
        return contains_natural_compound_term(haystack, needle);
    }

    let compact_haystack = compact_ascii_whitespace_if_code_like(haystack);
    compact_haystack.contains(&compact_needle) || contains_natural_compound_term(haystack, needle)
}

fn contains_natural_compound_term(haystack: &str, needle: &str) -> bool {
    let parts: Vec<_> = needle
        .split_whitespace()
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    if parts.len() < 2 || !parts.iter().any(|part| !part.is_ascii()) {
        return false;
    }

    let haystack = haystack.to_lowercase();
    let compact_haystack = compact_all_whitespace(&haystack);
    parts.iter().all(|part| {
        let part = part.to_lowercase();
        haystack.contains(&part) || compact_haystack.contains(&compact_all_whitespace(&part))
    })
}

fn compact_ascii_whitespace_if_code_like(value: &str) -> String {
    let trimmed = value.trim();
    let has_ascii_alnum = trimmed
        .chars()
        .any(|character| character.is_ascii_alphanumeric());
    let has_whitespace = trimmed.chars().any(char::is_whitespace);
    let code_like = trimmed.chars().all(|character| {
        character.is_ascii_alphanumeric()
            || character.is_whitespace()
            || matches!(
                character,
                '_' | '-' | '.' | '/' | ':' | '+' | '#' | '@' | '=' | '(' | ')'
            )
    });

    if has_ascii_alnum && has_whitespace && code_like {
        trimmed
            .chars()
            .filter(|character| !character.is_whitespace())
            .flat_map(char::to_lowercase)
            .collect()
    } else {
        trimmed.to_lowercase()
    }
}

fn missing_required_marker_groups(output: &str, marker_groups: &[String]) -> Vec<String> {
    marker_groups
        .iter()
        .filter(|group| {
            group
                .split('|')
                .map(str::trim)
                .filter(|marker| !marker.is_empty())
                .all(|marker| !contains_marker_text(output, marker))
        })
        .cloned()
        .collect()
}

fn contains_marker_text(haystack: &str, marker: &str) -> bool {
    if contains_text(haystack, marker) {
        return true;
    }

    const ALIAS_GROUPS: &[&[&str]] = &[
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
            "残す",
        ],
        &[
            "増やさない",
            "増えない",
            "増加させない",
            "回避",
            "抑制",
            "最小限",
        ],
        &[
            "避ける",
            "避け",
            "回避",
            "禁止",
            "しない",
            "行わない",
            "不可",
            "不要",
            "なし",
        ],
        &["のみ", "だけ", "only"],
    ];

    ALIAS_GROUPS.iter().any(|aliases| {
        aliases.iter().any(|alias| contains_text(marker, alias))
            && aliases.iter().any(|alias| contains_text(haystack, alias))
    })
}

fn contains_text(haystack: &str, needle: &str) -> bool {
    haystack
        .to_lowercase()
        .contains(&needle.trim().to_lowercase())
}

fn compact_all_whitespace(value: &str) -> String {
    value
        .trim()
        .chars()
        .filter(|character| !character.is_whitespace())
        .flat_map(char::to_lowercase)
        .collect()
}

fn trim_preview(value: &str, max_chars: usize) -> String {
    let mut preview: String = value.chars().take(max_chars).collect();
    if value.chars().count() > max_chars {
        preview.push('…');
    }
    preview
}

fn read_input(cli: &Cli) -> Result<String> {
    let mut sources = 0;
    if cli.prompt.is_some() {
        sources += 1;
    }
    if cli.file.is_some() {
        sources += 1;
    }
    if cli.stdin {
        sources += 1;
    }

    if sources == 0 {
        bail!("provide a prompt, --file, or --stdin");
    }
    if sources > 1 {
        bail!("use only one input source at a time");
    }

    if let Some(prompt) = &cli.prompt {
        return Ok(prompt.clone());
    }

    if let Some(file) = &cli.file {
        return fs::read_to_string(file)
            .with_context(|| format!("failed to read input file: {}", file.display()));
    }

    let mut buffer = String::new();
    io::stdin()
        .read_to_string(&mut buffer)
        .context("failed to read stdin")?;
    Ok(buffer)
}

fn resolve_settings_dir(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }

    let cwd = std::env::current_dir().context("failed to determine current directory")?;
    find_upward_settings_dir(&cwd).ok_or_else(|| {
        anyhow::anyhow!(
            "could not find ./application/config directory from {}",
            cwd.display()
        )
    })
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

fn map_mode(mode: ModeArg) -> CompressionMode {
    match mode {
        ModeArg::Lossless => CompressionMode::Lossless,
        ModeArg::InstructionExtract => CompressionMode::InstructionExtract,
        ModeArg::CodexOptimized => CompressionMode::CodexOptimized,
        ModeArg::PrivacyRedaction => CompressionMode::PrivacyRedaction,
        ModeArg::DeveloperMode => CompressionMode::DeveloperMode,
    }
}

#[cfg(test)]
mod tests {
    use super::{contains_marker_text, contains_required_term, missing_required_marker_groups};

    #[test]
    fn treats_keep_marker_aliases_as_equivalent() {
        assert!(contains_marker_text(
            "既存処理と監査ログは変更なし。",
            "変更しない"
        ));
        assert!(contains_marker_text(
            "既存コードとテスト名は変更せず。",
            "変更しない"
        ));
        assert!(contains_marker_text(
            "既存処理と監査ログは変更なし。",
            "維持"
        ));
        assert!(contains_marker_text(
            "時刻/requestId/エラー文字列は改変せず。",
            "変更しない"
        ));
        assert!(contains_marker_text(
            "大規模リファクタリングは回避。",
            "避ける"
        ));
        assert!(contains_marker_text("再レンダリング回避。", "増やさない"));
        assert!(contains_marker_text(
            "再レンダリングを最小限に。",
            "増やさない"
        ));
        assert!(contains_marker_text("個人情報外部送信なし。", "送信しない"));
    }

    #[test]
    fn accepts_marker_group_when_alias_is_present() {
        let missing = missing_required_marker_groups(
            "既存処理と監査ログは変更なし。",
            &["変更しない|維持".to_string()],
        );

        assert!(missing.is_empty());
    }

    #[test]
    fn treats_code_like_required_terms_as_whitespace_insensitive() {
        assert!(contains_required_term(
            "HTTP400 と INVALID_CUSTOMER を返す。",
            "HTTP 400"
        ));
        assert!(!contains_required_term(
            "INVALID_CUSTOMER を返す。",
            "HTTP 400"
        ));
        assert!(contains_required_term(
            "月額コスト3万円以下。",
            "3 万円以下"
        ));
    }

    #[test]
    fn treats_natural_compound_required_terms_as_parts() {
        assert!(contains_required_term(
            "通知はWindowsのみ。アプリ内通知は禁止。",
            "Windows 通知"
        ));
        assert!(!contains_required_term(
            "通知はアプリ内のみ。",
            "Windows 通知"
        ));
    }
}
