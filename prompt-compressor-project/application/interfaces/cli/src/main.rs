use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use prompt_compressor_core::{
    CompressionConstraints, CompressionLevel, CompressionRequest, CompressionService,
    ConfiguredRuntimeBackend, ProfileRegistry, RequestSource, RequestTarget, RuntimeBackend,
    RuntimeTransformation,
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

    #[arg(long, value_enum, default_value_t = FormatArg::Text)]
    format: FormatArg,

    #[arg(long, default_value_t = 2)]
    level: u8,

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

    #[arg(long, value_enum, default_value_t = EvaluationStageArg::FinalPipeline)]
    eval_stage: EvaluationStageArg,
}

#[derive(Clone, Debug, ValueEnum)]
enum FormatArg {
    Text,
    Json,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum EvaluationStageArg {
    RawModel,
    FinalPipeline,
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
            EvaluationOptions {
                fixture_path: eval_fixture,
                fallback_profile: &cli.profile,
                requested_levels: &cli.eval_levels,
                case_limit: cli.eval_case_limit,
                case_offset: cli.eval_case_offset,
                progress: cli.eval_progress,
                evaluation_stage: cli.eval_stage,
            },
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
    min_case_character_ratio: Option<f32>,
    #[serde(default)]
    max_case_character_ratio: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct EvaluationCase {
    id: String,
    input_text: String,
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
    evaluation_stage: EvaluationStageArg,
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
    min_case_character_ratio: Option<f32>,
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
    input_text: String,
    character_ratio: Option<f32>,
    should_send_original: Option<bool>,
    fallback_reason: Option<String>,
    output_preview: Option<String>,
    raw_model_character_ratio: Option<f32>,
    runtime_character_ratio: Option<f32>,
    final_character_ratio: Option<f32>,
    raw_model_output: Option<String>,
    runtime_output: Option<String>,
    final_output: Option<String>,
    raw_model_removed_content_summary: Option<Vec<String>>,
    runtime_removed_content_summary: Option<Vec<String>>,
    final_removed_content_summary: Option<Vec<String>>,
    runtime_transformations: Vec<RuntimeTransformation>,
    application_fallback_applied: bool,
    final_should_send_original: Option<bool>,
    final_fallback_reason: Option<String>,
}

#[derive(Debug, Default)]
struct EvaluationLevelAccumulator {
    run_count: usize,
    failure_count: usize,
    ratio_sum: f32,
}

struct EvaluationOptions<'a> {
    fixture_path: &'a Path,
    fallback_profile: &'a str,
    requested_levels: &'a [u8],
    case_limit: Option<usize>,
    case_offset: usize,
    progress: bool,
    evaluation_stage: EvaluationStageArg,
}

fn run_evaluation<B: RuntimeBackend>(
    service: &CompressionService<B>,
    options: EvaluationOptions<'_>,
) -> Result<EvaluationReport> {
    let EvaluationOptions {
        fixture_path,
        fallback_profile,
        requested_levels,
        case_limit,
        case_offset,
        progress,
        evaluation_stage,
    } = options;
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
    validate_evaluation_fixture(&fixture)?;

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
                compression_level: CompressionLevel::from_u8(level)?,
                profile: profile.clone(),
                constraints: CompressionConstraints::default(),
                target: RequestTarget::codex_default(),
                source: RequestSource::Cli,
            };

            let run = service.compress_with_observation(request);
            let mut reasons = Vec::new();
            let mut missing_terms = Vec::new();
            let mut missing_marker_groups = Vec::new();
            let mut character_ratio = None;
            let mut should_send_original = None;
            let mut fallback_reason = None;
            let mut output_preview = None;
            let mut raw_model_character_ratio = None;
            let mut runtime_character_ratio = None;
            let mut final_character_ratio = None;
            let mut raw_model_output = None;
            let mut runtime_output = None;
            let mut final_output = None;
            let mut raw_model_removed_content_summary = None;
            let mut runtime_removed_content_summary = None;
            let mut final_removed_content_summary = None;
            let mut runtime_transformations = Vec::new();
            let mut application_fallback_applied = false;
            let mut final_should_send_original = None;
            let mut final_fallback_reason = None;

            match run {
                Ok(observed) => {
                    application_fallback_applied = observed.application_fallback_applied;
                    let result = observed.result;
                    final_character_ratio = Some(result.metrics.character_ratio);
                    final_output = Some(result.distilled_prompt.trim().to_string());
                    final_removed_content_summary = Some(result.removed_content_summary);
                    final_should_send_original = Some(result.should_send_original);
                    final_fallback_reason = result.fallback_reason;

                    if let Some(runtime) = observed.runtime_observation {
                        if let Some(raw_model_draft) = runtime.raw_model_draft {
                            let output = raw_model_draft.distilled_prompt.trim().to_string();
                            raw_model_character_ratio =
                                Some(text_character_ratio(&case.input_text, &output));
                            raw_model_output = Some(output);
                            raw_model_removed_content_summary =
                                Some(raw_model_draft.removed_content_summary);
                        }

                        let output = runtime.final_draft.distilled_prompt.trim().to_string();
                        runtime_character_ratio =
                            Some(text_character_ratio(&case.input_text, &output));
                        runtime_output = Some(output);
                        runtime_removed_content_summary =
                            Some(runtime.final_draft.removed_content_summary);
                        runtime_transformations = runtime.transformations;
                    }

                    let selected_output = match evaluation_stage {
                        EvaluationStageArg::RawModel => raw_model_output.as_deref(),
                        EvaluationStageArg::FinalPipeline => final_output.as_deref(),
                    };
                    character_ratio = match evaluation_stage {
                        EvaluationStageArg::RawModel => raw_model_character_ratio,
                        EvaluationStageArg::FinalPipeline => final_character_ratio,
                    };
                    if evaluation_stage == EvaluationStageArg::FinalPipeline {
                        should_send_original = final_should_send_original;
                        fallback_reason = final_fallback_reason.clone();
                        if final_should_send_original == Some(true) {
                            reasons.push("should_send_original".to_string());
                        }
                    }

                    if let Some(output) = selected_output {
                        output_preview = Some(trim_preview(output, 120));
                        if let Some(ratio) = character_ratio {
                            reasons
                                .extend(case_character_ratio_failure_reasons(ratio, level_config));
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
                    } else {
                        reasons.push("selected_output_unavailable".to_string());
                    }
                    accumulator.ratio_sum += character_ratio.unwrap_or(1.0);
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
                input_text: case.input_text.clone(),
                character_ratio,
                should_send_original,
                fallback_reason,
                output_preview,
                raw_model_character_ratio,
                runtime_character_ratio,
                final_character_ratio,
                raw_model_output,
                runtime_output,
                final_output,
                raw_model_removed_content_summary,
                runtime_removed_content_summary,
                final_removed_content_summary,
                runtime_transformations,
                application_fallback_applied,
                final_should_send_original,
                final_fallback_reason,
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
                .is_none_or(|minimum| average_ratio >= minimum)
            && config
                .max_average_character_ratio
                .is_none_or(|maximum| average_ratio <= maximum);
        passed &= level_passed;
        level_reports.push(EvaluationLevelReport {
            level,
            run_count: accumulator.run_count,
            failure_count: accumulator.failure_count,
            failure_rate,
            average_character_ratio: average_ratio,
            min_average_character_ratio: config.min_average_character_ratio,
            max_average_character_ratio: config.max_average_character_ratio,
            min_case_character_ratio: config.min_case_character_ratio,
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
        evaluation_stage,
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

fn text_character_ratio(input: &str, output: &str) -> f32 {
    let input_characters = input.chars().count();
    if input_characters == 0 {
        1.0
    } else {
        output.chars().count() as f32 / input_characters as f32
    }
}

fn validate_evaluation_fixture(fixture: &EvaluationFixture) -> Result<()> {
    if let Some(policy) = &fixture.failure_policy {
        validate_ratio("failure_policy.max_failure_rate", policy.max_failure_rate)?;
    }

    for (level, config) in &fixture.levels {
        for (name, value) in [
            (
                "min_average_character_ratio",
                config.min_average_character_ratio,
            ),
            (
                "max_average_character_ratio",
                config.max_average_character_ratio,
            ),
            ("min_case_character_ratio", config.min_case_character_ratio),
            ("max_case_character_ratio", config.max_case_character_ratio),
        ] {
            if let Some(value) = value {
                validate_ratio(&format!("levels.{level}.{name}"), value)?;
            }
        }

        if let (Some(minimum), Some(maximum)) = (
            config.min_average_character_ratio,
            config.max_average_character_ratio,
        ) {
            if minimum > maximum {
                bail!(
                    "levels.{level}.min_average_character_ratio must not exceed max_average_character_ratio"
                );
            }
        }
        if let (Some(minimum), Some(maximum)) = (
            config.min_case_character_ratio,
            config.max_case_character_ratio,
        ) {
            if minimum > maximum {
                bail!(
                    "levels.{level}.min_case_character_ratio must not exceed max_case_character_ratio"
                );
            }
        }
    }
    Ok(())
}

fn validate_ratio(name: &str, value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        bail!("{name} must be a finite value between 0 and 1");
    }
    Ok(())
}

fn case_character_ratio_failure_reasons(ratio: f32, config: &EvaluationLevelConfig) -> Vec<String> {
    let mut reasons = Vec::new();
    if let Some(minimum) = config.min_case_character_ratio {
        if ratio < minimum {
            reasons.push(format!(
                "character_ratio {ratio:.3} is below min_case {minimum:.3}"
            ));
        }
    }
    if let Some(maximum) = config.max_case_character_ratio {
        if ratio > maximum {
            reasons.push(format!(
                "character_ratio {ratio:.3} exceeds max_case {maximum:.3}"
            ));
        }
    }
    reasons
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
        &["壊れない", "壊れず", "壊さない", "破損しない"],
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        case_character_ratio_failure_reasons, contains_marker_text, contains_required_term,
        missing_required_marker_groups, validate_evaluation_fixture, EvaluationFixture,
        EvaluationLevelConfig,
    };

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
        assert!(contains_marker_text(
            "既存クライアントが壊れず、互換性を確認。",
            "壊れない"
        ));
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

    #[test]
    fn applies_inclusive_per_case_character_ratio_bounds() {
        let config = EvaluationLevelConfig {
            min_average_character_ratio: None,
            max_average_character_ratio: None,
            min_case_character_ratio: Some(0.4),
            max_case_character_ratio: Some(0.9),
        };

        assert!(case_character_ratio_failure_reasons(0.4, &config).is_empty());
        assert!(case_character_ratio_failure_reasons(0.9, &config).is_empty());
        assert_eq!(
            case_character_ratio_failure_reasons(0.399, &config),
            vec!["character_ratio 0.399 is below min_case 0.400"]
        );
        assert_eq!(
            case_character_ratio_failure_reasons(0.901, &config),
            vec!["character_ratio 0.901 exceeds max_case 0.900"]
        );
    }

    #[test]
    fn rejects_inverted_or_out_of_range_evaluation_bounds() {
        let fixture_with = |config| EvaluationFixture {
            schema_version: 1,
            id: "test".to_string(),
            profile: None,
            failure_policy: None,
            levels: BTreeMap::from([("2".to_string(), config)]),
            cases: Vec::new(),
        };
        let inverted = fixture_with(EvaluationLevelConfig {
            min_average_character_ratio: None,
            max_average_character_ratio: None,
            min_case_character_ratio: Some(0.8),
            max_case_character_ratio: Some(0.7),
        });
        let out_of_range = fixture_with(EvaluationLevelConfig {
            min_average_character_ratio: None,
            max_average_character_ratio: None,
            min_case_character_ratio: Some(-0.1),
            max_case_character_ratio: Some(0.9),
        });

        assert!(validate_evaluation_fixture(&inverted)
            .expect_err("inverted bounds")
            .to_string()
            .contains("must not exceed"));
        assert!(validate_evaluation_fixture(&out_of_range)
            .expect_err("out of range bound")
            .to_string()
            .contains("between 0 and 1"));
    }
}
