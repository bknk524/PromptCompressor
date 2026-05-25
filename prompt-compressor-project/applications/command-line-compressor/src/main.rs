use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use prompt_compressor_core::{
    CompressionConstraints, CompressionLevel, CompressionMode, CompressionRequest,
    CompressionService, LlamaCppProcessBackend, ProfileRegistry, RequestSource, RequestTarget,
    TaskType,
};

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

    #[arg(long, default_value = "standard")]
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
        for profile in registry.list() {
            println!("{}\t{}", profile.id, profile.label);
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

    let backend = LlamaCppProcessBackend::from_settings_dir(&settings_dir)
        .context("failed to initialize llama.cpp process backend")?;
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
        anyhow::anyhow!("could not find ./settings directory from {}", cwd.display())
    })
}

fn find_upward_settings_dir(start: &Path) -> Option<PathBuf> {
    for ancestor in start.ancestors() {
        let candidate = ancestor.join("settings");
        if candidate.is_dir() {
            return Some(candidate);
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
