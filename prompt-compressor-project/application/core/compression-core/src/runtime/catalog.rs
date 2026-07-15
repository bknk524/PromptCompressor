use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{CompressionError, Result};

use super::model_download::ModelDownloadSpec;

#[derive(Debug, Clone)]
pub(super) struct ModelRegistry {
    models: BTreeMap<String, ModelDefinition>,
}

impl ModelRegistry {
    pub(super) fn from_path(path: impl AsRef<Path>) -> Result<Self> {
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
            .map(|(id, entry)| -> Result<_> {
                validate_catalog_id("model", &id)?;
                let download = entry
                    .hugging_face
                    .map(|source| {
                        ModelDownloadSpec::new(
                            source.repository,
                            source.revision,
                            source.filename,
                            source.sha256,
                            source.size_bytes,
                        )
                    })
                    .transpose()?;
                Ok((
                    id.clone(),
                    ModelDefinition {
                        id,
                        label: entry.label,
                        adapter: entry.adapter,
                        runtime_ref: entry.runtime_ref,
                        model_path: entry.model_path.map(PathBuf::from),
                        download,
                        api_model: entry.api_model,
                        quantization: entry.quantization,
                        context_length: entry.context_length,
                        thinking: entry.thinking,
                        default_max_output: entry.default_max_output,
                        prompt_template: entry.prompt_template,
                        prompt_style: entry.prompt_style,
                        supports_json_schema: entry.supports_json_schema,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        if models.is_empty() {
            return Err(CompressionError::InvalidConfig(
                "model catalog must define at least one model".into(),
            ));
        }
        Ok(Self { models })
    }

    pub(super) fn resolve(&self, id: &str) -> Result<&ModelDefinition> {
        self.models
            .get(id)
            .ok_or_else(|| CompressionError::UnknownModel(id.to_string()))
    }
}

#[derive(Debug, Clone)]
pub(super) struct RuntimeRegistry {
    runtimes: BTreeMap<String, RuntimeDefinition>,
}

impl RuntimeRegistry {
    pub(super) fn from_path(path: impl AsRef<Path>) -> Result<Self> {
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
            .map(|(id, entry)| -> Result<_> {
                validate_catalog_id("runtime", &id)?;
                let launch_mode = entry.launch_mode.unwrap_or_else(|| {
                    if entry.backend_kind == "llama.cpp" {
                        RuntimeLaunchMode::OneShot
                    } else {
                        RuntimeLaunchMode::External
                    }
                });
                Ok((
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
                        timeout_ms: entry.timeout_ms,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        if runtimes.is_empty() {
            return Err(CompressionError::InvalidConfig(
                "runtime catalog must define at least one runtime".into(),
            ));
        }
        Ok(Self { runtimes })
    }

    pub(super) fn resolve(&self, id: &str) -> Result<&RuntimeDefinition> {
        self.runtimes
            .get(id)
            .ok_or_else(|| CompressionError::UnknownRuntime(id.to_string()))
    }
}

#[derive(Debug, Clone)]
pub(super) struct PromptProfileRegistry {
    shared_instruction: String,
    profiles: BTreeMap<u8, PromptProfileDefinition>,
}

impl PromptProfileRegistry {
    pub(super) fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let file: PromptProfilesFile = serde_yaml::from_str(&contents)?;
        if file.schema_version != 1 {
            return Err(CompressionError::InvalidConfig(format!(
                "unsupported level prompt profiles schema_version: {}",
                file.schema_version
            )));
        }
        if file.id.trim().is_empty() {
            return Err(CompressionError::InvalidConfig(
                "level prompt profile id cannot be empty".into(),
            ));
        }

        let mut profiles = BTreeMap::new();
        for (id, entry) in file.profiles {
            validate_catalog_id("prompt profile", &id)?;
            if !(1..=3).contains(&entry.level) {
                return Err(CompressionError::InvalidConfig(format!(
                    "prompt profile '{id}' has unsupported level {}",
                    entry.level
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
        for level in 1..=3 {
            if !profiles.contains_key(&level) {
                return Err(CompressionError::InvalidConfig(format!(
                    "level prompt profiles do not define compression level {level}"
                )));
            }
        }

        Ok(Self {
            shared_instruction: file.shared_instruction,
            profiles,
        })
    }

    pub(super) fn shared_instruction(&self) -> &str {
        &self.shared_instruction
    }

    pub(super) fn resolve(&self, level: u8) -> Result<&PromptProfileDefinition> {
        self.profiles.get(&level).ok_or_else(|| {
            CompressionError::InvalidConfig(format!(
                "level prompt profiles do not define compression level {level}"
            ))
        })
    }
}

fn validate_catalog_id(kind: &str, id: &str) -> Result<()> {
    if id.trim().is_empty() || id.chars().any(char::is_control) {
        return Err(CompressionError::InvalidConfig(format!(
            "{kind} id cannot be empty or contain control characters"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(super) struct PromptProfileDefinition {
    pub(super) target_ratio: String,
    pub(super) instruction: String,
    pub(super) format_instruction: String,
    pub(super) allow_semantic_shortening: bool,
}

#[derive(Debug, Clone)]
pub(super) struct ModelDefinition {
    pub(super) id: String,
    #[allow(dead_code)]
    pub(super) label: String,
    #[allow(dead_code)]
    pub(super) adapter: String,
    pub(super) runtime_ref: String,
    pub(super) model_path: Option<PathBuf>,
    pub(super) download: Option<ModelDownloadSpec>,
    pub(super) api_model: Option<String>,
    #[allow(dead_code)]
    pub(super) quantization: String,
    pub(super) context_length: u32,
    #[allow(dead_code)]
    pub(super) thinking: bool,
    pub(super) default_max_output: u32,
    pub(super) prompt_template: String,
    pub(super) prompt_style: String,
    pub(super) supports_json_schema: bool,
}

#[derive(Debug, Clone)]
pub(super) struct RuntimeDefinition {
    pub(super) id: String,
    pub(super) backend_kind: String,
    pub(super) launch_mode: RuntimeLaunchMode,
    pub(super) executable_path: Option<PathBuf>,
    pub(super) base_url: Option<String>,
    pub(super) api_token_env: Option<String>,
    pub(super) health_path: Option<String>,
    pub(super) startup_timeout_ms: u64,
    pub(super) threads: String,
    pub(super) timeout_ms: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelsFile {
    schema_version: u32,
    models: BTreeMap<String, ModelEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelEntry {
    label: String,
    adapter: String,
    #[serde(rename = "runtime")]
    runtime_ref: String,
    #[serde(default)]
    model_path: Option<String>,
    #[serde(default)]
    hugging_face: Option<HuggingFaceModelEntry>,
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
#[serde(deny_unknown_fields)]
struct HuggingFaceModelEntry {
    repository: String,
    revision: String,
    filename: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimesFile {
    schema_version: u32,
    runtimes: BTreeMap<String, RuntimeEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptProfilesFile {
    schema_version: u32,
    id: String,
    shared_instruction: String,
    profiles: BTreeMap<String, PromptProfileEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptProfileEntry {
    level: u8,
    target_ratio: String,
    instruction: String,
    format_instruction: String,
    #[serde(default)]
    allow_semantic_shortening: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
    #[serde(default = "default_timeout_ms")]
    timeout_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum RuntimeLaunchMode {
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

#[cfg(test)]
mod tests {
    use super::{ModelsFile, PromptProfilesFile, RuntimesFile};

    #[test]
    fn rejects_unknown_model_catalog_keys() {
        let yaml = r#"
schema_version: 1
models:
  example:
    label: Example
    adapter: llama
    runtime: embedded
    prompt_template: test
    threadz: 4
"#;
        assert!(serde_yaml::from_str::<ModelsFile>(yaml).is_err());
    }

    #[test]
    fn rejects_unknown_runtime_catalog_keys() {
        let yaml = r#"
schema_version: 1
runtimes:
  embedded:
    backend: llama.cpp
    threadz: auto
"#;
        assert!(serde_yaml::from_str::<RuntimesFile>(yaml).is_err());
    }

    #[test]
    fn rejects_unknown_prompt_profile_keys() {
        let yaml = r#"
schema_version: 1
id: test
shared_instruction: shared
profiles:
  level_1:
    level: 1
    target_ratio: 90%
    instruction: test
    format_instruction: test
    extra_instruction: ignored-before
"#;
        assert!(serde_yaml::from_str::<PromptProfilesFile>(yaml).is_err());
    }
}
