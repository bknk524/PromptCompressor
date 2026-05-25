use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{CompressionError, Result};

#[derive(Debug, Clone)]
pub struct ProfileDefinition {
    pub id: String,
    pub label: String,
    pub model_ref: String,
    pub policy_ref: String,
    pub runtime_ref: String,
    pub fallback_profile: String,
    pub target_tokenizer_profile: String,
}

#[derive(Debug, Clone)]
pub struct ProfileRegistry {
    profiles: BTreeMap<String, ProfileDefinition>,
}

impl ProfileRegistry {
    pub fn bootstrap() -> Self {
        let mut profiles = BTreeMap::new();

        for profile in [
            ProfileDefinition {
                id: "standard".to_string(),
                label: "Standard".to_string(),
                model_ref: "qwen3_1_7b".to_string(),
                policy_ref: "balanced-codex-compression-policy-v1".to_string(),
                runtime_ref: "llama_cpp_default".to_string(),
                fallback_profile: "lightweight_safe".to_string(),
                target_tokenizer_profile: "codex_default".to_string(),
            },
            ProfileDefinition {
                id: "code_focused".to_string(),
                label: "Code Focused".to_string(),
                model_ref: "qwen2_5_coder_1_5b".to_string(),
                policy_ref: "code-focused-codex-compression-policy-v1".to_string(),
                runtime_ref: "llama_cpp_default".to_string(),
                fallback_profile: "lightweight_safe".to_string(),
                target_tokenizer_profile: "codex_default".to_string(),
            },
            ProfileDefinition {
                id: "lightweight_safe".to_string(),
                label: "Lightweight Safe".to_string(),
                model_ref: "gemma3_1b_it".to_string(),
                policy_ref: "safe-codex-compression-policy-v1".to_string(),
                runtime_ref: "llama_cpp_default".to_string(),
                fallback_profile: "lightweight_safe".to_string(),
                target_tokenizer_profile: "codex_default".to_string(),
            },
            ProfileDefinition {
                id: "lmstudio".to_string(),
                label: "LM Studio".to_string(),
                model_ref: "lmstudio_auto".to_string(),
                policy_ref: "balanced-codex-compression-policy-v1".to_string(),
                runtime_ref: "lmstudio_default".to_string(),
                fallback_profile: "lightweight_safe".to_string(),
                target_tokenizer_profile: "codex_default".to_string(),
            },
        ] {
            profiles.insert(profile.id.clone(), profile);
        }

        Self { profiles }
    }

    pub fn resolve(&self, id: &str) -> Result<&ProfileDefinition> {
        self.profiles
            .get(id)
            .ok_or_else(|| CompressionError::UnknownProfile(id.to_string()))
    }

    pub fn list(&self) -> Vec<&ProfileDefinition> {
        self.profiles.values().collect()
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let file: ProfilesFile = serde_yaml::from_str(&contents)?;

        if file.schema_version != 1 {
            return Err(CompressionError::InvalidConfig(format!(
                "unsupported profiles schema_version: {}",
                file.schema_version
            )));
        }

        let profiles = file
            .profiles
            .into_iter()
            .map(|(id, entry)| {
                (
                    id.clone(),
                    ProfileDefinition {
                        id,
                        label: entry.label,
                        model_ref: entry.model_ref,
                        policy_ref: entry.policy_ref,
                        runtime_ref: entry.runtime_ref,
                        fallback_profile: entry.fallback_profile,
                        target_tokenizer_profile: entry.target_tokenizer_profile,
                    },
                )
            })
            .collect();

        Ok(Self { profiles })
    }
}

#[derive(Debug, Deserialize)]
struct ProfilesFile {
    schema_version: u32,
    profiles: BTreeMap<String, ProfileEntry>,
}

#[derive(Debug, Deserialize)]
struct ProfileEntry {
    label: String,
    model_ref: String,
    policy_ref: String,
    runtime_ref: String,
    fallback_profile: String,
    target_tokenizer_profile: String,
}
