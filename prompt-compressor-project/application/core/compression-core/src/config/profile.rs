use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::error::{CompressionError, Result};

#[derive(Debug, Clone)]
pub struct ProfileDefinition {
    pub id: String,
    pub label: String,
    pub selectable: bool,
    pub model_ref: String,
    pub policy_ref: String,
    pub runtime_ref: String,
    pub fallback_profile: String,
    pub target_tokenizer_profile: String,
}

#[derive(Debug, Clone)]
pub struct ProfileRegistry {
    profiles: BTreeMap<String, ProfileDefinition>,
    default_profile: Option<String>,
}

impl ProfileRegistry {
    pub fn bootstrap() -> Self {
        let mut profiles = BTreeMap::new();

        let internal_profile = ProfileDefinition {
            id: "internal_llm".to_string(),
            label: "アプリ内モデル（Sarashina 2.2 3B）".to_string(),
            selectable: true,
            model_ref: "sarashina_2_2_3b_instruct_q4_k_s".to_string(),
            policy_ref: "balanced-codex-compression-policy-v1".to_string(),
            runtime_ref: "llama_cpp_embedded".to_string(),
            fallback_profile: "internal_llm".to_string(),
            target_tokenizer_profile: "codex_default".to_string(),
        };
        profiles.insert(internal_profile.id.clone(), internal_profile);

        let lmstudio_profile = ProfileDefinition {
            id: "lmstudio_local".to_string(),
            label: "LM Studio（ローカルモデル自由選択）".to_string(),
            selectable: true,
            model_ref: "lmstudio_local_model".to_string(),
            policy_ref: "balanced-codex-compression-policy-v1".to_string(),
            runtime_ref: "lmstudio_external".to_string(),
            fallback_profile: "lmstudio_local".to_string(),
            target_tokenizer_profile: "codex_default".to_string(),
        };
        profiles.insert(lmstudio_profile.id.clone(), lmstudio_profile);

        Self {
            profiles,
            default_profile: Some("internal_llm".to_string()),
        }
    }

    pub fn resolve(&self, id: &str) -> Result<&ProfileDefinition> {
        self.profiles
            .get(id)
            .ok_or_else(|| CompressionError::UnknownProfile(id.to_string()))
    }

    pub fn list(&self) -> Vec<&ProfileDefinition> {
        self.profiles.values().collect()
    }

    pub fn list_selectable(&self) -> Vec<&ProfileDefinition> {
        self.profiles
            .values()
            .filter(|profile| profile.selectable)
            .collect()
    }

    pub fn default_profile_id(&self) -> Option<&str> {
        self.default_profile
            .as_deref()
            .filter(|id| {
                self.profiles
                    .get(*id)
                    .is_some_and(|profile| profile.selectable)
            })
            .or_else(|| {
                self.profiles
                    .values()
                    .find(|profile| profile.selectable)
                    .map(|profile| profile.id.as_str())
            })
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let file: ProfilesFile = serde_yaml::from_str(&contents)?;
        let ProfilesFile {
            schema_version,
            default_profile,
            profiles: entries,
        } = file;

        if schema_version != 1 {
            return Err(CompressionError::InvalidConfig(format!(
                "unsupported profiles schema_version: {}",
                schema_version
            )));
        }

        let profiles: BTreeMap<_, _> = entries
            .into_iter()
            .map(|(id, entry)| -> Result<_> {
                if id.trim().is_empty() || id.chars().any(char::is_control) {
                    return Err(CompressionError::InvalidConfig(
                        "profile id cannot be empty or contain control characters".into(),
                    ));
                }
                Ok((
                    id.clone(),
                    ProfileDefinition {
                        id,
                        label: entry.label,
                        selectable: entry.selectable,
                        model_ref: entry.model_ref,
                        policy_ref: entry.policy_ref,
                        runtime_ref: entry.runtime_ref,
                        fallback_profile: entry.fallback_profile,
                        target_tokenizer_profile: entry.target_tokenizer_profile,
                    },
                ))
            })
            .collect::<Result<_>>()?;
        if profiles.is_empty() {
            return Err(CompressionError::InvalidConfig(
                "profile catalog must define at least one profile".into(),
            ));
        }
        if let Some(default_profile) = &default_profile {
            if !profiles.contains_key(default_profile) {
                return Err(CompressionError::InvalidConfig(format!(
                    "default profile '{default_profile}' is not defined"
                )));
            }
        }
        for profile in profiles.values() {
            if !profiles.contains_key(&profile.fallback_profile) {
                return Err(CompressionError::InvalidConfig(format!(
                    "profile '{}' references unknown fallback profile '{}'",
                    profile.id, profile.fallback_profile
                )));
            }
        }

        Ok(Self {
            profiles,
            default_profile,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfilesFile {
    schema_version: u32,
    #[serde(default)]
    default_profile: Option<String>,
    profiles: BTreeMap<String, ProfileEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileEntry {
    label: String,
    #[serde(default)]
    selectable: bool,
    model_ref: String,
    policy_ref: String,
    runtime_ref: String,
    fallback_profile: String,
    target_tokenizer_profile: String,
}

#[cfg(test)]
mod tests {
    use super::ProfilesFile;

    #[test]
    fn rejects_unknown_profile_keys() {
        let yaml = r#"
schema_version: 1
profiles:
  internal:
    label: Internal
    selectable: true
    model_ref: model
    policy_ref: policy
    runtime_ref: runtime
    fallback_profile: internal
    target_tokenizer_profile: codex_default
    runtime_threads: 4
"#;

        assert!(serde_yaml::from_str::<ProfilesFile>(yaml).is_err());
    }
}
