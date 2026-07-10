use std::env;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::compression::token_estimator::SimpleTokenEstimator;
use crate::compression::verifier::SimpleVerifier;
use crate::config::profile::ProfileRegistry;
use crate::error::{CompressionError, Result};
use crate::runtime::backend::{CompressionDraft, RuntimeBackend};
use crate::types::{
    CompressionMetrics, CompressionRequest, CompressionResult, PreservedRequirement, RiskFlag,
    RiskSeverity,
};

#[derive(Debug)]
pub struct CompressionService<B> {
    registry: ProfileRegistry,
    backend: B,
    token_estimator: SimpleTokenEstimator,
    verifier: SimpleVerifier,
}

impl<B> CompressionService<B>
where
    B: RuntimeBackend,
{
    pub fn new(registry: ProfileRegistry, backend: B) -> Self {
        Self {
            registry,
            backend,
            token_estimator: SimpleTokenEstimator,
            verifier: SimpleVerifier,
        }
    }

    pub fn compress(&self, request: CompressionRequest) -> Result<CompressionResult> {
        let started_at = Instant::now();
        if request.input_text.trim().is_empty() {
            return Err(CompressionError::EmptyInput);
        }

        let profile_started_at = Instant::now();
        let requested_profile = self.registry.resolve(&request.profile)?;
        trace_service_timing("profile_resolve", profile_started_at.elapsed());
        let request_id = Uuid::new_v4();
        let before_characters = count_characters(&request.input_text);

        let backend_started_at = Instant::now();
        let mut runtime_fallback_reason = None;
        let active_profile = requested_profile;
        let draft = if request.compression_level.is_original() {
            CompressionDraft {
                distilled_prompt: request.input_text.trim().to_string(),
                removed_content_summary: vec![
                    "Original level selected; no compression applied.".to_string()
                ],
            }
        } else {
            match self.backend.compress(&request, requested_profile) {
                Ok(draft) => draft,
                Err(error) => {
                    runtime_fallback_reason = Some(format!(
                        "Profile '{}' failed: {error}; original prompt returned without retry.",
                        requested_profile.id
                    ));
                    original_prompt_draft(&request)
                }
            }
        };
        trace_service_timing("backend_or_original", backend_started_at.elapsed());

        let verification_started_at = Instant::now();
        let mut verification = self.verifier.verify(&request, &draft.distilled_prompt);
        trace_service_timing("verification", verification_started_at.elapsed());
        if let Some(reason) = runtime_fallback_reason {
            verification.should_send_original = true;
            verification.fallback_reason = Some(reason);
            verification.risk_flags.push(RiskFlag {
                code: "RUNTIME_FALLBACK".to_string(),
                severity: RiskSeverity::High,
                message: "Runtime backend failed; sending the original prompt is safer."
                    .to_string(),
            });
        }

        let metrics_started_at = Instant::now();
        let before_tokens = self.token_estimator.estimate(
            &request.input_text,
            &active_profile.target_tokenizer_profile,
        );
        let after_tokens = self.token_estimator.estimate(
            &draft.distilled_prompt,
            &active_profile.target_tokenizer_profile,
        );
        let after_characters = count_characters(&draft.distilled_prompt);

        let metrics = CompressionMetrics {
            input_tokens_est: before_tokens,
            output_tokens_est: after_tokens,
            compression_ratio: if before_tokens == 0 {
                1.0
            } else {
                after_tokens as f32 / before_tokens as f32
            },
            input_characters: before_characters,
            output_characters: after_characters,
            character_ratio: if before_characters == 0 {
                1.0
            } else {
                after_characters as f32 / before_characters as f32
            },
            latency_ms: started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        };
        trace_service_timing("metrics", metrics_started_at.elapsed());
        trace_service_timing("total", started_at.elapsed());

        Ok(CompressionResult {
            request_id: request_id.to_string(),
            profile: active_profile.id.clone(),
            model_id: active_profile.model_ref.clone(),
            runtime: active_profile.runtime_ref.clone(),
            distilled_prompt: draft.distilled_prompt,
            preserved_requirements: verification.preserved_requirements,
            removed_content_summary: draft.removed_content_summary,
            risk_flags: verification.risk_flags,
            should_send_original: verification.should_send_original,
            fallback_reason: verification.fallback_reason,
            metrics,
        })
    }

    pub fn prepare(&self, request: CompressionRequest) -> Result<bool> {
        if request.compression_level.is_original() {
            return Ok(false);
        }

        let profile_started_at = Instant::now();
        let requested_profile = self.registry.resolve(&request.profile)?;
        trace_service_timing("prepare_profile_resolve", profile_started_at.elapsed());

        let backend_started_at = Instant::now();
        let prepared = self.backend.prepare(&request, requested_profile)?;
        trace_service_timing("prepare_backend", backend_started_at.elapsed());
        Ok(prepared)
    }

    pub fn list_profiles(&self) -> Vec<String> {
        self.registry
            .list()
            .into_iter()
            .map(|profile| profile.id.clone())
            .collect()
    }

    pub fn compare_profiles(
        &self,
        base_request: CompressionRequest,
        profiles: &[String],
    ) -> Vec<Result<CompressionResult>> {
        profiles
            .iter()
            .map(|profile| {
                let mut request = base_request.clone();
                request.profile = profile.clone();
                self.compress(request)
            })
            .collect()
    }
}

fn trace_service_timing(stage: &str, elapsed: Duration) {
    if env::var_os("PROMPT_COMPRESSOR_TRACE").is_some() {
        eprintln!(
            "trace.service.{stage}_ms={}",
            elapsed.as_millis().min(u128::from(u64::MAX))
        );
    }
}

fn original_prompt_draft(request: &CompressionRequest) -> CompressionDraft {
    CompressionDraft {
        distilled_prompt: request.input_text.trim().to_string(),
        removed_content_summary: vec![
            "Runtime backend failed; returned the original prompt.".to_string()
        ],
    }
}

fn count_characters(text: &str) -> usize {
    text.chars().count()
}

#[cfg(test)]
mod tests {
    use super::{count_characters, CompressionService};
    use crate::config::profile::{ProfileDefinition, ProfileRegistry};
    use crate::error::{CompressionError, Result};
    use crate::runtime::backend::{CompressionDraft, RuntimeBackend};
    use crate::types::{
        CompressionConstraints, CompressionLevel, CompressionMode, CompressionRequest,
        RequestSource, RequestTarget, TaskType,
    };

    #[test]
    fn counts_japanese_and_ascii_characters() {
        assert_eq!(count_characters("検索 API"), 6);
    }

    #[test]
    fn returns_original_without_profile_retry_when_runtime_fails() {
        let service = CompressionService::new(ProfileRegistry::bootstrap(), FallbackBackend);
        let result = service
            .compress(test_request("lmstudio_local"))
            .expect("service should fail open with original prompt");

        assert_eq!(result.profile, "lmstudio_local");
        assert_eq!(
            result.distilled_prompt,
            "テスト用の依頼を短くしてください。"
        );
        assert!(result.should_send_original);
        assert!(result
            .risk_flags
            .iter()
            .any(|risk| risk.code == "RUNTIME_FALLBACK"));
        assert!(!result
            .risk_flags
            .iter()
            .any(|risk| risk.code == "PROFILE_FALLBACK"));
    }

    #[test]
    fn returns_original_when_runtime_fails() {
        let service = CompressionService::new(ProfileRegistry::bootstrap(), AlwaysFailBackend);
        let result = service
            .compress(test_request("lmstudio_local"))
            .expect("service should fail open with original prompt");

        assert_eq!(result.profile, "lmstudio_local");
        assert_eq!(
            result.distilled_prompt,
            "テスト用の依頼を短くしてください。"
        );
        assert!(result.should_send_original);
        assert!(result
            .risk_flags
            .iter()
            .any(|risk| risk.code == "RUNTIME_FALLBACK"));
    }

    #[derive(Debug, Clone)]
    struct FallbackBackend;

    impl RuntimeBackend for FallbackBackend {
        fn compress(
            &self,
            _request: &CompressionRequest,
            profile: &ProfileDefinition,
        ) -> Result<CompressionDraft> {
            if profile.id == "internal_llm" {
                Ok(CompressionDraft {
                    distilled_prompt: "圧縮済み".to_string(),
                    removed_content_summary: vec![],
                })
            } else {
                Err(CompressionError::Runtime("primary unavailable".to_string()))
            }
        }
    }

    #[derive(Debug, Clone)]
    struct AlwaysFailBackend;

    impl RuntimeBackend for AlwaysFailBackend {
        fn compress(
            &self,
            _request: &CompressionRequest,
            _profile: &ProfileDefinition,
        ) -> Result<CompressionDraft> {
            Err(CompressionError::Runtime("runtime unavailable".to_string()))
        }
    }

    fn test_request(profile: &str) -> CompressionRequest {
        CompressionRequest {
            input_text: "テスト用の依頼を短くしてください。".to_string(),
            task_type: TaskType::Coding,
            compression_mode: CompressionMode::CodexOptimized,
            compression_level: CompressionLevel::from_u8(2).expect("valid level"),
            profile: profile.to_string(),
            constraints: CompressionConstraints::default(),
            target: RequestTarget::codex_default(),
            source: RequestSource::Desktop,
        }
    }
}

#[allow(dead_code)]
fn _keep_types_used(_a: PreservedRequirement, _b: RiskFlag) {}
