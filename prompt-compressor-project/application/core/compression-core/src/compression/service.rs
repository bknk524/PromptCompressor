use std::env;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::compression::token_estimator::SimpleTokenEstimator;
use crate::compression::verifier::SimpleVerifier;
use crate::config::profile::ProfileRegistry;
use crate::error::{CompressionError, Result};
use crate::runtime::backend::{CompressionDraft, RuntimeBackend, RuntimeCompressionObservation};
use crate::types::{
    CompressionMetrics, CompressionRequest, CompressionResult, RiskFlag, RiskSeverity,
};

#[derive(Debug)]
pub struct CompressionService<B> {
    registry: ProfileRegistry,
    backend: B,
    token_estimator: SimpleTokenEstimator,
    verifier: SimpleVerifier,
}

#[derive(Debug, Clone)]
pub struct ObservedCompression {
    pub result: CompressionResult,
    pub runtime_observation: Option<RuntimeCompressionObservation>,
    pub application_fallback_applied: bool,
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
        self.compress_internal(request, false)
            .map(|observed| observed.result)
    }

    pub fn compress_with_observation(
        &self,
        request: CompressionRequest,
    ) -> Result<ObservedCompression> {
        self.compress_internal(request, true)
    }

    fn compress_internal(
        &self,
        request: CompressionRequest,
        observe_runtime: bool,
    ) -> Result<ObservedCompression> {
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
        let mut runtime_observation = None;
        let mut application_fallback_applied = false;
        let active_profile = requested_profile;
        let mut draft = if request.compression_level.is_original() {
            CompressionDraft {
                distilled_prompt: request.input_text.trim().to_string(),
                removed_content_summary: vec![
                    "Original level selected; no compression applied.".to_string()
                ],
            }
        } else {
            let backend_result = if observe_runtime {
                self.backend
                    .compress_observed(&request, requested_profile)
                    .map(|observation| (observation.final_draft.clone(), Some(observation)))
            } else {
                self.backend
                    .compress(&request, requested_profile)
                    .map(|draft| (draft, None))
            };
            match backend_result {
                Ok((draft, observation)) => {
                    runtime_observation = observation;
                    draft
                }
                Err(error) => {
                    application_fallback_applied = true;
                    runtime_fallback_reason = Some(format!(
                        "原文返し理由: 圧縮ランタイムが失敗しました（profile: '{}', error: {error}）。再推論せず原文を返しました。",
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
                message: "圧縮ランタイムが失敗したため、原文返しを選択しました。".to_string(),
            });
        }
        if verification.should_send_original
            && draft.distilled_prompt.trim() != request.input_text.trim()
        {
            draft = verification_fallback_draft(&request);
            application_fallback_applied = true;
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

        let result = CompressionResult {
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
        };

        Ok(ObservedCompression {
            result,
            runtime_observation,
            application_fallback_applied,
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
        removed_content_summary: vec!["圧縮ランタイムが失敗したため原文を返しました。".to_string()],
    }
}

fn verification_fallback_draft(request: &CompressionRequest) -> CompressionDraft {
    CompressionDraft {
        distilled_prompt: request.input_text.trim().to_string(),
        removed_content_summary: vec![
            "圧縮結果の要件保持を確認できなかったため原文を返しました。".to_string()
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
    use crate::runtime::backend::{
        CompressionDraft, RuntimeBackend, RuntimeCompressionObservation, RuntimeTransformation,
    };
    use crate::types::{
        CompressionConstraints, CompressionLevel, CompressionRequest, RequestSource, RequestTarget,
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
            .fallback_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("圧縮ランタイムが失敗")));
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

    #[test]
    fn returns_original_when_output_verification_fails() {
        let service =
            CompressionService::new(ProfileRegistry::bootstrap(), MissingConstraintBackend);
        let mut request = test_request("internal_llm");
        request.input_text = "config.yamlのHTTP 400処理を変更しないでください。".to_string();

        let result = service
            .compress(request.clone())
            .expect("verification failure should return the original prompt");

        assert_eq!(result.distilled_prompt, request.input_text);
        assert!(result.should_send_original);
        assert!(result
            .fallback_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("原文返し理由")));
        assert!(result
            .risk_flags
            .iter()
            .any(|risk| risk.code == "VERIFICATION_FAILED"));
        assert!(!result
            .preserved_requirements
            .iter()
            .any(|requirement| requirement.text == "preserve_file_names"));
    }

    #[test]
    fn exposes_raw_and_runtime_outputs_for_evaluation() {
        let service = CompressionService::new(ProfileRegistry::bootstrap(), ObservingBackend);
        let mut request = test_request("internal_llm");
        request.input_text = "README.mdの説明を短く整理してください。".to_string();

        let observed = service
            .compress_with_observation(request)
            .expect("observed compression should succeed");
        let runtime = observed
            .runtime_observation
            .expect("runtime observation should be retained");

        assert_eq!(
            runtime
                .raw_model_draft
                .expect("raw model draft should be available")
                .distilled_prompt,
            "出力: README.mdの説明を簡潔にする。"
        );
        assert_eq!(
            runtime.final_draft.distilled_prompt,
            "README.mdの説明を簡潔にする。"
        );
        assert_eq!(
            runtime.transformations,
            [RuntimeTransformation::PolishedModelOutput]
        );
        assert!(!observed.application_fallback_applied);
    }

    #[test]
    fn regular_compression_uses_the_non_observed_backend_path() {
        let service = CompressionService::new(ProfileRegistry::bootstrap(), ObservingBackend);
        let mut request = test_request("internal_llm");
        request.input_text = "README.mdの説明を短く整理してください。".to_string();

        let result = service.compress(request).expect("regular compression");

        assert_eq!(result.distilled_prompt, "README.mdを短くする。");
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

    #[derive(Debug, Clone)]
    struct MissingConstraintBackend;

    impl RuntimeBackend for MissingConstraintBackend {
        fn compress(
            &self,
            _request: &CompressionRequest,
            _profile: &ProfileDefinition,
        ) -> Result<CompressionDraft> {
            Ok(CompressionDraft {
                distilled_prompt: "HTTP処理を整理する。".to_string(),
                removed_content_summary: vec![],
            })
        }
    }

    #[derive(Debug, Clone)]
    struct ObservingBackend;

    impl RuntimeBackend for ObservingBackend {
        fn compress(
            &self,
            _request: &CompressionRequest,
            _profile: &ProfileDefinition,
        ) -> Result<CompressionDraft> {
            Ok(CompressionDraft {
                distilled_prompt: "README.mdを短くする。".to_string(),
                removed_content_summary: vec![],
            })
        }

        fn compress_observed(
            &self,
            _request: &CompressionRequest,
            _profile: &ProfileDefinition,
        ) -> Result<RuntimeCompressionObservation> {
            Ok(RuntimeCompressionObservation {
                raw_model_draft: Some(CompressionDraft {
                    distilled_prompt: "出力: README.mdの説明を簡潔にする。".to_string(),
                    removed_content_summary: vec![],
                }),
                final_draft: CompressionDraft {
                    distilled_prompt: "README.mdの説明を簡潔にする。".to_string(),
                    removed_content_summary: vec![],
                },
                transformations: vec![RuntimeTransformation::PolishedModelOutput],
            })
        }
    }

    fn test_request(profile: &str) -> CompressionRequest {
        CompressionRequest {
            input_text: "テスト用の依頼を短くしてください。".to_string(),
            compression_level: CompressionLevel::from_u8(2).expect("valid level"),
            profile: profile.to_string(),
            constraints: CompressionConstraints::default(),
            target: RequestTarget::codex_default(),
            source: RequestSource::Desktop,
        }
    }
}
