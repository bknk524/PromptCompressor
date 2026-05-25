use uuid::Uuid;

use crate::backend::{CompressionDraft, RuntimeBackend};
use crate::error::{CompressionError, Result};
use crate::profile::ProfileRegistry;
use crate::token_estimator::SimpleTokenEstimator;
use crate::types::{
    CompressionMetrics, CompressionRequest, CompressionResult, PreservedRequirement, RiskFlag,
    RiskSeverity,
};
use crate::verifier::SimpleVerifier;

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
        if request.input_text.trim().is_empty() {
            return Err(CompressionError::EmptyInput);
        }

        let profile = self.registry.resolve(&request.profile)?;
        let request_id = Uuid::new_v4();
        let before_tokens = self
            .token_estimator
            .estimate(&request.input_text, &profile.target_tokenizer_profile);

        let mut runtime_fallback_reason = None;
        let draft = if request.compression_level.is_original() {
            CompressionDraft {
                distilled_prompt: request.input_text.trim().to_string(),
                removed_content_summary: vec![
                    "Original level selected; no compression applied.".to_string()
                ],
            }
        } else {
            match self.backend.compress(&request, profile) {
                Ok(draft) => draft,
                Err(error) => {
                    runtime_fallback_reason = Some(error.to_string());
                    CompressionDraft {
                        distilled_prompt: request.input_text.trim().to_string(),
                        removed_content_summary: vec![
                            "Runtime backend failed; returned the original prompt.".to_string(),
                        ],
                    }
                }
            }
        };

        let mut verification = self.verifier.verify(&request, &draft.distilled_prompt);
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
        let after_tokens = self
            .token_estimator
            .estimate(&draft.distilled_prompt, &profile.target_tokenizer_profile);

        let metrics = CompressionMetrics {
            input_tokens_est: before_tokens,
            output_tokens_est: after_tokens,
            compression_ratio: if before_tokens == 0 {
                1.0
            } else {
                after_tokens as f32 / before_tokens as f32
            },
            latency_ms: 0,
        };

        Ok(CompressionResult {
            request_id: request_id.to_string(),
            profile: profile.id.clone(),
            model_id: profile.model_ref.clone(),
            runtime: profile.runtime_ref.clone(),
            distilled_prompt: draft.distilled_prompt,
            preserved_requirements: verification.preserved_requirements,
            removed_content_summary: draft.removed_content_summary,
            risk_flags: verification.risk_flags,
            should_send_original: verification.should_send_original,
            fallback_reason: verification.fallback_reason,
            metrics,
        })
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

#[allow(dead_code)]
fn _keep_types_used(_a: PreservedRequirement, _b: RiskFlag) {}
