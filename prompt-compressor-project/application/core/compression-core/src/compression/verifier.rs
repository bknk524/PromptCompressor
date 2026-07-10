use crate::runtime::backend::CompressionDraft;
use crate::types::{CompressionRequest, PreservedRequirement, RiskFlag, RiskSeverity};

#[derive(Debug, Clone)]
pub struct VerificationSummary {
    pub preserved_requirements: Vec<PreservedRequirement>,
    pub risk_flags: Vec<RiskFlag>,
    pub should_send_original: bool,
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SimpleVerifier;

impl SimpleVerifier {
    pub fn verify(
        &self,
        request: &CompressionRequest,
        distilled_prompt: &str,
    ) -> VerificationSummary {
        let mut preserved_requirements = Vec::new();
        let mut risk_flags = Vec::new();

        if request.constraints.preserve_file_names {
            preserved_requirements.push(PreservedRequirement {
                kind: "constraint".to_string(),
                text: "preserve_file_names".to_string(),
            });
        }

        if request.constraints.preserve_error_messages {
            preserved_requirements.push(PreservedRequirement {
                kind: "constraint".to_string(),
                text: "preserve_error_messages".to_string(),
            });
        }

        if request.constraints.preserve_numbers {
            preserved_requirements.push(PreservedRequirement {
                kind: "constraint".to_string(),
                text: "preserve_numbers".to_string(),
            });
        }

        if request.constraints.preserve_negations {
            preserved_requirements.push(PreservedRequirement {
                kind: "constraint".to_string(),
                text: "preserve_negations".to_string(),
            });
        }

        let should_send_original = distilled_prompt.trim().is_empty();
        let fallback_reason = should_send_original.then(|| "empty_distilled_prompt".to_string());

        if should_send_original {
            risk_flags.push(RiskFlag {
                code: "EMPTY_OUTPUT".to_string(),
                severity: RiskSeverity::High,
                message: "Compression produced an empty output; sending the original is safer."
                    .to_string(),
            });
        }

        VerificationSummary {
            preserved_requirements,
            risk_flags,
            should_send_original,
            fallback_reason,
        }
    }
}

#[allow(dead_code)]
fn _keep_draft_used(_draft: &CompressionDraft) {}
