use serde::{Deserialize, Serialize};

use crate::error::{CompressionError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Coding,
    LogAnalysis,
    Refactor,
    DesignDiscussion,
    General,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionMode {
    Lossless,
    InstructionExtract,
    CodexOptimized,
    PrivacyRedaction,
    DeveloperMode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CompressionLevel(u8);

impl CompressionLevel {
    pub fn from_u8(value: u8) -> Result<Self> {
        if value <= 4 {
            Ok(Self(value))
        } else {
            Err(CompressionError::InvalidCompressionLevel(value))
        }
    }

    pub fn is_original(self) -> bool {
        self.0 == 0
    }

    pub fn value(self) -> u8 {
        self.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConstraints {
    pub preserve_code_blocks: bool,
    pub preserve_file_names: bool,
    pub preserve_error_messages: bool,
    pub preserve_numbers: bool,
    pub preserve_negations: bool,
}

impl Default for CompressionConstraints {
    fn default() -> Self {
        Self {
            preserve_code_blocks: true,
            preserve_file_names: true,
            preserve_error_messages: true,
            preserve_numbers: true,
            preserve_negations: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTarget {
    pub destination: String,
    pub tokenizer_profile: String,
}

impl RequestTarget {
    pub fn codex_default() -> Self {
        Self {
            destination: "codex".to_string(),
            tokenizer_profile: "codex_default".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestSource {
    Cli,
    Mcp,
    Desktop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionRequest {
    pub input_text: String,
    pub task_type: TaskType,
    pub compression_mode: CompressionMode,
    pub compression_level: CompressionLevel,
    pub profile: String,
    pub constraints: CompressionConstraints,
    pub target: RequestTarget,
    pub source: RequestSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreservedRequirement {
    pub kind: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFlag {
    pub code: String,
    pub severity: RiskSeverity,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetrics {
    pub input_tokens_est: usize,
    pub output_tokens_est: usize,
    pub compression_ratio: f32,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionResult {
    pub request_id: String,
    pub profile: String,
    pub model_id: String,
    pub runtime: String,
    pub distilled_prompt: String,
    pub preserved_requirements: Vec<PreservedRequirement>,
    pub removed_content_summary: Vec<String>,
    pub risk_flags: Vec<RiskFlag>,
    pub should_send_original: bool,
    pub fallback_reason: Option<String>,
    pub metrics: CompressionMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    Text,
    Json,
}
