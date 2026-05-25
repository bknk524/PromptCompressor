pub mod backend;
pub mod error;
pub mod profile;
pub mod service;
pub mod token_estimator;
pub mod types;
pub mod verifier;

pub use backend::{CompressionDraft, LlamaCppProcessBackend, NoopRuntimeBackend, RuntimeBackend};
pub use error::{CompressionError, Result};
pub use profile::{ProfileDefinition, ProfileRegistry};
pub use service::CompressionService;
pub use types::{
    CompressionConstraints, CompressionLevel, CompressionMetrics, CompressionMode,
    CompressionRequest, CompressionResult, OutputFormat, PreservedRequirement, RequestSource,
    RequestTarget, RiskFlag, RiskSeverity, TaskType,
};
