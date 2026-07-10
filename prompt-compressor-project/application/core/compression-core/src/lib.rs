pub mod compression {
    pub mod service;
    pub mod token_estimator;
    pub mod verifier;
}
pub mod config {
    pub mod profile;
}
pub mod error;
pub mod runtime {
    pub mod backend;
}
pub mod types;

pub use compression::service::CompressionService;
pub use config::profile::{ProfileDefinition, ProfileRegistry};
pub use error::{CompressionError, Result};
pub use runtime::backend::{
    CompressionDraft, ConfiguredRuntimeBackend, NoopRuntimeBackend, RuntimeBackend,
};
pub use types::{
    CompressionConstraints, CompressionLevel, CompressionMetrics, CompressionMode,
    CompressionRequest, CompressionResult, OutputFormat, PreservedRequirement, RequestSource,
    RequestTarget, RiskFlag, RiskSeverity, TaskType,
};
