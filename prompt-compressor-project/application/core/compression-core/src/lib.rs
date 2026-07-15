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
    mod catalog;
    mod model_download;
    pub(crate) mod prompt_structure;
}
pub mod types;

pub use compression::service::{CompressionService, ObservedCompression};
pub use config::profile::{ProfileDefinition, ProfileRegistry};
pub use error::{CompressionError, Result};
pub use runtime::backend::{
    CompressionDraft, ConfiguredRuntimeBackend, ModelDownloadCancellation, ModelDownloadProgress,
    NoopRuntimeBackend, ProfileModelStatus, RuntimeBackend, RuntimeCompressionObservation,
    RuntimeTransformation,
};
pub use types::{
    CompressionConstraints, CompressionLevel, CompressionMetrics, CompressionRequest,
    CompressionResult, OutputFormat, PreservedRequirement, RequestSource, RequestTarget, RiskFlag,
    RiskSeverity,
};
