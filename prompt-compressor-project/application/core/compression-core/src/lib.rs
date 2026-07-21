#[cfg(all(
    feature = "cpu-profile-strict",
    feature = "embedded-llama-compatible",
    not(target_feature = "sse4.2")
))]
compile_error!("the compatible CPU runtime must compile Rust with SSE4.2 enabled");

#[cfg(all(
    feature = "cpu-profile-strict",
    feature = "embedded-llama-avx2",
    not(all(
        target_feature = "avx",
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c",
        target_feature = "bmi2"
    ))
))]
compile_error!(
    "the AVX2 CPU runtime must compile Rust with AVX, AVX2, FMA, F16C, and BMI2 enabled"
);

#[cfg(all(
    feature = "cpu-profile-strict",
    feature = "embedded-llama-avx512",
    not(all(
        target_feature = "avx",
        target_feature = "avx2",
        target_feature = "fma",
        target_feature = "f16c",
        target_feature = "bmi2",
        target_feature = "avx512f",
        target_feature = "avx512cd",
        target_feature = "avx512bw",
        target_feature = "avx512dq",
        target_feature = "avx512vl"
    ))
))]
compile_error!("the AVX-512 CPU runtime must compile Rust with the complete configured AVX-512 feature group enabled");

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
    NoopRuntimeBackend, ProfileModelStatus, ProfileThreadStatus, RuntimeBackend,
    RuntimeCompressionObservation, RuntimeTransformation,
};
pub use types::{
    CompressionConstraints, CompressionLevel, CompressionMetrics, CompressionRequest,
    CompressionResult, OutputFormat, PreservedRequirement, RequestSource, RequestTarget, RiskFlag,
    RiskSeverity,
};
