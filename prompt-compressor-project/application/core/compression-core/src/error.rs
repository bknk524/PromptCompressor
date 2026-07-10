use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompressionError {
    #[error("unknown profile: {0}")]
    UnknownProfile(String),

    #[error("invalid compression level: {0}")]
    InvalidCompressionLevel(u8),

    #[error("empty input is not allowed")]
    EmptyInput,

    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to parse yaml config: {0}")]
    ConfigYaml(#[from] serde_yaml::Error),

    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("unknown model: {0}")]
    UnknownModel(String),

    #[error("unknown runtime: {0}")]
    UnknownRuntime(String),

    #[error("runtime timed out after {0} ms")]
    RuntimeTimeout(u64),

    #[error("runtime error: {0}")]
    Runtime(String),
}

pub type Result<T> = std::result::Result<T, CompressionError>;
