use camino::Utf8PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

/// Preserve recovery decisions such as a cache miss versus an invalid request.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("cannot read {path}: {source}")]
    Io {
        path: Utf8PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("invalid checkpoint: {0}")]
    InvalidCheckpoint(String),
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    #[error("checkpoint is missing {0}")]
    MissingArtifact(Utf8PathBuf),
    #[error("Hub operation failed: {0}")]
    Hub(#[from] Box<hf_hub::HFError>),
    #[error("authentication requires a token; set HF_TOKEN or log in with the Hugging Face CLI")]
    TokenRequired,
    #[error("device unavailable: {0}")]
    Device(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("invalid weights: {0}")]
    Weights(String),
    #[error("inference returned invalid data: {0}")]
    Inference(String),
}

impl From<hf_hub::HFError> for Error {
    fn from(value: hf_hub::HFError) -> Self {
        Self::Hub(Box::new(value))
    }
}
