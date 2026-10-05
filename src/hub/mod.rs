//! Preserve Hub loading options while keeping artifact selection and I/O separate.
mod artifacts;
mod download;
mod environment;

use crate::Metadata;
use camino::Utf8PathBuf;
use std::fmt;

pub(crate) use artifacts::{Family, resolve_auto, resolve_clef, resolve_vev, resolve_wald};
pub(crate) use download::{Artifacts, resolve};

/// Mirror token=None/False/True/string without exposing credentials through Debug.
#[derive(Clone, Default)]
pub enum Token {
    #[default]
    Auto,
    Anonymous,
    Required,
    Explicit(String),
}

impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Auto => "Auto",
            Self::Anonymous => "Anonymous",
            Self::Required => "Required",
            Self::Explicit(_) => "Explicit([redacted])",
        })
    }
}

/// Supply a repo ID directly, e.g. HubOptions::new("convaiinnovations/laya-multilingual").
#[derive(Debug, Clone, bon::Builder)]
#[builder(on(String, into))]
pub struct HubOptions {
    pub repo_id: String,
    pub revision: Option<String>,
    pub subfolder: Option<String>,
    pub cache_dir: Option<Utf8PathBuf>,
    pub endpoint: Option<String>,
    #[builder(default)]
    pub token: Token,
    #[builder(default)]
    pub local_files_only: bool,
    #[builder(default)]
    pub force_download: bool,
}

impl HubOptions {
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self::builder().repo_id(repo_id).build()
    }

    // Keep both cache hits and downloads tied to the same requested revision.
    // A moving branch such as main can resolve to a different commit SHA.
    fn metadata(&self, revision: &str, commit: &str) -> Metadata {
        Metadata {
            model_id: self.repo_id.clone(),
            revision: Some(revision.into()),
            commit_sha: Some(commit.into()),
            subfolder: self.subfolder.clone(),
            architecture: String::new(),
            device: String::new(),
        }
    }
}

/// Keep local loading distinct from downloading into a local_dir.
#[derive(Debug, Clone)]
pub enum ModelSource {
    Hub(HubOptions),
    Local(Utf8PathBuf),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credentials_are_redacted() {
        assert!(!format!("{:?}", Token::Explicit("secret".into())).contains("secret"));
    }
}
