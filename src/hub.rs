//! Resolve Python-compatible Hub options before delegating downloads to hf-hub.
use camino::{Utf8Component, Utf8Path, Utf8PathBuf};
use hf_hub::HFClient;
use std::fmt;

use crate::{Error, Metadata, Result, utils::read};

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
#[derive(Debug, Clone)]
pub struct HubOptions {
    pub repo_id: String,
    pub revision: Option<String>,
    pub subfolder: Option<String>,
    pub cache_dir: Option<Utf8PathBuf>,
    pub endpoint: Option<String>,
    pub token: Token,
    pub local_files_only: bool,
    pub force_download: bool,
}
impl HubOptions {
    pub fn new(repo_id: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            revision: None,
            subfolder: None,
            cache_dir: None,
            endpoint: None,
            token: Token::Auto,
            local_files_only: false,
            force_download: false,
        }
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

pub(crate) struct Artifacts {
    pub root: Utf8PathBuf,
    pub metadata: Metadata,
}

/// Resolve the files declared by a model, pinning downloads to the first file's commit.
pub(crate) fn resolve(source: &ModelSource, required: &[&str]) -> Result<Artifacts> {
    match source {
        ModelSource::Local(root) => {
            validate_artifacts(root, required)?;
            Ok(Artifacts {
                root: root.clone(),
                metadata: Metadata {
                    model_id: root.to_string(),
                    revision: None,
                    commit_sha: None,
                    subfolder: None,
                    architecture: String::new(),
                    device: String::new(),
                },
            })
        }
        ModelSource::Hub(options) => resolve_hub(options, required),
    }
}

fn resolve_hub(options: &HubOptions, required: &[&str]) -> Result<Artifacts> {
    let first = required
        .first()
        .ok_or_else(|| Error::InvalidCheckpoint("model must declare required artifacts".into()))?;
    let (owner, name) = repo_parts(&options.repo_id)?;
    let subfolder = options.subfolder.as_deref().unwrap_or("");
    relative_path(subfolder)?;
    let revision = options.revision.as_deref().unwrap_or("main");
    relative_path(revision)?;
    if revision.is_empty() {
        return Err(Error::InvalidCheckpoint(
            "revision must not be empty".into(),
        ));
    }
    let defaults = HubDefaults::from_env(&options.token, options.cache_dir.as_deref())?;
    let offline = options.local_files_only || defaults.offline;
    if offline && options.force_download {
        return Err(Error::InvalidCheckpoint(
            "force_download conflicts with offline/local_files_only".into(),
        ));
    }
    let cache = defaults.cache;
    let token = match &options.token {
        Token::Anonymous => None,
        Token::Explicit(token) if token.trim().is_empty() || token.contains(['\r', '\n']) => {
            return Err(Error::InvalidCheckpoint(
                "explicit token must be non-empty and contain no newlines".into(),
            ));
        }
        Token::Explicit(token) => Some(token.trim().to_owned()),
        Token::Auto => defaults.token,
        Token::Required => Some(defaults.token.ok_or(Error::TokenRequired)?),
    };
    let mut builder = HFClient::builder().cache_dir(cache.into_std_path_buf());
    if let Some(endpoint) = &options.endpoint {
        builder = builder.endpoint(endpoint);
    }
    // hf-hub 1.0 falls back to environment auth when no token is set. Its header
    // builder omits invalid header values, so this sentinel suppresses implicit
    // credentials for token=False without changing process-wide environment.
    builder = builder.token(token.unwrap_or_else(|| "\n".into()));
    let client = builder.build_sync()?;
    let repo = client.model(owner, name);
    let prefix = if subfolder.is_empty() {
        String::new()
    } else {
        format!("{subfolder}/")
    };
    let filename = format!("{prefix}{first}");
    // Pin every subsequent download to the first artifact's commit, even if main moves.
    let config = repo
        .download_file()
        .filename(filename.clone())
        .revision(revision)
        .local_files_only(offline)
        .force_download(options.force_download)
        .send()?;
    // Check the external path once so nested artifacts, e.g. encoder/config.json,
    // keep their exact UTF-8 spelling throughout model loading.
    let config = Utf8PathBuf::try_from(config).map_err(|source| {
        Error::InvalidCheckpoint(format!("Hub returned a non-UTF-8 snapshot path: {source}"))
    })?;
    // Strip exactly the requested path, e.g. snapshots/variant/config.json, rather
    // than searching directory names. This also works for offline branch cache hits.
    let snapshot = config
        .ancestors()
        .nth(Utf8Path::new(&filename).components().count())
        .ok_or_else(|| Error::InvalidCheckpoint("Hub did not return a snapshot path".into()))?;
    let commit = snapshot
        .file_name()
        .filter(|commit| is_commit(commit))
        .ok_or_else(|| Error::InvalidCheckpoint("invalid snapshot commit".into()))?
        .to_owned();
    let root = snapshot.join(subfolder);
    // Exact filenames keep paths such as v[1]/model.safetensors literal. hf-hub
    // owns SHA cache hits, offline resolution, locking, and forced downloads.
    for file in required.iter().skip(1) {
        repo.download_file()
            .filename(format!("{prefix}{file}"))
            .revision(commit.clone())
            .local_files_only(offline)
            .force_download(options.force_download)
            .send()?;
    }
    validate_artifacts(&root, required)?;
    Ok(Artifacts {
        root,
        metadata: options.metadata(revision, &commit),
    })
}

// Reject invalid repo components before constructing cache paths or HTTP requests.
// For example, owner/repo is valid while owner/../repo is not.
fn repo_parts(repo_id: &str) -> Result<(&str, &str)> {
    let (owner, name) = repo_id
        .split_once('/')
        .ok_or_else(|| Error::InvalidCheckpoint("repo_id must be owner/name".into()))?;
    for part in [owner, name] {
        if part.is_empty()
            || !part
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || "-_.".contains(character))
            || part == "."
            || part == ".."
        {
            return Err(Error::InvalidCheckpoint(
                "repo_id must be a valid owner/name".into(),
            ));
        }
    }
    Ok((owner, name))
}

fn validate_artifacts(root: &Utf8Path, required: &[&str]) -> Result<()> {
    for file in required {
        let path = root.join(file);
        if !path.is_file() {
            return Err(Error::MissingArtifact(path));
        }
    }
    Ok(())
}

fn relative_path(value: &str) -> Result<()> {
    if value.contains('\\')
        || Utf8Path::new(value)
            .components()
            .any(|part| !matches!(part, Utf8Component::Normal(_)))
    {
        return Err(Error::InvalidCheckpoint(
            "revision/subfolder must be relative paths without traversal".into(),
        ));
    }
    Ok(())
}
fn is_commit(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

struct HubDefaults {
    cache: Utf8PathBuf,
    token: Option<String>,
    offline: bool,
}
impl HubDefaults {
    fn from_env(policy: &Token, cache_dir: Option<&Utf8Path>) -> Result<Self> {
        Self::resolve(
            |key| match std::env::var(key) {
                Ok(value) => Ok(Some(value)),
                Err(std::env::VarError::NotPresent) => Ok(None),
                Err(std::env::VarError::NotUnicode(_)) => Err(Error::InvalidCheckpoint(format!(
                    "{key} must contain valid UTF-8"
                ))),
            },
            dirs::home_dir(),
            policy,
            cache_dir,
        )
    }
    fn resolve(
        env: impl Fn(&str) -> Result<Option<String>>,
        home: Option<std::path::PathBuf>,
        policy: &Token,
        cache_dir: Option<&Utf8Path>,
    ) -> Result<Self> {
        // Resolve fallback paths only when used, e.g. anonymous access with a
        // chosen cache does not require a home or token-file path.
        let hf_home = || -> Result<Utf8PathBuf> {
            if let Some(path) = env("HF_HOME")? {
                Ok(Utf8PathBuf::from(path))
            } else if let Some(path) = env("XDG_CACHE_HOME")? {
                Ok(Utf8PathBuf::from(path).join("huggingface"))
            } else {
                let home = home.as_ref().ok_or_else(|| {
                    Error::InvalidCheckpoint(
                        "cannot locate home; set HF_HOME or explicit cache and credentials".into(),
                    )
                })?;
                // Actual I/O paths still require UTF-8; never replace bytes such as 0xff.
                Ok(Utf8PathBuf::try_from(home.clone())
                    .map_err(|source| {
                        Error::InvalidCheckpoint(format!("home path must be valid UTF-8: {source}"))
                    })?
                    .join(".cache/huggingface"))
            }
        };
        let cache = match cache_dir {
            Some(path) => path.to_owned(),
            None => match env("HF_HUB_CACHE")? {
                Some(path) => Utf8PathBuf::from(path),
                None => hf_home()?.join("hub"),
            },
        };
        let needs_token = matches!(policy, Token::Required)
            || (matches!(policy, Token::Auto) && !truthy(env("HF_HUB_DISABLE_IMPLICIT_TOKEN")?));
        let token = if needs_token {
            read_token(&env, hf_home)?
        } else {
            None
        };
        Ok(Self {
            cache,
            token,
            offline: truthy(env("HF_HUB_OFFLINE")?),
        })
    }
}

// Prefer environment credentials, then the Python-compatible token file.
// Missing or blank files mean anonymous access; unreadable files still report errors.
fn read_token(
    env: &impl Fn(&str) -> Result<Option<String>>,
    hf_home: impl FnOnce() -> Result<Utf8PathBuf>,
) -> Result<Option<String>> {
    if let Some(token) = match env("HF_TOKEN")? {
        Some(token) => Some(token),
        None => env("HUGGING_FACE_HUB_TOKEN")?,
    }
    .filter(|token| !token.trim().is_empty())
    {
        return Ok(Some(token.trim().into()));
    }
    let path = match env("HF_TOKEN_PATH")? {
        Some(path) => Utf8PathBuf::from(path),
        None => hf_home()?.join("token"),
    };
    match read(&path) {
        Ok(bytes) => Ok(String::from_utf8(bytes)
            .ok()
            .filter(|token| !token.trim().is_empty())
            .map(|token| token.trim().into())),
        Err(Error::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn truthy(value: Option<String>) -> bool {
    value.is_some_and(|s| matches!(s.to_ascii_uppercase().as_str(), "1" | "ON" | "YES" | "TRUE"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    #[test]
    fn local_resolution_uses_the_callers_artifact_layout() {
        let directory = tempfile::tempdir().unwrap();
        // Keep local metadata and file lookup exact, e.g. a Korean directory with spaces.
        let root = Utf8Path::from_path(directory.path())
            .unwrap()
            .join("로컬 모델");
        std::fs::create_dir(&root).unwrap();
        let source = ModelSource::Local(root.clone());
        std::fs::write(root.join("config.json"), "{}").unwrap();
        let artifacts = resolve(&source, &["config.json"]).unwrap();
        assert_eq!(artifacts.root, root);
        assert_eq!(artifacts.metadata.model_id, root.as_str());
        assert!(matches!(
            resolve(&source, &["config.json", "weights.bin"]),
            Err(Error::MissingArtifact(_))
        ));
    }
    #[rstest]
    #[case("../escape")]
    #[case("/absolute")]
    #[case("branch/../../escape")]
    #[case("C:\\escape")]
    fn rejects_traversal(#[case] path: &str) {
        assert!(relative_path(path).is_err());
    }
    #[test]
    fn python_cache_precedence_and_token() {
        let env = |key: &str| {
            Ok(match key {
                "HF_HOME" => Some("/unused".into()),
                "HF_HUB_CACHE" => Some("/chosen".into()),
                "HF_TOKEN" => Some(" secret\n".into()),
                "HF_HUB_OFFLINE" => Some("yes".into()),
                _ => None,
            })
        };
        let config = HubDefaults::resolve(env, None, &Token::Auto, None).unwrap();
        assert_eq!(config.cache, Utf8PathBuf::from("/chosen"));
        assert_eq!(config.token.as_deref(), Some("secret"));
        assert!(config.offline);
    }
    #[test]
    fn credentials_are_redacted() {
        assert!(!format!("{:?}", Token::Explicit("secret".into())).contains("secret"));
    }

    #[rstest]
    #[case::anonymous(Token::Anonymous, false)]
    #[case::explicit(Token::Explicit("chosen".into()), false)]
    #[case::automatic(Token::Auto, false)]
    #[case::implicit_disabled(Token::Auto, true)]
    #[case::required(Token::Required, true)]
    fn explicit_cache_skips_unused_paths(#[case] policy: Token, #[case] disable_implicit: bool) {
        // Fail if an overridden path is read, e.g. a non-UTF-8 HF_HUB_CACHE.
        let env = |key: &str| match key {
            "HF_TOKEN" => Ok(Some("environment-token".into())),
            "HF_HUB_DISABLE_IMPLICIT_TOKEN" => Ok(Some(disable_implicit.to_string())),
            "HF_HUB_OFFLINE" => Ok(Some("1".into())),
            _ => Err(Error::InvalidCheckpoint(format!("unused path: {key}"))),
        };
        let cache = Utf8Path::new("chosen-cache");
        let config = HubDefaults::resolve(env, None, &policy, Some(cache)).unwrap();
        assert_eq!(config.cache, cache);
        let needs_token = matches!(policy, Token::Required)
            || (matches!(policy, Token::Auto) && !disable_implicit);
        assert_eq!(
            config.token.as_deref(),
            needs_token.then_some("environment-token")
        );
        assert!(config.offline);
    }

    #[test]
    fn explicit_token_file_and_cache_do_not_require_home() {
        let directory = tempfile::tempdir().unwrap();
        let cache = Utf8Path::from_path(directory.path()).unwrap();
        let token_path = cache.join("token");
        std::fs::write(&token_path, "file-token\n").unwrap();
        // Use the chosen credential file even when no home is available.
        let env = |key: &str| Ok((key == "HF_TOKEN_PATH").then(|| token_path.to_string()));
        let config = HubDefaults::resolve(env, None, &Token::Required, Some(cache)).unwrap();
        assert_eq!(config.token.as_deref(), Some("file-token"));
    }

    #[test]
    #[cfg(unix)]
    fn rejects_non_utf8_home_unless_hf_home_overrides_it() {
        use std::{ffi::OsString, os::unix::ffi::OsStringExt};

        // A byte such as 0xff must never redirect cache I/O to a replacement path.
        let home = std::path::PathBuf::from(OsString::from_vec(b"/home/\xff".to_vec()));
        let result =
            HubDefaults::resolve(|_| Ok(None), Some(home.clone()), &Token::Anonymous, None);
        assert!(
            matches!(result, Err(Error::InvalidCheckpoint(message)) if message.contains("UTF-8"))
        );
        let config = HubDefaults::resolve(
            |key| Ok((key == "HF_HOME").then(|| "/valid".into())),
            Some(home),
            &Token::Anonymous,
            None,
        )
        .unwrap();
        assert_eq!(config.cache, Utf8PathBuf::from("/valid/hub"));
    }
    #[test]
    fn anonymous_and_explicit_auth_do_not_read_token_files() {
        let directory = tempfile::tempdir().unwrap();
        // A directory cannot be read as a token file. Unused credentials should
        // not break token=False, an explicit token, or disabled implicit auth.
        let env = |key: &str| {
            Ok(match key {
                "HF_HOME" | "HF_TOKEN_PATH" => {
                    Some(Utf8Path::from_path(directory.path()).unwrap().to_string())
                }
                "HF_HUB_DISABLE_IMPLICIT_TOKEN" => Some("1".into()),
                _ => None,
            })
        };
        for policy in [
            Token::Anonymous,
            Token::Explicit("chosen".into()),
            Token::Auto,
        ] {
            assert!(
                HubDefaults::resolve(env, None, &policy, None)
                    .unwrap()
                    .token
                    .is_none()
            );
        }
        assert!(HubDefaults::resolve(env, None, &Token::Required, None).is_err());
    }
}
