//! Resolve local files and pinned Hub snapshots, e.g. every shard at one SHA.
use super::{HubOptions, ModelSource, Token, environment::HubDefaults};
use crate::{Error, Metadata, Result};
use camino::{Utf8Component, Utf8Path, Utf8PathBuf};
use hf_hub::HFClient;

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
}
