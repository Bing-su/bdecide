//! Apply Python-compatible cache and credential precedence, e.g. HF_TOKEN before a file.
use std::env::{VarError, var};
use std::io::ErrorKind;
use std::path::PathBuf;

use camino::{Utf8Path, Utf8PathBuf};
use dirs::home_dir;

use super::Token;
use crate::utils::read;
use crate::{Error, Result};

pub(super) struct HubDefaults {
    pub(super) cache: Utf8PathBuf,
    pub(super) token: Option<String>,
    pub(super) offline: bool,
}

impl HubDefaults {
    pub(super) fn from_env(policy: &Token, cache_dir: Option<&Utf8Path>) -> Result<Self> {
        Self::resolve(
            |key| match var(key) {
                Ok(value) => Ok(Some(value)),
                Err(VarError::NotPresent) => Ok(None),
                Err(VarError::NotUnicode(_)) => Err(Error::InvalidCheckpoint(format!(
                    "{key} must contain valid UTF-8"
                ))),
            },
            home_dir(),
            policy,
            cache_dir,
        )
    }

    fn resolve(
        env: impl Fn(&str) -> Result<Option<String>>,
        home: Option<PathBuf>,
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
        Err(Error::Io { source, .. }) if source.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn truthy(value: Option<String>) -> bool {
    value.is_some_and(|s| matches!(s.to_ascii_uppercase().as_str(), "1" | "ON" | "YES" | "TRUE"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use rstest::rstest;
    use tempfile::tempdir;

    use super::*;

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
        let directory = tempdir().unwrap();
        let cache = Utf8Path::from_path(directory.path()).unwrap();
        let token_path = cache.join("token");
        fs::write(&token_path, "file-token\n").unwrap();
        // Use the chosen credential file even when no home is available.
        let env = |key: &str| Ok((key == "HF_TOKEN_PATH").then(|| token_path.to_string()));
        let config = HubDefaults::resolve(env, None, &Token::Required, Some(cache)).unwrap();
        assert_eq!(config.token.as_deref(), Some("file-token"));
    }

    #[test]
    #[cfg(unix)]
    fn rejects_non_utf8_home_unless_hf_home_overrides_it() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        // A byte such as 0xff must never redirect cache I/O to a replacement path.
        let home = PathBuf::from(OsString::from_vec(b"/home/\xff".to_vec()));
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
        let directory = tempdir().unwrap();
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
