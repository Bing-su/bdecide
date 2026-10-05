//! Share checkpoint file handling, e.g. report malformed JSON with its artifact path.

use std::fs;

use camino::Utf8Path;
use serde::de::DeserializeOwned;

use crate::{Error, Result};

pub(crate) fn read(path: &Utf8Path) -> Result<Vec<u8>> {
    fs::read(path).map_err(|source| Error::Io {
        path: path.into(),
        source,
    })
}

pub(crate) fn read_checkpoint_json<T: DeserializeOwned>(path: &Utf8Path) -> Result<T> {
    // Keep syntax and schema failures with the model artifact, e.g. encoder/config.json.
    serde_json::from_slice(&read(path)?)
        .map_err(|source| Error::InvalidCheckpoint(format!("{path}: {source}")))
}
