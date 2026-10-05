//! Validate the merged Vev release layout, e.g. vev.json's base must be '.'.
use crate::{Error, Result};
use serde::Deserialize;

#[derive(Deserialize)]
pub(crate) struct VevConfig {
    vev_version: String,
    base: String,
}
impl VevConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.vev_version != "0.1.0" || self.base != "." {
            return Err(Error::UnsupportedModel(
                "Vev requires a merged 0.1.0 checkpoint".into(),
            ));
        }
        Ok(())
    }
}
