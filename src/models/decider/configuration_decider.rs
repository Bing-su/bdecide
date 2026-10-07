//! Validate the release's prompt and calibration before loading weights.
use indexmap::IndexMap;
use serde::Deserialize;

use crate::{Error, Question, Result};

#[derive(Deserialize)]
pub(crate) struct DeciderConfig {
    #[serde(default = "one")]
    temperature: f64,
    #[serde(default)]
    temperature_by_type: IndexMap<String, f64>,
    #[serde(default = "plain")]
    layout: String,
    #[serde(default)]
    chat_template: bool,
    #[serde(default = "yes")]
    pub(super) neutralize_none: bool,
    #[serde(default)]
    pub(super) isolated_levels: bool,
    #[serde(default = "options")]
    pub(super) max_options: usize,
    #[serde(default = "state_tokens")]
    pub(super) max_state_tokens: usize,
    // Refuse a calibration variant we cannot apply, e.g. Gemma's option-count fit.
    #[serde(default)]
    temperature_by_options: Option<serde_json::Value>,
}
fn one() -> f64 {
    1.0
}
fn plain() -> String {
    "plain".into()
}
fn yes() -> bool {
    true
}
fn options() -> usize {
    255
}
fn state_tokens() -> usize {
    32768
}

impl DeciderConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.layout != "plain" || self.chat_template || self.temperature_by_options.is_some() {
            return Err(Error::UnsupportedModel(
                "Decider requires a plain dense Qwen3.5 release with scalar/by-type calibration"
                    .into(),
            ));
        }
        if !self.temperature.is_finite()
            || self.temperature <= 0.0
            || self.temperature_by_type.iter().any(|(kind, value)| {
                !matches!(kind.as_str(), "choice" | "score" | "noul")
                    || !value.is_finite()
                    || *value <= 0.0
            })
            || !(2..=255).contains(&self.max_options)
            || self.max_state_tokens < 4
        {
            return Err(Error::InvalidCheckpoint(
                "invalid Decider option, context or temperature configuration".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn temperature(&self, question: &Question) -> f64 {
        let kind = match question {
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
            Question::Noul { .. } => "noul",
        };
        self.temperature_by_type
            .get(kind)
            .copied()
            .unwrap_or(self.temperature)
    }
}
