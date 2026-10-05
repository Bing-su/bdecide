//! Validate Wald's serving budget and calibration, e.g. choice|3-4 temperatures.
use crate::{Error, Question, Result, utils::read_checkpoint_json};
use camino::Utf8Path;
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
pub(crate) struct WaldConfig {
    pub(super) prompt_format: String,
    pub(super) max_model_len: usize,
    temperature: String,
}
impl WaldConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if !matches!(self.prompt_format.as_str(), "plain" | "repeat_state_plain") {
            return Err(Error::UnsupportedModel(format!(
                "Wald prompt format: {}",
                self.prompt_format
            )));
        }
        if self.max_model_len < 4 || self.temperature != "temperature.json" {
            return Err(Error::InvalidCheckpoint(
                "invalid Wald context budget or calibration filename".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct Bucket {
    temperature: f64,
}
#[derive(Deserialize)]
pub(super) struct Temperature {
    #[serde(default = "one")]
    single: f64,
    #[serde(default)]
    buckets: IndexMap<String, Bucket>,
}
fn one() -> f64 {
    1.0
}
impl Temperature {
    pub(super) fn load(root: &Utf8Path) -> Result<Self> {
        let value: Value = read_checkpoint_json(&root.join("temperature.json"))?;
        let table: Self = serde_json::from_value(value.get("A").unwrap_or(&value).clone())
            .map_err(|error| Error::InvalidCheckpoint(format!("Wald temperature.json: {error}")))?;
        if std::iter::once(table.single)
            .chain(table.buckets.values().map(|bucket| bucket.temperature))
            .any(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(Error::InvalidCheckpoint(
                "Wald temperatures must be finite and positive".into(),
            ));
        }
        Ok(table)
    }
    pub(super) fn get(&self, question: &Question, count: usize) -> f64 {
        let kind = match question {
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
            Question::Noul { .. } => "noul",
        };
        let bucket = match count {
            0..=2 => "2",
            3..=4 => "3-4",
            5..=8 => "5-8",
            _ => "9+",
        };
        self.buckets
            .get(&format!("{kind}|{bucket}"))
            .map_or(self.single, |bucket| bucket.temperature)
    }
}
