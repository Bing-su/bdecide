use bon::Builder;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::SystemOneQuestion;
use crate::media::{AudioInput, ImageInput};
use crate::{Error, Result};

/// Preserve question and criterion insertion order because it changes token IDs.
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[serde(deny_unknown_fields)]
pub struct SystemOneRequest {
    #[builder(into)]
    pub state: Value,
    #[builder(default)]
    pub questions: IndexMap<String, SystemOneQuestion>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default)]
    pub images: Vec<ImageInput>,
    /// Keep clip order; each model checks its limit, e.g. D1Omni accepts at most one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[builder(default)]
    pub audio: Vec<AudioInput>,
    #[serde(default)]
    #[builder(default)]
    pub options: PredictOptions,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Truncation {
    #[default]
    Error,
    Truncate,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Builder)]
#[serde(deny_unknown_fields)]
pub struct PredictOptions {
    #[serde(default)]
    #[builder(default)]
    pub truncation: Truncation,
    pub max_len: Option<usize>,
    pub head_max_len: Option<usize>,
}

impl PredictOptions {
    /// Keep truncation opt-in, e.g. `PredictOptions::new()` rejects lost tokens.
    pub fn new() -> Self {
        Self::default()
    }
}

impl SystemOneRequest {
    /// Preserve ordered questions with default options, e.g. `SystemOneRequest::new(state, questions)`.
    pub fn new(state: impl Into<Value>, questions: IndexMap<String, SystemOneQuestion>) -> Self {
        Self::builder().state(state).questions(questions).build()
    }

    /// Insert a question by ID, e.g. `.question("cancel", SystemOneQuestion::noul("cancel?"))`.
    /// Reusing an ID replaces its question without changing insertion order.
    #[must_use]
    pub fn question(mut self, id: impl Into<String>, question: SystemOneQuestion) -> Self {
        self.questions.insert(id.into(), question);
        self
    }

    /// Insert a boolean question with default labels, e.g. `.noul("cancel", "cancel?")`.
    #[must_use]
    pub fn noul(self, id: impl Into<String>, instructions: impl Into<String>) -> Self {
        self.question(id, SystemOneQuestion::noul(instructions))
    }

    /// Insert ordered choice IDs and descriptions, e.g. `.choice("route", "Route", [("billing", "Payments")])`.
    #[must_use]
    pub fn choice(
        self,
        id: impl Into<String>,
        instructions: impl Into<String>,
        criteria: impl IntoIterator<Item = (impl Into<String>, impl Into<Value>)>,
    ) -> Self {
        self.question(id, SystemOneQuestion::choice(instructions, criteria))
    }

    /// Insert ordered score levels, e.g. `.score("severity", "Rate", ["low", "high"])`.
    #[must_use]
    pub fn score(
        self,
        id: impl Into<String>,
        instructions: impl Into<String>,
        criteria: impl IntoIterator<Item = impl Into<Value>>,
    ) -> Self {
        self.question(id, SystemOneQuestion::score(instructions, criteria))
    }

    pub fn validate(&self) -> Result<()> {
        if !matches!(
            self.state,
            Value::String(_) | Value::Object(_) | Value::Array(_)
        ) && !(self.state.is_null() && (!self.images.is_empty() || !self.audio.is_empty()))
        {
            return Err(Error::InvalidRequest(
                "state must be text, a JSON object, or a conversation array".into(),
            ));
        }
        if !self.images.is_empty() && !self.audio.is_empty() {
            return Err(Error::InvalidRequest(
                "a request carries images or audio, not both".into(),
            ));
        }
        for image in &self.images {
            image.validate()?;
        }
        for audio in &self.audio {
            audio.validate()?;
        }
        for (id, q) in &self.questions {
            if id.trim().is_empty() {
                return Err(Error::InvalidRequest(
                    "question IDs must not be empty".into(),
                ));
            }
            q.validate()
                .map_err(|message| Error::InvalidRequest(format!("question {id:?}: {message}")))?;
        }
        // Reject impossible budgets before model I/O, e.g. max_len=3 cannot be used.
        // The processor checks model-specific upper bounds after loading.
        if self.options.max_len.is_some_and(|max_len| max_len < 4) {
            return Err(Error::InvalidRequest("max_len must be at least 4".into()));
        }
        if self
            .options
            .head_max_len
            .is_some_and(|head_max_len| head_max_len < 16)
        {
            return Err(Error::InvalidRequest(
                "head_max_len must be at least 16".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_text_only(&self) -> Result<()> {
        self.validate()?;
        if !self.images.is_empty() || !self.audio.is_empty() {
            return Err(Error::InvalidRequest("this model accepts text only".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::omitted(serde_json::json!({"state": "alpha", "questions": {}}))]
    #[case::empty(serde_json::json!({"state": "alpha", "questions": {}, "audio": []}))]
    fn empty_audio_defaults_and_is_omitted(#[case] value: Value) {
        let request: SystemOneRequest = serde_json::from_value(value).expect("valid request");
        request
            .validate_text_only()
            .expect("empty audio is text-only");
        assert!(request.audio.is_empty());
        assert!(
            serde_json::to_value(&request)
                .expect("serializable request")
                .get("audio")
                .is_none()
        );
        assert!(
            SystemOneRequest::new(request.state, request.questions)
                .audio
                .is_empty()
        );
    }

    #[test]
    fn multiple_audio_clips_preserve_order_and_validate_every_clip() {
        let audio = serde_json::json!([
            {"samples": [0.25], "sample_rate": 16000},
            {"pcm16": [123], "sample_rate": 16000}
        ]);
        let mut request = SystemOneRequest::builder()
            .state(Value::Null)
            .questions(IndexMap::new())
            .audio(serde_json::from_value(audio.clone()).expect("valid audio clips"))
            .build();
        request
            .validate()
            .expect("common request accepts multiple clips");
        request
            .validate_text_only()
            .expect_err("text-only models reject audio");
        let serialized = serde_json::to_value(&request).expect("serializable request");
        assert_eq!(serialized.get("audio"), Some(&audio));
        let restored: SystemOneRequest =
            serde_json::from_value(serialized).expect("valid audio list");
        assert_eq!(
            serde_json::to_value(restored.audio).expect("serializable clips"),
            audio
        );
        request.audio.push(AudioInput::Samples {
            samples: vec![0.0],
            sample_rate: 8000,
        });
        request
            .validate()
            .expect_err("validate clips beyond the first");
    }

    #[rstest]
    #[case::defaults(None, None)]
    #[case::minimum_max_len(Some(4), None)]
    #[case::minimum_head_max_len(None, Some(16))]
    #[case::minimum_budgets(Some(4), Some(16))]
    #[case::model_dependent_upper_bounds(Some(usize::MAX), Some(usize::MAX))]
    fn valid_static_budgets(#[case] max_len: Option<usize>, #[case] head_max_len: Option<usize>) {
        let request = SystemOneRequest {
            state: Value::String("alpha".into()),
            questions: IndexMap::new(),
            images: Vec::new(),
            audio: Vec::new(),
            options: PredictOptions {
                max_len,
                head_max_len,
                ..Default::default()
            },
        };
        request
            .validate()
            .expect("static budgets should be valid before model-specific checks");
    }
}
