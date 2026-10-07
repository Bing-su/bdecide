use bon::Builder;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Error, Result};

/// Preserve question and criterion insertion order because it changes token IDs.
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub state: Value,
    pub questions: IndexMap<String, Question>,
    #[serde(default)]
    #[builder(default)]
    pub options: PredictOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum Question {
    Choice {
        instructions: String,
        criteria: IndexMap<String, Value>,
    },
    Score {
        instructions: String,
        criteria: Vec<Value>,
    },
    Noul {
        instructions: String,
        #[serde(default)]
        criteria: IndexMap<String, Value>,
        #[serde(default)]
        labels: NoulLabels,
    },
}

/// Labels change the prompt text; the returned probability always means true.
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[serde(deny_unknown_fields)]
pub struct NoulLabels {
    #[builder(default = "false", into)]
    pub r#false: String,
    #[builder(default = "true", into)]
    pub r#true: String,
}

impl NoulLabels {
    /// Use the standard boolean labels, e.g. `NoulLabels::new()`.
    pub fn new() -> Self {
        Self::builder().build()
    }
}

impl Default for NoulLabels {
    fn default() -> Self {
        Self::new()
    }
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

impl Request {
    /// Preserve ordered questions with default options, e.g. `Request::new(state, questions)`.
    pub fn new(state: Value, questions: IndexMap<String, Question>) -> Self {
        Self::builder().state(state).questions(questions).build()
    }

    pub fn validate(&self) -> Result<()> {
        if !matches!(
            self.state,
            Value::String(_) | Value::Object(_) | Value::Array(_)
        ) {
            return Err(Error::InvalidRequest(
                "state must be text, a JSON object, or a conversation array".into(),
            ));
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
}

impl Question {
    pub(crate) fn instructions(&self) -> &str {
        match self {
            Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::Noul { instructions, .. } => instructions,
        }
    }

    fn validate(&self) -> std::result::Result<(), String> {
        if self.instructions().trim().is_empty() {
            return Err("instructions must not be empty".into());
        }
        match self {
            Self::Choice { criteria, .. } if criteria.is_empty() => {
                Err("choice needs at least one option".into())
            }
            Self::Score { criteria, .. }
                if criteria.is_empty() || criteria.iter().any(Value::is_null) =>
            {
                Err("score needs non-null ordered levels".into())
            }
            Self::Noul {
                criteria, labels, ..
            } => {
                if criteria.keys().any(|k| k != "false" && k != "true") {
                    return Err("noul criteria only accept false and true".into());
                }
                if labels.r#false.trim().is_empty()
                    || labels.r#true.trim().is_empty()
                    || labels.r#false.trim() == labels.r#true.trim()
                {
                    return Err("noul labels must be distinct non-empty strings".into());
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::defaults(None, None)]
    #[case::minimum_max_len(Some(4), None)]
    #[case::minimum_head_max_len(None, Some(16))]
    #[case::minimum_budgets(Some(4), Some(16))]
    #[case::model_dependent_upper_bounds(Some(usize::MAX), Some(usize::MAX))]
    fn valid_static_budgets(#[case] max_len: Option<usize>, #[case] head_max_len: Option<usize>) {
        let request = Request {
            state: Value::String("alpha".into()),
            questions: IndexMap::new(),
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
