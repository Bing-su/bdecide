use bon::Builder;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum SystemOneQuestion {
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

impl SystemOneQuestion {
    /// Start a boolean question with standard labels, e.g. `SystemOneQuestion::noul("cancel?")`.
    pub fn noul(instructions: impl Into<String>) -> Self {
        Self::Noul {
            instructions: instructions.into(),
            criteria: IndexMap::new(),
            labels: NoulLabels::default(),
        }
    }

    /// Preserve choice ID order and JSON descriptions, e.g. `SystemOneQuestion::choice("Route", [("billing", "Payments")])`.
    pub fn choice(
        instructions: impl Into<String>,
        criteria: impl IntoIterator<Item = (impl Into<String>, impl Into<Value>)>,
    ) -> Self {
        Self::Choice {
            instructions: instructions.into(),
            criteria: criteria
                .into_iter()
                .map(|(id, value)| (id.into(), value.into()))
                .collect(),
        }
    }

    /// Preserve ordered JSON levels, e.g. `SystemOneQuestion::score("Rate", ["low", "high"])`.
    pub fn score(
        instructions: impl Into<String>,
        criteria: impl IntoIterator<Item = impl Into<Value>>,
    ) -> Self {
        Self::Score {
            instructions: instructions.into(),
            criteria: criteria.into_iter().map(Into::into).collect(),
        }
    }

    pub(crate) fn instructions(&self) -> &str {
        match self {
            Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::Noul { instructions, .. } => instructions,
        }
    }

    pub(super) fn validate(&self) -> std::result::Result<(), String> {
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
