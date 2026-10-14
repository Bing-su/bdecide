use serde::{Deserialize, Serialize};

/// Preserve value types, e.g. `"true"` and `true` are different choices.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DecisionChoiceValue {
    String(String),
    Boolean(bool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionChoice {
    pub value: DecisionChoiceValue,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionLevel {
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum DecisionQuestion {
    Predicate {
        instructions: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Choice {
        instructions: String,
        choices: Vec<DecisionChoice>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Score {
        instructions: String,
        levels: Vec<DecisionLevel>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
}

impl DecisionQuestion {
    /// Build an unnamed predicate, e.g. `DecisionQuestion::predicate("cancel?")`.
    pub fn predicate(instructions: impl Into<String>) -> Self {
        Self::Predicate {
            instructions: instructions.into(),
            name: None,
        }
    }

    /// Build an unnamed choice from values or described options, e.g. `DecisionQuestion::choice("Select", [true, false])`.
    pub fn choice(
        instructions: impl Into<String>,
        choices: impl IntoIterator<Item = impl Into<DecisionChoice>>,
    ) -> Self {
        Self::Choice {
            instructions: instructions.into(),
            choices: choices.into_iter().map(Into::into).collect(),
            name: None,
        }
    }

    /// Build an unnamed score from labels or described levels, e.g. `DecisionQuestion::score("Rate", ["low", "high"])`.
    pub fn score(
        instructions: impl Into<String>,
        levels: impl IntoIterator<Item = impl Into<DecisionLevel>>,
    ) -> Self {
        Self::Score {
            instructions: instructions.into(),
            levels: levels.into_iter().map(Into::into).collect(),
            name: None,
        }
    }

    /// Set the echoed name, e.g. `DecisionQuestion::predicate("cancel?").with_name("cancel")`.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        let (Self::Predicate { name: slot, .. }
        | Self::Choice { name: slot, .. }
        | Self::Score { name: slot, .. }) = &mut self;
        *slot = Some(name.into());
        self
    }
}

impl DecisionChoice {
    /// Keep the supplied value's type, e.g. `DecisionChoice::new(true)` differs from `DecisionChoice::new("true")`.
    pub fn new(value: impl Into<DecisionChoiceValue>) -> Self {
        Self {
            value: value.into(),
            description: None,
        }
    }

    /// Describe a choice without changing its value, e.g. `DecisionChoice::new("billing").with_description("Payments")`.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

impl<T: Into<DecisionChoiceValue>> From<T> for DecisionChoice {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl From<String> for DecisionChoiceValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for DecisionChoiceValue {
    fn from(value: &str) -> Self {
        Self::String(value.into())
    }
}

impl From<bool> for DecisionChoiceValue {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

impl DecisionLevel {
    /// Start a rubric level with its label, e.g. `DecisionLevel::new("high")`.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            description: None,
        }
    }

    /// Add rubric criteria, e.g. `DecisionLevel::new("high").with_description("Fully blocked")`.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

impl<T: Into<String>> From<T> for DecisionLevel {
    fn from(label: T) -> Self {
        Self::new(label)
    }
}
