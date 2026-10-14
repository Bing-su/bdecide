use bon::Builder;
use serde::{Deserialize, Serialize};

/// Match OpenAI's request body, e.g. deserialize it and call `model.decisions(&request)`.
#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[serde(deny_unknown_fields)]
pub struct DecisionsRequest {
    /// Select the already loaded model by its exact `metadata().model_id`.
    #[builder(into)]
    pub model: String,
    #[builder(into)]
    pub input: DecisionInput,
    pub questions: Vec<DecisionQuestion>,
    /// Accept caller metadata; local inference has no hosted abuse tracking policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DecisionInput {
    Text(String),
    Messages(Vec<DecisionInputMessage>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionInputMessage {
    pub role: UserRole,
    pub content: DecisionInputContent,
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub message_type: Option<MessageType>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UserRole {
    User,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageType {
    Message,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DecisionInputContent {
    Text(String),
    Parts(Vec<DecisionInputPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DecisionInputPart {
    InputText {
        text: String,
    },
    /// shortcut: text adapter rejects images; add decoding and media ordering when needed.
    InputImage {
        image_url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Low,
    High,
    Auto,
    Original,
}

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

impl DecisionsRequest {
    /// Start with no questions, e.g. `DecisionsRequest::new("local", "alpha").predicate("q", "cancel?")`.
    pub fn new(model: impl Into<String>, input: impl Into<DecisionInput>) -> Self {
        Self::builder()
            .model(model)
            .input(input)
            .questions(Vec::new())
            .build()
    }

    /// Append a question without changing its name, e.g. `.question(DecisionQuestion::predicate("cancel?"))`.
    #[must_use]
    pub fn question(mut self, question: DecisionQuestion) -> Self {
        self.questions.push(question);
        self
    }

    /// Append a named predicate, e.g. `.predicate("cancel", "Does the customer want to cancel?")`.
    #[must_use]
    pub fn predicate(self, name: impl Into<String>, instructions: impl Into<String>) -> Self {
        self.question(DecisionQuestion::predicate(instructions).with_name(name))
    }

    /// Append a named choice, e.g. `.choice("department", "Route", ["billing", "shipping"])`.
    #[must_use]
    pub fn choice(
        self,
        name: impl Into<String>,
        instructions: impl Into<String>,
        choices: impl IntoIterator<Item = impl Into<DecisionChoice>>,
    ) -> Self {
        self.question(DecisionQuestion::choice(instructions, choices).with_name(name))
    }

    /// Append a named score, e.g. `.score("severity", "Rate", ["low", "high"])`.
    #[must_use]
    pub fn score(
        self,
        name: impl Into<String>,
        instructions: impl Into<String>,
        levels: impl IntoIterator<Item = impl Into<DecisionLevel>>,
    ) -> Self {
        self.question(DecisionQuestion::score(instructions, levels).with_name(name))
    }
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

impl From<String> for DecisionInput {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for DecisionInput {
    fn from(text: &str) -> Self {
        Self::Text(text.into())
    }
}

impl From<Vec<DecisionInputMessage>> for DecisionInput {
    fn from(messages: Vec<DecisionInputMessage>) -> Self {
        Self::Messages(messages)
    }
}

impl DecisionInputMessage {
    /// Build a user turn from text or parts, e.g. `DecisionInputMessage::user("alpha")`.
    pub fn user(content: impl Into<DecisionInputContent>) -> Self {
        Self {
            role: UserRole::User,
            content: content.into(),
            message_type: None,
        }
    }
}

impl DecisionInputPart {
    /// Build a text part, e.g. `DecisionInputPart::text("alpha ")` followed by `DecisionInputPart::text("beta")`.
    pub fn text(text: impl Into<String>) -> Self {
        Self::InputText { text: text.into() }
    }
}

impl From<String> for DecisionInputContent {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for DecisionInputContent {
    fn from(text: &str) -> Self {
        Self::Text(text.into())
    }
}

impl From<Vec<DecisionInputPart>> for DecisionInputContent {
    fn from(parts: Vec<DecisionInputPart>) -> Self {
        Self::Parts(parts)
    }
}
