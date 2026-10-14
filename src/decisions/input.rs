use serde::{Deserialize, Serialize};

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
