use bon::Builder;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
#[builder(on(String, into))]
pub struct Metadata {
    pub model_id: String,
    pub revision: Option<String>,
    pub commit_sha: Option<String>,
    pub subfolder: Option<String>,
    pub architecture: String,
    pub device: String,
}

impl Metadata {
    /// Describe a model before resolving its revision, e.g. `Metadata::new("local", "laya", "cpu")`.
    pub fn new(
        model_id: impl Into<String>,
        architecture: impl Into<String>,
        device: impl Into<String>,
    ) -> Self {
        Self::builder()
            .model_id(model_id)
            .architecture(architecture)
            .device(device)
            .build()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct Response {
    #[builder(into)]
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    #[builder(default)]
    pub usage: Usage,
    pub metadata: Metadata,
}

impl Response {
    /// Assemble ordered answers with empty usage, e.g. `Response::new("local", answers, metadata)`.
    pub fn new(
        model: impl Into<String>,
        answers: IndexMap<String, Answer>,
        metadata: Metadata,
    ) -> Self {
        Self::builder()
            .model(model)
            .answers(answers)
            .metadata(metadata)
            .build()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Builder)]
#[builder(on(_, default))]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub state_tokens: usize,
    pub state_tokens_dropped: usize,
    pub truncated: bool,
    pub truncated_questions: Vec<String>,
    /// Record head truncation too, e.g. a long option description losing its suffix.
    pub truncated_head_questions: Vec<String>,
}

impl Usage {
    /// Start token accounting at zero, e.g. `Usage::new()` before processing.
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct Action {
    pub act_probability: f64,
}

impl Action {
    /// Record the model's action probability, e.g. `Action::new(0.8)`.
    pub fn new(act_probability: f64) -> Self {
        Self { act_probability }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice {
        choice: String,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
        answer_confidence: f64,
        action: Action,
    },
    Score {
        score: f64,
        legend: IndexMap<String, String>,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
        answer_confidence: f64,
        action: Action,
    },
    Noul {
        noul: f64,
        confidence: f64,
        answer_confidence: f64,
        action: Action,
    },
}
