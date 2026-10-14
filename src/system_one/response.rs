use bon::Builder;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::{Action, SystemOneAnswer};

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
pub struct SystemOneResponse {
    #[builder(into)]
    pub model: String,
    pub answers: IndexMap<String, SystemOneAnswer>,
    #[builder(default)]
    pub usage: Usage,
    pub metadata: Metadata,
}

impl SystemOneResponse {
    /// Assemble ordered answers with empty usage, e.g. `SystemOneResponse::new("local", answers, metadata)`.
    pub fn new(
        model: impl Into<String>,
        answers: IndexMap<String, SystemOneAnswer>,
        metadata: Metadata,
    ) -> Self {
        Self::builder()
            .model(model)
            .answers(answers)
            .metadata(metadata)
            .build()
    }

    /// Borrow an answer by question ID, e.g. `response.answer("severity")`.
    pub fn answer(&self, id: &str) -> Option<&SystemOneAnswer> {
        self.answers.get(id)
    }

    /// Read the probability of true, e.g. `response.noul("cancel")`; other types return `None`.
    pub fn noul(&self, id: &str) -> Option<f64> {
        self.answer(id)?.noul()
    }

    /// Borrow the selected choice ID, e.g. `response.choice("route")`; other types return `None`.
    pub fn choice(&self, id: &str) -> Option<&str> {
        self.answer(id)?.choice()
    }

    /// Read a score, e.g. `response.score("severity")`; other types return `None`.
    pub fn score(&self, id: &str) -> Option<f64> {
        self.answer(id)?.score()
    }

    /// Read model confidence for any answer type, e.g. `response.confidence("cancel")`.
    pub fn confidence(&self, id: &str) -> Option<f64> {
        Some(self.answer(id)?.confidence())
    }

    /// Read answer confidence separately, e.g. `response.answer_confidence("route")`.
    pub fn answer_confidence(&self, id: &str) -> Option<f64> {
        Some(self.answer(id)?.answer_confidence())
    }

    /// Borrow the action for an answer, e.g. `response.action("cancel")`.
    pub fn action(&self, id: &str) -> Option<&Action> {
        Some(self.answer(id)?.action())
    }

    /// Read the action probability directly, e.g. `response.act_probability("cancel")`.
    pub fn act_probability(&self, id: &str) -> Option<f64> {
        Some(self.answer(id)?.act_probability())
    }

    /// Read an option's probability by choice ID, e.g. `response.choice_probability("route", "billing")`.
    pub fn choice_probability(&self, id: &str, choice: &str) -> Option<f64> {
        self.answer(id)?.choice_probability(choice)
    }

    /// Read a zero-based score level's probability, e.g. `response.score_probability("severity", 2)`.
    pub fn score_probability(&self, id: &str, index: usize) -> Option<f64> {
        self.answer(id)?.score_probability(index)
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
