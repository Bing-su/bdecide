use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub model_id: String,
    pub revision: Option<String>,
    pub commit_sha: Option<String>,
    pub subfolder: Option<String>,
    pub architecture: String,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub act_probability: f64,
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
