use serde::{Deserialize, Serialize};

use super::DecisionChoiceValue;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DecisionAnswer {
    Predicate {
        name: Option<String>,
        probability: f64,
    },
    Choice {
        name: Option<String>,
        choice: DecisionChoiceValue,
        probabilities: Vec<DecisionChoiceProbability>,
        confidence: f64,
    },
    Score {
        name: Option<String>,
        score: f64,
        probabilities: Vec<DecisionScoreProbability>,
        confidence: f64,
    },
    /// Read hosted responses too; local inference does not invent refusal decisions.
    Refusal { name: Option<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionChoiceProbability {
    pub value: DecisionChoiceValue,
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionScoreProbability {
    pub value: usize,
    pub label: String,
    pub probability: f64,
}

impl DecisionAnswer {
    /// Read the echoed optional name for any answer, e.g. `answer.name() == Some("severity")`.
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Predicate { name, .. }
            | Self::Choice { name, .. }
            | Self::Score { name, .. }
            | Self::Refusal { name } => name.as_deref(),
        }
    }

    /// Read a predicate's probability, e.g. `response.answers.first()?.probability()` for an unnamed question.
    pub fn probability(&self) -> Option<f64> {
        match self {
            Self::Predicate { probability, .. } => Some(*probability),
            _ => None,
        }
    }

    /// Borrow a selected string or boolean value, e.g. `answer.choice()`.
    pub fn choice(&self) -> Option<&DecisionChoiceValue> {
        match self {
            Self::Choice { choice, .. } => Some(choice),
            _ => None,
        }
    }

    /// Read a score without matching the answer enum, e.g. `answer.score()`.
    pub fn score(&self) -> Option<f64> {
        match self {
            Self::Score { score, .. } => Some(*score),
            _ => None,
        }
    }

    /// Read confidence when present, e.g. `answer.confidence()` for Choice or Score.
    pub fn confidence(&self) -> Option<f64> {
        match self {
            Self::Choice { confidence, .. } | Self::Score { confidence, .. } => Some(*confidence),
            _ => None,
        }
    }

    /// Compare typed values rather than text, e.g. `answer.choice_probability(true)` differs from `answer.choice_probability("true")`.
    pub fn choice_probability(&self, value: impl Into<DecisionChoiceValue>) -> Option<f64> {
        let Self::Choice { probabilities, .. } = self else {
            return None;
        };
        let value = value.into();
        probabilities
            .iter()
            .find(|entry| entry.value == value)
            .map(|entry| entry.probability)
    }

    /// Find the supplied level index, e.g. `answer.score_probability(2)` even if entries are reordered.
    pub fn score_probability(&self, value: usize) -> Option<f64> {
        let Self::Score { probabilities, .. } = self else {
            return None;
        };
        probabilities
            .iter()
            .find(|entry| entry.value == value)
            .map(|entry| entry.probability)
    }

    /// Distinguish refusals from missing scalar values, e.g. `answer.is_refusal()` before inspecting a hosted result.
    pub fn is_refusal(&self) -> bool {
        matches!(self, Self::Refusal { .. })
    }
}
