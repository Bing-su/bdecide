use serde::{Deserialize, Serialize};

use super::DecisionChoiceValue;

/// Return only OpenAI fields, e.g. an ordered answer array instead of System One metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionsResponse {
    pub model: String,
    pub answers: Vec<DecisionAnswer>,
    pub usage: DecisionsUsage,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionsUsage {
    pub input_tokens: usize,
    pub input_tokens_details: DecisionInputTokensDetails,
    pub output_tokens: usize,
    pub output_tokens_details: DecisionOutputTokensDetails,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecisionInputTokensDetails {
    pub cached_tokens: usize,
    pub cache_write_tokens: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecisionOutputTokensDetails {
    pub reasoning_tokens: usize,
}

impl DecisionsResponse {
    /// Find the first answer with this name, e.g. `response.answer("severity")`.
    /// Duplicate names use request order; unnamed answers remain accessible through `answers`.
    pub fn answer(&self, name: &str) -> Option<&DecisionAnswer> {
        self.answers
            .iter()
            .find(|answer| answer.name() == Some(name))
    }

    /// Iterate every answer with a name in request order, e.g. `response.answers_named("q")`.
    pub fn answers_named<'a>(
        &'a self,
        name: &'a str,
    ) -> impl Iterator<Item = &'a DecisionAnswer> + 'a {
        self.answers
            .iter()
            .filter(move |answer| answer.name() == Some(name))
    }

    /// Read a predicate directly, e.g. `response.probability("cancel")`; missing or other types return `None`.
    pub fn probability(&self, name: &str) -> Option<f64> {
        self.answer(name)?.probability()
    }

    /// Read a typed selection, e.g. `response.choice("department")`; missing or other types return `None`.
    pub fn choice(&self, name: &str) -> Option<&DecisionChoiceValue> {
        self.answer(name)?.choice()
    }

    /// Read a string selection without matching an enum, e.g. `response.choice_str("department")`.
    pub fn choice_str(&self, name: &str) -> Option<&str> {
        match self.choice(name)? {
            DecisionChoiceValue::String(value) => Some(value),
            DecisionChoiceValue::Boolean(_) => None,
        }
    }

    /// Read a boolean selection without conflating it with text, e.g. `response.choice_bool("approved")`.
    pub fn choice_bool(&self, name: &str) -> Option<bool> {
        match self.choice(name)? {
            DecisionChoiceValue::Boolean(value) => Some(*value),
            DecisionChoiceValue::String(_) => None,
        }
    }

    /// Read a score directly, e.g. `response.score("severity")`; missing or other types return `None`.
    pub fn score(&self, name: &str) -> Option<f64> {
        self.answer(name)?.score()
    }

    /// Read choice/score confidence, e.g. `response.confidence("department")`; predicates and refusals return `None`.
    pub fn confidence(&self, name: &str) -> Option<f64> {
        self.answer(name)?.confidence()
    }

    /// Read a typed option's probability, e.g. `response.choice_probability("approved", true)`.
    pub fn choice_probability(
        &self,
        name: &str,
        value: impl Into<DecisionChoiceValue>,
    ) -> Option<f64> {
        self.answer(name)?.choice_probability(value)
    }

    /// Read a level's probability by zero-based index, e.g. `response.score_probability("severity", 2)`.
    pub fn score_probability(&self, name: &str, value: usize) -> Option<f64> {
        self.answer(name)?.score_probability(value)
    }
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
