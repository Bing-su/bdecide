use serde::{Deserialize, Serialize};

use super::{DecisionAnswer, DecisionChoiceValue};

/// Return only OpenAI fields, e.g. an ordered answer array instead of System One metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionsResponse {
    pub model: String,
    pub answers: Vec<DecisionAnswer>,
    pub usage: DecisionsUsage,
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
