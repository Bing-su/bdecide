use bon::Builder;
use serde::{Deserialize, Serialize};

use super::{DecisionChoice, DecisionInput, DecisionLevel, DecisionQuestion};

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
