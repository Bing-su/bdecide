use indexmap::IndexMap;
use serde_json::{Value, json};

use super::{
    DecisionAnswer,
    DecisionChoiceProbability,
    DecisionInput,
    DecisionInputContent,
    DecisionInputPart,
    DecisionQuestion,
    DecisionScoreProbability,
    DecisionsRequest,
    DecisionsResponse,
    DecisionsUsage,
};
use crate::{
    Error,
    NoulLabels,
    Result,
    SystemOneAnswer,
    SystemOneQuestion,
    SystemOneRequest,
    SystemOneResponse,
};

impl DecisionsRequest {
    /// Validate without loading a model, e.g. reject duplicate typed choices before model I/O.
    pub fn validate(&self) -> Result<()> {
        self.to_system_one().map(|_request| ())
    }

    pub(crate) fn to_system_one(&self) -> Result<SystemOneRequest> {
        if self.model.trim().is_empty() {
            return Err(Error::InvalidRequest("model must not be empty".into()));
        }
        if self
            .safety_identifier
            .as_ref()
            .is_some_and(|id| id.chars().count() > 128)
        {
            return Err(Error::InvalidRequest(
                "safety_identifier must be at most 128 characters".into(),
            ));
        }
        let state = match &self.input {
            DecisionInput::Text(text) => Value::String(text.clone()),
            DecisionInput::Messages(messages) => {
                let mut turns = Vec::with_capacity(messages.len());
                for message in messages {
                    let content = match &message.content {
                        DecisionInputContent::Text(text) => text.clone(),
                        DecisionInputContent::Parts(parts) => {
                            let mut text = String::new();
                            for part in parts {
                                match part {
                                    DecisionInputPart::InputText { text: part } => {
                                        text.push_str(part)
                                    }
                                    DecisionInputPart::InputImage { .. } => {
                                        return Err(Error::InvalidRequest(
                                            "decisions currently accepts text only; use system_one for local images".into(),
                                        ));
                                    }
                                }
                            }
                            text
                        }
                    };
                    // Preserve turn boundaries while joining adjacent text parts verbatim.
                    turns.push(json!({"role": "user", "content": content}));
                }
                Value::Array(turns)
            }
        };
        let questions = self
            .questions
            .iter()
            .enumerate()
            .map(|(index, question)| Ok((index.to_string(), question.to_system_one()?)))
            .collect::<Result<IndexMap<_, _>>>()?;
        let request = SystemOneRequest::new(state, questions);
        request.validate()?;
        Ok(request)
    }

    pub(crate) fn decode(&self, response: SystemOneResponse) -> Result<DecisionsResponse> {
        if response.answers.len() != self.questions.len() {
            return Err(Error::Inference("decisions answer count mismatch".into()));
        }
        let answers = self
            .questions
            .iter()
            .enumerate()
            .map(|(index, question)| {
                let answer = response.answers.get(&index.to_string()).ok_or_else(|| {
                    Error::Inference(format!("missing decisions answer at index {index}"))
                })?;
                question.decode(answer)
            })
            .collect::<Result<Vec<_>>>()?;
        let total_tokens = response
            .usage
            .input_tokens
            .checked_add(response.usage.output_tokens)
            .ok_or_else(|| Error::Inference("decisions token count overflow".into()))?;
        Ok(DecisionsResponse {
            model: response.model,
            answers,
            usage: DecisionsUsage {
                input_tokens: response.usage.input_tokens,
                input_tokens_details: Default::default(),
                output_tokens: response.usage.output_tokens,
                output_tokens_details: Default::default(),
                total_tokens,
            },
        })
    }
}

impl DecisionQuestion {
    fn to_system_one(&self) -> Result<SystemOneQuestion> {
        Ok(match self {
            Self::Predicate { instructions, .. } => SystemOneQuestion::Noul {
                instructions: instructions.clone(),
                criteria: IndexMap::new(),
                labels: NoulLabels::default(),
            },
            Self::Choice {
                instructions,
                choices,
                ..
            } => {
                if !(2..=255).contains(&choices.len()) {
                    return Err(Error::InvalidRequest("choice needs 2..255 options".into()));
                }
                let mut criteria = IndexMap::new();
                for choice in choices {
                    // JSON literals keep boolean true distinct from the string "true" in prompts.
                    let key = serde_json::to_string(&choice.value)?;
                    if criteria
                        .insert(key, serde_json::to_value(choice)?)
                        .is_some()
                    {
                        return Err(Error::InvalidRequest("choice values must be unique".into()));
                    }
                }
                SystemOneQuestion::Choice {
                    instructions: instructions.clone(),
                    criteria,
                }
            }
            Self::Score {
                instructions,
                levels,
                ..
            } => SystemOneQuestion::Score {
                instructions: instructions.clone(),
                // Render levels as text so model-specific object schemas cannot drop labels/descriptions.
                criteria: levels
                    .iter()
                    .map(|level| serde_json::to_string(level).map(Value::String))
                    .collect::<std::result::Result<_, _>>()?,
            },
        })
    }

    fn decode(&self, answer: &SystemOneAnswer) -> Result<DecisionAnswer> {
        Ok(match (self, answer) {
            (Self::Predicate { name, .. }, SystemOneAnswer::Noul { noul, .. }) => {
                DecisionAnswer::Predicate {
                    name: name.clone(),
                    probability: *noul,
                }
            }
            (
                Self::Choice { name, choices, .. },
                SystemOneAnswer::Choice {
                    choice,
                    probabilities,
                    confidence,
                    ..
                },
            ) => {
                if probabilities.len() != choices.len() {
                    return Err(Error::Inference("decisions choice count mismatch".into()));
                }
                let mut selected = None;
                let mut decoded = Vec::with_capacity(choices.len());
                for option in choices {
                    let key = serde_json::to_string(&option.value)?;
                    if &key == choice {
                        selected = Some(option.value.clone());
                    }
                    decoded.push(DecisionChoiceProbability {
                        value: option.value.clone(),
                        probability: probability(probabilities, &key)?,
                    });
                }
                DecisionAnswer::Choice {
                    name: name.clone(),
                    choice: selected
                        .ok_or_else(|| Error::Inference("unknown decisions choice value".into()))?,
                    probabilities: decoded,
                    confidence: *confidence,
                }
            }
            (
                Self::Score { name, levels, .. },
                SystemOneAnswer::Score {
                    score,
                    probabilities,
                    confidence,
                    ..
                },
            ) => {
                if probabilities.len() != levels.len() {
                    return Err(Error::Inference("decisions score count mismatch".into()));
                }
                DecisionAnswer::Score {
                    name: name.clone(),
                    score: *score,
                    probabilities: levels
                        .iter()
                        .enumerate()
                        .map(|(index, level)| {
                            Ok(DecisionScoreProbability {
                                value: index,
                                label: level.label.clone(),
                                probability: probability(probabilities, &index.to_string())?,
                            })
                        })
                        .collect::<Result<_>>()?,
                    confidence: *confidence,
                }
            }
            _ => return Err(Error::Inference("decisions answer type mismatch".into())),
        })
    }
}

fn probability(probabilities: &IndexMap<String, f64>, key: &str) -> Result<f64> {
    probabilities
        .get(key)
        .copied()
        .ok_or_else(|| Error::Inference(format!("missing decisions probability for {key}")))
}
