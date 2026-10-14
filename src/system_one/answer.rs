use bon::Builder;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

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
pub enum SystemOneAnswer {
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

impl SystemOneAnswer {
    /// Read a Noul answer's probability of true, e.g. `answer.noul()`.
    pub fn noul(&self) -> Option<f64> {
        match self {
            Self::Noul { noul, .. } => Some(*noul),
            _ => None,
        }
    }

    /// Borrow a selected choice ID, e.g. `answer.choice() == Some("billing")`.
    pub fn choice(&self) -> Option<&str> {
        match self {
            Self::Choice { choice, .. } => Some(choice),
            _ => None,
        }
    }

    /// Read a score without matching the enum, e.g. `answer.score()`.
    pub fn score(&self) -> Option<f64> {
        match self {
            Self::Score { score, .. } => Some(*score),
            _ => None,
        }
    }

    /// Read model confidence, present on every System One answer, e.g. `answer.confidence()`.
    pub fn confidence(&self) -> f64 {
        match self {
            Self::Choice { confidence, .. }
            | Self::Score { confidence, .. }
            | Self::Noul { confidence, .. } => *confidence,
        }
    }

    /// Read answer confidence without conflating it with model confidence, e.g. `answer.answer_confidence()`.
    pub fn answer_confidence(&self) -> f64 {
        match self {
            Self::Choice {
                answer_confidence, ..
            }
            | Self::Score {
                answer_confidence, ..
            }
            | Self::Noul {
                answer_confidence, ..
            } => *answer_confidence,
        }
    }

    /// Borrow the action, present on every System One answer, e.g. `answer.action()`.
    pub fn action(&self) -> &Action {
        match self {
            Self::Choice { action, .. }
            | Self::Score { action, .. }
            | Self::Noul { action, .. } => action,
        }
    }

    /// Read the action probability without unpacking Action, e.g. `answer.act_probability()`.
    pub fn act_probability(&self) -> f64 {
        self.action().act_probability
    }

    /// Look up an exact choice ID, e.g. `answer.choice_probability("billing")`.
    pub fn choice_probability(&self, choice: &str) -> Option<f64> {
        let Self::Choice { probabilities, .. } = self else {
            return None;
        };
        probabilities.get(choice).copied()
    }

    /// Look up a numeric level key independently of map order, e.g. `answer.score_probability(2)` reads `"2"`.
    pub fn score_probability(&self, index: usize) -> Option<f64> {
        let Self::Score { probabilities, .. } = self else {
            return None;
        };
        probabilities.get(&index.to_string()).copied()
    }
}
