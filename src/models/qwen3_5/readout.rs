//! Share selected-token inference, e.g. Vev's Yes/No and Wald's A/B.
use std::collections::BTreeSet;
use std::ops::Range;

use burn::tensor::Device as BurnDevice;
use camino::Utf8Path;
use indexmap::IndexMap;
use tokenizers::Tokenizer;

use super::{Qwen3_5ForCausalLM, Qwen3_5TextConfig};
use crate::utils::{load_tokenizer, token_ids, tokenize};
use crate::{Action, Answer, Error, Question, Request, Result, Truncation, Usage};

pub(crate) struct Readout {
    pub(crate) model: Qwen3_5ForCausalLM,
    tokenizer: Tokenizer,
    pub(crate) max_positions: usize,
    vocab_size: usize,
    device: BurnDevice,
}

impl Readout {
    pub(crate) fn load(root: &Utf8Path, device: &BurnDevice) -> Result<Self> {
        let config = Qwen3_5TextConfig::from_pretrained(root)?;
        let tokenizer = load_tokenizer(&root.join("tokenizer.json"))?;
        Ok(Self {
            model: Qwen3_5ForCausalLM::from_pretrained(root, device)?,
            tokenizer,
            max_positions: config.max_position_embeddings,
            vocab_size: config.vocab_size,
            device: device.clone(),
        })
    }

    pub(crate) fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        token_ids(&self.tokenizer, text, self.vocab_size, "Qwen3.5")
    }

    // Accept separately tokenized prompt pieces, e.g. Decider's wide option labels.
    pub(crate) fn logits(&self, input: &[u32], answers: &[u32]) -> Result<Vec<f64>> {
        self.model
            .forward_selected(input, answers, &self.device)
            .map(|logits| logits.into_iter().map(f64::from).collect())
    }

    pub(crate) fn validate(&self, request: &Request) -> Result<usize> {
        request.validate()?;
        if request.questions.is_empty() || request.options.head_max_len.is_some() {
            return Err(Error::InvalidRequest(
                "at least one question is required; head_max_len is a Laya-only option".into(),
            ));
        }
        for question in request.questions.values() {
            if let Question::Noul { labels, .. } = question
                && (labels.r#true != "true" || labels.r#false != "false")
            {
                return Err(Error::InvalidRequest(
                    "custom noul labels are unsupported; use criteria descriptions".into(),
                ));
            }
        }
        let max_len = request.options.max_len.unwrap_or(self.max_positions);
        if max_len > self.max_positions {
            return Err(Error::InvalidRequest(format!(
                "max_len exceeds {} positions",
                self.max_positions
            )));
        }
        Ok(max_len)
    }

    pub(crate) fn probabilities(
        &self,
        prompt: &str,
        state: &[Range<usize>],
        groups: &[Vec<u32>],
        request: &Request,
        id: &str,
        usage: &mut Usage,
    ) -> Result<Vec<f64>> {
        let max_len = request.options.max_len.unwrap_or(self.max_positions);
        let encoded = tokenize(&self.tokenizer, prompt)?;
        let mut input = encoded.get_ids().to_vec();
        let dropped = input.len().saturating_sub(max_len);
        if dropped > 0 {
            if request.options.truncation == Truncation::Error {
                return Err(Error::InvalidRequest(format!(
                    "question {id:?} exceeds max_len={max_len}; select truncation=truncate explicitly"
                )));
            }
            let mut copies: Vec<Vec<usize>> = state
                .iter()
                .map(|range| {
                    encoded
                        .get_offsets()
                        .iter()
                        .enumerate()
                        .filter(|(_, (start, end))| {
                            start < end && range.start <= *start && *end <= range.end
                        })
                        .map(|(index, _)| index)
                        .collect()
                })
                .collect();
            if copies.iter().map(Vec::len).sum::<usize>() < dropped {
                return Err(Error::InvalidRequest(format!(
                    "question {id:?} exceeds the token budget even without state"
                )));
            }
            // Remove state suffixes only; preserve every option and the final readout cue.
            // Repeated Wald states are shortened in alternating copies, e.g. two copies lose equally.
            let mut remove = BTreeSet::new();
            let mut remaining = dropped;
            while remaining > 0 {
                for copy in &mut copies {
                    if remaining > 0
                        && let Some(index) = copy.pop()
                    {
                        remove.insert(index);
                        remaining -= 1;
                    }
                }
            }
            input = input
                .into_iter()
                .enumerate()
                .filter_map(|(i, token)| (!remove.contains(&i)).then_some(token))
                .collect();
            usage.truncated = true;
            usage.state_tokens_dropped = usage
                .state_tokens_dropped
                .max(dropped.div_ceil(state.len().max(1)));
            if !usage
                .truncated_questions
                .iter()
                .any(|question| question == id)
            {
                usage.truncated_questions.push(id.into());
            }
        }
        usage.input_tokens += input.len();
        let ids: Vec<_> = groups.iter().flatten().copied().collect();
        if groups.iter().any(Vec::is_empty)
            || ids.iter().collect::<BTreeSet<_>>().len() != ids.len()
        {
            return Err(Error::InvalidCheckpoint(
                "answer token groups must be nonempty and distinct".into(),
            ));
        }
        let logits = self.model.forward_selected(&input, &ids, &self.device)?;
        let token_probabilities = softmax(
            &logits
                .iter()
                .map(|&value| f64::from(value))
                .collect::<Vec<_>>(),
        )?;
        let mut tokens = token_probabilities.into_iter();
        Ok(groups
            .iter()
            .map(|group| tokens.by_ref().take(group.len()).sum())
            .collect())
    }
}

pub(crate) fn softmax(logits: &[f64]) -> Result<Vec<f64>> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(Error::Inference("non-finite or empty answer logits".into()));
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let values: Vec<_> = logits.iter().map(|value| (value - max).exp()).collect();
    let sum: f64 = values.iter().sum();
    Ok(values.into_iter().map(|value| value / sum).collect())
}

pub(crate) fn answer(
    question: &Question,
    probabilities: Vec<f64>,
    confidence: f64,
    legend: Vec<String>,
    true_index: usize,
) -> Result<Answer> {
    let max = probabilities.iter().copied().fold(0.0, f64::max);
    let action = Action::new(1.0);
    Ok(match question {
        Question::Noul { .. } => Answer::Noul {
            noul: *probabilities
                .get(true_index)
                .ok_or_else(|| Error::Inference("missing boolean probability".into()))?,
            confidence,
            answer_confidence: max,
            action,
        },
        Question::Choice { criteria, .. } => {
            let best = argmax(&probabilities);
            Answer::Choice {
                choice: criteria
                    .get_index(best)
                    .ok_or_else(|| Error::Inference("missing choice probability".into()))?
                    .0
                    .clone(),
                probabilities: criteria.keys().cloned().zip(probabilities).collect(),
                confidence,
                answer_confidence: max,
                action,
            }
        }
        Question::Score { .. } => Answer::Score {
            score: probabilities
                .iter()
                .enumerate()
                .map(|(i, probability)| i as f64 * probability)
                .sum(),
            probabilities: probabilities
                .into_iter()
                .enumerate()
                .map(|(i, probability)| (i.to_string(), probability))
                .collect(),
            legend: legend
                .into_iter()
                .enumerate()
                .map(|(i, text)| (i.to_string(), text))
                .collect::<IndexMap<_, _>>(),
            confidence,
            answer_confidence: max,
            action,
        },
    })
}

pub(crate) fn argmax(probabilities: &[f64]) -> usize {
    // Keep insertion order on ties, e.g. equal A/B probabilities select A.
    probabilities
        .iter()
        .enumerate()
        .fold(
            (0, -1.0),
            |best, (i, &p)| if p > best.1 { (i, p) } else { best },
        )
        .0
}

pub(crate) fn choice_confidence(probabilities: &[f64]) -> f64 {
    if probabilities.len() == 1 {
        return 1.0;
    }
    let prior = 1.0 / probabilities.len() as f64;
    (probabilities.iter().copied().fold(0.0, f64::max) - prior) / (1.0 - prior)
}
