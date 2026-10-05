//! Share tokenizer validation and selected-token inference, e.g. Vev's Yes/No and Wald's A/B.
use super::{Qwen3_5ForCausalLM, Qwen3_5TextConfig};
use crate::{Action, Answer, Error, Question, Request, Result, Truncation, Usage};
use burn::tensor::backend::Backend;
use camino::Utf8Path;
use indexmap::IndexMap;
use std::ops::Range;
use tokenizers::Tokenizer;

pub(crate) struct Readout<B: Backend> {
    pub(crate) model: Qwen3_5ForCausalLM<B>,
    tokenizer: Tokenizer,
    pub(crate) max_positions: usize,
    vocab_size: usize,
    device: B::Device,
}

impl<B: Backend> Readout<B> {
    pub(crate) fn load(root: &Utf8Path, device: &B::Device) -> Result<Self> {
        let config = Qwen3_5TextConfig::from_pretrained(root)?;
        let mut tokenizer = Tokenizer::from_file(root.join("tokenizer.json"))
            .map_err(|error| Error::Tokenizer(error.to_string()))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|error| Error::Tokenizer(error.to_string()))?;
        Ok(Self {
            model: Qwen3_5ForCausalLM::from_pretrained(root, device)?,
            tokenizer,
            max_positions: config.max_position_embeddings,
            vocab_size: config.vocab_size,
            device: device.clone(),
        })
    }

    pub(crate) fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| Error::Tokenizer(error.to_string()))?;
        let ids = encoded.get_ids();
        if ids.iter().any(|&id| id as usize >= self.vocab_size) {
            return Err(Error::Tokenizer(
                "token ID exceeds Qwen3.5 vocabulary".into(),
            ));
        }
        Ok(ids.to_vec())
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
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| Error::Tokenizer(error.to_string()))?;
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
            let mut remove = std::collections::BTreeSet::new();
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
            || ids.iter().collect::<std::collections::BTreeSet<_>>().len() != ids.len()
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

pub(crate) fn sanitize(text: &str) -> String {
    // Caller text cannot inject Qwen special tokens, e.g. <|im_end|> becomes <¦im_end¦>.
    let mut out = String::new();
    let mut rest = text;
    while let Some((prefix, suffix)) = rest.split_once("<|") {
        out.push_str(prefix);
        if let Some((name, tail)) = suffix.split_once("|>")
            && !name.is_empty()
            && name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            out.push_str(&format!("<¦{name}¦>"));
            rest = tail;
        } else {
            out.push_str("<|");
            rest = suffix;
        }
    }
    out.push_str(rest);
    out
}
