//! Share selected-token inference, e.g. Vev's Yes/No and Wald's A/B.
use std::collections::BTreeSet;
use std::ops::Range;

use burn::tensor::Device as BurnDevice;
use camino::Utf8Path;
use tokenizers::Tokenizer;

use super::{Qwen3_5ForCausalLM, Qwen3_5TextConfig};
use crate::utils::decision::softmax;
use crate::utils::{load_tokenizer, token_ids, tokenize};
use crate::{Error, Question, Request, Result, Truncation, Usage};

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
        request.validate_text_only()?;
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
