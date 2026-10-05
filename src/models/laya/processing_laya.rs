//! Assemble typed questions using Laya's token and truncation conventions.
use super::configuration_laya::LayaConfig;
use crate::{
    Error, Question, Request, Result, Truncation, Usage,
    models::modernbert::ModernBertConfig,
    utils::{load_tokenizer, read_checkpoint_json, render, token_ids},
};
use camino::Utf8Path;
use tokenizers::Tokenizer;

// Match Laya's reference head budgets; e.g. an option keeps its marker plus 48 tokens.
const MAX_OPTION_TOKENS: usize = 48;
const RESERVED_INSTRUCTION_TOKENS: usize = 16;
const MIN_OPTION_TOKENS: usize = 4;
const MIN_INSTRUCTION_TOKENS: usize = 8;

pub(crate) struct LayaProcessor {
    tokenizer: Tokenizer,
    cls: u32,
    sep: u32,
    pub pad: u32,
    mask: u32,
    mask_text: String,
    max_len: usize,
    head_max_len: usize,
    positions: usize,
    vocab: usize,
}
pub(crate) struct Encoded {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub kind: usize,
}
pub(crate) struct Batch {
    pub rows: Vec<Encoded>,
    pub usage: Usage,
}

impl LayaProcessor {
    pub fn load(root: &Utf8Path, config: &LayaConfig, encoder: &ModernBertConfig) -> Result<Self> {
        let tokenizer = load_tokenizer(&root.join("tokenizer/tokenizer.json"))?;
        let config_json: serde_json::Value =
            read_checkpoint_json(&root.join("tokenizer/tokenizer_config.json"))?;
        let token = |name: &str| -> Result<(String, u32)> {
            let value = config_json
                .get(name)
                .ok_or_else(|| Error::Tokenizer(format!("missing {name}")))?;
            let text = value
                .as_str()
                .or_else(|| value.get("content")?.as_str())
                .ok_or_else(|| Error::Tokenizer(format!("invalid {name}")))?;
            let id = tokenizer
                .token_to_id(text)
                .ok_or_else(|| Error::Tokenizer(format!("{name} is absent from vocabulary")))?;
            if id as usize >= encoder.vocab_size {
                return Err(Error::Tokenizer(format!(
                    "{name} exceeds encoder vocabulary"
                )));
            }
            Ok((text.into(), id))
        };
        let (_, cls) = token("cls_token")?;
        let (_, sep) = token("sep_token")?;
        let (_, pad) = token("pad_token")?;
        let (mask_text, mask) = token("mask_token")?;
        Ok(Self {
            tokenizer,
            cls,
            sep,
            pad,
            mask,
            mask_text,
            max_len: config.max_len,
            head_max_len: config.head_max_len,
            positions: encoder.max_position_embeddings,
            vocab: encoder.vocab_size,
        })
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Keep mask markers under processor control, e.g. user text cannot add an option marker.
        token_ids(
            &self.tokenizer,
            &text.replace(&self.mask_text, " "),
            self.vocab,
            "encoder",
        )
    }
    pub fn process(&self, request: &Request) -> Result<Batch> {
        request.validate()?;
        let max_len = request.options.max_len.unwrap_or(self.max_len);
        let head_max_len = request.options.head_max_len.unwrap_or(self.head_max_len);
        if max_len > self.positions || head_max_len > self.positions {
            return Err(Error::InvalidRequest(format!(
                "max_len must be 4..={} and head_max_len 16..={}",
                self.positions, self.positions
            )));
        }
        let mut batch = Batch {
            rows: Vec::new(),
            usage: Usage::default(),
        };
        if request.questions.is_empty() {
            return Ok(batch);
        }
        let state_ids = self.encode(&render(&request.state)?)?;
        batch.usage.state_tokens = state_ids.len();
        for (id, question) in &request.questions {
            let (mut row, head_truncated) = self.encode_head(question, head_max_len)?;
            // Refuse a head that cannot fit instead of silently losing option markers.
            if row.ids.len() >= max_len {
                return Err(Error::InvalidRequest(format!(
                    "question {id:?}: question head needs {} tokens plus a closing separator, max_len={max_len}",
                    row.ids.len()
                )));
            }
            let room = max_len - row.ids.len() - 1;
            let dropped = state_ids.len().saturating_sub(room);
            if request.options.truncation == Truncation::Error && (dropped > 0 || head_truncated) {
                return Err(Error::InvalidRequest(format!(
                    "question {id:?}: token budget truncates the input; select truncation=truncate explicitly"
                )));
            }
            if request.state.is_array() {
                // Conversation arrays keep newest turns, e.g. the user's last request.
                row.ids.extend(state_ids.iter().skip(dropped).copied());
            } else {
                row.ids.extend(state_ids.iter().take(room).copied());
            }
            row.ids.push(self.sep);
            batch.usage.input_tokens += row.ids.len();
            batch.usage.state_tokens_dropped = batch.usage.state_tokens_dropped.max(dropped);
            if dropped > 0 {
                batch.usage.truncated_questions.push(id.clone());
            }
            if head_truncated {
                batch.usage.truncated_head_questions.push(id.clone());
            }
            batch.rows.push(row);
        }
        batch.usage.truncated = !batch.usage.truncated_questions.is_empty()
            || !batch.usage.truncated_head_questions.is_empty();
        Ok(batch)
    }

    // Build only the question head so state truncation can use its final token count.
    // Option markers stay at the start of each option, e.g. [MASK] followed by its text.
    fn encode_head(&self, question: &Question, head_max_len: usize) -> Result<(Encoded, bool)> {
        let (name, kind) = kind(question);
        let mut instruction =
            self.encode(&format!("{name} question: {}", question.instructions()))?;
        let options = rendered_options(question)?;
        let mut head_truncated = false;
        let mut option_ids = Vec::with_capacity(options.len());
        for option in options {
            let mut tokens = self.encode(&format!(" {option}"))?;
            head_truncated |= tokens.len() > MAX_OPTION_TOKENS;
            tokens.truncate(MAX_OPTION_TOKENS);
            tokens.insert(0, self.mask);
            option_ids.push(tokens);
        }

        let mut option_tokens: usize = option_ids.iter().map(Vec::len).sum();
        if head_max_len.saturating_sub(option_tokens) < RESERVED_INSTRUCTION_TOKENS {
            let option_budget = ((head_max_len - RESERVED_INSTRUCTION_TOKENS)
                / option_ids.len().max(1))
            .max(MIN_OPTION_TOKENS);
            for tokens in &mut option_ids {
                head_truncated |= tokens.len() > option_budget;
                tokens.truncate(option_budget);
            }
            option_tokens = option_ids.iter().map(Vec::len).sum();
        }
        let instruction_budget = head_max_len
            .saturating_sub(option_tokens)
            .max(MIN_INSTRUCTION_TOKENS);
        head_truncated |= instruction.len() > instruction_budget;
        instruction.truncate(instruction_budget);

        let mut ids = vec![self.cls];
        ids.extend(instruction);
        ids.push(self.sep);
        let mut markers = Vec::with_capacity(option_ids.len());
        for tokens in option_ids {
            markers.push(ids.len());
            ids.extend(tokens);
        }
        ids.push(self.sep);
        Ok((Encoded { ids, markers, kind }, head_truncated))
    }
}

// Keep Laya's question type IDs and prompt wording with its preprocessing.
pub(super) fn kind(question: &Question) -> (&'static str, usize) {
    match question {
        Question::Choice { .. } => ("choice", 0),
        Question::Score { .. } => ("score", 1),
        Question::Noul { .. } => ("noul", 2),
    }
}
fn rendered_options(question: &Question) -> Result<Vec<String>> {
    match question {
        Question::Choice { criteria, .. } => criteria
            .iter()
            .map(|(key, value)| {
                if value.is_null() || value.as_str() == Some("") {
                    Ok(key.clone())
                } else {
                    Ok(format!("{key}: {}", render(value)?))
                }
            })
            .collect(),
        Question::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(i, value)| Ok(format!("level {i}: {}", render(value)?)))
            .collect(),
        Question::Noul {
            criteria, labels, ..
        } => [
            (
                "false",
                labels.r#false.trim(),
                "no, the statement does not hold",
            ),
            ("true", labels.r#true.trim(), "yes, the statement holds"),
        ]
        .into_iter()
        .map(|(key, label, fallback)| {
            let description = match criteria.get(key) {
                Some(value) if !value.is_null() && value.as_str() != Some("") => render(value)?,
                _ => fallback.into(),
            };
            Ok(format!("{label}: {description}"))
        })
        .collect(),
    }
}
