use super::super::qwen3_5::Qwen3_5Config;
use crate::{Error, Question, Request, Result, Truncation, Usage};
use camino::Utf8Path;
use serde_json::{Map, Value};
use std::ops::Range;
use tokenizers::Tokenizer;

const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";

#[derive(Debug)]
pub struct EncodedQuestion {
    pub question_id: String,
    pub question_type: usize,
    pub question_span: Range<usize>,
    pub option_spans: Vec<Range<usize>>,
    pub option_ids: Vec<String>,
}

#[derive(Debug)]
pub struct EncodedRecord {
    pub input_ids: Vec<u32>,
    pub questions: Vec<EncodedQuestion>,
    pub usage: Usage,
}

/// Match Clef's `encode_record` prompt and span conventions for text/JSON inputs.
pub struct ClefProcessor {
    tokenizer: Tokenizer,
    vocab_size: usize,
    max_positions: usize,
}
impl ClefProcessor {
    pub fn from_pretrained(root: &Utf8Path, config: &Qwen3_5Config) -> Result<Self> {
        config.validate()?;
        let mut tokenizer = Tokenizer::from_file(root.join("tokenizer.json"))
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        // Assemble the full record ourselves, e.g. do not truncate instruction spans.
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(Self {
            tokenizer,
            vocab_size: config.text_config.vocab_size,
            max_positions: config.text_config.max_position_embeddings,
        })
    }
    fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        if encoded
            .get_ids()
            .iter()
            .any(|&id| id as usize >= self.vocab_size)
        {
            return Err(Error::Tokenizer(
                "token ID exceeds Qwen3.5 vocabulary".into(),
            ));
        }
        Ok(encoded.get_ids().to_vec())
    }
    pub fn process(&self, request: &Request) -> Result<EncodedRecord> {
        request.validate()?;
        if request.questions.is_empty() {
            return Err(Error::InvalidRequest(
                "Clef needs at least one question".into(),
            ));
        }
        if request.options.head_max_len.is_some() {
            return Err(Error::InvalidRequest(
                "Clef uses a joint schema; head_max_len is a Laya-only option".into(),
            ));
        }
        let max_len = request
            .options
            .max_len
            .unwrap_or(16384.min(self.max_positions));
        if max_len > self.max_positions {
            return Err(Error::InvalidRequest(format!(
                "max_len exceeds {} positions",
                self.max_positions
            )));
        }
        let mut schema = self.tokens("\n\nSCHEMA FIELDS:\n")?;
        let mut questions = Vec::new();
        for (index, (id, question)) in request.questions.iter().enumerate() {
            let (kind, question_type) = match question {
                Question::Noul { labels, .. } => {
                    if labels.r#true != "true" || labels.r#false != "false" {
                        return Err(Error::InvalidRequest("Clef uses fixed true/false option IDs; customize noul criteria instead of labels".into()));
                    }
                    ("noul", 0)
                }
                Question::Choice { .. } => ("choice", 1),
                Question::Score { .. } => ("score", 2),
            };
            schema.extend(self.tokens(&format!(
                "\nFIELD {}\nID: {id}\nTYPE: {kind}\nINSTRUCTION: ",
                index + 1
            ))?);
            let start = schema.len();
            schema.extend(self.tokens(question.instructions())?);
            let question_span = start..schema.len();
            schema.extend(self.tokens("\nALLOWED OPTIONS:\n")?);
            let options = options(question);
            let mut option_spans = Vec::new();
            let mut option_ids = Vec::new();
            for (index, (option_id, description)) in options.into_iter().enumerate() {
                schema.extend(self.tokens(&format!("OPTION {}: ", index + 1))?);
                let start = schema.len();
                let mut semantics = Map::new();
                semantics.insert("option_id".into(), Value::String(option_id.clone()));
                if !description.is_null() {
                    semantics.insert("description".into(), description);
                }
                schema.extend(self.tokens(&render(&Value::Object(semantics))?)?);
                option_spans.push(start..schema.len());
                option_ids.push(option_id);
                schema.extend(self.tokens("\n")?);
            }
            schema.extend(self.tokens("END FIELD\n")?);
            questions.push(EncodedQuestion {
                question_id: id.clone(),
                question_type,
                question_span,
                option_spans,
                option_ids,
            });
        }
        let mut prefix = self.tokens(&format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
        ))?;
        let suffix = self.tokens(
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
        )?;
        let fixed = prefix.len() + schema.len() + suffix.len();
        if fixed > max_len {
            return Err(Error::InvalidRequest(format!(
                "joint schema requires {fixed} tokens before state; max_len={max_len}"
            )));
        }
        let mut state = self.tokens(&render(&request.state)?)?;
        let state_tokens = state.len();
        let dropped = state_tokens.saturating_sub(max_len - fixed);
        if dropped > 0 && request.options.truncation == Truncation::Error {
            return Err(Error::InvalidRequest(
                "token budget truncates state; select truncation=truncate explicitly".into(),
            ));
        }
        // Clef keeps the start of every state, including conversation arrays.
        // Shift all spans after truncation so options still address their original text.
        state.truncate(max_len - fixed);
        let offset = prefix.len() + state.len();
        for question in &mut questions {
            question.question_span =
                question.question_span.start + offset..question.question_span.end + offset;
            for span in &mut question.option_spans {
                *span = span.start + offset..span.end + offset;
            }
        }
        prefix.extend(state);
        prefix.extend(schema);
        prefix.extend(suffix);
        let usage = Usage {
            input_tokens: prefix.len(),
            state_tokens,
            state_tokens_dropped: dropped,
            truncated: dropped > 0,
            truncated_questions: if dropped > 0 {
                request.questions.keys().cloned().collect()
            } else {
                Vec::new()
            },
            ..Default::default()
        };
        Ok(EncodedRecord {
            input_ids: prefix,
            questions,
            usage,
        })
    }
}

fn options(question: &Question) -> Vec<(String, Value)> {
    match question {
        Question::Noul { criteria, .. } => [
            ("true", "The proposition is true or the answer is yes."),
            ("false", "The proposition is false or the answer is no."),
        ]
        .into_iter()
        .map(|(key, default)| {
            (
                key.into(),
                criteria
                    .get(key)
                    .cloned()
                    .unwrap_or_else(|| Value::String(default.into())),
            )
        })
        .collect(),
        Question::Choice { criteria, .. } => {
            let mut options: Vec<_> = criteria
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            options.sort_by(|a, b| a.0.cmp(&b.0));
            options
        }
        Question::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(i, value)| (i.to_string(), value.clone()))
            .collect(),
    }
}

// Match Python's sorted compact JSON, including nested keys and float exponents.
// For example {"z":1,"a":0.000001} renders as {"a":1e-06,"z":1}.
fn render(value: &Value) -> Result<String> {
    fn json(value: &Value) -> Result<String> {
        match value {
            Value::Object(fields) => {
                let mut entries: Vec<_> = fields.iter().collect();
                entries.sort_by(|a, b| a.0.cmp(b.0));
                Ok(format!(
                    "{{{}}}",
                    entries
                        .into_iter()
                        .map(|(key, value)| Ok(format!(
                            "{}:{}",
                            serde_json::to_string(key)?,
                            json(value)?
                        )))
                        .collect::<Result<Vec<_>>>()?
                        .join(",")
                ))
            }
            Value::Array(values) => Ok(format!(
                "[{}]",
                values
                    .iter()
                    .map(json)
                    .collect::<Result<Vec<_>>>()?
                    .join(",")
            )),
            Value::Number(number) if number.is_f64() => {
                let number = number
                    .as_f64()
                    .ok_or_else(|| Error::InvalidRequest("invalid JSON float".into()))?;
                if number != 0.0 && (number.abs() < 1e-4 || number.abs() >= 1e16) {
                    let scientific = format!("{number:e}");
                    let (mantissa, exponent) = scientific
                        .split_once('e')
                        .ok_or_else(|| Error::InvalidRequest("invalid float exponent".into()))?;
                    let exponent: i32 = exponent.parse().map_err(|source| {
                        Error::InvalidRequest(format!("invalid float exponent: {source}"))
                    })?;
                    Ok(format!("{mantissa}e{exponent:+03}"))
                } else {
                    let text = number.to_string();
                    Ok(if text.contains('.') {
                        text
                    } else {
                        format!("{text}.0")
                    })
                }
            }
            _ => Ok(serde_json::to_string(value)?),
        }
    }
    match value {
        Value::String(text) => Ok(text.clone()),
        _ => json(value),
    }
}
