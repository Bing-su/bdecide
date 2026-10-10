//! Render each released model's trained question format and selected answer positions.
use std::collections::BTreeSet;

use camino::Utf8Path;
use serde_json::Value;
use tokenizers::Tokenizer;

use super::D1OmniConfig;
use crate::models::lfm2_vl::Lfm2VlConfig;
use crate::models::lfm2_vl::vision::ImageCrops;
use crate::utils::{load_tokenizer, render, sanitize, token_ids};
use crate::{Error, Question, Request, Result, Truncation, Usage};

pub(super) struct Processor {
    tokenizer: Tokenizer,
    vocab: usize,
    bos: String,
}
pub(super) struct Row {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub groups: Vec<Vec<u32>>,
    pub qtype: usize,
}

impl Processor {
    fn load(root: &Utf8Path, bos_token_id: u32, vocab: usize) -> Result<Self> {
        let tokenizer = load_tokenizer(&root.join("tokenizer.json"))?;
        let bos = tokenizer
            .id_to_token(bos_token_id)
            .ok_or_else(|| Error::Tokenizer("missing BOS token".into()))?;
        Ok(Self {
            tokenizer,
            vocab,
            bos,
        })
    }

    pub(super) fn load_omni(root: &Utf8Path, config: &D1OmniConfig) -> Result<Self> {
        let processor = Self::load(root, config.bos_token_id, config.text_config.vocab_size)?;
        for token in [
            "<|reserved_7|>",
            "<|reserved_8|>",
            "<|reserved_9|>",
            "<|reserved_10|>",
            "<|reserved_11|>",
            "<|mask|>",
        ] {
            processor.special(token)?;
        }
        Ok(processor)
    }

    pub(super) fn load_causal(root: &Utf8Path, config: &Lfm2VlConfig) -> Result<Self> {
        let processor = Self::load(root, config.bos_token_id, config.text_config.vocab_size)?;
        for token in [
            "<|im_start|>",
            "<|im_end|>",
            "<|image_start|>",
            "<|image_end|>",
            "<|img_thumbnail|>",
            "<image>",
        ] {
            processor.special(token)?;
        }
        if processor.special("<image>")? != config.image_token_id {
            return Err(Error::Tokenizer(
                "image_token_id does not match tokenizer".into(),
            ));
        }
        Ok(processor)
    }

    pub(super) fn tokens(&self, text: &str) -> Result<Vec<u32>> {
        token_ids(&self.tokenizer, text, self.vocab, "d1")
    }

    fn special(&self, token: &str) -> Result<u32> {
        self.tokenizer
            .token_to_id(token)
            .filter(|&id| (id as usize) < self.vocab)
            .ok_or_else(|| Error::Tokenizer(format!("missing or out-of-vocabulary token {token}")))
    }

    pub(super) fn validate(&self, max_positions: usize, request: &Request) -> Result<usize> {
        request.validate()?;
        if request.questions.is_empty() || request.options.head_max_len.is_some() {
            return Err(Error::InvalidRequest(
                "at least one question is required; head_max_len is a Laya-only option".into(),
            ));
        }
        for q in request.questions.values() {
            match q {
                Question::Choice { criteria, .. } if criteria.len() < 2 => {
                    return Err(Error::InvalidRequest(
                        "d1 choice needs at least two options".into(),
                    ));
                }
                Question::Score { criteria, .. } if !(2..=10).contains(&criteria.len()) => {
                    return Err(Error::InvalidRequest("d1 score needs 2..10 levels".into()));
                }
                Question::Noul { labels, .. }
                    if labels.r#false != "false" || labels.r#true != "true" =>
                {
                    return Err(Error::InvalidRequest(
                        "d1 uses fixed boolean labels; use criteria descriptions".into(),
                    ));
                }
                _ => {}
            }
        }
        let limit = request.options.max_len.unwrap_or(max_positions);
        if limit > max_positions {
            return Err(Error::InvalidRequest(
                "max_len exceeds the d1 context".into(),
            ));
        }
        Ok(limit)
    }

    pub(super) fn causal(
        &self,
        request: &Request,
        q: &Question,
        images: &[ImageCrops],
        limit: usize,
        id: &str,
        usage: &mut Usage,
    ) -> Result<Row> {
        let mut markup = String::new();
        for image in images {
            markup.push_str("<|image_start|>");
            if image.rows * image.cols > 1 {
                for row in 0..image.rows {
                    for col in 0..image.cols {
                        markup.push_str(&format!("<|img_row_{}_col_{}|>", row + 1, col + 1));
                        markup.push_str(&"<image>".repeat(256));
                    }
                }
                markup.push_str("<|img_thumbnail|>");
            }
            if let Some(crop) = image.crops.last() {
                markup.push_str(&"<image>".repeat(crop.height * crop.width / 4));
            }
            markup.push_str("<|image_end|>");
        }
        let state = if request.state.is_null() {
            String::new()
        } else {
            let text = if let Value::String(text) = &request.state {
                text.clone()
            } else {
                serde_json::to_string_pretty(&request.state)?
            };
            safe_text(&text)
        };
        let header = format!("{}<|im_start|>user\n{markup}", self.bos);
        let (question, groups) = self.causal_question(q)?;
        let cue = if request.state.is_null() {
            ""
        } else {
            "\n\n\nQUESTION:\n"
        };
        let suffix = format!("{cue}{question}<|im_end|>\n<|im_start|>assistant\n");
        let prefix = format!("{header}{state}");
        let mut ids = self.tokens(&format!("{prefix}{suffix}"))?;
        usage.state_tokens = self.tokens(&state)?.len();
        if ids.len() > limit {
            let mut state_ids = self.tokens(&state)?;
            let header_ids = self.tokens(&header)?;
            let suffix_ids = self.tokens(&suffix)?;
            let keep = limit
                .checked_sub(header_ids.len() + suffix_ids.len())
                .ok_or_else(|| {
                    Error::InvalidRequest("images and options do not fit in max_len".into())
                })?;
            let dropped = state_ids.len().saturating_sub(keep);
            self.truncated(request, id, dropped, false, usage)?;
            state_ids.truncate(keep);
            ids = [header_ids, state_ids, suffix_ids].concat();
        }
        let image_id = self.special("<image>")?;
        let expected: usize = images
            .iter()
            .flat_map(|image| &image.crops)
            .map(|crop| crop.height * crop.width / 4)
            .sum();
        if ids.iter().filter(|&&token| token == image_id).count() != expected {
            return Err(Error::InvalidRequest(
                "image placeholders do not match media features".into(),
            ));
        }
        usage.input_tokens += ids.len();
        Ok(Row {
            ids,
            markers: Vec::new(),
            groups,
            qtype: qtype(q),
        })
    }

    fn causal_question(&self, q: &Question) -> Result<(String, Vec<Vec<u32>>)> {
        let instructions = safe_text(q.instructions());
        Ok(match q {
            Question::Noul { criteria, .. } => {
                let extra = if criteria.is_empty() {
                    String::new()
                } else {
                    let value = |key| {
                        criteria
                            .get(key)
                            .map(render)
                            .transpose()
                            .map(|text| text.unwrap_or_else(|| "None".into()))
                    };
                    format!(
                        "\nYes: {}\nNo: {}",
                        safe_text(&value("true")?),
                        safe_text(&value("false")?)
                    )
                };
                (
                    format!("{instructions}{extra}\n\nReply with yes or no only."),
                    vec![
                        self.forms(&["yes", "Yes", "YES"])?,
                        self.forms(&["no", "No", "NO"])?,
                    ],
                )
            }
            Question::Score { criteria, .. } => {
                let lines = criteria
                    .iter()
                    .enumerate()
                    .map(|(i, value)| Ok(format!("{i} {}", safe_text(&render(value)?))))
                    .collect::<Result<Vec<_>>>()?
                    .join("\n");
                let groups = (0..criteria.len())
                    .map(|i| self.forms(&[&i.to_string()]))
                    .collect::<Result<Vec<_>>>()?;
                (
                    format!(
                        "{instructions}\n\n{lines}\n\nReply with a single digit 0-{} only.",
                        criteria.len() - 1
                    ),
                    groups,
                )
            }
            Question::Choice { criteria, .. } => {
                let labels: Vec<_> = criteria.keys().map(|x| x.trim()).collect();
                let native = labels
                    .iter()
                    .all(|x| x.chars().count() == 1 && x.chars().all(char::is_alphabetic));
                let pool: Vec<_> = ('A'..='Z')
                    .map(|x| x.to_string())
                    .chain((0..100).map(|i| format!("{i:02}")))
                    .chain(('a'..='z').map(|x| x.to_string()))
                    .chain((0..200).map(|i| format!("#{i}")))
                    .chain(('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}"))))
                    .collect();
                let mut used = BTreeSet::new();
                let mut groups = Vec::new();
                let mut lines = Vec::new();
                for (i, (label, value)) in criteria.iter().enumerate() {
                    let code = if native {
                        label.trim().to_owned()
                    } else if labels.len() <= 26 {
                        char::from(b'A' + i as u8).to_string()
                    } else {
                        format!("{i:02}")
                    };
                    let mut selected = None;
                    for candidate in std::iter::once(&code).chain(pool.iter()) {
                        if let [tid] = self.tokens(candidate)?.as_slice()
                            && !used.contains(tid)
                        {
                            selected = Some((candidate.clone(), *tid));
                            break;
                        }
                    }
                    let (code, tid) = selected.ok_or_else(|| {
                        Error::Tokenizer("no single-token option alias left".into())
                    })?;
                    used.insert(tid);
                    let mut group = vec![tid];
                    if let [extra] = self.tokens(&format!(" {code}"))?.as_slice()
                        && *extra != tid
                    {
                        group.push(*extra);
                    }
                    groups.push(group);
                    let text = if value.is_null() || value.as_str() == Some("") {
                        label.replace('_', " ")
                    } else {
                        render(value)?
                    };
                    lines.push(format!("{code} {}", safe_text(&text)));
                }
                (
                    format!(
                        "{instructions}\n\nOptions:\n{}\n\nReply with the option code only.",
                        lines.join("\n")
                    ),
                    groups,
                )
            }
        })
    }

    fn forms(&self, texts: &[&str]) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for text in texts {
            if let [id] = self.tokens(text)?.as_slice()
                && !ids.contains(id)
            {
                ids.push(*id);
            }
        }
        if ids.is_empty() {
            return Err(Error::Tokenizer(
                "answer requires a single-token verbalizer".into(),
            ));
        }
        Ok(ids)
    }

    pub(super) fn omni(
        &self,
        config: &D1OmniConfig,
        request: &Request,
        q: &Question,
        prefix: usize,
        id: &str,
        usage: &mut Usage,
    ) -> Result<Row> {
        let context = request.options.max_len.unwrap_or(config.max_length);
        let available = context
            .checked_sub(prefix)
            .ok_or_else(|| Error::InvalidRequest("media exceed max_len".into()))?;
        let limit = available.min(if !request.images.is_empty() {
            config.image_text_length
        } else if request.audio.is_some() {
            config.audio_text_length
        } else {
            config.max_length
        });
        if limit < 64 {
            return Err(Error::InvalidRequest(
                "media leave fewer than 64 text positions; send fewer images".into(),
            ));
        }
        let options = omni_options(q, !request.images.is_empty(), request.audio.is_some())?;
        let budget = (options.len().saturating_mul(24).saturating_add(32))
            .min(limit / 2)
            .max(96);
        let per = (budget.saturating_sub(3 * options.len()) / options.len()).max(2);
        let mut question = vec![self.special("<|reserved_8|>")?];
        question.extend(self.tokens(&sanitize(q.instructions()))?);
        let mut dropped = question.len().saturating_sub(budget.max(16));
        question.truncate(budget.max(16));
        let mut markers = Vec::new();
        for text in options {
            markers.push(question.len() + 1);
            let mut tokens = self.tokens(&sanitize(&format!(" {text}")))?;
            dropped += tokens.len().saturating_sub(per);
            tokens.truncate(per);
            question.extend([self.special("<|reserved_9|>")?, self.special("<|mask|>")?]);
            question.extend(tokens);
            question.push(self.special("<|reserved_10|>")?);
        }
        question.push(self.special("<|reserved_11|>")?);
        self.truncated(request, id, dropped, true, usage)?;
        let state = if request.state.is_null() {
            if request.audio.is_some() {
                "{}".into()
            } else {
                String::new()
            }
        } else {
            render(&request.state)?
        };
        let mut state_ids = self.tokens(&sanitize(&state))?;
        usage.state_tokens = state_ids.len();
        let room = limit.saturating_sub(question.len() + 2);
        self.truncated(
            request,
            id,
            state_ids.len().saturating_sub(room),
            false,
            usage,
        )?;
        state_ids.truncate(room);
        let offset = 2 + state_ids.len();
        for marker in &mut markers {
            *marker += offset;
        }
        if markers.last().is_none_or(|&marker| marker >= limit) {
            return Err(Error::InvalidRequest(
                "options do not fit in the d1 context".into(),
            ));
        }
        let mut ids = vec![self.special(&self.bos)?, self.special("<|reserved_7|>")?];
        ids.extend(state_ids);
        ids.extend(question);
        self.truncated(request, id, ids.len().saturating_sub(limit), true, usage)?;
        ids.truncate(limit);
        usage.input_tokens += prefix + ids.len();
        Ok(Row {
            ids,
            markers,
            groups: Vec::new(),
            qtype: qtype(q),
        })
    }

    fn truncated(
        &self,
        request: &Request,
        id: &str,
        count: usize,
        head: bool,
        usage: &mut Usage,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        if request.options.truncation == Truncation::Error {
            return Err(Error::InvalidRequest(format!(
                "question {id:?} loses {count} {} tokens; select truncation=truncate explicitly",
                if head { "instruction/option" } else { "state" }
            )));
        }
        usage.truncated = true;
        if head {
            if !usage.truncated_head_questions.iter().any(|key| key == id) {
                usage.truncated_head_questions.push(id.into());
            }
        } else {
            usage.state_tokens_dropped = usage.state_tokens_dropped.max(count);
            if !usage.truncated_questions.iter().any(|key| key == id) {
                usage.truncated_questions.push(id.into());
            }
        }
        Ok(())
    }
}

fn safe_text(text: &str) -> String {
    sanitize(text).replace("<image>", "<¦image¦>")
}
fn qtype(q: &Question) -> usize {
    match q {
        Question::Choice { .. } => 0,
        Question::Score { .. } => 1,
        Question::Noul { .. } => 2,
    }
}

fn omni_options(q: &Question, image: bool, audio: bool) -> Result<Vec<String>> {
    Ok(match q {
        Question::Choice { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(i, (name, value))| {
                let empty = value.is_null() || value.as_str() == Some("");
                if audio {
                    Ok(format!(
                        "option_{i:03}: {}",
                        if empty { name.clone() } else { render(value)? }
                    ))
                } else if empty {
                    Ok(name.clone())
                } else {
                    Ok(format!("{name}: {}", render(value)?))
                }
            })
            .collect::<Result<Vec<_>>>()?,
        Question::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(i, value)| Ok(format!("level {i}: {}", render(value)?)))
            .collect::<Result<Vec<_>>>()?,
        Question::Noul { criteria, .. } => {
            if audio {
                vec!["false: no".into(), "true: yes".into()]
            } else {
                let value = |key: &str, fallback: &str| -> Result<String> {
                    match criteria.get(key) {
                        Some(value) if !value.is_null() && value.as_str() != Some("") => {
                            render(value)
                        }
                        _ => Ok(fallback.into()),
                    }
                };
                let defaults = image && criteria.is_empty();
                vec![
                    format!(
                        "false: {}",
                        value(
                            "false",
                            if defaults {
                                "no"
                            } else {
                                "no, the statement does not hold"
                            }
                        )?
                    ),
                    format!(
                        "true: {}",
                        value(
                            "true",
                            if defaults {
                                "yes"
                            } else {
                                "yes, the statement holds"
                            }
                        )?
                    ),
                ]
            }
        }
    })
}
