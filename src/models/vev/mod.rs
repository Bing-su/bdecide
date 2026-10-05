//! Vev's trained text decision protocol, e.g. CountingSheep/vev-4b and vev-9b.
use crate::{
    DecisionModel, Error, Metadata, Question, Request, Response, Result, Usage,
    hub::{self, ModelSource},
    models::qwen3_5::readout::{Readout, answer, choice_confidence, sanitize},
    utils::read_checkpoint_json,
};
use burn::tensor::backend::Backend;
use camino::Utf8Path;
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::Value;
use std::ops::Range;

const SYSTEM: &str = "You are a careful judge. Read the state, then answer the question about it. Reply with only the answer token, nothing else.";

#[derive(Deserialize)]
pub(crate) struct VevConfig {
    vev_version: String,
    base: String,
}
impl VevConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.vev_version != "0.1.0" || self.base != "." {
            return Err(Error::UnsupportedModel(
                "Vev requires a merged 0.1.0 checkpoint".into(),
            ));
        }
        Ok(())
    }
}

pub struct VevModel<B: Backend> {
    readout: Readout<B>,
    labels: Vec<String>,
    label_ids: Vec<Vec<u32>>,
    metadata: Metadata,
}

impl<B: Backend> VevModel<B> {
    /// Load a merged release, e.g. `VevModel::<Flex>::from_pretrained(&source, &device)?`.
    pub fn from_pretrained(source: &ModelSource, device: &B::Device) -> Result<Self> {
        let artifacts = hub::resolve_vev(source)?;
        Self::load(&artifacts.root, device, artifacts.metadata)
    }

    pub(crate) fn load(
        root: &Utf8Path,
        device: &B::Device,
        mut metadata: Metadata,
    ) -> Result<Self> {
        let config: VevConfig = read_checkpoint_json(&root.join("vev.json"))?;
        config.validate()?;
        let mut readout = Readout::load(root, device)?;
        readout.max_positions = readout.max_positions.min(32768);
        let mut labels = Vec::new();
        let mut label_ids = Vec::new();
        let candidates =
            (b'A'..=b'Z')
                .map(|a| char::from(a).to_string())
                .chain((b'A'..=b'Z').flat_map(|a| {
                    (b'A'..=b'Z').map(move |b| format!("{}{}", char::from(a), char::from(b)))
                }));
        for label in candidates {
            let ids = readout.tokens(&label)?;
            if ids.len() == 1 && !label_ids.contains(&ids) {
                labels.push(label);
                label_ids.push(ids);
            }
            if labels.len() == 255 {
                break;
            }
        }
        if labels.len() != 255 {
            return Err(Error::InvalidCheckpoint(
                "Vev needs 255 distinct single-token choice labels".into(),
            ));
        }
        metadata.architecture = "vev".into();
        if metadata.device.is_empty() {
            metadata.device = B::name(device);
        }
        Ok(Self {
            readout,
            labels,
            label_ids,
            metadata,
        })
    }
}

impl<B: Backend> DecisionModel for VevModel<B> {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn predict(&self, request: &Request) -> Result<Response> {
        self.readout.validate(request)?;
        for question in request.questions.values() {
            validate_question(question)?;
        }
        let state = sanitize(&render_state(&request.state)?);
        let mut usage = Usage {
            state_tokens: self.readout.tokens(&state)?.len(),
            ..Default::default()
        };
        let mut answers = IndexMap::new();
        for (id, question) in &request.questions {
            let (prompt, range, legend) = prompt(&state, question, &self.labels)?;
            let groups: Vec<Vec<u32>> = match question {
                Question::Choice { criteria, .. } => self
                    .label_ids
                    .iter()
                    .take(criteria.len())
                    .cloned()
                    .collect(),
                Question::Score { criteria, .. } => (0..criteria.len())
                    .map(|i| single(&self.readout, &i.to_string()))
                    .collect::<Result<_>>()?,
                Question::Noul { .. } => ["Yes", "No"]
                    .iter()
                    .map(|label| single(&self.readout, label))
                    .collect::<Result<_>>()?,
            };
            let probabilities =
                self.readout
                    .probabilities(&prompt, &[range], &groups, request, id, &mut usage)?;
            let confidence = choice_confidence(&probabilities);
            answers.insert(
                id.clone(),
                answer(question, probabilities, confidence, legend, 0)?,
            );
        }
        Ok(Response {
            model: self.metadata.model_id.clone(),
            answers,
            usage,
            metadata: self.metadata.clone(),
        })
    }
}

fn single<B: Backend>(readout: &Readout<B>, label: &str) -> Result<Vec<u32>> {
    let ids = readout.tokens(label)?;
    if ids.len() != 1 {
        return Err(Error::InvalidCheckpoint(format!(
            "Vev answer {label:?} must be a single token"
        )));
    }
    Ok(ids)
}

fn validate_question(question: &Question) -> Result<()> {
    let invalid = match question {
        Question::Choice { criteria, .. } => criteria.len() > 255,
        Question::Score { criteria, .. } => criteria.len() > 10,
        Question::Noul { .. } => false,
    };
    if invalid {
        return Err(Error::InvalidRequest(
            "Vev supports at most 255 choice options and 10 score levels".into(),
        ));
    }
    Ok(())
}

fn prompt(
    state: &str,
    question: &Question,
    labels: &[String],
) -> Result<(String, Range<usize>, Vec<String>)> {
    let mut text = format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nState:\n");
    let start = text.len();
    text.push_str(state);
    let range = start..text.len();
    text.push_str(&format!(
        "\n\nQuestion: {}\n\n",
        sanitize(question.instructions().trim())
    ));
    let mut legend = Vec::new();
    match question {
        Question::Choice { criteria, .. } => {
            text.push_str("Options:\n");
            for ((key, value), label) in criteria.iter().zip(labels) {
                let description = desc(value)?;
                text.push_str(&format!("{label}. {}", sanitize(key)));
                if !description.is_empty() {
                    text.push_str(&format!(" - {}", sanitize(&description)));
                }
                text.push('\n');
            }
            text.push_str("\nAnswer with the letter of the single best option.");
        }
        Question::Score { criteria, .. } => {
            text.push_str(&format!(
                "Scale, from lowest (0) to highest ({}):\n",
                criteria.len() - 1
            ));
            for (i, value) in criteria.iter().enumerate() {
                let description = desc(value)?;
                text.push_str(&format!(
                    "{}\n",
                    format!("{i}. {}", sanitize(&description)).trim_end()
                ));
                legend.push(description);
            }
            text.push_str("\nAnswer with the level number only.");
        }
        Question::Noul { criteria, .. } => {
            text.push_str("Answer Yes or No.");
            for (key, label) in [("true", "Yes"), ("false", "No")] {
                if let Some(value) = criteria.get(key).filter(|value| !value.is_null()) {
                    text.push_str(&format!("\n{label} means: {}", sanitize(&desc(value)?)));
                }
            }
        }
    }
    text.push_str("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    Ok((text, range, legend))
}

fn render_state(value: &Value) -> Result<String> {
    if let Value::String(text) = value {
        return Ok(text.trim_end_matches('\n').into());
    }
    fn tree(value: &Value, depth: usize, out: &mut String) -> Result<()> {
        let pad = "  ".repeat(depth);
        let entries: Vec<_> = match value {
            Value::Object(fields) => fields
                .iter()
                .map(|(key, value)| (format!("{key}:"), value))
                .collect(),
            Value::Array(values) => values
                .iter()
                .enumerate()
                .map(|(i, value)| (format!("{}.", i + 1), value))
                .collect(),
            _ => {
                out.push_str(&format!("{pad}{}\n", desc(value)?));
                return Ok(());
            }
        };
        for (key, value) in entries {
            if key == "image:" && value.get("url").is_some_and(Value::is_string) {
                return Err(Error::InvalidRequest(
                    "Vev image inputs require a vision backend; this loader supports text states"
                        .into(),
                ));
            }
            out.push_str(&format!("{pad}{key}"));
            if value.as_object().is_some_and(|v| !v.is_empty())
                || value.as_array().is_some_and(|v| !v.is_empty())
            {
                out.push('\n');
                tree(value, depth + 1, out)?;
            } else {
                let text = if value.is_string() {
                    desc(value)?
                } else {
                    json_text(value)?
                };
                if text.contains('\n') {
                    out.push('\n');
                    for line in text.split('\n') {
                        out.push_str(&format!("{}{}\n", "  ".repeat(depth + 1), line));
                    }
                } else {
                    out.push_str(&format!(" {text}\n"));
                }
            }
        }
        Ok(())
    }
    let mut out = String::new();
    tree(value, 0, &mut out)?;
    Ok(out.trim_end_matches('\n').into())
}

fn desc(value: &Value) -> Result<String> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        _ => json_text(value),
    }
}

fn json_text(value: &Value) -> Result<String> {
    // Match Python's ensure_ascii=False JSON spacing, e.g. {"z": [true, null]}.
    match value {
        Value::Object(fields) => Ok(format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(key, value)| Ok(format!(
                    "{}: {}",
                    serde_json::to_string(key)?,
                    json_text(value)?
                )))
                .collect::<Result<Vec<_>>>()?
                .join(", ")
        )),
        Value::Array(values) => Ok(format!(
            "[{}]",
            values
                .iter()
                .map(json_text)
                .collect::<Result<Vec<_>>>()?
                .join(", ")
        )),
        Value::Number(number) => Ok(crate::utils::number_text(number)),
        _ => Ok(serde_json::to_string(value)?),
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;

    #[test]
    fn prompts_match_pinned_vev_renderer() {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-vev-4b");
        let model = VevModel::<burn::backend::Flex>::load(
            &root,
            &Default::default(),
            Metadata::new("fixture", "vev", "cpu"),
        )
        .unwrap();
        let reference: Value = read_checkpoint_json(&root.join("reference.json")).unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let request: Request = serde_json::from_value(case["request"].clone()).unwrap();
            let state = sanitize(&render_state(&request.state).unwrap());
            for (question, row) in request
                .questions
                .values()
                .zip(case["rows"].as_array().unwrap())
            {
                assert_eq!(
                    prompt(&state, question, &model.labels).unwrap().0,
                    row["prompt"]
                );
            }
        }
    }
}
