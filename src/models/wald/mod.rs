//! Wald's calibrated one-pass readout, e.g. org2ai/Wald-4B v1.2 (effort=none).
use crate::{
    DecisionModel, Error, Metadata, Question, Request, Response, Result, Usage,
    hub::{self, ModelSource},
    models::qwen3_5::readout::{Readout, answer, argmax, choice_confidence, sanitize, softmax},
    utils::read_checkpoint_json,
};
use burn::tensor::backend::Backend;
use camino::Utf8Path;
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::Value;
use std::ops::Range;

const REPEAT: &str = "\n\nRead the same context again before answering. This is a repeated copy, not additional events or independent evidence:\n";

#[derive(Deserialize)]
pub(crate) struct WaldConfig {
    prompt_format: String,
    max_model_len: usize,
    temperature: String,
}
impl WaldConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if !matches!(self.prompt_format.as_str(), "plain" | "repeat_state_plain") {
            return Err(Error::UnsupportedModel(format!(
                "Wald prompt format: {}",
                self.prompt_format
            )));
        }
        if self.max_model_len < 4 || self.temperature != "temperature.json" {
            return Err(Error::InvalidCheckpoint(
                "invalid Wald context budget or calibration filename".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct Bucket {
    temperature: f64,
}
#[derive(Deserialize)]
struct Temperature {
    #[serde(default = "one")]
    single: f64,
    #[serde(default)]
    buckets: IndexMap<String, Bucket>,
}
fn one() -> f64 {
    1.0
}
impl Temperature {
    fn load(root: &Utf8Path) -> Result<Self> {
        let value: Value = read_checkpoint_json(&root.join("temperature.json"))?;
        let table: Self = serde_json::from_value(value.get("A").unwrap_or(&value).clone())
            .map_err(|error| Error::InvalidCheckpoint(format!("Wald temperature.json: {error}")))?;
        if std::iter::once(table.single)
            .chain(table.buckets.values().map(|bucket| bucket.temperature))
            .any(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(Error::InvalidCheckpoint(
                "Wald temperatures must be finite and positive".into(),
            ));
        }
        Ok(table)
    }
    fn get(&self, question: &Question, count: usize) -> f64 {
        let kind = match question {
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
            Question::Noul { .. } => "noul",
        };
        let bucket = match count {
            0..=2 => "2",
            3..=4 => "3-4",
            5..=8 => "5-8",
            _ => "9+",
        };
        self.buckets
            .get(&format!("{kind}|{bucket}"))
            .map_or(self.single, |bucket| bucket.temperature)
    }
}

pub struct WaldModel<B: Backend> {
    readout: Readout<B>,
    letters: Vec<Vec<u32>>,
    config: WaldConfig,
    temperature: Temperature,
    metadata: Metadata,
}

impl<B: Backend> WaldModel<B> {
    /// Load the one-pass protocol, e.g. `WaldModel::<Flex>::from_pretrained(&source, &device)?`.
    pub fn from_pretrained(source: &ModelSource, device: &B::Device) -> Result<Self> {
        let artifacts = hub::resolve_wald(source)?;
        Self::load(&artifacts.root, device, artifacts.metadata)
    }

    pub(crate) fn load(
        root: &Utf8Path,
        device: &B::Device,
        mut metadata: Metadata,
    ) -> Result<Self> {
        let config: WaldConfig = read_checkpoint_json(&root.join("serving.json"))?;
        config.validate()?;
        let temperature = Temperature::load(root)?;
        let mut readout = Readout::load(root, device)?;
        readout.max_positions = readout.max_positions.min(config.max_model_len);
        let mut letters = Vec::new();
        for letter in b'A'..=b'Z' {
            let mut variants = Vec::new();
            for text in [
                char::from(letter).to_string(),
                format!(" {}", char::from(letter)),
            ] {
                let ids = readout.tokens(&text)?;
                if let [id] = ids.as_slice()
                    && !variants.contains(id)
                {
                    variants.push(*id);
                }
            }
            if variants.is_empty()
                || letters
                    .iter()
                    .any(|previous: &Vec<u32>| previous.iter().any(|id| variants.contains(id)))
            {
                return Err(Error::InvalidCheckpoint(
                    "Wald requires distinct single-token A..Z labels".into(),
                ));
            }
            letters.push(variants);
        }
        metadata.architecture = "wald".into();
        if metadata.device.is_empty() {
            metadata.device = B::name(device);
        }
        Ok(Self {
            readout,
            letters,
            config,
            temperature,
            metadata,
        })
    }

    fn read(
        &self,
        state: &str,
        question: &Question,
        options: &[String],
        request: &Request,
        id: &str,
        usage: &mut Usage,
    ) -> Result<Vec<f64>> {
        let (prompt, ranges) = prompt(
            state,
            question,
            options,
            self.config.prompt_format == "repeat_state_plain",
        );
        self.readout.probabilities(
            &prompt,
            &ranges,
            &self
                .letters
                .iter()
                .take(options.len())
                .cloned()
                .collect::<Vec<_>>(),
            request,
            id,
            usage,
        )
    }

    fn wide_read(
        &self,
        state: &str,
        question: &Question,
        options: &[String],
        request: &Request,
        id: &str,
        usage: &mut Usage,
    ) -> Result<Vec<f64>> {
        if options.len() <= 26 {
            return self.read(state, question, options, request, id, usage);
        }
        // Match the released balanced knockout: compare each group's winner once,
        // then distribute its mass within that group, e.g. 27 options split 14/13.
        let count = options.len().div_ceil(26);
        let base = options.len() / count;
        let extra = options.len() % count;
        let mut groups = Vec::new();
        let mut winners = Vec::new();
        let mut start = 0;
        for group in 0..count {
            let end = start + base + usize::from(group < extra);
            let probabilities = self.read(
                state,
                question,
                options
                    .get(start..end)
                    .ok_or_else(|| Error::Inference("invalid Wald group".into()))?,
                request,
                id,
                usage,
            )?;
            winners.push(
                options
                    .get(start + argmax(&probabilities))
                    .ok_or_else(|| Error::Inference("missing Wald group winner".into()))?
                    .clone(),
            );
            groups.push(probabilities);
            start = end;
        }
        let top = self.read(state, question, &winners, request, id, usage)?;
        let probabilities: Vec<_> = groups
            .into_iter()
            .zip(top)
            .flat_map(|(group, mass)| group.into_iter().map(move |probability| probability * mass))
            .collect();
        let sum: f64 = probabilities.iter().sum();
        Ok(probabilities.into_iter().map(|value| value / sum).collect())
    }
}

impl<B: Backend> DecisionModel for WaldModel<B> {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    #[expect(
        clippy::float_cmp,
        reason = "temperature=1 exactly preserves the upstream uncalibrated probabilities"
    )]
    fn predict(&self, request: &Request) -> Result<Response> {
        self.readout.validate(request)?;
        for question in request.questions.values() {
            if matches!(question, Question::Choice { criteria, .. } if criteria.len() > 255)
                || matches!(question, Question::Score { criteria, .. } if criteria.len() > 255)
            {
                return Err(Error::InvalidRequest(
                    "Wald supports at most 255 options per question".into(),
                ));
            }
        }
        let state = sanitize(&render(&request.state, 0));
        let mut usage = Usage {
            state_tokens: self.readout.tokens(&state)?.len(),
            ..Default::default()
        };
        let mut answers = IndexMap::new();
        for (id, question) in &request.questions {
            let options = match question {
                Question::Choice { criteria, .. } => criteria
                    .iter()
                    .map(|(key, value)| option_text(key, Some(value)))
                    .collect(),
                Question::Score { criteria, .. } => {
                    criteria.iter().map(|value| render(value, 0)).collect()
                }
                Question::Noul { criteria, .. } => vec![
                    option_text("no", criteria.get("false")),
                    option_text("yes", criteria.get("true")),
                ],
            };
            let probabilities =
                self.wide_read(&state, question, &options, request, id, &mut usage)?;
            let temperature = self.temperature.get(question, probabilities.len());
            let probabilities = if temperature == 1.0 {
                probabilities
            } else {
                softmax(
                    &probabilities
                        .iter()
                        .map(|value| value.max(1e-12).ln() / temperature)
                        .collect::<Vec<_>>(),
                )?
            };
            let confidence = if matches!(question, Question::Choice { .. }) {
                choice_confidence(&probabilities)
            } else {
                probabilities.iter().copied().fold(0.0, f64::max)
            };
            let legend = if matches!(question, Question::Score { .. }) {
                options
            } else {
                Vec::new()
            };
            answers.insert(
                id.clone(),
                answer(question, probabilities, confidence, legend, 1)?,
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

fn option_text(key: &str, value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => key.into(),
        Some(Value::String(text)) if text.is_empty() => key.into(),
        Some(value) => format!("{key}: {}", render(value, 0)),
    }
}

fn prompt(
    state: &str,
    question: &Question,
    options: &[String],
    repeat: bool,
) -> (String, Vec<Range<usize>>) {
    let mut text = String::new();
    let mut ranges = Vec::new();
    if !state.trim().is_empty() {
        text.push_str("State:\n");
        let start = text.len();
        text.push_str(state);
        ranges.push(start..text.len());
        if repeat {
            text.push_str(REPEAT);
            let start = text.len();
            text.push_str(state);
            ranges.push(start..text.len());
        }
        text.push_str("\n\n");
    }
    text.push_str(&format!("Question: {}", sanitize(question.instructions())));
    let slug = |text: &str| {
        !text.is_empty()
            && text.len() <= 40
            && text
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
            && text.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"_.-".contains(&byte)
            })
    };
    let keys: Vec<_> = if options.iter().all(|text| slug(text))
        && options
            .iter()
            .enumerate()
            .all(|(i, text)| !options.iter().take(i).any(|previous| previous == text))
    {
        options.to_vec()
    } else {
        (b'a'..=b'z')
            .take(options.len())
            .map(|letter| char::from(letter).to_string())
            .collect()
    };
    let positional = keys.iter().all(|key| {
        key.len() == 1
            && key
                .bytes()
                .next()
                .is_some_and(|byte| byte.is_ascii_lowercase())
            || key.strip_prefix('o').is_some_and(|suffix| {
                !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
            })
    });
    for ((option, key), letter) in options.iter().zip(&keys).zip(b'A'..=b'Z') {
        let option = if matches!(question, Question::Choice { .. }) && positional {
            option.strip_prefix(&format!("{key}: ")).unwrap_or(option)
        } else {
            option
        };
        text.push_str(&format!("\n({}) {}", char::from(letter), sanitize(option)));
    }
    text.push_str("\nAnswer: (");
    (text, ranges)
}

fn render(value: &Value, depth: usize) -> String {
    // Preserve Wald's training renderer, e.g. arrays use '-' and booleans use Python's True/False.
    let pad = "  ".repeat(depth);
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        Value::Number(number) => crate::utils::number_text(number),
        Value::Array(values) => values
            .iter()
            .map(|value| format!("{pad}- {}", render(value, depth + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(fields) => fields
            .iter()
            .map(|(key, value)| {
                if value.is_array() || value.is_object() {
                    format!("{pad}{key}:\n{}", render(value, depth + 1))
                } else {
                    format!("{pad}{key}: {}", render(value, 0))
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_match_pinned_wald_renderer() {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-wald");
        let reference: Value = read_checkpoint_json(&root.join("reference.json")).unwrap();
        for case in reference["cases"].as_array().unwrap().iter().take(5) {
            let request: Request = serde_json::from_value(case["request"].clone()).unwrap();
            let state = sanitize(&render(&request.state, 0));
            for (question, row) in request
                .questions
                .values()
                .zip(case["rows"].as_array().unwrap())
            {
                let options: Vec<_> = match question {
                    Question::Choice { criteria, .. } => criteria
                        .iter()
                        .map(|(key, value)| option_text(key, Some(value)))
                        .collect(),
                    Question::Score { criteria, .. } => {
                        criteria.iter().map(|value| render(value, 0)).collect()
                    }
                    Question::Noul { criteria, .. } => vec![
                        option_text("no", criteria.get("false")),
                        option_text("yes", criteria.get("true")),
                    ],
                };
                assert_eq!(prompt(&state, question, &options, true).0, row["prompt"]);
            }
        }
    }
}
