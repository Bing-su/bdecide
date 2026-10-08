//! Evaluate independent Decider rows without generating answer tokens.
use burn::tensor::Device as BurnDevice;
use camino::Utf8Path;
use indexmap::IndexMap;

use super::configuration_decider::DeciderConfig;
use super::processing_decider::{neutralize, option, state, strip_level};
use crate::hub::{self, ModelSource};
use crate::models::qwen3_5::readout::{Readout, answer, argmax, choice_confidence, softmax};
use crate::utils::{read_checkpoint_json, render, sanitize};
use crate::{
    DecisionModel,
    Error,
    Metadata,
    Question,
    Request,
    Response,
    Result,
    Truncation,
    Usage,
};

pub struct DeciderModel {
    readout: Readout,
    labels: Vec<(String, u32)>,
    config: DeciderConfig,
    metadata: Metadata,
}
impl DeciderModel {
    /// Load merged weights, e.g. `DeciderModel::from_pretrained(&source, &device)?`.
    pub fn from_pretrained(source: &ModelSource, device: &BurnDevice) -> Result<Self> {
        let artifacts = hub::resolve_decider(source)?;
        Self::load(&artifacts.root, device, artifacts.metadata)
    }

    pub(crate) fn load(
        root: &Utf8Path,
        device: &BurnDevice,
        mut metadata: Metadata,
    ) -> Result<Self> {
        let config: DeciderConfig = read_checkpoint_json(&root.join("decider_config.json"))?;
        config.validate()?;
        let readout = Readout::load(root, device)?;
        // Score always supports A..J, e.g. max_options=2 only limits Choice.
        let label_count = config.max_options.max(10);
        let names = (b'A'..=b'Z')
            .map(|letter| char::from(letter).to_string())
            .chain((b'A'..=b'Z').flat_map(|a| {
                (b'A'..=b'Z').map(move |b| format!("{}{}", char::from(a), char::from(b)))
            }));
        let mut labels = Vec::new();
        for name in names {
            if let [id] = readout.tokens(&name)?.as_slice() {
                if labels.iter().any(|(_, previous)| previous == id) {
                    return Err(Error::InvalidCheckpoint(
                        "duplicate Decider label tokens".into(),
                    ));
                }
                labels.push((name, *id));
            }
            if labels.len() == label_count {
                break;
            }
        }
        if labels.len() != label_count
            || labels
                .iter()
                .take(10)
                .enumerate()
                .any(|(i, (name, _))| name != &char::from(b'A' + i as u8).to_string())
        {
            return Err(Error::InvalidCheckpoint(
                "Decider requires distinct single-token A..J and wide labels".into(),
            ));
        }
        metadata.architecture = "decider".into();
        if metadata.device.is_empty() {
            metadata.device = format!("{device:?}");
        }
        Ok(Self {
            readout,
            labels,
            config,
            metadata,
        })
    }

    fn read(
        &self,
        context: &str,
        row: (&str, &[String]),
        temperature: f64,
        request: &Request,
        id: &str,
        usage: &mut Usage,
    ) -> Result<Vec<f64>> {
        let (instructions, options) = row;
        let mut prefix = self.readout.tokens(&format!("Context:\n{context}"))?;
        let head = format!("\n\nQuestion: {}\nOptions:", sanitize(instructions));
        let tail = "\nAnswer: (";
        let options: Vec<_> = options
            .iter()
            .map(|text| sanitize(&neutralize(text.clone(), self.config.neutralize_none)))
            .collect();
        let mut suffix = if options.len() <= 10 {
            let mut piece = head;
            for (i, text) in options.iter().enumerate() {
                piece.push_str(&format!("\n({}) {text}", char::from(b'A' + i as u8)));
            }
            piece.push_str(tail);
            self.readout.tokens(&piece)?
        } else {
            let mut piece = self.readout.tokens(&head)?;
            let open = self.readout.tokens("\n(")?;
            for ((_, label), text) in self.labels.iter().zip(&options) {
                piece.extend_from_slice(&open);
                piece.push(*label);
                piece.extend(self.readout.tokens(&format!(") {text}"))?);
            }
            piece.extend(self.readout.tokens(tail)?);
            piece
        };
        let max_len = request
            .options
            .max_len
            .unwrap_or(self.readout.max_positions);
        // Preserve the context header and all answer slots; only state may be lost.
        let header = self.readout.tokens("Context:\n")?.len();
        let keep = self
            .config
            .max_state_tokens
            .min(max_len.saturating_sub(suffix.len()));
        if keep < header {
            return Err(Error::InvalidRequest(format!(
                "question {id:?} exceeds the budget even without state"
            )));
        }
        let dropped = prefix.len().saturating_sub(keep);
        if dropped > 0 {
            if request.options.truncation == Truncation::Error {
                return Err(Error::InvalidRequest(format!(
                    "question {id:?} exceeds the state or context budget; select truncation=truncate explicitly"
                )));
            }
            prefix.truncate(keep);
            usage.truncated = true;
            usage.state_tokens_dropped = usage.state_tokens_dropped.max(dropped);
            if !usage.truncated_questions.iter().any(|value| value == id) {
                usage.truncated_questions.push(id.into());
            }
        }
        prefix.append(&mut suffix);
        usage.input_tokens += prefix.len();
        let ids: Vec<_> = self
            .labels
            .iter()
            .take(options.len())
            .map(|(_, id)| *id)
            .collect();
        softmax(
            &self
                .readout
                .logits(&prefix, &ids)?
                .into_iter()
                .map(|logit| logit / temperature)
                .collect::<Vec<_>>(),
        )
    }
}
impl DecisionModel for DeciderModel {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn system_one(&self, request: &Request) -> Result<Response> {
        self.readout.validate(request)?;
        for question in request.questions.values() {
            let valid = match question {
                Question::Choice { criteria, .. } => {
                    (2..=self.config.max_options).contains(&criteria.len())
                }
                Question::Score { criteria, .. } => (2..=10).contains(&criteria.len()),
                Question::Noul { .. } => true,
            };
            if !valid {
                return Err(Error::InvalidRequest(
                    "Decider choice requires 2..255 options; score requires 2..10 levels".into(),
                ));
            }
        }
        let context = sanitize(&state(&request.state)?);
        let mut usage = Usage {
            state_tokens: self.readout.tokens(&context)?.len(),
            ..Default::default()
        };
        let mut answers = IndexMap::new();
        for (id, question) in &request.questions {
            let legend = if let Question::Score { criteria, .. } = question {
                criteria.iter().map(render).collect::<Result<Vec<_>>>()?
            } else {
                Vec::new()
            };
            let temperature = self.config.temperature(question);
            let probabilities =
                if self.config.isolated_levels && matches!(question, Question::Score { .. }) {
                    let mut fit = Vec::new();
                    for level in &legend {
                        let instructions = format!(
                            "{}\nProposed answer: {}\nDoes the proposed answer fit?",
                            question.instructions(),
                            strip_level(level)?
                        );
                        let row = self.read(
                            &context,
                            (&instructions, &["no".into(), "yes".into()]),
                            temperature,
                            request,
                            id,
                            &mut usage,
                        )?;
                        fit.push(
                            *row.get(1)
                                .ok_or_else(|| Error::Inference("missing level fit".into()))?,
                        );
                    }
                    let sum = fit.iter().sum::<f64>();
                    let sum = if sum > 0.0 { sum } else { 1e-9 };
                    fit.into_iter().map(|value| value / sum).collect()
                } else {
                    let options = match question {
                        Question::Choice { criteria, .. } => criteria
                            .iter()
                            .map(|(name, desc)| option(name, Some(desc)))
                            .collect::<Result<Vec<_>>>()?,
                        Question::Score { .. } => legend
                            .iter()
                            .enumerate()
                            .map(|(i, text)| format!("{i}: {text}"))
                            .collect(),
                        Question::Noul { criteria, .. } => vec![
                            option("no", criteria.get("false"))?,
                            option("yes", criteria.get("true"))?,
                        ],
                    };
                    self.read(
                        &context,
                        (question.instructions(), &options),
                        temperature,
                        request,
                        id,
                        &mut usage,
                    )?
                };
            let confidence = match question {
                Question::Choice { .. } => choice_confidence(&probabilities),
                Question::Score { .. } => {
                    let best = argmax(&probabilities);
                    let n = probabilities.len() as f64;
                    let spread: f64 = probabilities
                        .iter()
                        .enumerate()
                        .map(|(i, p)| *p * i.abs_diff(best) as f64)
                        .sum();
                    let uniform: f64 = (0..probabilities.len())
                        .map(|i| (i as f64 - (n - 1.0) / 2.0).abs())
                        .sum::<f64>()
                        / n;
                    (1.0 - spread / uniform).clamp(0.0, 1.0)
                }
                Question::Noul { .. } => probabilities.iter().copied().fold(0.0, f64::max),
            };
            answers.insert(
                id.clone(),
                answer(question, probabilities, confidence, legend, 1)?,
            );
        }
        Ok(Response::builder()
            .model(self.metadata.model_id.clone())
            .answers(answers)
            .usage(usage)
            .metadata(self.metadata.clone())
            .build())
    }
}
