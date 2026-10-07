//! Load and evaluate Wald's calibrated one-pass decisions, e.g. effort=none.
use burn::tensor::backend::Backend;
use camino::Utf8Path;
use indexmap::IndexMap;

use super::configuration_wald::{Temperature, WaldConfig};
use super::processing_wald::{option_text, prompt, render};
use crate::hub::{self, ModelSource};
use crate::models::qwen3_5::readout::{Readout, answer, argmax, choice_confidence, softmax};
use crate::utils::{read_checkpoint_json, sanitize};
use crate::{DecisionModel, Error, Metadata, Question, Request, Response, Result, Usage};

/// Reuse the loaded text readout and calibration, e.g. successive Wald requests.
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
