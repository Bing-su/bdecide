//! Execute Von's published encoder/scorer and posterior calibration without token generation.
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::{Bool, Device as BurnDevice, Int, Tensor, TensorData, activation};
use camino::Utf8Path;
use indexmap::IndexMap;
use tokenizers::Tokenizer;

use super::configuration_von::VonConfig;
use super::processing_von::{self, SpecialTokens};
use crate::hub::{self, ModelSource};
use crate::models::modernbert::{ModernBertConfig, ModernBertModel};
use crate::models::qwen3_5::readout::{answer, choice_confidence, softmax};
use crate::models::weights;
use crate::utils::{load_tokenizer, read_checkpoint_json};
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

#[derive(Module, Debug)]
struct OptionMarkerModel {
    encoder: ModernBertModel,
    scorer: OptionMarkerScorer,
}
#[derive(Module, Debug)]
struct OptionMarkerScorer {
    input_norm: LayerNorm,
    dense: Linear,
    norm: LayerNorm,
    out_proj: Linear,
}
impl OptionMarkerScorer {
    fn init(hidden: usize, device: &BurnDevice) -> Self {
        Self {
            input_norm: LayerNormConfig::new(hidden).with_epsilon(1e-5).init(device),
            dense: LinearConfig::new(hidden, hidden / 2).init(device),
            norm: LayerNormConfig::new(hidden / 2)
                .with_epsilon(1e-5)
                .init(device),
            out_proj: LinearConfig::new(hidden / 2, 1).init(device),
        }
    }

    fn forward(&self, hidden: Tensor<3>) -> Tensor<3> {
        self.out_proj.forward(self.norm.forward(activation::gelu(
            self.dense.forward(self.input_norm.forward(hidden)),
        )))
    }
}
pub struct VonModel {
    model: OptionMarkerModel,
    tokenizer: Tokenizer,
    special: SpecialTokens,
    config: VonConfig,
    encoder_config: ModernBertConfig,
    metadata: Metadata,
    device: BurnDevice,
}
impl VonModel {
    /// Load the original state_dict, e.g. `VonModel::from_pretrained(&source, &device)?`.
    pub fn from_pretrained(source: &ModelSource, device: &BurnDevice) -> Result<Self> {
        let artifacts = hub::resolve_von(source)?;
        Self::load(&artifacts.root, device, artifacts.metadata)
    }

    pub(crate) fn load(
        root: &Utf8Path,
        device: &BurnDevice,
        mut metadata: Metadata,
    ) -> Result<Self> {
        let config: VonConfig = read_checkpoint_json(&root.join("marker_calibration.json"))?;
        config.validate()?;
        let encoder_config: ModernBertConfig = read_checkpoint_json(&root.join("config.json"))?;
        encoder_config.validate()?;
        let special: SpecialTokens = read_checkpoint_json(&root.join("tokenizer_config.json"))?;
        if special.mask_token.is_empty()
            || special.sep_token.is_empty()
            || special.mask_token == special.sep_token
        {
            return Err(Error::InvalidCheckpoint(
                "invalid Von special tokens".into(),
            ));
        }
        let tokenizer = load_tokenizer(&root.join("tokenizer.json"))?;
        let sep = tokenizer
            .token_to_id(&special.sep_token)
            .ok_or_else(|| Error::InvalidCheckpoint("Von separator token is absent".into()))?;
        let empty = tokenizer
            .encode("", true)
            .map_err(|error| Error::Tokenizer(error.to_string()))?;
        if empty.get_ids().len() < 2 || empty.get_ids().last() != Some(&sep) {
            return Err(Error::InvalidCheckpoint(
                "Von tokenizer must add a leading token and trailing separator".into(),
            ));
        }
        let mut model = OptionMarkerModel {
            encoder: ModernBertModel::init(&encoder_config, device)?,
            scorer: OptionMarkerScorer::init(encoder_config.hidden_size, device),
        };
        // The .pt contains the complete fine-tuned encoder too; do not substitute model.safetensors.
        weights::load(
            &mut model,
            root,
            &["option_marker.pt".into()],
            &[],
            weights::identity_name,
        )?;
        metadata.architecture = "von".into();
        if metadata.device.is_empty() {
            metadata.device = format!("{device:?}");
        }
        Ok(Self {
            model,
            tokenizer,
            special,
            config,
            encoder_config,
            metadata,
            device: device.clone(),
        })
    }

    fn read(
        &self,
        state: &str,
        question: &str,
        options: &[String],
        request: &Request,
        id: &str,
        usage: &mut Usage,
    ) -> Result<(Vec<f64>, usize)> {
        let mut encoded = processing_von::encode(
            &self.tokenizer,
            &self.special,
            self.encoder_config.vocab_size,
            state,
            question,
            options,
            self.config.digit_split,
        )?;
        let max_len = request
            .options
            .max_len
            .unwrap_or(self.encoder_config.max_position_embeddings);
        let dropped = encoded.ids.len().saturating_sub(max_len);
        let state_tokens = if dropped == 0 {
            processing_von::count_state(&self.tokenizer, state, self.encoder_config.vocab_size)?
        } else {
            // Calibration must see the state retained by this row, e.g. short budgets.
            encoded.state_positions.len().saturating_sub(dropped)
        };
        if dropped > 0 {
            if request.options.truncation == Truncation::Error {
                return Err(Error::InvalidRequest(format!(
                    "question {id:?} exceeds max_len={max_len}; select truncation=truncate explicitly"
                )));
            }
            if dropped > encoded.state_positions.len() {
                return Err(Error::InvalidRequest(format!(
                    "question {id:?} exceeds the token budget even without state"
                )));
            }
            let remove: std::collections::BTreeSet<_> = encoded
                .state_positions
                .iter()
                .rev()
                .take(dropped)
                .copied()
                .collect();
            encoded.markers = encoded
                .markers
                .iter()
                .map(|marker| marker - remove.range(..*marker).count())
                .collect();
            encoded.ids = encoded
                .ids
                .into_iter()
                .enumerate()
                .filter_map(|(i, token)| (!remove.contains(&i)).then_some(token))
                .collect();
            usage.truncated = true;
            usage.state_tokens_dropped = usage.state_tokens_dropped.max(dropped);
            if !usage.truncated_questions.iter().any(|value| value == id) {
                usage.truncated_questions.push(id.into());
            }
        }
        let length = encoded.ids.len();
        usage.input_tokens += length;
        let ids =
            Tensor::<2, Int>::from_data(TensorData::new(encoded.ids, [1, length]), &self.device);
        let hidden = if self.config.independent_options {
            let prefix = *encoded
                .markers
                .first()
                .ok_or_else(|| Error::Inference("missing Von marker".into()))?;
            let mut positions: Vec<_> = (0..length).collect();
            let mut groups = vec![None; length];
            for (i, start) in encoded.markers.iter().copied().enumerate() {
                let end = encoded.markers.get(i + 1).copied().unwrap_or(length - 1);
                for position in start..end {
                    if let (Some(slot), Some(group)) =
                        (positions.get_mut(position), groups.get_mut(position))
                    {
                        *slot = prefix + position - start;
                        *group = Some(i);
                    }
                }
            }
            // The trailing SEP belongs to the premise, as in upstream; options see only
            // premise plus their own span, with sliding windows measured on restarted IDs.
            let mask: Vec<_> = groups
                .iter()
                .flat_map(|query| groups.iter().map(move |key| key.is_some() && query != key))
                .collect();
            self.model.encoder.forward_with_positions(
                ids,
                Tensor::<4, Bool>::from_data(
                    TensorData::new(mask, [1, 1, length, length]),
                    &self.device,
                ),
                &positions,
            )
        } else {
            self.model.encoder.forward(
                ids,
                Tensor::<4, Bool>::from_data(
                    TensorData::new(vec![false; length], [1, 1, 1, length]),
                    &self.device,
                ),
            )
        };
        let indices: Vec<i32> = encoded
            .markers
            .iter()
            .map(|index| {
                i32::try_from(*index).map_err(|error| {
                    Error::InvalidRequest(format!(
                        "Von marker exceeds backend index range: {error}"
                    ))
                })
            })
            .collect::<Result<_>>()?;
        let indices =
            Tensor::<1, Int>::from_data(TensorData::new(indices, [options.len()]), &self.device);
        let logits = self
            .model
            .scorer
            .forward(hidden.select(1, indices))
            .try_into_vec_as::<f32>()
            .map_err(|error| Error::Inference(error.to_string()))?;
        Ok((logits.into_iter().map(f64::from).collect(), state_tokens))
    }
}
impl DecisionModel for VonModel {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn system_one(&self, request: &Request) -> Result<Response> {
        request.validate()?;
        if request.questions.is_empty()
            || request.options.head_max_len.is_some()
            || request
                .options
                .max_len
                .is_some_and(|max| max > self.encoder_config.max_position_embeddings)
        {
            return Err(Error::InvalidRequest(
                "Von requires questions, a valid context budget and no head_max_len".into(),
            ));
        }
        let state = processing_von::state(&request.state)?;
        let mut usage = Usage {
            state_tokens: processing_von::count_state(
                &self.tokenizer,
                &state,
                self.encoder_config.vocab_size,
            )?,
            ..Default::default()
        };
        let mut answers = IndexMap::new();
        for (id, question) in &request.questions {
            if let Question::Noul { labels, .. } = question
                && (labels.r#true != "true" || labels.r#false != "false")
            {
                return Err(Error::InvalidRequest(
                    "custom noul labels are unsupported; use criteria descriptions".into(),
                ));
            }
            let descriptions = processing_von::descriptions(question)?;
            let (mut logits, state_tokens) = self.read(
                &state,
                question.instructions(),
                &descriptions,
                request,
                id,
                &mut usage,
            )?;
            if let Question::Noul { criteria, .. } = question {
                let explicit = criteria.values().any(|value| {
                    !value.is_null() && value.as_str().is_none_or(|text| !text.is_empty())
                });
                if !explicit {
                    let null = self
                        .read(
                            "",
                            question.instructions(),
                            &descriptions,
                            request,
                            id,
                            &mut usage,
                        )?
                        .0;
                    let bias = null
                        .first()
                        .zip(null.get(1))
                        .map(|(yes, no)| yes - no)
                        .ok_or_else(|| Error::Inference("missing Von prior logits".into()))?;
                    let correction = self
                        .config
                        .noul_zero_shot_prior
                        .as_ref()
                        .map_or(0.7 * bias, |prior| prior.a * bias + prior.b);
                    if let Some(logit) = logits.first_mut() {
                        *logit -= correction;
                    }
                }
            }
            let temperature = self
                .config
                .effective_temperature(&logits, state_tokens)?
                .max(1e-4);
            let probabilities = softmax(
                &logits
                    .into_iter()
                    .map(|value| value / temperature)
                    .collect::<Vec<_>>(),
            )?;
            let confidence = choice_confidence(&probabilities);
            let legend = if matches!(question, Question::Score { .. }) {
                descriptions
            } else {
                Vec::new()
            };
            // Report the calibrated posterior (VON_NOUL_DECISION=raw), e.g. 0.55 stays
            // 0.55 instead of the server's optional forced-commit band remapping.
            answers.insert(
                id.clone(),
                answer(question, probabilities, confidence, legend, 0)?,
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
