//! Run Omni's bidirectional encoder and calibrated head, e.g. speech decisions.
use burn::module::Module;
use burn::nn::{
    Embedding,
    EmbeddingConfig,
    Gelu,
    LayerNorm,
    LayerNormConfig,
    Linear,
    LinearConfig,
};
use burn::tensor::activation::relu;
use burn::tensor::{Device, Int, Tensor, TensorData};
use burn_std::s;
use camino::Utf8Path;
use indexmap::IndexMap;

use super::D1OmniConfig;
use super::audio::Audio;
use super::processing_d1::{Processor, Row};
use super::readout::{answer, values};
use crate::hub::{self, ModelSource};
use crate::models::lfm2::Lfm2Model;
use crate::models::lfm2_vl::vision::{self, Vision};
use crate::models::lfm2_vl::weights::weight_name;
use crate::models::weights;
use crate::utils::attention::attend;
use crate::{DecisionModel, Metadata, Question, Request, Response, Result, Usage};

/// D1Omni-600M's encoder and decision API, e.g. speech or image input.
pub struct D1OmniModel {
    pub(super) network: OmniModel,
    processor: Processor,
    config: D1OmniConfig,
    device: Device,
    metadata: Metadata,
}

#[derive(Module, Debug)]
pub(super) struct OmniModel {
    pub(super) encoder: Lfm2Model,
    head: DecisionHead,
    vision: Vision,
    audio: Audio,
}

impl OmniModel {
    pub(super) fn new(config: &D1OmniConfig, device: &Device) -> Result<Self> {
        config.validate()?;
        let d = config.text_config.hidden_size;
        Ok(Self {
            encoder: Lfm2Model::new_encoder(&config.text_config, device)?,
            head: DecisionHead::new(d, config.head_layers, device),
            vision: Vision::new(
                &config.vision_config,
                config.projector_hidden_size,
                d,
                device,
            )?,
            audio: Audio::new(&config.audio_config, d, device),
        })
    }
}

impl D1OmniModel {
    /// Resolve Omni's artifacts, e.g. a d1_omni config and indexed shards.
    pub fn from_pretrained(source: &ModelSource, device: &Device) -> Result<Self> {
        let artifacts = hub::resolve_d1_omni(source)?;
        Self::load(&artifacts.root, device, artifacts.metadata)
    }

    pub(crate) fn load(root: &Utf8Path, device: &Device, mut metadata: Metadata) -> Result<Self> {
        let config = D1OmniConfig::from_pretrained(root)?;
        let processor = Processor::load_omni(root, &config)?;
        let mut network = OmniModel::new(&config, device)?;
        let files = weights::checkpoint_files(root, |_| true)?;
        weights::load(&mut network, root, &files, &[], |name| {
            // The training batch counter is unused; affine and running statistics remain required.
            if name.starts_with("audio.encoder.layers.")
                && name.ends_with(".conv.batch_norm.num_batches_tracked")
            {
                Ok(None)
            } else {
                weight_name(name)
            }
        })?;
        metadata.architecture = "d1_omni".into();
        if metadata.device.is_empty() {
            metadata.device = format!("{device:?}");
        }
        Ok(Self {
            network,
            processor,
            config,
            device: device.clone(),
            metadata,
        })
    }
}

impl DecisionModel for D1OmniModel {
    fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    fn system_one(&self, request: &Request) -> Result<Response> {
        self.processor.validate(self.config.max_length, request)?;
        let images = vision::preprocess(&request.images, false)?;
        let media = if !images.is_empty() {
            Some(self.network.vision.forward(&images, &self.device))
        } else if let Some(input) = &request.audio {
            Some(self.network.audio.forward(&input.decode()?, &self.device))
        } else {
            None
        };
        let prefix = media.as_ref().map_or(0, |x| x.dims()[1]);
        let mut usage = Usage::default();
        let mut answers = IndexMap::new();
        for (id, q) in &request.questions {
            let row = self
                .processor
                .omni(&self.config, request, q, prefix, id, &mut usage)?;
            let h = self.network.encoder.embed(&row.ids, &self.device);
            let h = if let Some(media) = &media {
                Tensor::cat(vec![media.clone(), h], 1)
            } else {
                h
            };
            let h = self
                .network
                .encoder
                .forward_prefix(h, prefix)
                .slice(s![.., prefix.., ..]);
            let mut logits = values(self.network.head.forward(h, &row, &self.device))?;
            if media.is_none() {
                let kind = match q {
                    Question::Choice { .. } => "choice",
                    Question::Score { .. } => "score",
                    Question::Noul { .. } => "noul",
                };
                let bucket = match logits.len() {
                    0..=2 => "2",
                    3..=5 => "3-5",
                    6..=10 => "6-10",
                    _ => "11+",
                };
                let temp = self
                    .config
                    .temperatures
                    .get(&format!("{kind}:{bucket}"))
                    .or_else(|| self.config.temperatures.get(kind))
                    .copied()
                    .unwrap_or(1.0);
                for logit in &mut logits {
                    *logit /= temp;
                }
            }
            answers.insert(id.clone(), answer(q, &logits, 1)?);
        }
        Ok(Response {
            model: self.metadata.model_id.clone(),
            answers,
            usage,
            metadata: self.metadata.clone(),
        })
    }
}

#[derive(Module, Debug)]
struct DecisionHead {
    type_emb: Embedding,
    head: HeadLayers,
    scorer: (LayerNorm, Linear, Gelu, Linear),
}

#[derive(Module, Debug)]
struct HeadLayers {
    layers: Vec<HeadLayer>,
}

#[derive(Module, Debug)]
struct HeadLayer {
    self_attn: HeadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

#[derive(Module, Debug)]
struct HeadAttention {
    in_proj: Linear,
    out_proj: Linear,
    heads: usize,
}

impl DecisionHead {
    fn new(d: usize, layers: usize, device: &Device) -> Self {
        let linear = |i, o| LinearConfig::new(i, o).init(device);
        let norm = || LayerNormConfig::new(d).with_epsilon(1e-5).init(device);
        Self {
            type_emb: EmbeddingConfig::new(3, d).init(device),
            head: HeadLayers {
                layers: (0..layers)
                    .map(|_| HeadLayer {
                        self_attn: HeadAttention {
                            in_proj: linear(d, 3 * d),
                            out_proj: linear(d, d),
                            heads: d / 64,
                        },
                        norm1: norm(),
                        norm2: norm(),
                        linear1: linear(d, 4 * d),
                        linear2: linear(4 * d, d),
                    })
                    .collect(),
            },
            scorer: (norm(), linear(d, d), Gelu::new(), linear(d, 1)),
        }
    }

    fn forward(&self, mut h: Tensor<3>, row: &Row, device: &Device) -> Tensor<3> {
        h = h + self.type_emb.forward(Tensor::<2, Int>::from_data(
            TensorData::new(vec![row.qtype as i64], [1, 1]),
            device,
        ));
        for layer in &self.head.layers {
            let x = layer.norm1.forward(h.clone());
            let qkv = layer.self_attn.in_proj.forward(x);
            let length = h.dims()[1];
            let padding = burn::tensor::Tensor::<4, burn::tensor::Bool>::from_data(
                TensorData::new(vec![false; length], [1, 1, 1, length]),
                device,
            );
            h = h + layer.self_attn.out_proj.forward(attend(
                qkv,
                layer.self_attn.heads,
                padding,
                None,
            ));
            let x = layer.norm2.forward(h.clone());
            h = h + layer.linear2.forward(relu(layer.linear1.forward(x)));
        }
        let markers = Tensor::<1, Int>::from_data(
            TensorData::new(
                row.markers.iter().map(|&i| i as i64).collect::<Vec<_>>(),
                [row.markers.len()],
            ),
            device,
        );
        let selected = h.select(1, markers);
        self.scorer.3.forward(
            self.scorer
                .2
                .forward(self.scorer.1.forward(self.scorer.0.forward(selected))),
        )
    }
}
