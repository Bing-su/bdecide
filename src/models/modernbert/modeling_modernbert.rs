//! Mirror ModernBERT's module hierarchy while constructing tensors from configuration.

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::{Bool, Device as BurnDevice, Int, Tensor};
use projections::{ModernBertAttention, ModernBertMLP};

use super::configuration_modernbert::ModernBertConfig;
use crate::Result;
use crate::utils::activation::HiddenActivation;
use crate::utils::attention::attend_with_positions;

#[derive(Module, Debug)]
pub(crate) struct ModernBertModel {
    embeddings: ModernBertEmbeddings,
    layers: Vec<ModernBertEncoderLayer>,
    final_norm: LayerNorm,
}

#[derive(Module, Debug)]
struct ModernBertEmbeddings {
    tok_embeddings: Embedding,
    norm: LayerNorm,
}

#[derive(Module, Debug)]
struct ModernBertEncoderLayer {
    attn_norm: Option<LayerNorm>,
    attn: ModernBertAttention,
    mlp_norm: LayerNorm,
    mlp: ModernBertMLP,
    heads: usize,
    theta: f64,
    window: Option<usize>,
}

// Preserve checkpoint names in projections and generated Burn records, e.g. attn.Wqkv.weight.
#[expect(non_snake_case, reason = "Match ModernBERT checkpoint parameter paths")]
mod projections {
    use super::*;

    #[derive(Module, Debug)]
    pub(super) struct ModernBertAttention {
        pub(super) Wqkv: Linear,
        pub(super) Wo: Linear,
    }

    #[derive(Module, Debug)]
    pub(super) struct ModernBertMLP {
        pub(super) Wi: Linear,
        pub(super) Wo: Linear,
        // Config owns this choice; checkpoints contain no activation tensors, e.g. relu.
        #[module(skip)]
        pub(super) act: HiddenActivation,
    }
}

impl ModernBertModel {
    pub fn init(config: &ModernBertConfig, device: &BurnDevice) -> Result<Self> {
        config.validate()?;
        let act = config.hidden_activation.parse::<HiddenActivation>()?;
        let hidden_size = config.hidden_size;
        let norm = || {
            LayerNormConfig::new(hidden_size)
                .with_bias(config.norm_bias)
                .with_epsilon(config.norm_eps)
                .init(device)
        };
        let linear = |input, output, bias| {
            LinearConfig::new(input, output)
                .with_bias(bias)
                .init(device)
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let (theta, window) = config.rope(i)?;
            layers.push(ModernBertEncoderLayer {
                attn_norm: if i == 0 { None } else { Some(norm()) },
                attn: ModernBertAttention {
                    Wqkv: linear(hidden_size, 3 * hidden_size, config.attention_bias),
                    Wo: linear(hidden_size, hidden_size, config.attention_bias),
                },
                mlp_norm: norm(),
                mlp: ModernBertMLP {
                    Wi: linear(hidden_size, 2 * config.intermediate_size, config.mlp_bias),
                    Wo: linear(config.intermediate_size, hidden_size, config.mlp_bias),
                    act,
                },
                heads: config.num_attention_heads,
                theta,
                window,
            });
        }
        Ok(Self {
            embeddings: ModernBertEmbeddings {
                tok_embeddings: EmbeddingConfig::new(config.vocab_size, hidden_size).init(device),
                norm: norm(),
            },
            layers,
            final_norm: norm(),
        })
    }

    pub fn forward(&self, ids: Tensor<2, Int>, padding: Tensor<4, Bool>) -> Tensor<3> {
        let mut hidden = self
            .embeddings
            .norm
            .forward(self.embeddings.tok_embeddings.forward(ids));
        for layer in &self.layers {
            hidden = layer.forward(hidden, padding.clone());
        }
        self.final_norm.forward(hidden)
    }

    // The same layer weights support isolated option spans, e.g. Von's masked prefix.
    pub(crate) fn forward_with_positions(
        &self,
        ids: Tensor<2, Int>,
        mask: Tensor<4, Bool>,
        positions: &[usize],
    ) -> Tensor<3> {
        let mut hidden = self
            .embeddings
            .norm
            .forward(self.embeddings.tok_embeddings.forward(ids));
        for layer in &self.layers {
            hidden = layer.forward_with_positions(hidden, mask.clone(), Some(positions));
        }
        self.final_norm.forward(hidden)
    }
}

impl ModernBertEncoderLayer {
    // Keep a layer's residual steps together without changing checkpoint paths,
    // e.g. attn.Wqkv and mlp.Wi still belong to the same encoder layer.
    fn forward(&self, hidden: Tensor<3>, padding: Tensor<4, Bool>) -> Tensor<3> {
        self.forward_with_positions(hidden, padding, None)
    }

    fn forward_with_positions(
        &self,
        hidden: Tensor<3>,
        padding: Tensor<4, Bool>,
        positions: Option<&[usize]>,
    ) -> Tensor<3> {
        let normalized = match &self.attn_norm {
            Some(norm) => norm.forward(hidden.clone()),
            None => hidden.clone(),
        };
        let attention = attend_with_positions(
            self.attn.Wqkv.forward(normalized),
            self.heads,
            padding,
            Some((self.theta, self.window)),
            positions,
        );
        let mut hidden = hidden + self.attn.Wo.forward(attention);
        let normalized = self.mlp_norm.forward(hidden.clone());
        let mut halves = self.mlp.Wi.forward(normalized).chunk(2, 2).into_iter();
        // Activate the first half before gating, e.g. GEGLU when the config selects gelu.
        if let (Some(input), Some(gate)) = (halves.next(), halves.next()) {
            hidden = hidden + self.mlp.Wo.forward(self.mlp.act.forward(input) * gate);
        }
        hidden
    }
}

#[cfg(test)]
mod activation_tests {
    use burn::tensor::TensorData;
    use burn_std::s;
    use camino::Utf8Path;

    use super::*;
    use crate::Error;
    use crate::models::laya::weights::load_laya;
    use crate::models::laya::{LayaConfig, LayaDecisionModel};
    use crate::utils::activation::tests::{assert_close, reference};
    use crate::utils::read_checkpoint_json;

    fn matches_python(device: BurnDevice) {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya");
        let config: LayaConfig = read_checkpoint_json(&root.join("rl_agent_config.json")).unwrap();
        let mut encoder: ModernBertConfig =
            read_checkpoint_json(&root.join("encoder/config.json")).unwrap();
        let reference = reference();
        let length = reference.input_ids.len();

        let ids =
            Tensor::<2, Int>::from_data(TensorData::new(reference.input_ids, [1, length]), &device);
        for case in reference.cases {
            encoder.hidden_activation.clone_from(&case.name);
            let mut model = LayaDecisionModel::init(&config, &encoder, &device).unwrap();
            // Loading existing weights must retain the config choice, e.g. hidden_activation="relu".
            load_laya(&mut model, &root.join("model.safetensors")).unwrap();
            let output = model.encoder.forward(
                ids.clone(),
                Tensor::from_data(
                    TensorData::new(vec![false; length], [1, 1, 1, length]),
                    &device,
                ),
            );
            let output = output.slice(s![.., length - 1..length, ..]).into_data();
            assert_close(output, &case.modernbert, &case.name, 2e-5);
        }
        for name in ["prelu", "xielu", "unknown"] {
            encoder.hidden_activation = name.into();
            let error = ModernBertModel::init(&encoder, &device).unwrap_err();
            assert!(matches!(error, Error::UnsupportedModel(_)), "{error}");
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn cpu_matches_python_activation_options() {
        matches_python(BurnDevice::flex());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a wgpu adapter"]
    fn wgpu_matches_python_activation_options() {
        matches_python(BurnDevice::wgpu(Default::default()));
    }
}
