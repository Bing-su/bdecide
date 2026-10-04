//! Mirror ModernBERT's module hierarchy while constructing tensors from configuration.

use super::configuration_modernbert::ModernBertConfig;
use crate::{
    Result,
    utils::{activation::HiddenActivation, attention::attend},
};
use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Bool, Int, Tensor, backend::Backend},
};
use projections::{ModernBertAttention, ModernBertMLP};

#[derive(Module, Debug)]
pub(crate) struct ModernBertModel<B: Backend> {
    embeddings: ModernBertEmbeddings<B>,
    layers: Vec<ModernBertEncoderLayer<B>>,
    final_norm: LayerNorm<B>,
}

#[derive(Module, Debug)]
struct ModernBertEmbeddings<B: Backend> {
    tok_embeddings: Embedding<B>,
    norm: LayerNorm<B>,
}

#[derive(Module, Debug)]
struct ModernBertEncoderLayer<B: Backend> {
    attn_norm: Option<LayerNorm<B>>,
    attn: ModernBertAttention<B>,
    mlp_norm: LayerNorm<B>,
    mlp: ModernBertMLP<B>,
    heads: usize,
    theta: f64,
    window: Option<usize>,
}

// Preserve checkpoint names in projections and generated Burn records, e.g. attn.Wqkv.weight.
#[expect(non_snake_case, reason = "Match ModernBERT checkpoint parameter paths")]
mod projections {
    use super::*;

    #[derive(Module, Debug)]
    pub(super) struct ModernBertAttention<B: Backend> {
        pub(super) Wqkv: Linear<B>,
        pub(super) Wo: Linear<B>,
    }

    #[derive(Module, Debug)]
    pub(super) struct ModernBertMLP<B: Backend> {
        pub(super) Wi: Linear<B>,
        pub(super) Wo: Linear<B>,
        // Config owns this choice; checkpoints contain no activation tensors, e.g. relu.
        #[module(skip)]
        pub(super) act: HiddenActivation,
    }
}

impl<B: Backend> ModernBertModel<B> {
    pub fn init(config: &ModernBertConfig, device: &B::Device) -> Result<Self> {
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
    pub fn forward(&self, ids: Tensor<B, 2, Int>, padding: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let mut hidden = self
            .embeddings
            .norm
            .forward(self.embeddings.tok_embeddings.forward(ids));
        for layer in &self.layers {
            hidden = layer.forward(hidden, padding.clone());
        }
        self.final_norm.forward(hidden)
    }
}

impl<B: Backend> ModernBertEncoderLayer<B> {
    // Keep a layer's residual steps together without changing checkpoint paths,
    // e.g. attn.Wqkv and mlp.Wi still belong to the same encoder layer.
    fn forward(&self, hidden: Tensor<B, 3>, padding: Tensor<B, 4, Bool>) -> Tensor<B, 3> {
        let normalized = match &self.attn_norm {
            Some(norm) => norm.forward(hidden.clone()),
            None => hidden.clone(),
        };
        let attention = attend(
            self.attn.Wqkv.forward(normalized),
            self.heads,
            padding,
            Some((self.theta, self.window)),
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
    use super::*;
    use crate::{
        models::laya::{LayaConfig, LayaDecisionModel, weights::load_laya},
        utils::{
            activation::tests::{assert_close, reference},
            read_checkpoint_json,
        },
    };
    use burn::tensor::TensorData;
    use camino::Utf8Path;

    fn matches_python<B: Backend>() {
        let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya");
        let config: LayaConfig = read_checkpoint_json(&root.join("rl_agent_config.json")).unwrap();
        let mut encoder: ModernBertConfig =
            read_checkpoint_json(&root.join("encoder/config.json")).unwrap();
        let reference = reference();
        let length = reference.input_ids.len();
        let device = B::Device::default();
        let ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(reference.input_ids, [1, length]),
            &device,
        );
        for case in reference.cases {
            encoder.hidden_activation.clone_from(&case.name);
            let mut model = LayaDecisionModel::<B>::init(&config, &encoder, &device).unwrap();
            // Loading existing weights must retain the config choice, e.g. hidden_activation="relu".
            load_laya(&mut model, &root.join("model.safetensors")).unwrap();
            let output = model.encoder.forward(
                ids.clone(),
                Tensor::from_data(
                    TensorData::new(vec![false; length], [1, 1, 1, length]),
                    &device,
                ),
            );
            let output = output
                .slice(burn_std::s![.., length - 1..length, ..])
                .into_data();
            assert_close(output, &case.modernbert, &case.name, 2e-5);
        }
        for name in ["prelu", "xielu", "unknown"] {
            encoder.hidden_activation = name.into();
            let error = ModernBertModel::<B>::init(&encoder, &device).unwrap_err();
            assert!(
                matches!(error, crate::Error::UnsupportedModel(_)),
                "{error}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn cpu_matches_python_activation_options() {
        matches_python::<burn::backend::Flex>();
    }

    #[test]
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a wgpu adapter"]
    fn wgpu_matches_python_activation_options() {
        matches_python::<burn::backend::Wgpu<f32, i32>>();
    }
}
