//! Mirror ModernBERT's module hierarchy while constructing tensors from configuration.

use super::configuration_modernbert::ModernBertConfig;
use crate::{Result, utils::attention::attend};
use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Int, Tensor, activation::gelu, backend::Backend},
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
    }
}

impl<B: Backend> ModernBertModel<B> {
    pub fn init(config: &ModernBertConfig, device: &B::Device) -> Result<Self> {
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
    pub fn forward(&self, ids: Tensor<B, 2, Int>, padding: Tensor<B, 4>) -> Tensor<B, 3> {
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
    fn forward(&self, hidden: Tensor<B, 3>, padding: Tensor<B, 4>) -> Tensor<B, 3> {
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
        // GEGLU activates the first half; swapping the halves changes the model.
        if let (Some(input), Some(gate)) = (halves.next(), halves.next()) {
            hidden = hidden + self.mlp.Wo.forward(gelu(input) * gate);
        }
        hidden
    }
}
