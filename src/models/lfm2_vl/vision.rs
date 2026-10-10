//! Encode NaFlex patches in crop order, e.g. tiles followed by their thumbnail.
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::{Device, Tensor, TensorData};
use image::RgbImage;
use image::imageops::{self, FilterType};
use itertools::Itertools;

use super::Siglip2VisionConfig;
use crate::utils::activation::HiddenActivation;
use crate::utils::attention::attention;
use crate::{Error, ImageInput, Result};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[derive(Module, Debug)]
pub(crate) struct Siglip2VisionModel {
    vision_model: VisionTransformer,
}
#[derive(Module, Debug)]
struct VisionTransformer {
    embeddings: Embeddings,
    encoder: Encoder,
    post_layernorm: LayerNorm,
}
#[derive(Module, Debug)]
struct Embeddings {
    patch_embedding: Linear,
    position_embedding: Embedding,
}
#[derive(Module, Debug)]
struct Encoder {
    layers: Vec<VisionLayer>,
}
#[derive(Module, Debug)]
struct VisionLayer {
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    self_attn: VisionAttention,
    mlp: VisionMlp,
}
#[derive(Module, Debug)]
struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    heads: usize,
}
#[derive(Module, Debug)]
struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
    #[module(skip)]
    activation: HiddenActivation,
}
#[derive(Module, Debug)]
pub(crate) struct Lfm2VlMultiModalProjector {
    linear_1: Linear,
    linear_2: Linear,
    // None preserves Omni's fixed F.gelu; Some follows LFM2-VL's projector_hidden_act.
    #[module(skip)]
    activation: Option<HiddenActivation>,
}
#[derive(Module, Debug)]
pub(crate) struct Vision {
    pub(crate) tower: Siglip2VisionModel,
    pub(crate) projector: Lfm2VlMultiModalProjector,
}

pub(crate) struct Crop {
    pub pixels: Vec<f32>,
    pub height: usize,
    pub width: usize,
}
pub(crate) struct ImageCrops {
    pub crops: Vec<Crop>,
    pub rows: usize,
    pub cols: usize,
}

impl Siglip2VisionModel {
    pub(crate) fn new(c: &Siglip2VisionConfig, device: &Device) -> Result<Self> {
        let activation = c.hidden_act.parse::<HiddenActivation>()?;
        let d = c.hidden_size;
        let linear = |i, o| LinearConfig::new(i, o).init(device);
        let norm = || {
            LayerNormConfig::new(d)
                .with_epsilon(c.layer_norm_eps)
                .init(device)
        };
        Ok(Self {
            vision_model: VisionTransformer {
                embeddings: Embeddings {
                    patch_embedding: linear(3 * c.patch_size.pow(2), d),
                    position_embedding: EmbeddingConfig::new(c.num_patches, d).init(device),
                },
                encoder: Encoder {
                    layers: (0..c.num_hidden_layers)
                        .map(|_| VisionLayer {
                            layer_norm1: norm(),
                            layer_norm2: norm(),
                            self_attn: VisionAttention {
                                q_proj: linear(d, d),
                                k_proj: linear(d, d),
                                v_proj: linear(d, d),
                                out_proj: linear(d, d),
                                heads: c.num_attention_heads,
                            },
                            mlp: VisionMlp {
                                fc1: linear(d, c.intermediate_size),
                                fc2: linear(c.intermediate_size, d),
                                activation,
                            },
                        })
                        .collect(),
                },
                post_layernorm: norm(),
            },
        })
    }

    fn forward(&self, crop: &Crop, device: &Device) -> Tensor<3> {
        let v = &self.vision_model;
        let length = crop.height * crop.width;
        let pixels = Tensor::<3>::from_data(
            TensorData::new(crop.pixels.clone(), [1, length, 768]),
            device,
        );
        let position = v.embeddings.position_embedding.weight.val();
        let [patches, d] = position.dims();
        let side = patches.isqrt();
        let horizontal = resize_weights(side, crop.width);
        let vertical = resize_weights(side, crop.height);
        // Match SigLIP2's antialiased position resize without reading weights back to the CPU.
        let weights: Vec<_> = vertical
            .iter()
            .cartesian_product(&horizontal)
            .flat_map(|(ys, xs)| ys.iter().cartesian_product(xs).map(|(&y, &x)| y * x))
            .collect();
        let position = Tensor::<2>::from_data(TensorData::new(weights, [length, patches]), device)
            .matmul(position)
            .reshape([1, length, d]);
        let mut h = v.embeddings.patch_embedding.forward(pixels) + position;
        for layer in &v.encoder.layers {
            let x = layer.layer_norm1.forward(h.clone());
            h = h + layer.self_attn.forward(x);
            let x = layer.layer_norm2.forward(h.clone());
            h = h + layer
                .mlp
                .fc2
                .forward(layer.mlp.activation.forward(layer.mlp.fc1.forward(x)));
        }
        v.post_layernorm.forward(h)
    }
}

impl VisionAttention {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, l, d] = x.dims();
        let shape = [b, l, self.heads, d / self.heads];
        self.out_proj.forward(
            attention(
                self.q_proj
                    .forward(x.clone())
                    .reshape(shape)
                    .swap_dims(1, 2),
                self.k_proj
                    .forward(x.clone())
                    .reshape(shape)
                    .swap_dims(1, 2),
                self.v_proj.forward(x).reshape(shape).swap_dims(1, 2),
                None,
                Default::default(),
            )
            .swap_dims(1, 2)
            .reshape([b, l, d]),
        )
    }
}

impl Lfm2VlMultiModalProjector {
    pub(crate) fn new(
        vision: usize,
        hidden: usize,
        out: usize,
        activation: Option<HiddenActivation>,
        device: &Device,
    ) -> Self {
        Self {
            linear_1: LinearConfig::new(4 * vision, hidden).init(device),
            linear_2: LinearConfig::new(hidden, out).init(device),
            activation,
        }
    }

    pub(crate) fn forward(&self, x: Tensor<3>, height: usize, width: usize) -> Tensor<3> {
        let [b, _, c] = x.dims();
        let x = x
            .reshape([b, height, width / 2, c * 2])
            .swap_dims(1, 2)
            .reshape([b, width / 2, height / 2, c * 4])
            .swap_dims(1, 2)
            .reshape([b, height * width / 4, c * 4]);
        let x = self.linear_1.forward(x);
        let x = match self.activation {
            Some(activation) => activation.forward(x),
            None => gelu(x),
        };
        self.linear_2.forward(x)
    }
}

impl Vision {
    pub(crate) fn new(
        c: &Siglip2VisionConfig,
        hidden: usize,
        out: usize,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            tower: Siglip2VisionModel::new(c, device)?,
            projector: Lfm2VlMultiModalProjector::new(c.hidden_size, hidden, out, None, device),
        })
    }

    pub(crate) fn forward(&self, images: &[ImageCrops], device: &Device) -> Tensor<3> {
        encode(&self.tower, &self.projector, images, device)
    }
}

pub(crate) fn encode(
    tower: &Siglip2VisionModel,
    projector: &Lfm2VlMultiModalProjector,
    images: &[ImageCrops],
    device: &Device,
) -> Tensor<3> {
    Tensor::cat(
        images
            .iter()
            .flat_map(|image| &image.crops)
            .map(|crop| projector.forward(tower.forward(crop, device), crop.height, crop.width))
            .collect(),
        1,
    )
}

fn resize_weights(input: usize, output: usize) -> Vec<Vec<f32>> {
    let scale = input as f32 / output as f32;
    let support = scale.max(1.0);
    (0..output)
        .map(|i| {
            let center = (i as f32 + 0.5) * scale;
            let mut weights: Vec<_> = (0..input)
                .map(|j| (1.0 - ((j as f32 + 0.5 - center) / support).abs()).max(0.0))
                .collect();
            let total: f32 = weights.iter().sum();
            for weight in &mut weights {
                *weight /= total;
            }
            weights
        })
        .collect()
}

#[expect(
    clippy::cast_sign_loss,
    reason = "nonempty image dimensions produce positive downscaled pixel sizes"
)]
pub(crate) fn preprocess(images: &[ImageInput], cap: bool) -> Result<Vec<ImageCrops>> {
    images
        .iter()
        .map(|input| {
            let mut image = input.decode()?;
            let (mut w, mut h) = image.dimensions();
            if cap && u64::from(w) * u64::from(h) > 1_048_576 {
                let scale = (1_048_576.0 / (f64::from(w) * f64::from(h))).sqrt();
                w = (f64::from(w) * scale).max(1.0) as u32;
                h = (f64::from(h) * scale).max(1.0) as u32;
                image = imageops::resize(&image, w, h, FilterType::CatmullRom);
            }
            let (rows, cols, th, tw) = layout(w as usize, h as usize)?;
            let mut crops = Vec::new();
            if rows * cols > 1 {
                let big = imageops::resize(
                    &image,
                    (cols * 512) as u32,
                    (rows * 512) as u32,
                    FilterType::Triangle,
                );
                for row in 0..rows {
                    for col in 0..cols {
                        crops.push(patchify(
                            &imageops::crop_imm(
                                &big,
                                (col * 512) as u32,
                                (row * 512) as u32,
                                512,
                                512,
                            )
                            .to_image(),
                            cap,
                        ));
                    }
                }
            }
            crops.push(patchify(
                &imageops::resize(&image, tw as u32, th as u32, FilterType::Triangle),
                cap,
            ));
            Ok(ImageCrops { crops, rows, cols })
        })
        .collect()
}

#[expect(
    clippy::cast_sign_loss,
    reason = "dimensions are checked positive before applying the upstream resize formula"
)]
pub(crate) fn layout(width: usize, height: usize) -> Result<(usize, usize, usize, usize)> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidRequest("empty image".into()));
    }
    let round = |x: usize| (x as f64 / 32.0).round_ties_even() as usize * 32;
    let (mut h, mut w) = (round(height).max(32), round(width).max(32));
    let area = height as f64 * width as f64;
    if h as f64 * w as f64 > 262_144.0 {
        let beta = (area / 262_144.0).sqrt();
        h = ((height as f64 / beta / 32.0).floor() as usize * 32).max(32);
        w = ((width as f64 / beta / 32.0).floor() as usize * 32).max(32);
    } else if (h * w) < 65_536 {
        let beta = (65_536.0 / area).sqrt();
        h = (height as f64 * beta / 32.0).ceil() as usize * 32;
        w = (width as f64 * beta / 32.0).ceil() as usize * 32;
    }
    if h.checked_mul(w).is_none_or(|size| size > 262_144) {
        return Err(Error::InvalidRequest(
            "image aspect ratio exceeds the NaFlex patch budget".into(),
        ));
    }
    let (mut rows, mut cols) = (1, 1);
    if round(height).max(16) as f64 * round(width).max(16) as f64 > 524_288.0 {
        let mut ratios: Vec<_> = (1..=10)
            .flat_map(|x| {
                (1..=10).filter_map(move |y| (2..=10).contains(&(x * y)).then_some((x, y)))
            })
            .collect();
        ratios.sort_by_key(|&(x, y)| x * y);
        let mut best = f64::INFINITY;
        for (x, y) in ratios {
            let diff = (width as f64 / height as f64 - x as f64 / y as f64).abs();
            if diff < best
                || (diff.total_cmp(&best).is_eq() && area > 0.5 * 512.0 * 512.0 * (x * y) as f64)
            {
                cols = x;
                rows = y;
                best = diff;
            }
        }
    }
    Ok((rows, cols, h, w))
}

fn patchify(image: &RgbImage, vl: bool) -> Crop {
    let (w, h) = (image.width() as usize / 16, image.height() as usize / 16);
    let mut pixels = Vec::with_capacity(h * w * 768);
    for row in 0..h {
        for col in 0..w {
            for y in 0..16 {
                for x in 0..16 {
                    for &channel in &image
                        .get_pixel((col * 16 + x) as u32, (row * 16 + y) as u32)
                        .0
                    {
                        // Transformers fuses 1/255 rescaling with normalization; Omni uses raw RGB / 127.5.
                        pixels.push(if vl {
                            f32::from(channel) * (1.0 / 255.0) * 2.0 - 1.0
                        } else {
                            (f32::from(channel) - 127.5) / 127.5
                        });
                    }
                }
            }
        }
    }
    Crop {
        pixels,
        height: h,
        width: w,
    }
}
