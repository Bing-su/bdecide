//! Share multi-head attention across encoders and decision heads.
use burn::tensor::{Tensor, TensorData, activation::softmax, backend::Backend};
use burn_std::s;

pub(crate) fn attend<B: Backend>(
    qkv: Tensor<B, 3>,
    heads: usize,
    padding: Tensor<B, 4>,
    rope: Option<(f64, Option<usize>)>,
) -> Tensor<B, 3> {
    let [batch, length, total] = qkv.dims();
    let hidden_size = total / 3;
    let head_width = hidden_size / heads;
    let device = qkv.device();
    let query = qkv
        .clone()
        .slice(s![.., .., 0..hidden_size])
        .reshape([batch, length, heads, head_width])
        .swap_dims(1, 2);
    let key = qkv
        .clone()
        .slice(s![.., .., hidden_size..2 * hidden_size])
        .reshape([batch, length, heads, head_width])
        .swap_dims(1, 2);
    let value = qkv
        .slice(s![.., .., 2 * hidden_size..3 * hidden_size])
        .reshape([batch, length, heads, head_width])
        .swap_dims(1, 2);
    let (query, key, mask) = if let Some((theta, window)) = rope {
        let mut cos = Vec::with_capacity(length * head_width);
        let mut sin = Vec::with_capacity(length * head_width);
        for position in 0..length {
            for channel in 0..head_width {
                let exponent = (2 * (channel % (head_width / 2))) as f32 / head_width as f32;
                let phase = position as f32 * (1.0 / (theta as f32).powf(exponent));
                cos.push(phase.cos());
                sin.push(phase.sin());
            }
        }
        let cos =
            Tensor::<B, 4>::from_data(TensorData::new(cos, [1, 1, length, head_width]), &device);
        let sin =
            Tensor::<B, 4>::from_data(TensorData::new(sin, [1, 1, length, head_width]), &device);
        let apply_rotary = |hidden: Tensor<B, 4>| {
            // Split rotary channels while keeping every batch, head, and position.
            let first_half = hidden.clone().slice(s![.., .., .., 0..head_width / 2]);
            let second_half = hidden
                .clone()
                .slice(s![.., .., .., head_width / 2..head_width]);
            hidden * cos.clone() + Tensor::cat(vec![-second_half, first_half], 3) * sin.clone()
        };
        let mask = match window {
            Some(window) => {
                let local: Vec<f32> = (0..length)
                    .flat_map(|query_position| {
                        (0..length).map(move |key_position| {
                            if query_position.abs_diff(key_position) > window {
                                f32::MIN
                            } else {
                                0.0
                            }
                        })
                    })
                    .collect();
                padding
                    + Tensor::<B, 4>::from_data(
                        TensorData::new(local, [1, 1, length, length]),
                        &device,
                    )
            }
            None => padding,
        };
        (apply_rotary(query), apply_rotary(key), mask)
    } else {
        (query, key, padding)
    };
    let scores = query.matmul(key.swap_dims(2, 3)) / (head_width as f64).sqrt() + mask;
    softmax(scores, 3)
        .matmul(value)
        .swap_dims(1, 2)
        .reshape([batch, length, hidden_size])
}
