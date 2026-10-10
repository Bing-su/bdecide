//! Apply Transformers' split-half RoPE without changing each model's frequency precision.
use burn::tensor::Tensor;
use burn_std::s;

// Share full and partial rotation, e.g. Qwen keeps channels beyond cos.dims()[3] unchanged.
pub(crate) fn apply_rotary(input: Tensor<4>, cos: Tensor<4>, sin: Tensor<4>) -> Tensor<4> {
    let dim = cos.dims()[3];
    let width = input.dims()[3];
    let first = input.clone().slice(s![.., .., .., ..dim / 2]);
    let second = input.clone().slice(s![.., .., .., dim / 2..dim]);
    let rotated = input.clone().slice(s![.., .., .., ..dim]) * cos
        + Tensor::cat(vec![-second, first], 3) * sin;
    if dim == width {
        rotated
    } else {
        Tensor::cat(vec![rotated, input.slice(s![.., .., .., dim..])], 3)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, TensorData};

    use super::*;

    fn full_and_partial_rotation(device: Device) {
        // Use exact 0/90 degree rotations to verify channel pairing and Q/K broadcasting.
        for (dim, rotated) in [
            (4, [-3.0_f32, -4.0, 1.0, 2.0, 5.0, 6.0]),
            (6, [-4.0_f32, -5.0, -6.0, 1.0, 2.0, 3.0]),
        ] {
            let row = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            let input = Tensor::from_data(TensorData::new(row.repeat(8), [2, 2, 2, 6]), &device);
            let cos = [vec![1.0_f32; dim], vec![0.0; dim]].concat();
            let sin = [vec![0.0_f32; dim], vec![1.0; dim]].concat();
            let actual = apply_rotary(
                input,
                Tensor::from_data(TensorData::new(cos, [1, 1, 2, dim]), &device),
                Tensor::from_data(TensorData::new(sin, [1, 1, 2, dim]), &device),
            )
            .try_into_vec_as::<f32>()
            .unwrap();
            assert_eq!(actual, [row, rotated].concat().repeat(4));
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_preserves_rotary_layout_and_tail() {
        full_and_partial_rotation(Device::flex());
    }

    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "requires a wgpu adapter"]
    fn wgpu_preserves_rotary_layout_and_tail() {
        full_and_partial_rotation(Device::wgpu(Default::default()));
    }
}
