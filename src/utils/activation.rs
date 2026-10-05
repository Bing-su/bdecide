//! Select parameterless Transformers activations without changing checkpoint tensors.

use std::{f64::consts::SQRT_2, str::FromStr};

use burn::tensor::{Tensor, activation, backend::Backend};

use crate::{Error, Result};

/// Keep activation math independent of model gating, e.g. GEGLU's split and multiply.
#[derive(Debug, Clone, Copy)]
pub(crate) enum HiddenActivation {
    Gelu,
    GeluPython,
    Gelu10,
    GeluTanh,
    GeluFast,
    QuickGelu,
    HardSwish,
    Laplace,
    LeakyRelu,
    Linear,
    Mish,
    Relu,
    Relu2,
    Relu6,
    Sigmoid,
    Silu,
    SqrtSoftplus,
    Tanh,
}

impl FromStr for HiddenActivation {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        Ok(match name {
            "gelu" => Self::Gelu,
            "gelu_python" => Self::GeluPython,
            "gelu_10" => Self::Gelu10,
            "gelu_new" | "gelu_accurate" | "gelu_pytorch_tanh" | "gelu_python_tanh" => {
                Self::GeluTanh
            }
            "gelu_fast" => Self::GeluFast,
            "quick_gelu" => Self::QuickGelu,
            "hardswish" => Self::HardSwish,
            "laplace" => Self::Laplace,
            "leaky_relu" => Self::LeakyRelu,
            "linear" => Self::Linear,
            "mish" => Self::Mish,
            "relu" => Self::Relu,
            "relu2" => Self::Relu2,
            "relu6" => Self::Relu6,
            "sigmoid" => Self::Sigmoid,
            "silu" | "swish" => Self::Silu,
            "sqrtsoftplus" => Self::SqrtSoftplus,
            "tanh" => Self::Tanh,
            "prelu" | "xielu" => {
                return Err(Error::UnsupportedModel(format!(
                    "hidden activation {name} requires learned parameters"
                )));
            }
            _ => {
                return Err(Error::UnsupportedModel(format!(
                    "unknown hidden activation {name}"
                )));
            }
        })
    }
}

impl HiddenActivation {
    pub(crate) fn forward<B: Backend, const D: usize>(self, input: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Self::Gelu => activation::gelu(input),
            Self::GeluPython => input.clone() * 0.5 * ((input / SQRT_2).erf() + 1.0),
            Self::Gelu10 => activation::gelu(input).clamp(-10.0, 10.0),
            Self::GeluTanh => activation::gelu_approximate(input),
            // Square once to reduce shared handles while retaining FastGELU's constant.
            // For example, the inner term is x*0.7978845608*(1+0.044715*x²).
            Self::GeluFast => {
                let square = input.clone().square();
                let inner = input.clone() * 0.7978845608 * (square * 0.044715 + 1.0);
                input * 0.5 * (inner.tanh() + 1.0)
            }
            Self::QuickGelu => input.clone() * activation::sigmoid(input * 1.702),
            Self::HardSwish => activation::hard_swish(input),
            Self::Laplace => {
                #[expect(
                    clippy::approx_constant,
                    reason = "Match Transformers' rounded Laplace mu, not exact 1/sqrt(2)"
                )]
                const MU: f64 = 0.707107;
                (((input - MU) / (0.282095 * SQRT_2)).erf() + 1.0) * 0.5
            }
            Self::LeakyRelu => activation::leaky_relu(input, 0.01),
            Self::Linear => input,
            Self::Mish => input.clone() * softplus(input).tanh(),
            Self::Relu => activation::relu(input),
            Self::Relu2 => activation::relu(input).square(),
            Self::Relu6 => input.clamp(0.0, 6.0),
            Self::Sigmoid => activation::sigmoid(input),
            Self::Silu => activation::silu(input),
            Self::SqrtSoftplus => softplus(input).sqrt(),
            Self::Tanh => input.tanh(),
        }
    }
}

// Burn 0.21's softplus uses log(1+exp(x)), which loses small tails and overflows in float32.
// Preserve PyTorch's beta=1, threshold=20 behavior, e.g. sqrtsoftplus(-20) stays nonzero
// and sqrtsoftplus(100) stays finite without evaluating exp(100).
fn softplus<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    let tail = (-input.clone().abs()).exp();
    // Some GPU log1p kernels round 1+tail to 1. Preserve those tails with a
    // second-order expansion whose omitted term is below float32 precision at tail<1e-4.
    let small = tail.clone() - tail.clone().square() * 0.5;
    let logarithm = tail
        .clone()
        .log1p()
        .mask_where(tail.lower_elem(1e-4), small);
    let result = input.clone().clamp_min(0.0) + logarithm;
    result.mask_where(input.clone().greater_elem(20.0), input)
}

#[cfg(test)]
pub(crate) mod tests {
    use approx::{abs_diff_eq, assert_relative_eq};
    #[cfg(feature = "cpu")]
    use burn::backend::Flex;
    #[cfg(feature = "wgpu")]
    use burn::backend::Wgpu;
    use burn::tensor::TensorData;
    use camino::Utf8Path;
    use serde::Deserialize;

    use super::*;
    use crate::utils::read_checkpoint_json;

    #[derive(Deserialize)]
    pub(crate) struct Reference {
        pub inputs: Vec<f32>,
        pub input_ids: Vec<i32>,
        pub cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    pub(crate) struct Case {
        pub name: String,
        pub values: Vec<f32>,
        pub modernbert: Vec<f32>,
        pub qwen3_5: Vec<f32>,
    }

    pub(crate) fn reference() -> Reference {
        read_checkpoint_json(
            &Utf8Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/activation-reference.json"),
        )
        .unwrap()
    }

    // Compare independent Python values with backend rounding allowance, e.g. erf on Wgpu.
    pub(crate) fn assert_close(data: TensorData, expected: &[f32], name: &str, tolerance: f32) {
        let actual = data.as_slice::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len(), "{name}");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            // Keep the reference-scaled budget, e.g. tolerance doubles at expected=1.
            assert!(
                actual.is_finite()
                    && abs_diff_eq!(
                        actual,
                        expected,
                        epsilon = tolerance * (1.0 + expected.abs())
                    ),
                "{name}[{index}]: {actual} != {expected}"
            );
        }
    }

    fn matches_python<B: Backend>() {
        let reference = reference();
        assert_eq!(reference.cases.len(), 22);
        let device = B::Device::default();
        let input = Tensor::<B, 1>::from_data(
            TensorData::new(reference.inputs.clone(), [reference.inputs.len()]),
            &device,
        );
        for case in reference.cases {
            let act = case.name.parse::<HiddenActivation>().unwrap();
            let output = act.forward(input.clone()).into_data();
            assert_close(output, &case.values, &case.name, 3e-6);
            // A zero tail can pass absolute tolerance; require relative accuracy at x=-20.
            if case.name == "sqrtsoftplus" {
                let output = act
                    .forward(Tensor::<B, 1>::from_data([-20.0], &device))
                    .into_data();
                let value = output.as_slice::<f32>().unwrap()[0];
                assert_relative_eq!(value, 0.00004539993, epsilon = 0.0, max_relative = 1e-4);
            }
        }
    }

    #[test]
    fn rejects_learned_and_unknown_activations() {
        for name in ["prelu", "xielu", "unknown", "", "GELU"] {
            let error = name.parse::<HiddenActivation>().unwrap_err();
            assert!(matches!(error, Error::UnsupportedModel(_)));
            assert!(error.to_string().contains(name));
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn cpu_matches_python_activations() {
        matches_python::<Flex>();
    }

    #[test]
    #[cfg(feature = "wgpu")]
    #[ignore = "requires a wgpu adapter"]
    fn wgpu_matches_python_activations() {
        matches_python::<Wgpu<f32, i32>>();
    }
}
