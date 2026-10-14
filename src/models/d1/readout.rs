//! Share typed decision output, e.g. each model supplies its own true-label index.
use burn::tensor::Tensor;

use crate::utils::decision::{self, softmax};
use crate::utils::render;
use crate::{Error, Result, SystemOneAnswer, SystemOneQuestion};

pub(super) fn values(tensor: Tensor<3>) -> Result<Vec<f64>> {
    tensor
        .try_into_vec_as::<f32>()
        .map(|values| values.into_iter().map(f64::from).collect())
        .map_err(|error| Error::Inference(error.to_string()))
}

pub(super) fn answer(
    q: &SystemOneQuestion,
    logits: &[f64],
    true_index: usize,
) -> Result<SystemOneAnswer> {
    let probabilities = softmax(logits)?;
    let confidence = probabilities.iter().copied().fold(0.0, f64::max);
    let legend = if let SystemOneQuestion::Score { criteria, .. } = q {
        criteria.iter().map(render).collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    decision::answer(q, probabilities, confidence, legend, true_index)
}
