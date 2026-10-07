//! Load the backbone and standalone joint head in their own checkpoint namespaces.
use burn::tensor::backend::Backend;
use camino::Utf8Path;

use super::ClefDecisionModel;
use crate::Result;
use crate::models::{qwen3_5, weights};

pub(super) fn load_clef<B: Backend>(
    model: &mut ClefDecisionModel<B>,
    root: &Utf8Path,
) -> Result<()> {
    let mut candidate = model.clone();
    qwen3_5::weights::load_causal_lm(&mut candidate.backbone, root)?;
    // Load hidden_norm directly, e.g. no synthetic "head." prefix for the standalone file.
    weights::load(
        &mut candidate.head,
        root,
        &["joint_head.safetensors".into()],
        &[],
        weights::identity_name,
    )?;
    // Preserve the caller's complete model if either checkpoint fails validation.
    *model = candidate;
    Ok(())
}
