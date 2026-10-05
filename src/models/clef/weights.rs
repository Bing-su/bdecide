use super::ClefDecisionModel;
use crate::{Result, models::qwen3_5::weights};
use burn::tensor::backend::Backend;
use camino::Utf8Path;
pub(crate) use weights::backbone_files;

pub(super) fn load_clef<B: Backend>(
    model: &mut ClefDecisionModel<B>,
    root: &Utf8Path,
) -> Result<()> {
    let mut files = backbone_files(root)?;
    let embeddings = weights::embedding_weights(root, &files, model.output_embeddings.is_none())?;
    let mut candidate = model.clone();
    if !embeddings.is_tied() && candidate.output_embeddings.is_none() {
        candidate.output_embeddings = Some(candidate.language_model.get_input_embeddings().clone());
    }
    files.push("joint_head.safetensors".into());
    // Keep Clef's head namespace while sharing strict validation, e.g. head.hidden_norm.gamma.
    weights::load(&mut candidate, root, &files, &[], |file, source| {
        if file == "joint_head.safetensors" {
            Ok(Some(weights::remap(format!("head.{source}"))))
        } else {
            // Clef's released joint model keeps its own namespace, e.g. output_embeddings.weight.
            embeddings.map_name(source).map(|name| {
                name.map(|name| {
                    if name == "lm_head.weight" {
                        "output_embeddings.weight".into()
                    } else {
                        name.replacen("model.", "language_model.", 1)
                    }
                })
            })
        }
    })?;
    *model = candidate;
    Ok(())
}
