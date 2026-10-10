//! Normalize Transformers SigLIP2 paths while preserving published checkpoint names.
use crate::Result;
use crate::models::weights;

pub(crate) fn weight_name(name: &str) -> Result<Option<String>> {
    // Transformers exports SigLIP2 without vision_model, e.g. vision.tower.embeddings.*.
    for prefix in ["model.vision_tower.", "vision.tower."] {
        if let Some(rest) = name.strip_prefix(prefix)
            && (rest.starts_with("embeddings.")
                || rest.starts_with("encoder.")
                || rest.starts_with("post_layernorm."))
        {
            return Ok(Some(format!("{prefix}vision_model.{rest}")));
        }
    }
    weights::identity_name(name)
}
