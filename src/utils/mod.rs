//! Re-export shared helpers so model code stays independent of their file layout.
pub(crate) mod activation;
pub(crate) mod attention;
mod helpers;

pub(crate) use helpers::{number_text, read, read_checkpoint_json, render};
