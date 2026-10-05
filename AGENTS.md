# Models

Follow the conventions of the Python `transformers` library.
Preserve weight names wherever possible so weights can be loaded without renaming.
As long as the values match, Python's JSON string output and Rust's JSON string output do not need to be identical.

# Rust

Keep code in `mod.rs` and `lib.rs` to a minimum.

# Python

Write Python code that passes `ruff` and `ty` checks.
