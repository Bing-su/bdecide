"""Check fixture plumbing without generating models, e.g. imports must stay read-only.

Run with: uv run python tests/test_reference_utils.py
"""

from __future__ import annotations

import importlib
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from reference_utils import save_sharded_weights, write_json
from safetensors.torch import load_file


def test_imports_do_not_generate_or_download() -> None:
    """Fail on import side effects, e.g. a missing main guard must not run inference."""


with ExitStack() as stack:
    for target in (
        "pathlib.Path.write_text",
        "pathlib.Path.write_bytes",
        "pathlib.Path.mkdir",
        "safetensors.torch.save_file",
        "huggingface_hub.hf_hub_download",
        "urllib.request.urlopen",
        "torch.manual_seed",
        "torch.set_num_threads",
    ):
        stack.enter_context(patch(target, side_effect=AssertionError(target)))
    for name in (
        "generate_reference",
        "generate_clef_reference",
        "generate_decision_reference",
        "generate_activation_reference",
        "generate_d1_reference",
    ):
        # Reload too so a cached import cannot hide generation, e.g. discovery imports.
        importlib.reload(importlib.import_module(name))


def test_json_and_shards_preserve_reference_values() -> None:
    """Check independent stored values, e.g. every tensor appears in exactly one shard."""


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    payload = {"한글": [True, None, -0.0], "z": 1, "a": 2}
    write_json(root / "reference.json", payload)
    text = (root / "reference.json").read_text(encoding="utf-8")
    assert "한글" in text
    assert json.loads(text) == payload
    try:
        write_json(root / "invalid.json", {"logit": float("nan")})
    except ValueError:
        pass
    else:
        message = "Non-finite fixture values must fail before writing"
        raise AssertionError(message)
    assert not (root / "invalid.json").exists()

    weights = {f"layer.{i}": torch.tensor([float(i)]) for i in range(5)}
    save_sharded_weights(root, weights)
    index = json.loads(
        (root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    assert set(index) == set(weights)
    assert set(index.values()) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    seen = set()
    for filename in set(index.values()):
        for name, tensor in load_file(root / filename).items():
            assert name not in seen
            seen.add(name)
            assert index[name] == filename
            torch.testing.assert_close(tensor, weights[name], rtol=0, atol=0)
    assert seen == set(weights)


if __name__ == "__main__":
    test_imports_do_not_generate_or_download()
    test_json_and_shards_preserve_reference_values()
    print("Reference import and storage checks passed")
