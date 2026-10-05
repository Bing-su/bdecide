"""Share fixture wire types and storage, e.g. keep question order in reference JSON."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
    from pydantic import JsonValue

import json

from safetensors.torch import save_file

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class Question(TypedDict):
    """Describe upstream questions without coercing criteria, e.g. nested JSON choices."""

    type: Literal["choice", "score", "noul"]
    instructions: str
    criteria: NotRequired[JsonValue]


class PredictOptions(TypedDict, total=False):
    """Keep truncation opt-in, e.g. max_len=40 in the long-state reference."""

    truncation: Literal["truncate"]
    max_len: int
    head_max_len: int


class FixtureRequest(TypedDict):
    """Use the Rust request shape, e.g. ordered question IDs with optional limits."""

    state: str | list[JsonValue] | dict[str, JsonValue]
    questions: dict[str, Question]
    options: NotRequired[PredictOptions]


def write_json(path: Path, value: object) -> None:
    """Reject non-finite reference values and preserve Unicode, e.g. Korean criteria."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_sharded_weights(root: Path, weights: Mapping[str, torch.Tensor]) -> None:
    """Exercise complete shard loading, e.g. alternating parameters across two files."""
    weight_map = {}
    for shard_index in range(2):
        filename = f"model-{shard_index + 1:05d}-of-00002.safetensors"
        selected = {
            name: tensor
            for i, (name, tensor) in enumerate(weights.items())
            if i % 2 == shard_index
        }
        save_file(selected, str(root / filename))
        weight_map.update(dict.fromkeys(selected, filename))
    write_json(root / "model.safetensors.index.json", {"weight_map": weight_map})
