"""Generate independent Laya/PyTorch fixtures, including local attention and padding.

Run with laya==0.3.26, transformers==5.18.0, and torch==2.14.1 (CPU).
For example: python tests/generate_reference.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from laya.agent import Agent

import laya
import torch
import transformers
from laya.agent import collate_items
from laya.common import DecisionModel
from reference_utils import FIXTURES, FixtureRequest, Question, write_json
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import ModernBertConfig, ModernBertModel


def create_checkpoint(root: Path) -> None:
    """Store the tiny encoder and head in release precision, e.g. FP16 weights."""
    torch.manual_seed(431)

    (root / "encoder").mkdir(parents=True, exist_ok=True)
    (root / "tokenizer").mkdir(exist_ok=True)

    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "choice",
        "score",
        "noul",
        "question",
        ":",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "level",
        "0",
        "1",
        "2",
        "false",
        "true",
        "no",
        "yes",
        "the",
        "statement",
        "does",
        "not",
        "hold",
        "holds",
        ",",
        "Select",
        "urgent",
        "Is",
        "this",
        "?",
        "z",
        "a",
        "한글",
        "cancel",
        "last",
    ]
    vocabulary.extend(f"extra{i}" for i in range(64 - len(vocabulary)))
    tokenizer = Tokenizer(
        # Reuse the vocabulary's unknown token so both tokenizer definitions agree.
        models.WordLevel(
            {word: i for i, word in enumerate(vocabulary)}, unk_token=vocabulary[1]
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(vocabulary[:5])
    tokenizer.save(str(root / "tokenizer" / "tokenizer.json"))
    special = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    write_json(root / "tokenizer" / "tokenizer_config.json", special)
    encoder = ModernBertConfig.from_dict(
        {
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "max_position_embeddings": 128,
            "local_attention": 4,
            "global_attn_every_n_layers": 2,
            "attention_bias": True,
            "mlp_bias": True,
            "norm_bias": True,
            "pad_token_id": 0,
            "bos_token_id": 2,
            "eos_token_id": 3,
            "cls_token_id": 2,
            "sep_token_id": 3,
            "mask_token_id": 4,
            "norm_eps": 1e-5,
        }
    )
    encoder.save_pretrained(root / "encoder")
    cfg = {
        "encoder": "fixture-modernbert",
        "head_layers": 2,
        "max_len": 96,
        "head_max_len": 64,
        "act_costs": {"escalate": 0.5},
        "temperature": [0.1, 2.0, 1.5],
        "temperature_by_options": {"choice:3-5": 0.25},
    }
    write_json(root / "rl_agent_config.json", cfg)
    model = DecisionModel(ModernBertModel(encoder), head_layers=2).eval()
    # Store half precision like public checkpoints, then let each runtime load FP32.
    weights = {
        name: value.contiguous().half() if name != "temperature" else value.contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(weights, str(root / "model.safetensors"))


def reference_requests() -> list[FixtureRequest]:
    """Cover rendering and truncation independently, e.g. newest-turn list retention."""
    questions: dict[str, Question] = {
        "department": {
            "type": "choice",
            "instructions": "Select",
            "criteria": {"z": "gamma", "a": "delta", "other": "epsilon"},
        },
        "urgency": {
            "type": "score",
            "instructions": "Is this urgent?",
            "criteria": ["alpha", "beta", "gamma"],
        },
        "risk": {"type": "noul", "instructions": "cancel?"},
    }
    return [
        {"state": "alpha beta last", "questions": questions},
        {
            "state": {"z": "한글", "a": [1, True, {"last": "cancel"}]},
            "questions": questions,
        },
        {
            "state": "alpha " * 80,
            "questions": questions,
            "options": {"truncation": "truncate", "max_len": 40},
        },
        {
            "state": ["alpha " * 50, {"last": "cancel"}],
            "questions": questions,
            "options": {"truncation": "truncate", "max_len": 40},
        },
        {
            "state": "alpha",
            "questions": {
                "only": {
                    "type": "choice",
                    "instructions": "Select",
                    "criteria": {"z": None},
                }
            },
        },
        {
            "state": {"float": 0.000001, "integer_float": 1.0, "negative_zero": -0.0},
            "questions": questions,
        },
        {
            "state": "alpha beta",
            "questions": questions,
            "options": {"truncation": "truncate", "head_max_len": 16},
        },
        {
            "state": "alpha",
            "questions": {
                "choice": {
                    "type": "choice",
                    "instructions": "Select",
                    "criteria": {"z": {"a": "한글", "level": 1}, "a": [True, "beta"]},
                },
                "score": {
                    "type": "score",
                    "instructions": "Is this urgent?",
                    "criteria": [{"alpha": 0}, ["beta", None], False],
                },
                "noul": {
                    "type": "noul",
                    "instructions": "cancel?",
                    "criteria": {"false": {"z": "alpha"}, "true": {"a": "beta"}},
                },
            },
        },
    ]


def generate_cases(agent: Agent, requests: list[FixtureRequest]) -> list[dict]:
    """Use upstream preprocessing and inference, e.g. retain native JSON rendering."""
    cases = []
    for request in requests:
        # Pass ordinary dictionaries to Laya's mutable API, e.g. TypedDict stays local.
        q = {key: dict(value) for key, value in request["questions"].items()}
        ids = list(q)
        internal = {key: agent._to_internal(value) for key, value in q.items()}
        options = request.get("options", {})
        items = agent._encode_state(
            request["state"],
            ids,
            internal,
            max_len=options.get("max_len"),
            head_max_len=options.get("head_max_len"),
        )
        response = agent.predict(
            request["state"],
            q,
            max_len=options.get("max_len"),
            head_max_len=options.get("head_max_len"),
        )
        batch = collate_items([items], agent.tok.pad_token_id)
        with torch.no_grad():
            logits, actions = agent._forward(batch)
        cases.append(
            {
                "request": request,
                "encoded": [
                    {"ids": item["ids"], "markers": item["markers"]} for item in items
                ],
                "logits": [
                    row[: len(item["markers"])].tolist()
                    for row, item in zip(logits, items)
                ],
                "actions": actions[:, 0].tolist(),
                "response": response,
            }
        )

    return cases


def main(fixtures: Path = FIXTURES) -> None:
    """Generate Laya references explicitly, e.g. importing this module writes no files."""
    torch.set_num_threads(2)
    root = fixtures / "tiny-laya"
    create_checkpoint(root)
    agent = laya.load(str(root), device="cpu")
    cases = generate_cases(agent, reference_requests())
    payload = {
        "reference": {
            "laya": laya.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
        "cases": cases,
    }
    write_json(fixtures / "reference.json", payload)
    print(f"Generated {len(cases)} independent reference cases in {fixtures}")


if __name__ == "__main__":
    main()
