"""Generate small native d1 parity fixtures from pinned LiquidAI implementations.

Run with the repository's torch/transformers environment and torchvision:
    uv run python tests/generate_d1_reference.py
"""

from __future__ import annotations

import importlib
import json
import sys
import types
import wave
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download, parse_safetensors_file_metadata
from PIL import Image
from reference_utils import FIXTURES, save_sharded_weights, write_json
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    Lfm2Config,
    Lfm2VlConfig,
    Lfm2VlForConditionalGeneration,
    Lfm2VlImageProcessor,
    Lfm2VlProcessor,
    PreTrainedTokenizerFast,
    Siglip2VisionConfig,
)

if TYPE_CHECKING:
    from types import ModuleType

RELEASES = {
    "3b": ("LiquidAI/d1-3B", "051bcc464b01b9f92942b364d9586b0ef5912432"),
    "omni": ("LiquidAI/d1-omni-600M", "02b55d7076f15129e59ab3f94783f32c4b088674"),
}


def reference(kind: str) -> ModuleType:
    """Import pinned upstream modules, e.g. Omni's unmodified audio frontend."""
    repo, revision = RELEASES[kind]
    files = (
        ["encoder.py", "vision.py", "audio.py", "prompt.py", "modeling_d1.py"]
        if kind == "omni"
        else [
            "hybrid.py",
            "lfm2_vl.py",
            "prompt.py",
            "api.py",
            "runner.py",
            "modeling_d1.py",
            "chat_template.jinja",
        ]
    )
    folder = None
    for filename in files:
        source = hf_hub_download(
            repo,
            filename,
            revision=revision,
            cache_dir=FIXTURES.parents[1] / ".cache" / "d1-reference",
        )
        folder = Path(source).parent
    assert folder is not None
    name = f"d1_reference_{kind}"
    package = types.ModuleType(name)
    package.__path__ = [str(folder)]
    sys.modules[name] = package
    return importlib.import_module(f"{name}.modeling_d1")


def tokenizer() -> PreTrainedTokenizerFast:
    """Keep all verbalizers atomic, e.g. yes/Yes/YES and A..Z."""
    words = list(
        dict.fromkeys(
            [
                "[UNK]",
                "<bos>",
                "<pad>",
                "alpha",
                "beta",
                "gamma",
                "Select",
                "urgent",
                "cancel",
                "last",
                "한글",
                "yes",
                "Yes",
                "YES",
                "no",
                "No",
                "NO",
                "false",
                "true",
                "level",
                *list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
                *[str(i) for i in range(100)],
                *[f"{i:02}" for i in range(100)],
            ]
        )
    )
    backend = Tokenizer(
        models.WordLevel({s: i for i, s in enumerate(words)}, unk_token=words[0])
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.add_special_tokens(
        [
            "<|im_start|>",
            "<|im_end|>",
            "<image>",
            "<|image_start|>",
            "<|image_end|>",
            "<|img_thumbnail|>",
            *[f"<|reserved_{i}|>" for i in range(7, 12)],
            "<|mask|>",
            *[f"<|img_row_{r}_col_{c}|>" for r in range(1, 11) for c in range(1, 11)],
        ]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=words[1],
        pad_token=words[2],
        unk_token=words[0],
    )


def media(root: Path) -> tuple[list[Image.Image], np.ndarray]:
    """Use varying pixels and PCM, e.g. no constant-image or silence-only false passes."""
    images = []
    for name, width, height in [
        ("square", 256, 256),
        ("rectangle", 352, 288),
        ("tiled", 1024, 768),
    ]:
        y, x = np.indices((height, width))
        rgb = np.stack(
            [(x * 3 + y) % 256, (x + y * 2) % 256, (x * 2 + y * 3) % 256], axis=-1
        ).astype(np.uint8)
        image = Image.fromarray(rgb)
        image.save(root / f"{name}.png")
        images.append(image)
    rng = np.random.default_rng(731)
    samples = rng.integers(-8000, 8000, 8131, dtype=np.int16)
    with wave.open(str(root / "speech.wav"), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(samples.tobytes())
    return images, samples


def requests() -> list[dict]:
    """Exercise ordered types and defaults, e.g. unequal calibration buckets."""
    questions = {
        "pick": {
            "type": "choice",
            "instructions": "Select",
            "criteria": {"z": "alpha", "a": "beta", "other": None},
        },
        "rating": {
            "type": "score",
            "instructions": "urgent?",
            "criteria": ["alpha", "beta", "gamma"],
        },
        "boolean": {"type": "noul", "instructions": "cancel?"},
    }
    return [
        {"state": "alpha beta last", "questions": questions},
        {"state": {"z": "한글", "a": [True, None, 2]}, "questions": questions},
        {"state": None, "questions": questions, "images": ["square.png"]},
        {
            "state": "alpha",
            "questions": questions,
            "images": ["rectangle.png", "square.png"],
        },
        {
            "state": "beta",
            "questions": {"boolean": questions["boolean"]},
            "images": ["tiled.png"],
        },
        {
            "state": "alpha",
            "questions": {
                "boolean": {
                    "type": "noul",
                    "instructions": "cancel?",
                    "criteria": {"false": "alpha", "true": "beta"},
                }
            },
        },
    ]


def generate(kind: str) -> None:
    """Run released encoders and readout, e.g. BF16 stored weights compared in F32."""
    torch.manual_seed(735)
    module = reference(kind)
    root = FIXTURES / f"tiny-d1-{kind}"
    root.mkdir(exist_ok=True)
    release_artifacts(kind, root)
    fast = tokenizer()
    fast.save_pretrained(root)
    pics, samples = media(root)
    text = {
        "vocab_size": len(fast),
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "layer_types": ["conv", "full_attention", "conv"],
        "norm_eps": 1e-5,
        "conv_L_cache": 3,
        "block_ffn_dim_multiplier": 1.0,
        "block_multiple_of": 8,
        "max_position_embeddings": 4096,
        "rope_theta": 1_000_000.0,
    }
    vision = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_patches": 16,
        "patch_size": 16,
        "num_channels": 3,
        "hidden_act": "gelu_pytorch_tanh",
        "layer_norm_eps": 1e-6,
        "vision_use_head": False,
    }
    if kind == "omni":
        config = module.D1OmniConfig(
            text_config=text,
            vision_config=vision,
            audio_config={
                "feat_in": 128,
                "n_layers": 2,
                "d_model": 16,
                "subsampling_conv_channels": 4,
                "ff_expansion_factor": 2,
                "n_heads": 2,
                "conv_kernel_size": 9,
                "residual_width": 16,
            },
            projector_hidden_size=32,
            head_layers=2,
            max_length=4096,
            image_text_length=896,
            audio_text_length=3072,
            bos_token_id=fast.bos_token_id,
            pad_token_id=fast.pad_token_id,
            temperatures={"choice:3-5": 1.4, "noul:2": 1.7, "score": 1.3},
        )
        model = module.D1OmniModel(config).eval()
        model.tokenizer = fast
        # Nontrivial running statistics reveal training-mode or missing-buffer mistakes.
        for layer in model.audio.encoder.layers:
            layer.conv.batch_norm.running_mean.fill_(0.2)
            layer.conv.batch_norm.running_var.fill_(1.3)
        engine = model
    else:
        cfg = Lfm2Config.from_dict(
            {
                **text,
                "block_auto_adjust_ff_dim": False,
                "conv_bias": False,
                "bos_token_id": fast.bos_token_id,
                "rope_parameters": {"rope_type": "default", "rope_theta": 1_000_000.0},
            }
        )
        config = Lfm2VlConfig.from_dict(
            {
                "text_config": cfg.to_dict(),
                "vision_config": Siglip2VisionConfig.from_dict(vision).to_dict(),
                "projector_hidden_size": 32,
                "image_token_id": fast.convert_tokens_to_ids("<image>"),
                "bos_token_id": fast.bos_token_id,
                "pad_token_id": fast.pad_token_id,
                "projector_bias": True,
                "projector_hidden_act": "gelu",
                "projector_use_layernorm": False,
            }
        )
        model = module.D1Model(config).eval()
        # The wrapper replaces the language stack; from_pretrained normally re-ties its new embedding.
        model.tie_weights()
        assert module.__file__ is not None
        template = (
            Path(module.__file__).parent.joinpath("chat_template.jinja").read_text()
        )
        runner = importlib.import_module("d1_reference_3b.runner")
        engine = runner.SystemOne(model=model, tokenizer=fast)
        engine.processor = Lfm2VlProcessor(
            Lfm2VlImageProcessor(), fast, chat_template=template
        )
    # Save the exact values seen by the Python reference, e.g. no random-initializer parity.
    model.bfloat16().float()
    weights = {name: tensor.contiguous() for name, tensor in model.state_dict().items()}
    if kind == "omni":
        save_sharded_weights(root, weights)
    else:
        weights.pop("lm_head.weight")
        save_file(weights, str(root / "model.safetensors"))
    config.save_pretrained(root)
    if kind == "3b":
        base_reference(root, model, fast, pics[1])
    cases = requests()
    if kind == "omni":
        cases.extend(
            [
                {
                    "state": None,
                    "questions": cases[0]["questions"],
                    "audio": "speech.wav",
                },
                {
                    "state": "alpha",
                    "questions": cases[0]["questions"],
                    "audio": {"pcm16": samples[:1000].tolist(), "sample_rate": 16000},
                },
                {
                    "state": "alpha " * 500,
                    "questions": cases[0]["questions"],
                    "options": {"max_len": 240, "truncation": "truncate"},
                },
                {
                    "state": "alpha",
                    "questions": {
                        "long": {
                            "type": "choice",
                            "instructions": "Select " * 100,
                            "criteria": {
                                "first": "alpha " * 100,
                                "second": "beta " * 100,
                            },
                        }
                    },
                    "options": {"truncation": "truncate"},
                },
            ]
        )
    image_map = dict(zip(["square.png", "rectangle.png", "tiled.png"], pics))
    output = evaluate(kind, engine, cases, image_map, samples)
    write_json(
        root / "reference.json",
        {
            "reference": {
                "revision": RELEASES[kind][1],
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "cases": output,
        },
    )
    if kind == "omni":
        frontend = importlib.import_module("d1_reference_omni.audio")
        mel, frames = frontend.MelFrontend()(frontend.waveform(samples))
        write_json(
            root / "mel.json",
            {
                "samples": samples.tolist(),
                "frames": int(frames[0]),
                "mel": mel.transpose(1, 2).flatten().tolist(),
            },
        )
    print(f"Generated {len(output)} independent {kind} reference cases")


def base_reference(root, model, fast, image) -> None:
    """Check the independent Transformers base, e.g. full and selected text/image logits."""
    base = Lfm2VlForConditionalGeneration(model.config).eval()
    base.load_state_dict(model.state_dict())
    base.tie_weights()
    tokens = [
        fast.bos_token_id,
        *fast.encode("alpha beta last", add_special_tokens=False),
    ]
    selected = fast.convert_tokens_to_ids(["yes", "Yes", "YES", "no", "No", "NO"])
    pixels = Lfm2VlImageProcessor()(images=[image], return_tensors="pt")
    with torch.no_grad():
        ids = torch.tensor([tokens])
        hidden = base.model.language_model(ids, use_cache=False).last_hidden_state
        text_logits = base(ids, use_cache=False).logits
        features = torch.cat(base.model.get_image_features(**pixels).pooler_output)
        image_tokens = [
            tokens[0],
            *[model.config.image_token_id] * len(features),
            *tokens[1:],
        ]
        image_logits = base(
            torch.tensor([image_tokens]), **pixels, use_cache=False
        ).logits
    write_json(
        root / "base-reference.json",
        {
            "tokens": tokens,
            "selected": selected,
            "hidden": hidden.flatten().tolist(),
            "text_logits": text_logits.flatten().tolist(),
            "image": "rectangle.png",
            "image_tokens": image_tokens,
            "image_features": features.flatten().tolist(),
            "image_logits": image_logits[0, -1, selected].tolist(),
        },
    )
    activation_cases = []
    # Reuse checkpoint weights to isolate each configured activation, e.g. a ReLU vision MLP.
    for vision_act, projector_act in [
        ("relu", "gelu"),
        ("gelu_pytorch_tanh", "silu"),
        ("gelu", "tanh"),
    ]:
        config = deepcopy(model.config)
        config.vision_config.hidden_act = vision_act
        config.projector_hidden_act = projector_act
        variant = Lfm2VlForConditionalGeneration(config).eval()
        variant.load_state_dict(model.state_dict())
        with torch.no_grad():
            features = torch.cat(
                variant.model.get_image_features(**pixels).pooler_output
            )
        activation_cases.append(
            {
                "vision_act": vision_act,
                "projector_act": projector_act,
                "image_features": features.flatten().tolist(),
            }
        )
    write_json(root / "activation-reference.json", activation_cases)


def evaluate(kind, engine, cases, image_map, samples) -> list[dict]:
    """Compare complete requests, e.g. restore the model context after a truncation case."""
    output = []
    for request in cases:
        images = [image_map[name] for name in request.get("images", [])] or None
        audio = request.get("audio")
        pcm = (
            samples
            if isinstance(audio, str)
            else np.array(audio["pcm16"], dtype=np.int16)
            if audio
            else None
        )
        with torch.no_grad():
            if kind == "omni":
                config = engine.config
                original = config.max_length
                config.max_length = request.get("options", {}).get("max_len", 4096)
                # Compare inference on the same compact JSON text used by Rust, e.g. [true,null].
                state = request["state"]
                if state is not None and not isinstance(state, str):
                    state = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
                result = engine.system_one(
                    state, request["questions"], images=images, audio=pcm
                )
                config.max_length = original
            else:
                result = engine.system_one(
                    request["state"], request["questions"], images=images
                )
        output.append({"request": request, "answers": result["answers"]})
    return output


def release_artifacts(kind: str, root: Path) -> None:
    """Record actual release shapes without downloading GBs of weights, e.g. tied LM head omission."""
    repo, revision = RELEASES[kind]
    source = hf_hub_download(
        repo,
        "config.json",
        revision=revision,
        cache_dir=FIXTURES.parents[1] / ".cache" / "d1-reference",
    )
    root.joinpath("release-config.json").write_text(Path(source).read_text())
    header = parse_safetensors_file_metadata(
        repo, "model.safetensors", revision=revision, token=False
    )
    write_json(
        root / "release-shapes.json",
        {name: info.shape for name, info in header.tensors.items()},
    )


def main() -> None:
    """Keep imports free of network and fixture writes, e.g. ty may inspect this file."""
    torch.set_num_threads(2)
    for kind in RELEASES:
        generate(kind)


if __name__ == "__main__":
    main()
