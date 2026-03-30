# Copyright © 2023 Apple Inc.

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from . import whisper


def _find_weights_file(model_path: Path) -> Path:
    """Find weights file, preferring SafeTensors over NPZ for security and speed."""
    safetensors = model_path / "model.safetensors"
    if safetensors.exists():
        return safetensors
    npz = model_path / "weights.npz"
    if npz.exists():
        return npz
    raise FileNotFoundError(
        f"No weights file found in {model_path}. "
        "Expected 'model.safetensors' or 'weights.npz'."
    )


def load_model(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float32,
) -> whisper.Whisper:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

    with open(str(model_path / "config.json"), "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)

    model_args = whisper.ModelDimensions(**config)

    weights_file = _find_weights_file(model_path)
    weights = mx.load(str(weights_file))
    weights = tree_unflatten(list(weights.items()))

    model = whisper.Whisper(model_args, dtype)

    if quantization is not None:
        # Flatten weights dict for scale lookup (fixes issue #11, #24)
        flat_weights = {}

        def _flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(v, key)
                else:
                    flat_weights[key] = v

        _flatten(weights)

        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in flat_weights
        )
        nn.quantize(model, **quantization, class_predicate=class_predicate)

    model.update(weights)
    # mx.eval materializes all model parameters on the GPU
    mx.eval(model.parameters())  # noqa: S307 — MLX GPU eval, not Python eval
    return model
