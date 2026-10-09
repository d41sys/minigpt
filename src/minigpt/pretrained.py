"""Loading pretrained GPT-2 weights from HuggingFace safetensors into MiniGPT."""

from __future__ import annotations

from pathlib import Path

import torch

from minigpt.model import GPTConfig, GPTModel

CONV1D_WEIGHT_PATTERNS: tuple[str, ...] = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def get_gpt2_config(model_type: str = "gpt2") -> GPTConfig:
    """Return GPTConfig corresponding to GPT-2 configurations."""
    configs = {
        "gpt2": GPTConfig(
            vocab_size=50257, context_length=1024, d_model=768, num_layers=12, num_heads=12
        ),
        "gpt2-medium": GPTConfig(
            vocab_size=50257, context_length=1024, d_model=1024, num_layers=24, num_heads=16
        ),
        "gpt2-large": GPTConfig(
            vocab_size=50257, context_length=1024, d_model=1280, num_layers=36, num_heads=20
        ),
        "gpt2-xl": GPTConfig(
            vocab_size=50257, context_length=1024, d_model=1600, num_layers=48, num_heads=25
        ),
    }
    if model_type not in configs:
        raise ValueError(f"Unknown model_type {model_type}, choose from {list(configs.keys())}")
    return configs[model_type]


def map_hf_key_to_minigpt(key: str) -> tuple[str, bool]:
    """Map a HuggingFace parameter key to its MiniGPT equivalent.

    Returns:
        (minigpt_key, needs_transpose)
    """
    cleaned = key
    if cleaned.startswith("transformer."):
        cleaned = cleaned[len("transformer.") :]

    needs_transpose = any(cleaned.endswith(p) for p in CONV1D_WEIGHT_PATTERNS)

    if cleaned.startswith("h."):
        parts = cleaned.split(".")
        layer_idx = parts[1]
        rest = ".".join(parts[2:])
        return f"blocks.{layer_idx}.{rest}", needs_transpose

    return cleaned, needs_transpose


def load_hf_gpt2_weights(
    model: GPTModel,
    safetensors_path: str | Path | None = None,
    repo_id: str = "gpt2",
    device: torch.device | None = None,
) -> GPTModel:
    """Load pretrained weights from safetensors file or HuggingFace Hub.

    Args:
        model: Target GPTModel initialized with matching config
        safetensors_path: Optional local path to model.safetensors
        repo_id: HuggingFace model repo ID (default "gpt2")
        device: Target torch device
    """
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise ImportError("safetensors is required: install the `minigpt[hf]` extra") from e

    if safetensors_path is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError("huggingface_hub is required: install the `minigpt[hf]` extra") from e
        file_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    else:
        file_path = str(safetensors_path)

    state_dict = load_file(file_path)
    model_state = model.state_dict()

    loaded_keys: list[str] = []
    skipped_keys: list[str] = []

    with torch.no_grad():
        for hf_key, hf_val in state_dict.items():
            minigpt_key, needs_transpose = map_hf_key_to_minigpt(hf_key)

            if minigpt_key == "lm_head.weight":
                # lm_head is weight-tied to wte, skip redundant load
                continue

            if minigpt_key not in model_state:
                skipped_keys.append(hf_key)
                continue

            tensor = hf_val
            if needs_transpose:
                tensor = tensor.t()

            target_shape = model_state[minigpt_key].shape
            if tensor.shape != target_shape:
                raise ValueError(
                    f"Shape mismatch for {minigpt_key}: expected {target_shape}, got {tensor.shape}"
                )

            model_state[minigpt_key].copy_(tensor)
            loaded_keys.append(minigpt_key)

    if device is not None:
        model.to(device)

    return model


def from_pretrained(
    model_type: str = "gpt2",
    safetensors_path: str | Path | None = None,
    device: torch.device | None = None,
) -> GPTModel:
    """Instantiate and load a pretrained GPT-2 model."""
    config = get_gpt2_config(model_type)
    model = GPTModel(config)
    return load_hf_gpt2_weights(
        model,
        safetensors_path=safetensors_path,
        repo_id=model_type,
        device=device,
    )
