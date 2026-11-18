from pathlib import Path

import pytest
import torch

from minigpt.model import GPTConfig, GPTModel
from minigpt.pretrained import (
    CONV1D_WEIGHT_PATTERNS,
    from_pretrained,
    load_hf_gpt2_weights,
    map_hf_key_to_minigpt,
)

TINY = GPTConfig(vocab_size=50, context_length=16, d_model=16, num_layers=2, num_heads=2)


def test_map_hf_key_to_minigpt() -> None:
    assert map_hf_key_to_minigpt("h.3.attn.c_attn.weight") == ("blocks.3.attn.c_attn.weight", True)
    assert map_hf_key_to_minigpt("h.0.attn.c_attn.bias") == ("blocks.0.attn.c_attn.bias", False)
    assert map_hf_key_to_minigpt("transformer.ln_f.bias") == ("ln_f.bias", False)
    assert map_hf_key_to_minigpt("wte.weight") == ("wte.weight", False)


def _write_hf_checkpoint(model: GPTModel, path: Path, drop: str | None = None) -> None:
    """Save `model` in HF GPT-2 layout: `h.N.` keys, Conv1D weights stored transposed."""
    from safetensors.torch import save_file

    tensors: dict[str, torch.Tensor] = {}
    for key, val in model.state_dict().items():
        if key == "lm_head.weight" or key == drop:
            continue
        hf_key = key.replace("blocks.", "h.", 1)
        if key.endswith(CONV1D_WEIGHT_PATTERNS):
            val = val.t()
        tensors[hf_key] = val.contiguous().clone()
    save_file(tensors, str(path))


def test_load_hf_gpt2_weights_roundtrip_and_missing_key(tmp_path: Path) -> None:
    pytest.importorskip("safetensors")
    torch.manual_seed(0)
    src = GPTModel(TINY)
    good = tmp_path / "good.safetensors"
    _write_hf_checkpoint(src, good)
    dst = load_hf_gpt2_weights(GPTModel(TINY), safetensors_path=good)
    for key, val in src.state_dict().items():
        assert torch.equal(dst.state_dict()[key], val), key

    bad = tmp_path / "bad.safetensors"
    _write_hf_checkpoint(src, bad, drop="blocks.1.mlp.c_fc.weight")
    with pytest.raises(ValueError, match="missing"):
        load_hf_gpt2_weights(GPTModel(TINY), safetensors_path=bad)


@pytest.mark.network
@pytest.mark.slow
def test_gpt2_greedy_matches_transformers() -> None:
    transformers = pytest.importorskip("transformers")
    hf_tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    hf_model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").eval()
    ids = hf_tok("Hello, I'm a language model,", return_tensors="pt").input_ids
    with torch.no_grad():
        expected = hf_model.generate(ids, max_new_tokens=10, do_sample=False)
    ours = from_pretrained("gpt2").eval()
    got = ours.generate(ids, max_new_tokens=10, temperature=0.0)
    assert got.tolist() == expected.tolist()
