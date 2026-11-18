import sys
from types import ModuleType

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


def _hf_state(model: GPTModel, drop: str | None = None) -> dict[str, torch.Tensor]:
    """`model` in HF GPT-2 layout: `h.N.` keys, Conv1D weights stored transposed."""
    tensors: dict[str, torch.Tensor] = {}
    for key, val in model.state_dict().items():
        if key == "lm_head.weight" or key == drop:
            continue
        if key.endswith(CONV1D_WEIGHT_PATTERNS):
            val = val.t()
        tensors[key.replace("blocks.", "h.", 1)] = val.clone()
    return tensors


def _serve_state(monkeypatch: pytest.MonkeyPatch, tensors: dict[str, torch.Tensor]) -> None:
    """Make `from safetensors.torch import load_file` return `tensors` (works without the extra)."""
    mod = ModuleType("safetensors.torch")
    mod.load_file = lambda _path: tensors
    monkeypatch.setitem(sys.modules, "safetensors", ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.torch", mod)


def test_load_hf_gpt2_weights_roundtrip_and_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    src = GPTModel(TINY)
    _serve_state(monkeypatch, _hf_state(src))
    dst = load_hf_gpt2_weights(GPTModel(TINY), safetensors_path="unused")
    for key, val in src.state_dict().items():
        assert torch.equal(dst.state_dict()[key], val), key

    _serve_state(monkeypatch, _hf_state(src, drop="blocks.1.mlp.c_fc.weight"))
    with pytest.raises(ValueError, match="missing"):
        load_hf_gpt2_weights(GPTModel(TINY), safetensors_path="unused")


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
