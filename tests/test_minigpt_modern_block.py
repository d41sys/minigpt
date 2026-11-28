import pytest
import torch
import torch.nn as nn

from minigpt.model import GPTConfig, GPTModel, count_parameters

MODERN = {"norm": "rmsnorm", "mlp_type": "swiglu", "pos_encoding": "rope"}


def make(**kw: object) -> GPTModel:
    base = {"vocab_size": 50, "context_length": 32, "d_model": 32, "num_layers": 2, "num_heads": 4}
    return GPTModel(GPTConfig(**{**base, **kw}))  # type: ignore[arg-type]


def expected_params(c: GPTConfig) -> int:
    d, kv = c.d_model, c.num_kv_heads or c.num_heads
    qkv = d + 2 * kv * (d // c.num_heads)
    attn = d * qkv + qkv + d * d + d
    width = 2 if c.mlp_type == "swiglu" else 1
    mlp = d * width * c.d_ff + width * c.d_ff + c.d_ff * d + d  # type: ignore[operator]
    norm = d if c.norm == "rmsnorm" else 2 * d
    pos = c.context_length * d if c.pos_encoding == "learned" else 0
    return c.vocab_size * d + pos + c.num_layers * (attn + mlp + 2 * norm) + norm


VARIANTS = [{}, MODERN, {**MODERN, "num_kv_heads": 2}, {"norm": "rmsnorm", "num_kv_heads": 1}]


@pytest.mark.parametrize("kw", VARIANTS)
def test_parameter_count_formula_and_shapes(kw: dict[str, object]) -> None:
    model = make(**kw)
    assert count_parameters(model) == expected_params(model.config)
    logits, loss = model(torch.randint(0, 50, (3, 7)), targets=torch.randint(0, 50, (3, 7)))
    assert logits.shape == (3, 7, 50) and loss is not None and loss.isfinite()


@pytest.mark.parametrize("kw", VARIANTS[1:])
def test_cached_generation_equals_uncached(kw: dict[str, object]) -> None:
    torch.manual_seed(0)
    model = make(**kw).eval()
    prompt = torch.randint(0, 50, (2, 5))
    cached = model.generate(prompt, 12, temperature=0.0, use_cache=True)
    uncached = model.generate(prompt, 12, temperature=0.0, use_cache=False)
    assert torch.equal(cached, uncached)
    cache: list[dict[str, torch.Tensor]] = [{} for _ in model.blocks]
    model(prompt[:, :4], cache=cache)
    step, _ = model(prompt[:, 4:], cache=cache)
    full, _ = model(prompt)
    assert torch.allclose(step[:, -1], full[:, -1], atol=1e-5)
    # grouped-query attention caches only the kv heads
    assert cache[0]["k"].shape[1] == (kw.get("num_kv_heads") or 4)


def test_defaults_are_the_gpt2_block() -> None:
    model = make()
    assert isinstance(model.ln_f, nn.LayerNorm) and model.config.num_kv_heads == 0
    assert model.blocks[0].mlp.c_fc.weight.shape == (4 * 32, 32)
    assert model.blocks[0].attn.c_attn.weight.shape == (3 * 32, 32)


def test_swiglu_default_width_rounds_up_to_64() -> None:
    assert GPTConfig(d_model=512, num_heads=8, mlp_type="swiglu").d_ff == 1408
    assert GPTConfig(d_model=512, num_heads=8).d_ff == 2048


@pytest.mark.parametrize(
    "kw",
    [
        {"norm": "batch"},
        {"mlp_type": "relu"},
        {"num_kv_heads": 3},
        {"attn_type": "csa", "num_kv_heads": 2},
    ],
)
def test_invalid_options_raise(kw: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make(**kw)
