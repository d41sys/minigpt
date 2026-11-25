"""CSA decode cache: incremental prefill/decode equals the full forward and uncached generate."""

import itertools
from typing import Any

import pytest
import torch

from minigpt.model import GPTConfig, GPTModel, LayerCache
from minigpt.reason import sample_group

BASE: dict[str, Any] = dict(
    vocab_size=64, context_length=16, d_model=32, num_layers=2, num_heads=4, d_ff=64
)
CSA: dict[str, Any] = dict(attn_type="csa", csa_block=2, csa_topk=2, csa_window=3)
V4: dict[str, Any] = dict(
    **CSA, n_experts=4, moe_top_k=2, moe_dense_first=True, hc_streams=4, mtp_depth=1
)
VARIANTS = {
    "dense-mlp": CSA,
    "block1": {**CSA, "csa_block": 1},
    "block3-win1": {**CSA, "csa_block": 3, "csa_window": 1},
    "block4-wide-window": {**CSA, "csa_block": 4, "csa_window": 40, "csa_topk": 9},
    "v4": V4,
    "v4-rope": {**V4, "pos_encoding": "rope"},
}


def model_for(extra: dict[str, Any], seed: int = 0) -> GPTModel:
    torch.manual_seed(seed)
    return GPTModel(GPTConfig(**{**BASE, **extra})).eval()


@torch.no_grad()
def cached_logits(model: GPTModel, idx: torch.Tensor, cuts: list[int]) -> torch.Tensor:
    """Feed `idx` through the cache in chunks ending at `cuts`; concatenate all logits."""
    cache: list[LayerCache] = [{} for _ in model.blocks]
    edges = [0, *cuts, idx.size(1)]
    return torch.cat(
        [model(idx[:, a:b], cache=cache)[0] for a, b in itertools.pairwise(edges)], dim=1
    )


@pytest.mark.parametrize("extra", VARIANTS.values(), ids=VARIANTS.keys())
def test_every_prefill_cut_then_single_token_steps_matches_full_forward(
    extra: dict[str, Any],
) -> None:
    model = model_for(extra)
    idx = torch.randint(0, 64, (3, 16))  # batch of 3, fills the whole context
    full = model(idx)[0]
    for cut in range(1, 16):  # prefill idx[:cut], then one token at a time
        got = cached_logits(model, idx, [*range(cut, 16)])
        torch.testing.assert_close(got, full, atol=1e-5, rtol=1e-4, msg=f"cut={cut}")


@pytest.mark.parametrize("extra", [CSA, V4], ids=["csa", "v4"])
@pytest.mark.parametrize("cuts", [[5], [2, 4, 9], [3, 7, 8, 13], [1, 2, 3, 4, 5, 6]])
def test_multi_token_chunks_across_block_boundaries_match_full_forward(
    extra: dict[str, Any], cuts: list[int]
) -> None:
    model = model_for(extra, seed=1)
    idx = torch.randint(0, 64, (2, 16))
    torch.testing.assert_close(cached_logits(model, idx, cuts), model(idx)[0], atol=1e-5, rtol=1e-4)


def test_cache_holds_finished_blocks_and_the_unfinished_tail() -> None:
    model = model_for(CSA)  # block m = 2
    cache: list[LayerCache] = [{} for _ in model.blocks]
    for n in range(1, 8):
        model(torch.randint(0, 64, (1, 1)), cache=cache)
        layer = cache[0]
        assert layer["k"].size(2) == n
        assert layer["pending"].size(1) == n % 2  # always fewer than m tokens pending
        if n < 2:
            assert "comp" not in layer
        else:
            assert layer["comp"].size(1) == layer["index"].size(1) == n // 2


@pytest.mark.parametrize("extra", VARIANTS.values(), ids=VARIANTS.keys())
@pytest.mark.parametrize("prompt_len", [1, 4, 15, 16, 19])  # 16 = full context, 19 = over it
def test_generate_matches_uncached_for_batches(extra: dict[str, Any], prompt_len: int) -> None:
    model = model_for(extra)
    idx = torch.randint(0, 64, (3, prompt_len))
    cached = model.generate(idx, 20, temperature=0.0)  # slides past the context limit
    assert cached.shape == (3, prompt_len + 20)
    assert torch.equal(cached, model.generate(idx, 20, temperature=0.0, use_cache=False))


def test_sampled_generate_matches_uncached_with_the_same_seed() -> None:
    model = model_for(V4)
    idx = torch.randint(0, 64, (4, 5))
    torch.manual_seed(7)
    cached = model.generate(idx, 10, temperature=1.0, top_k=8)
    torch.manual_seed(7)
    assert torch.equal(cached, model.generate(idx, 10, temperature=1.0, top_k=8, use_cache=False))


def test_decode_steps_feed_one_token_not_the_prefix() -> None:
    model = model_for(V4)
    seen: list[int] = []
    model.blocks[1].attn.register_forward_pre_hook(lambda _m, args: seen.append(args[0].size(1)))
    model.generate(torch.randint(0, 64, (2, 4)), 8, temperature=0.0)  # 12 <= context: no slide
    assert seen == [4] + [1] * 7  # the prompt once, then single tokens (the last is not fed)


def test_sample_group_uses_the_cache_and_matches_uncached_rollouts() -> None:
    model = model_for(V4)
    prompt, eos = [3, 9, 5], 0
    seen: list[int] = []
    model.blocks[0].attn.register_forward_pre_hook(lambda _m, args: seen.append(args[0].size(1)))
    torch.manual_seed(3)
    group = sample_group(model, prompt, 4, 8, eos, temperature=1.0)
    assert seen[0] == len(prompt) and set(seen[1:]) <= {1}
    seen.clear()
    torch.manual_seed(3)
    x = torch.tensor([prompt] * 4)
    ref = model.generate(x, 8, temperature=1.0, eos_token_id=eos, use_cache=False)
    expected = [r[: r.index(eos) + 1] if eos in r else r for r in ref[:, 3:].tolist()]
    assert group == expected


def test_training_forward_is_unchanged_by_the_cache_path() -> None:
    model = GPTModel(GPTConfig(**{**BASE, **V4})).train()
    x = torch.randint(0, 64, (2, 12))
    _, loss = model(x, x)
    assert loss is not None
    loss.backward()
    attn = model.blocks[0].attn
    assert attn.index_q.weight.grad is not None and attn.index_q.weight.grad.abs().sum() > 0
