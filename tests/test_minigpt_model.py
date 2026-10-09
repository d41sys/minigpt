import pytest
import torch
from minigpt.data import SlidingWindowDataset
from minigpt.model import GPTConfig, GPTModel, count_parameters
from minigpt.pretrained import map_hf_key_to_minigpt
from minigpt.train import get_lr_multiplier

TINY = GPTConfig(vocab_size=64, context_length=16, d_model=32, num_layers=2, num_heads=4)


def test_sliding_window_shapes_and_shift() -> None:
    ds = SlidingWindowDataset(list(range(20)), context_length=4, stride=2)
    assert len(ds) == 8
    x, y = ds[1]
    assert x.tolist() == [2, 3, 4, 5]
    assert y.tolist() == [3, 4, 5, 6]


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_causal_mask_future_does_not_change_past(use_sdpa: bool) -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(**{**TINY.__dict__, "use_sdpa": use_sdpa})
    model = GPTModel(cfg).eval()
    a = torch.randint(0, 64, (1, 10))
    b = a.clone()
    b[0, 6:] = torch.randint(0, 64, (4,))
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[0, :6], lb[0, :6], atol=1e-5)
    assert not torch.allclose(la[0, 6:], lb[0, 6:])


def test_sdpa_matches_manual_attention() -> None:
    torch.manual_seed(0)
    manual = GPTModel(TINY).eval()
    sdpa = GPTModel(GPTConfig(**{**TINY.__dict__, "use_sdpa": True})).eval()
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randint(0, 64, (2, 12))
    assert torch.allclose(manual(x)[0], sdpa(x)[0], atol=1e-5)


def test_gpt2_124m_param_count_tied() -> None:
    with torch.device("meta"):
        model = GPTModel(GPTConfig())
    assert model.lm_head.weight is model.wte.weight
    assert count_parameters(model) == 124_439_808


def test_generate_greedy_is_deterministic() -> None:
    torch.manual_seed(0)
    model = GPTModel(TINY)
    x = torch.tensor([[1, 2, 3]])
    out1 = model.generate(x, 20, temperature=0.0)
    out2 = model.generate(x, 20, temperature=0.0)
    assert out1.shape == (1, 23)
    assert torch.equal(out1, out2)
    sampled = model.generate(x, 5, temperature=1.0, top_k=1)
    assert torch.equal(sampled, out1[:, :8])


def test_lr_schedule_warmup_and_cosine() -> None:
    assert get_lr_multiplier(0, 10, 100) == pytest.approx(0.1)
    assert get_lr_multiplier(9, 10, 100) == pytest.approx(1.0)
    assert get_lr_multiplier(100, 10, 100) == pytest.approx(0.1)
    assert get_lr_multiplier(55, 10, 100) == pytest.approx(0.55)


def test_hf_key_mapping() -> None:
    assert map_hf_key_to_minigpt("h.3.attn.c_attn.weight") == ("blocks.3.attn.c_attn.weight", True)
    assert map_hf_key_to_minigpt("transformer.h.0.ln_1.bias") == ("blocks.0.ln_1.bias", False)
    assert map_hf_key_to_minigpt("wte.weight") == ("wte.weight", False)


def test_overfit_single_batch() -> None:
    torch.manual_seed(0)
    model = GPTModel(TINY)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 64, (4, 16))
    y = torch.roll(x, -1, dims=1)
    loss_value = float("inf")
    for _ in range(200):
        opt.zero_grad()
        _, loss = model(x, targets=y)
        assert loss is not None
        loss.backward()
        opt.step()
        loss_value = loss.item()
        if loss_value < 0.1:
            break
    assert loss_value < 0.1
