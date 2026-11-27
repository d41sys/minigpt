"""LoRA: zero-init B is a no-op, merging is exact, only A/B train, the checkpoint stays plain."""

from pathlib import Path

import pytest
import torch

from minigpt.lora import LoRALinear, apply_lora, merge_lora
from minigpt.model import GPTConfig, GPTModel
from minigpt.train import load_model, save_checkpoint

TINY = GPTConfig(vocab_size=50, context_length=8, d_model=16, num_layers=2, num_heads=2)
X = torch.randint(0, 50, (2, 8))


def lora_model(targets: str = "all", embed: bool = False) -> tuple[GPTModel, GPTModel, int]:
    torch.manual_seed(0)
    base = GPTModel(TINY).eval()
    model = GPTModel(TINY).eval()
    model.load_state_dict(base.state_dict())
    return base, model, apply_lora(model, 2, alpha=4.0, targets=targets, embed=embed)


def randomize_b(model: GPTModel) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            torch.nn.init.normal_(m.B, std=0.1)  # B = 0 would leave A without gradient


def test_zero_b_leaves_the_output_unchanged() -> None:
    base, model, _ = lora_model()
    assert torch.equal(model(X)[0], base(X)[0])


def test_merged_forward_equals_lora_forward() -> None:
    _, model, _ = lora_model()
    randomize_b(model)
    out = model(X)[0]
    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    torch.testing.assert_close(model(X)[0], out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("targets", "wrapped", "params"), [("all", 8, 1024), ("attn", 4, 384)])
def test_only_a_and_b_get_gradients(targets: str, wrapped: int, params: int) -> None:
    _, model, trainable = lora_model(targets)
    randomize_b(model)
    model.train()
    model(X, targets=X)[1].backward()
    named = dict(model.named_parameters())
    assert all(p.grad is not None for n, p in named.items() if n.endswith((".A", ".B")))
    assert all(p.grad is None for n, p in named.items() if not n.endswith((".A", ".B")))
    assert sum(isinstance(m, LoRALinear) for m in model.modules()) == wrapped
    assert trainable == params  # rank 2: A is 2 x in, B is out x 2


def test_embed_flag_also_trains_the_tied_embedding() -> None:
    _, model, trainable = lora_model(embed=True)
    randomize_b(model)
    model.train()
    model(X, targets=X)[1].backward()
    assert model.wte.weight.grad is not None and model.lm_head.weight is model.wte.weight
    assert trainable == 1024 + 50 * 16  # vocab x d_model, shared with the LM head


def test_merged_checkpoint_loads_as_a_plain_model(tmp_path: Path) -> None:
    _, model, _ = lora_model()
    randomize_b(model)
    out = model(X)[0]
    merge_lora(model)
    save_checkpoint(tmp_path / "ckpt.pt", model, None, step=1, val_loss=0.0)
    torch.testing.assert_close(load_model(tmp_path / "ckpt.pt").eval()(X)[0], out)
