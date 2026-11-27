import dataclasses
import random

import pytest
import torch

from minigpt.merge import task_arithmetic
from minigpt.model import GPTConfig, GPTModel
from minigpt.reason import make_task, reasoning_tokenizer
from minigpt.rft import sample_by_op, sample_tasks

TINY = GPTConfig(vocab_size=258, context_length=64, d_model=16, num_layers=1, num_heads=2)


def _model(seed: int, cfg: GPTConfig = TINY) -> GPTModel:
    torch.manual_seed(seed)
    return GPTModel(cfg).eval()


def _same(a: GPTModel, b: GPTModel) -> bool:
    pairs = zip(a.state_dict().values(), b.state_dict().values(), strict=True)
    return all(torch.allclose(x, y, atol=1e-6) for x, y in pairs)


def test_zero_scale_is_the_base_and_one_expert_at_scale_one_is_the_expert() -> None:
    base, e1, e2 = _model(0), _model(1), _model(2)
    assert _same(task_arithmetic(base, [e1, e2], 0.0), base)
    assert _same(task_arithmetic(base, [e1], 1.0), e1)  # also checks the tied head, once


def test_default_scale_averages_the_deltas() -> None:
    base, e1, e2 = _model(0), _model(1), _model(2)
    merged = task_arithmetic(base, [e1, e2]).state_dict()
    for k, w in base.state_dict().items():
        want = w + 0.5 * ((e1.state_dict()[k] - w) + (e2.state_dict()[k] - w))
        torch.testing.assert_close(merged[k], want)


def test_mismatched_configs_and_no_experts_are_refused() -> None:
    other = dataclasses.replace(TINY, d_model=32)
    with pytest.raises(ValueError, match="differs"):
        task_arithmetic(_model(0), [_model(1, other)])
    with pytest.raises(ValueError, match="at least one"):
        task_arithmetic(_model(0), [])


def test_sample_by_op_samples_each_task_from_its_ops_model() -> None:
    tok = reasoning_tokenizer()
    plus, minus = _model(1), _model(2)
    tasks = [make_task(random.Random(i), 2, "+-"[i % 2]) for i in range(6)]
    got = sample_by_op({"+": plus, "-": minus}, tok, tasks, k=1, max_new_tokens=6, temperature=0.0)
    want = sample_tasks(plus, tok, tasks[0::2], 1, 6, 0.0) + sample_tasks(
        minus, tok, tasks[1::2], 1, 6, 0.0
    )
    assert [s.texts for s in got] == [s.texts for s in want]
    assert [s.task for s in got] == [s.task for s in want]
