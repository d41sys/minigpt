"""OPD rollouts batch tasks per (sampler, prompt length); results must match the per-task path."""

import random
from collections.abc import Mapping

import pytest
import torch

from minigpt import opd
from minigpt.model import GPTConfig, GPTModel
from minigpt.opd import _rollouts, evaluate_student
from minigpt.reason import build_batch, reasoning_tokenizer, sample_group, sample_task

BASE = dict(vocab_size=258, context_length=64, d_model=32, num_layers=2, num_heads=4)
CSA = dict(attn_type="csa", csa_block=2, csa_topk=2, csa_window=3, n_experts=4, hc_streams=2)
CPU = torch.device("cpu")


def model(seed: int, **extra: object) -> GPTModel:
    torch.manual_seed(seed)
    return GPTModel(GPTConfig(**{**BASE, **extra})).eval()  # type: ignore[arg-type]


def tasks(n: int = 12) -> list:
    rng = random.Random(0)
    return [sample_task(rng, rng.randint(1, 3), "+-") for _ in range(n)]


def per_task(samplers: Mapping[str, GPTModel], ts: list, group: int, temp: float):  # type: ignore[no-untyped-def]
    """The pre-batching reference: one `sample_group` call per task."""
    tok = reasoning_tokenizer()
    prompts, comps, ops = [], [], []
    for t in ts:
        ids = tok.encode(t.prompt)
        for c in sample_group(samplers[t.op], ids, group, 10, tok.eos_token_id, temp):
            prompts.append(ids)
            comps.append(c)
            ops.append(t.op)
    x, y = build_batch(prompts, comps, tok.special_to_id["<|pad|>"])
    return x, y, comps, ops


@pytest.mark.parametrize("extra", [{}, CSA], ids=["dense", "v4"])
@pytest.mark.parametrize("group", [1, 3])
def test_greedy_batched_rollouts_equal_per_task_rollouts(extra: dict, group: int) -> None:
    plus, minus = model(1, **extra), model(2, **extra)  # one sampler per op, like off-policy
    for samplers in ({"+": plus, "-": minus}, {"+": plus, "-": plus}):
        ts = tasks()
        got = _rollouts(samplers, ts, group, 10, 0.0, CPU)
        want = per_task(samplers, ts, group, 0.0)
        assert got[2] == want[2] and got[3] == want[3]
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_one_generate_call_per_sampler_and_prompt_length(monkeypatch: pytest.MonkeyPatch) -> None:
    m = model(1)
    calls: list[tuple[int, int]] = []
    real = GPTModel.generate

    def spy(self: GPTModel, idx: torch.Tensor, *a: object, **k: object) -> torch.Tensor:
        calls.append((idx.size(0), idx.size(1)))
        return real(self, idx, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(GPTModel, "generate", spy)
    ts = tasks(20)
    _rollouts({"+": m, "-": m}, ts, 2, 6, 1.0, CPU)
    lengths = {len(reasoning_tokenizer().encode(t.prompt)) for t in ts}
    assert sorted(c[1] for c in calls) == sorted(lengths)  # 1 call per prompt length, not 20
    assert sum(c[0] for c in calls) == 20 * 2  # every task x group row is generated once


def test_sampled_rollouts_keep_order_ops_and_cut_at_first_eos() -> None:
    m = model(3)
    ts = tasks(9)
    tok = reasoning_tokenizer()
    torch.manual_seed(0)
    x, y, comps, ops = _rollouts({"+": m, "-": m}, ts, 2, 8, 1.0, CPU)
    assert ops == [t.op for t in ts for _ in range(2)]
    assert x.size(0) == y.size(0) == len(comps) == 18
    assert all(tok.eos_token_id not in c[:-1] and 1 <= len(c) <= 8 for c in comps)
    # each row's prompt prefix is its task's prompt
    for i, t in enumerate(ts):
        prompt = tok.encode(t.prompt)
        assert x[2 * i, : len(prompt)].tolist() == prompt == x[2 * i + 1, : len(prompt)].tolist()


def test_evaluate_student_is_unchanged_by_batching() -> None:
    student = model(1, **CSA)
    teachers = {"+": model(2), "-": model(3)}
    ts = tasks(10)
    got = evaluate_student(student, teachers, ts, 10, CPU)
    x, y, _, ops = per_task({"+": student, "-": student}, ts, 1, 0.0)
    want, _ = opd.routed_opd_loss(student, teachers, x, y, ops, "reverse")
    assert got["kl"] == pytest.approx(want.item(), rel=1e-4)
