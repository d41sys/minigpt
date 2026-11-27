from typing import Any

import torch

from minigpt.model import GPTConfig, GPTModel
from minigpt.opd import train_opd
from minigpt.reason import evaluate_pass_at_k, reasoning_tokenizer

CFG = GPTConfig(vocab_size=258, context_length=64, d_model=32, num_layers=2, num_heads=4)


def _model(seed: int) -> GPTModel:
    torch.manual_seed(seed)
    return GPTModel(CFG).eval()


def _run(patience: int) -> dict[str, Any]:
    return train_opd(
        _model(2),
        _model(3),
        steps=4,
        batch_size=2,
        digits=1,
        max_new_tokens=8,
        eval_every=1,
        eval_tasks=4,
        device=torch.device("cpu"),
        patience=patience,
    )


def test_patience_stops_when_greedy_accuracy_never_improves() -> None:
    # A random model never answers correctly: accuracy is 0 at step 0 and 1, so one stale eval.
    res = _run(patience=1)
    assert res["steps_done"] == 1 and len(res["history"]) == 2


def test_no_patience_runs_every_step() -> None:
    res = _run(patience=0)
    assert res["steps_done"] == 4 and len(res["history"]) == 5


def test_pass_at_k_shape_and_range() -> None:
    p = evaluate_pass_at_k(_model(0), reasoning_tokenizer(), 3, 2, max_new_tokens=8, ops="+-")
    assert sorted(p) == [1, 2, 3] and all(0.0 <= v <= 1.0 for v in p.values())
