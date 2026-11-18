from typing import Any

import pytest
import torch
import torch.nn as nn

from minigpt import evaluate
from minigpt.bpe import BPETokenizer
from minigpt.cli import sample_response
from minigpt.evaluate import RANK_DISTRACTOR_OFFSET, likelihood_ranking

CPU = torch.device("cpu")


def _rows(*outputs: str) -> list[dict[str, str]]:
    return [{"instruction": f"Say {i}.", "input": "", "output": o} for i, o in enumerate(outputs)]


def _bytes_tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    tok.train("abcd", target_vocab_size=258)  # no merges: one token per byte, plus EOS
    return tok


class _Uniform(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.zeros(*x.shape, 300), None


def test_ranking_counts_wins_per_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    # Response lengths in tokens (bytes + EOS): 3, 9, 5, 2; the distractor of row i is row i+1.
    rows = _rows("aa", "bbbbbbbb", "cccc", "d")
    assert RANK_DISTRACTOR_OFFSET % len(rows) == 1
    scripted = iter(
        [
            (-3.0, -18.0),  # summed and per-token win
            (-9.0, -6.0),  # summed loses (longer answer), per-token wins (-1.0 vs -1.2)
            (-5.0, -9.0),  # both win
            (-3.0, -3.0),  # summed tie loses, per-token loses (-1.5 vs -1.0)
        ]
    )
    monkeypatch.setattr(
        evaluate, "compute_sequence_log_probs", lambda *_: torch.tensor(next(scripted))
    )
    assert likelihood_ranking(nn.Module(), _bytes_tokenizer(), rows, CPU, 512) == (2, 3)


def test_uniform_model_ranks_by_length_only() -> None:
    # With no information in the logits, summed log-prob just prefers the shorter answer.
    rows = _rows("alpha beta gamma", "one two three four five six", "red green blue pink", "cat")
    shorter = sum(
        len(r["output"]) < len(rows[(i + RANK_DISTRACTOR_OFFSET) % len(rows)]["output"])
        for i, r in enumerate(rows)
    )
    assert 0 < shorter < len(rows)
    wins_sum, _ = likelihood_ranking(_Uniform(), _bytes_tokenizer(), rows, CPU, 512)
    assert wins_sum == shorter


class _FixedGenerate:
    def __init__(self, tail: list[int]) -> None:
        self.tail = tail

    def generate(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.cat([x, torch.tensor([self.tail])], dim=1)


@pytest.mark.parametrize(("emit_eos", "stopped"), [(True, True), (False, False)])
def test_sample_response_reports_stop(emit_eos: bool, stopped: bool) -> None:
    tok = _bytes_tokenizer()
    tail = [tok.inv_vocab[b"a"], tok.eos_token_id if emit_eos else tok.inv_vocab[b"b"]]
    model: Any = _FixedGenerate(tail)
    text, did_stop = sample_response(model, tok, _rows("x")[0], CPU)
    assert (text, did_stop) == ("a" if emit_eos else "ab", stopped)
