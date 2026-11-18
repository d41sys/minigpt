import io
import json
import math
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from minigpt import evaluate
from minigpt.cli import paired_f1_table
from minigpt.evaluate import (
    EvalReport,
    MockJudge,
    OllamaJudge,
    compute_perplexity,
    exact_match,
    holm_adjust,
    normalize_answer,
    paired_bootstrap_test,
    token_f1,
    wilson_interval,
)


def test_normalize_answer_drops_case_punctuation_articles() -> None:
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("an apple, a pear") == "apple pear"


def test_exact_match_after_normalisation() -> None:
    assert exact_match("Paris.", "paris") == 1.0
    assert exact_match("The Paris", "Paris") == 1.0
    assert exact_match("Paris, France", "Paris") == 0.0


def test_token_f1_hand_computed() -> None:
    # pred: cat sat mat (3)   ref: cat sat on mat today (5)   overlap 3
    # P = 1, R = 3/5, F1 = 2 * 0.6 / 1.6 = 0.75
    assert token_f1("the cat sat mat", "the cat sat on mat today") == pytest.approx(0.75)
    assert token_f1("dog", "cat") == 0.0
    assert token_f1("cat cat", "cat") == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_mock_judge_bounds() -> None:
    judge = MockJudge()
    assert judge.score("q", "", "anything") == 1
    assert judge.score("q", "zzz", "the cat sat") == 1
    assert judge.score("q", "the cat sat", "the cat sat") == 5


def _fake_ollama(monkeypatch: pytest.MonkeyPatch, content: str) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    def urlopen(req: Any, timeout: float) -> io.BytesIO:
        sent.append({"url": req.full_url, **json.loads(req.data)})
        return io.BytesIO(json.dumps({"message": {"content": content}}).encode())

    monkeypatch.setattr(evaluate.urllib.request, "urlopen", urlopen)
    return sent


def test_ollama_judge_sends_deterministic_schema_request(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _fake_ollama(monkeypatch, '{"score": 4}')
    judge = OllamaJudge("http://h:1/", "qwen", seed=7)
    assert judge.score("Name a fruit.", "An apple.", "Apple") == 4
    (req,) = sent
    assert req["url"] == "http://h:1/api/chat"
    assert req["model"] == "qwen" and req["think"] is False and req["stream"] is False
    assert req["options"] == {"temperature": 0, "seed": 7, "num_ctx": 4096}
    assert req["format"]["properties"]["score"] == {"type": "integer", "minimum": 1, "maximum": 5}
    prompt = req["messages"][0]["content"]
    assert "Name a fruit." in prompt and "Apple" in prompt and "An apple." in prompt


def test_ollama_judge_rejects_bad_reply_and_skips_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _fake_ollama(monkeypatch, '{"score": 9}')
    judge = OllamaJudge("http://h:1", "qwen")
    with pytest.raises(ValueError, match="1-5"):
        judge.score("q", "something", "ref")
    assert judge.score("q", "   ", "ref") == 1
    assert len(sent) == 1  # the empty response made no call


def test_ollama_judge_retries_server_errors_only(monkeypatch: pytest.MonkeyPatch) -> None:
    codes = [500, 500, None]  # two transient failures, then a reply
    http_error = evaluate.urllib.error.HTTPError

    def urlopen(req: Any, timeout: float) -> io.BytesIO:
        code = codes.pop(0)
        if code is not None:
            raise http_error(req.full_url, code, "err", None, None)  # type: ignore[arg-type]
        return io.BytesIO(json.dumps({"message": {"content": '{"score": 3}'}}).encode())

    monkeypatch.setattr(evaluate.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(evaluate.time, "sleep", lambda s: None)
    assert OllamaJudge("http://h:1", "qwen").score("q", "a", "b") == 3
    codes[:] = [404]
    with pytest.raises(evaluate.urllib.error.HTTPError):
        OllamaJudge("http://h:1", "qwen").score("q", "a", "b")
    assert codes == []  # 404 was not retried


def _report(**kw: float) -> EvalReport:
    base = dict(
        perplexity=32.0,
        vocab_size=1024,
        exact_match=0.2,
        token_f1=0.4,
        judge_mean=3.0,
    )
    base.update(kw)
    return EvalReport(**base)  # type: ignore[arg-type]


def test_perplexity_score_endpoints() -> None:
    assert _report(perplexity=1.0).perplexity_score == 1.0
    assert _report(perplexity=1024.0).perplexity_score == pytest.approx(0.0)
    assert _report(perplexity=32.0).perplexity_score == pytest.approx(0.5)  # log 32 / log 1024
    assert _report(perplexity=math.nan).perplexity_score == 0.0


def test_aggregate_is_weighted_mean() -> None:
    r = _report()  # parts: ppl 0.5, EM 0.2, F1 0.4, judge (3-1)/4 = 0.5
    assert r.aggregate() == pytest.approx((0.5 + 0.2 + 0.4 + 0.5) / 4)
    assert r.aggregate((1.0, 0.0, 0.0, 0.0)) == pytest.approx(0.5)
    assert r.aggregate((0.0, 1.0, 3.0, 0.0)) == pytest.approx((0.2 + 3 * 0.4) / 4)


def test_holm_adjust_hand_computed() -> None:
    # sorted p: 0.01 (x3), 0.02 (x2), 0.04 (x1) -> 0.03, 0.04, 0.04 (monotone)
    assert holm_adjust([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm_adjust([0.9, 0.6]) == pytest.approx([1.0, 1.0])  # 2 * 0.6 capped, then monotone


def test_paired_bootstrap_test_separates_shift_from_noise() -> None:
    rng = np.random.default_rng(1)
    shifted = rng.normal(0.5, 0.1, 100)
    mean, lo, hi, p = paired_bootstrap_test(shifted)
    assert mean == pytest.approx(shifted.mean()) and 0 < lo < mean < hi and p < 0.001
    centred = rng.normal(0.0, 0.1, 100)
    centred -= centred.mean()
    _, lo, hi, p = paired_bootstrap_test(centred)
    assert lo < 0 < hi and p > 0.5


def test_wilson_interval_hand_computed() -> None:
    # 0/12 has upper bound z^2/n / (1 + z^2/n) = 0.2425 (the README's EM bound)
    assert wilson_interval(0, 12) == pytest.approx((0.0, 0.24249), abs=1e-4)
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(1 - hi) and lo == pytest.approx(0.4038, abs=1e-4)


class _CopyInput(nn.Module):
    """Stub LM: logit 10 on the input token, so loss is ~0 iff target == input."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.one_hot(x, 20).float() * 10, None


def test_compute_perplexity_is_token_mean_over_uneven_batches() -> None:
    torch.manual_seed(0)
    x = torch.randint(0, 20, (2, 4, 8))
    y = x.clone()  # batch 0: near-zero loss, 4 active tokens
    y[0, :, 1:] = -100
    y[1] = (x[1] + 1) % 20  # batch 1: loss ~10 per token, 28 active tokens
    y[1, 0, 4:] = -100
    batches = [(x[0], y[0]), (x[1], y[1])]
    cpu = torch.device("cpu")
    logits = torch.cat([_CopyInput()(b)[0].view(-1, 20) for b, _ in batches])
    targets = torch.cat([t.view(-1) for _, t in batches])
    expected = math.exp(F.cross_entropy(logits, targets, ignore_index=-100).item())
    got = compute_perplexity(_CopyInput(), batches, cpu)
    assert got == pytest.approx(expected, rel=1e-5)
    # a mean of per-batch means would land near exp(5), far from the token mean
    first = math.exp(F.cross_entropy(logits[:32], targets[:32], ignore_index=-100).item())
    got_first = compute_perplexity(_CopyInput(), batches, cpu, max_batches=1)
    assert got_first == pytest.approx(first, rel=1e-5)
    assert got_first < expected / 10


def test_paired_f1_table_verdicts() -> None:
    base = np.linspace(0.2, 0.6, 60)
    sign = np.where(np.arange(60) % 2 == 0, 1.0, -1.0)
    f1_by_name = {
        "base": base.tolist(),
        "better": (base + 0.2 + 0.02 * sign).tolist(),
        "worse": (base - 0.2 + 0.02 * sign).tolist(),
        "noisy": (base + 0.3 * sign).tolist(),  # mean diff exactly 0
    }
    rows = {r["model"]: r for r in paired_f1_table(f1_by_name, "base", n_boot=500)}
    assert set(rows) == {"better", "worse", "noisy"}
    assert rows["better"]["verdict"] == "different (better)"
    assert rows["worse"]["verdict"] == "different (worse)"
    assert rows["noisy"]["verdict"] == "no detectable difference"
    # f1_first averages only the first DEFAULT_EVAL_N rows, f1 all of them
    better = f1_by_name["better"]
    assert rows["better"]["f1_first"] == pytest.approx(float(np.mean(better[:50])))
    assert rows["better"]["f1"] == pytest.approx(float(np.mean(better)))
    assert rows["better"]["f1_first"] != pytest.approx(rows["better"]["f1"])
