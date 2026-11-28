import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import minigpt.data as data
from minigpt.bpe import BPETokenizer
from minigpt.data import hf_rows, tokenize_cached
from minigpt.evaluate import score_multiple_choice, to_mc_item


class Favour(nn.Module):
    """Logits that put most of the mass on byte `fav` at every position."""

    def __init__(self, fav: int, vocab: int) -> None:
        super().__init__()
        self.fav, self.vocab = fav, vocab

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits = torch.zeros(*idx.shape, self.vocab)
        logits[..., self.fav] = 10.0
        return logits, None


def test_scorer_on_a_hand_made_set() -> None:
    tok = BPETokenizer()
    tok.initialize()
    model = Favour(ord("a"), tok.vocab_size)
    items = [("Q:", ["a", "b"], 0), ("Q:", ["b", "a"], 1), ("Q:", ["b", "a"], 0)]
    r = score_multiple_choice(model, tok, items, torch.device("cpu"), max_length=16)
    assert r["correct"] == [1, 1, 0] and r["n"] == 3
    assert r["acc"] == pytest.approx(2 / 3) and r["chance"] == 0.5
    assert r["stderr"] == pytest.approx(math.sqrt(2 / 3 * 1 / 3 / 3))
    # an option longer than the window is cut to its last tokens, the scorer still runs
    long = [("Q:", ["a" * 40, "b" * 40], 0)]
    assert score_multiple_choice(model, tok, long, torch.device("cpu"), 16)["acc"] == 1.0


def test_rows_become_items() -> None:
    arc = {
        "question": "Which is wet?",
        "choices": {"text": ["sand", "water"], "label": ["A", "B"]},
        "answerKey": "B",
    }
    assert to_mc_item("arc_easy", arc) == ("Question: Which is wet?\nAnswer:", ["sand", "water"], 1)
    obqa = {"question_stem": "A cat", "choices": arc["choices"], "answerKey": "A"}
    assert to_mc_item("openbookqa", obqa) == ("A cat", ["sand", "water"], 0)
    piqa = {"goal": "g", "sol1": "x", "sol2": "y", "label": 1}
    assert to_mc_item("piqa", piqa) == ("Question: g\nAnswer:", ["x", "y"], 1)
    hella = {
        "activity_label": "Baking",
        "ctx_a": "A man mixes.",
        "ctx_b": "then he [step] adds",
        "endings": ["flour [title] eggs", "none"],
        "label": "0",
    }
    ctx, endings, label = to_mc_item("hellaswag", hella)
    assert ctx == "Baking: A man mixes. Then he adds" and endings == ["flour. eggs", "none"]
    assert label == 0


def test_tokenize_cached_equals_encode(tmp_path: Path) -> None:
    text = "".join(f"story {i} the cat sat on the mat<|endoftext|>" for i in range(30))
    tok = BPETokenizer()
    tok.train(text.replace("<|endoftext|>", "\n"), target_vocab_size=300)
    src = tmp_path / "t.txt"
    src.write_text(text, encoding="utf-8")
    ids = tokenize_cached(tok, src, tmp_path / "t.npy", workers=2, piece_chars=100)
    assert ids.tolist() == tok.encode(text, allow_special=True)
    src.unlink()  # the second call reads the cache
    assert tokenize_cached(tok, src, tmp_path / "t.npy").tolist() == ids.tolist()


def test_hf_rows_pages_by_100(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    class Resp:
        def __init__(self, n: int) -> None:
            self.n = n

        def __enter__(self) -> "Resp":
            return self

        def __exit__(self, *a: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"rows": [{"row": {"i": i}} for i in range(self.n)]}).encode()

    def fake_urlopen(req: Any, timeout: int) -> Resp:
        urls.append(req.full_url)
        return Resp(100 if "offset=0" in req.full_url else 20)  # the split ends at 120 rows

    monkeypatch.setattr(data.urllib.request, "urlopen", fake_urlopen)
    assert len(list(hf_rows("d", "c", "s", 500, pause=0))) == 120
    assert len(urls) == 2 and "length=100" in urls[1]
