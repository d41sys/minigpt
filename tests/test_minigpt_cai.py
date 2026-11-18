import json
import random
import urllib.error
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from minigpt import cai
from minigpt.cai import (
    JudgeError,
    OllamaChat,
    adherence_report,
    critique_and_revise,
    judge_pair,
    json_str,
    label_pairs,
    load_principles,
    revise_rows,
    score_adherence,
    split_prompts,
)
from minigpt.dpo import PreferenceDataset, PreferenceSample
from minigpt.reason import reasoning_tokenizer
from minigpt.sft import SFTDataset

DOCS = Path(__file__).parents[1] / "docs" / "cai"


class ScriptedJudge:
    """Returns the queued replies in order and records every prompt."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)


class SlotJudge:
    """Reads the A/B slots out of the prompt and answers with `pick(a, b)`."""

    def __init__(self, pick: Callable[[str, str], str]) -> None:
        self.pick = pick
        self.calls = 0

    def chat(self, prompt: str) -> str:
        self.calls += 1
        a = prompt.split("Response A: ")[1].split("\n\nResponse B: ")[0]
        b = prompt.split("Response B: ")[1].split("\n\nWhich")[0]
        return json.dumps({"better": self.pick(a, b)})


def test_shipped_constitution_and_prompts_parse() -> None:
    principles = load_principles((DOCS / "constitution.md").read_text(encoding="utf-8"))
    assert 6 <= len(principles) <= 8 and all(len(p) > 20 for p in principles)
    prompts = json.loads((DOCS / "prompts.json").read_text(encoding="utf-8"))
    assert len(prompts) >= 100 and len(set(prompts)) == len(prompts)
    assert all(isinstance(p, str) and p.strip() for p in prompts)
    train, held = split_prompts(prompts, 20)
    assert held == prompts[-20:] and len(train) == len(prompts) - 20
    assert split_prompts(prompts, 0) == (prompts, [])  # heldout 0 must not empty the train set


def test_json_str_rejects_every_malformed_shape() -> None:
    assert json_str('{"revision": "  ok  "}', "revision") == "ok"
    for bad in ["", "not json", "[1]", '{"revision": ""}', '{"revision": 5}', '{"other": "x"}']:
        assert json_str(bad, "revision") is None


def test_critique_revise_skips_malformed_rows_without_crashing() -> None:
    good = [json.dumps({"critique": "too blunt"}), json.dumps({"revision": "A kind answer."})]
    judge = ScriptedJudge(
        [
            *good,  # row 0: fine
            "I think the answer is bad.",  # row 1: critique is not JSON -> skipped, no 2nd call
            json.dumps({"critique": "meh"}),  # row 2: critique ok ...
            json.dumps({"revision": "   "}),  # ... empty revision -> skipped
            *good,  # row 3: fine again, the run continued
        ]
    )
    rows = [("p0", "a0"), ("p1", "a1"), ("p2", "a2"), ("p3", "a3")]
    records, stats = revise_rows(judge, ["be kind"], rows, random.Random(0))
    assert [r["instruction"] for r in records] == ["p0", "p3"]
    assert records[0]["output"] == "A kind answer." and records[0]["initial"] == "a0"
    assert stats == {"n_rows": 4, "n_records": 2}
    assert len(judge.prompts) == 2 + 1 + 2 + 2  # row 1 stopped after its failed critique
    assert critique_and_revise(ScriptedJudge(["{}"]), "p", "a", "x") is None


def test_judge_pair_maps_label_back_when_order_is_swapped() -> None:
    judge = SlotJudge(lambda a, b: "A" if a == "GOOD" else "B")  # always prefers GOOD
    rng = random.Random(0)
    shown_first: set[bool] = set()
    for _ in range(40):
        v = judge_pair(judge, "q", "GOOD", "BAD", "p", rng)
        assert v is not None and (v.chosen, v.rejected) == ("GOOD", "BAD")
        shown_first.add(v.picked_a)
    assert shown_first == {True, False}  # both display orders occurred
    # argument order must not matter: x and y swapped still yields GOOD as chosen
    v = judge_pair(judge, "q", "BAD", "GOOD", "p", random.Random(1))
    assert v is not None and v.chosen == "GOOD"


def test_judge_pair_malformed_or_unknown_label_is_none() -> None:
    for reply in ["A", '{"better": "C"}', '{"better": ""}', "{}", '{"better": 1}']:
        assert judge_pair(ScriptedJudge([reply]), "q", "x", "y", "p", random.Random(0)) is None
    v = judge_pair(ScriptedJudge(['{"better": "b"}']), "q", "x", "y", "p", random.Random(0))
    assert v is not None and v.picked_a is False  # lower case accepted


def test_position_bias_counts_slot_not_content() -> None:
    rows = [(f"q{i}", "short", "a much longer answer") for i in range(200)]
    # content-driven judge: picks the longer text wherever it is shown -> about half "A"
    fair = SlotJudge(lambda a, b: "A" if len(a) > len(b) else "B")
    pairs, stats = label_pairs(fair, ["p"], rows, random.Random(0))
    assert stats["n_pairs"] == 200 and all(p.chosen == "a much longer answer" for p in pairs)
    assert 0.35 < stats["position_bias"] < 0.65
    # slot-driven judge: always "A" -> bias 1.0, and the chosen text is therefore random
    lazy = SlotJudge(lambda a, b: "A")
    pairs, stats = label_pairs(lazy, ["p"], rows, random.Random(0))
    assert stats["position_bias"] == 1.0 and stats["picked_a"] == 200
    assert {p.chosen for p in pairs} == {"short", "a much longer answer"}


def test_label_pairs_skips_identical_and_malformed_and_excludes_them_from_bias() -> None:
    judge = ScriptedJudge(['{"better": "B"}', "garbage", '{"better": "B"}'])
    rows = [("q0", "x", "y"), ("q1", "same", "same"), ("q2", "x", "y"), ("q3", "x", "y")]
    pairs, stats = label_pairs(judge, ["p"], rows, random.Random(0))
    assert len(judge.prompts) == 3  # the identical pair was never sent to the judge
    assert [p.prompt for p in pairs] == ["q0", "q3"]
    assert stats["n_rows"] == 4 and stats["n_pairs"] == 2 and stats["picked_a"] == 0
    assert stats["position_bias"] == 0.0  # 0 of 2 valid verdicts, not 0 of 4 rows


def test_records_round_trip_into_sft_and_preference_datasets(tmp_path: Path) -> None:
    tok = reasoning_tokenizer()
    judge = ScriptedJudge(['{"critique": "c"}', '{"revision": "No, that is unkind."}'])
    rows = [("Insult Tom.", "You are dumb.")]
    records, _ = revise_rows(judge, ["be kind"], rows, random.Random(0))
    path = tmp_path / "sl.json"
    path.write_text(json.dumps(records))
    ds = SFTDataset(json.loads(path.read_text()), tok, max_length=512)
    assert len(ds) == 1
    _, y = ds[0]
    assert int((y != -100).sum()) == len(tok.encode("No, that is unkind.")) + 1  # + EOS
    assert records[0]["initial"] == "You are dumb."  # audit fields survive but are not trained on

    pairs = [PreferenceSample("Insult Tom.", chosen="No, that is unkind.", rejected="Tom is dumb")]
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps([vars(p) for p in pairs]))
    loaded = [PreferenceSample(**d) for d in json.loads(path.read_text())]
    assert loaded == pairs
    pds = PreferenceDataset(loaded, tok, max_length=512)
    assert len(pds) == 1
    item = pds[0]
    assert int((item["chosen_labels"] != -100).sum()) == len(tok.encode(pairs[0].chosen)) + 1
    assert int((item["rejected_labels"] != -100).sum()) == len(tok.encode(pairs[0].rejected)) + 1


def test_score_adherence_accepts_only_integers_1_to_5() -> None:
    def score(reply: str) -> int | None:
        return score_adherence(ScriptedJudge([reply]), ["p"], "q", "an answer")

    assert score('{"score": 4}') == 4
    bad_replies = ['{"score": 0}', '{"score": 6}', '{"score": true}', '{"score": "4"}', "4", "[]"]
    for bad in bad_replies:
        assert score(bad) is None
    empty = ScriptedJudge([])
    assert score_adherence(empty, ["p"], "q", "  ") == 1 and empty.prompts == []  # no call


def test_adherence_report_mean_ignores_malformed_replies() -> None:
    judge = ScriptedJudge(['{"score": 5}', "oops", '{"score": 2}'])
    rep = adherence_report(judge, ["p"], ["a", "b", "c"], ["r1", "r2", "r3"])
    assert rep == {"mean": 3.5, "n_scored": 2, "n_invalid": 1}


def test_ollama_chat_request_and_unreachable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []

    class Resp:
        def __enter__(self) -> "Resp":
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"message": {"content": '{"ok": true}'}}).encode()

    def ok(req: Any, timeout: float) -> Resp:
        sent.append({"url": req.full_url, **json.loads(req.data)})
        return Resp()

    monkeypatch.setattr(cai.urllib.request, "urlopen", ok)
    assert OllamaChat("http://h:1/", "qwen", seed=3).chat("hi") == '{"ok": true}'
    (req,) = sent
    assert req["url"] == "http://h:1/api/chat" and req["think"] is False and req["stream"] is False
    assert req["format"] == "json" and req["options"]["temperature"] == 0
    assert req["options"]["seed"] == 3

    def refuse(req: Any, timeout: float) -> Resp:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(cai.urllib.request, "urlopen", refuse)
    with pytest.raises(JudgeError, match=r"http://h:1/api/chat.*ssh -L 11434"):
        OllamaChat("http://h:1", "qwen").chat("hi")
