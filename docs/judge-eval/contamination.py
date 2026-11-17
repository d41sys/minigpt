"""Contamination check: word 8-gram overlap of each held-out Alpaca row with the 900 SFT/DPO
training rows and with the pretraining corpus (tiny-Shakespeare). A row is flagged when any of
its 8-grams appears in the training text; token F1 is then recomputed on the clean rows."""

import json
import re
from pathlib import Path

import numpy as np

from minigpt.cli import SFT_HELDOUT, download_tinyshakespeare, load_alpaca

N = 8


def grams(text: str) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i : i + N]) for i in range(len(w) - N + 1)}


def row_text(r: dict[str, str]) -> str:
    return " ".join((r["instruction"], r.get("input", ""), r["output"]))


rows = load_alpaca()
train, held = rows[:-SFT_HELDOUT], rows[-SFT_HELDOUT:]
sources = {
    "sft_train": set().union(*(grams(row_text(r)) for r in train)),
    "pretrain": grams(download_tinyshakespeare().read_text("utf-8", errors="replace")),
}
out: dict[str, object] = {"n_gram": N, "heldout_rows": len(held)}
flagged: set[int] = set()
for name, bank in sources.items():
    hits = [i for i, r in enumerate(held) if grams(row_text(r)) & bank]
    flagged.update(hits)
    out[name] = {"flagged_rows": hits, "count": len(hits)}
out["exact_duplicate_instructions"] = sum(
    r["instruction"].strip().lower() in {t["instruction"].strip().lower() for t in train}
    for r in held
)
ev = json.loads(Path("runs/eval/rows.json").read_text())
clean = [i for i in range(len(next(iter(ev["token_f1"].values())))) if i not in flagged]
out["token_f1_all_vs_clean"] = {
    m: [float(np.mean(v)), float(np.mean([v[i] for i in clean]))] for m, v in ev["token_f1"].items()
}
print(json.dumps(out, indent=1))
Path("docs/judge-eval/contamination.json").write_text(json.dumps(out, indent=1) + "\n")
