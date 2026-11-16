"""Judge sanity check: references vs themselves (expect 5), vs another row's answer (expect low),
and rank agreement of judge scores with token F1 on the eval rows."""

import json
from pathlib import Path

import numpy as np

from minigpt.cli import SFT_HELDOUT, load_alpaca
from minigpt.evaluate import OllamaJudge


def avg_ranks(x: np.ndarray) -> np.ndarray:
    order = x.argsort(kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for v in np.unique(x):  # ties get their average rank
        ranks[x == v] = ranks[x == v].mean()
    return ranks


rows = load_alpaca()[-SFT_HELDOUT:]
judge = OllamaJudge("http://localhost:11434", "qwen3.6:27b-ctx4k")
same = [judge.score(r["instruction"], r["output"], r["output"]) for r in rows]
other = [
    judge.score(r["instruction"], rows[(i + 37) % len(rows)]["output"], r["output"])
    for i, r in enumerate(rows)
]
ev = json.loads(Path("runs/eval/rows.json").read_text())
f1 = np.concatenate([ev["token_f1"][m] for m in ev["token_f1"]])
js = np.concatenate([np.asarray(ev["judge"][m], float) for m in ev["judge"]])
# Spearman is undefined (null) when either side is constant, e.g. every judge score is 1.
rho = None
if js.std() > 0 and f1.std() > 0:
    rho = float(np.corrcoef(avg_ranks(f1), avg_ranks(js))[0, 1])
out = {
    "reference_vs_itself": {
        "mean": float(np.mean(same)),
        "share_5": float(np.mean(np.array(same) == 5)),
        "scores": same,
    },
    "other_rows_answer": {
        "mean": float(np.mean(other)),
        "share_1": float(np.mean(np.array(other) == 1)),
        "scores": other,
    },
    "spearman_judge_vs_token_f1": rho,
    "n_pairs": int(f1.size),
    "judge_score_counts": {str(k): int((js == k).sum()) for k in range(1, 6)},
}
Path("runs/eval/judge_check.json").write_text(json.dumps(out, indent=2))
print("spearman:", rho, "counts:", out["judge_score_counts"])
print("self:", out["reference_vs_itself"]["mean"], out["reference_vs_itself"]["share_5"])
print("other:", out["other_rows_answer"]["mean"], out["other_rows_answer"]["share_1"])
