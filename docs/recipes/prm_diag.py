"""Why a verifier selects badly: within-task ranking and what the argmax of 16 looks like.

Post-hoc analysis of the cached `best-of-n` samples (`runs/prm/eval_samples.json`). For each
verifier it prints, over the 3-digit tasks that have both a right and a wrong sample among the 16:
the within-task AUC (chance that a right sample outscores a wrong one, ties 0.5), the share of
well-formed completions among the picks at N=16, and their mean length.

    python docs/recipes/prm_diag.py runs/prm/eval_samples.json runs/prm_s0 [runs/prm_s1 ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from minigpt.prm import load_samples, score_samples
from minigpt.reason import accuracy_reward, format_reward, reasoning_tokenizer
from minigpt.rlhf import load_reward_model


def auc(scores: list[float], right: list[float]) -> float | None:
    pos = [s for s, r in zip(scores, right, strict=True) if r]
    neg = [s for s, r in zip(scores, right, strict=True) if not r]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def main(samples_path: str, dirs: list[str]) -> None:
    tok = reasoning_tokenizer()
    samples = [s for s in load_samples(Path(samples_path), tok) if len(str(s.task.a)) == 3]
    right = [[accuracy_reward(s.task, t) for t in s.texts] for s in samples]
    n_right = sum(map(sum, right))
    print(f"{len(samples)} three-digit tasks; sample accuracy {n_right / 16 / len(samples):.3f}")
    for d in dirs:
        prm = load_reward_model(Path(d) / "prm.pt").eval()
        orm = load_reward_model(Path(d) / "orm.pt").eval()
        scores = score_samples(prm, orm, tok, samples, torch.device("cpu"))
        for name, per_task in scores.items():
            aucs = [a for p, r in zip(per_task, right, strict=True) if (a := auc(p, r)) is not None]
            picks = [
                (s, max(range(16), key=lambda j, p=p: p[j]))
                for s, p in zip(samples, per_task, strict=True)
            ]
            fmt = sum(format_reward(s.texts[j]) for s, j in picks) / len(picks)
            length = sum(len(s.ids[j]) for s, j in picks) / len(picks)
            print(
                f"{d} {name:8s} within-task AUC {sum(aucs) / len(aucs):.3f} ({len(aucs)} tasks) "
                f"picked: well-formed {fmt:.3f} mean tokens {length:.1f}"
            )
    mean_len = sum(len(i) for s in samples for i in s.ids) / (16 * len(samples))
    print(f"all samples mean tokens {mean_len:.1f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
