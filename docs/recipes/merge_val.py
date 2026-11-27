"""Validation scores for the merge recipe: pick the scale L, or score checkpoints.

Validation tasks are drawn from `Random(1)` for `+` and `-`, 1-3 digits, up to 200 distinct tasks
per (operator, digit count), and every task `reason-eval --n 200 --seed 0` scores is removed
first, so no validation task is a test task. 1-digit tasks are few (100 pairs, 55 for `-`) and
the test set covers most of them: that group comes out small or empty.

    python docs/recipes/merge_val.py --base B --expert E1 --expert E2 --out OUT  # sweep L
    python docs/recipes/merge_val.py CKPT [CKPT ...]                             # score only
"""

from __future__ import annotations

import argparse
import random

from minigpt.merge import task_arithmetic
from minigpt.model import GPTModel
from minigpt.reason import Task, accuracy_reward, make_task, reasoning_tokenizer
from minigpt.rft import sample_tasks
from minigpt.train import get_device, load_model, save_checkpoint

OPS, DIGITS, N = "+-", (1, 2, 3), 200


def test_tasks() -> set[Task]:
    """What `reason-eval --n 200 --ops OP --seed 0` scores, for either operator, 1-5 digits."""
    out: set[Task] = set()
    for op in OPS:
        rng = random.Random(0)
        out |= {make_task(rng, d, op) for d in range(1, 6) for _ in range(N)}
    return out


def val_tasks() -> dict[tuple[str, int], list[Task]]:
    test = test_tasks()
    groups: dict[tuple[str, int], list[Task]] = {}
    for op in OPS:
        for d in DIGITS:
            rng, seen = random.Random(1), set[Task]()
            for _ in range(50 * N):
                t = make_task(rng, d, op)
                if t not in test:
                    seen.add(t)
                if len(seen) == N:
                    break
            groups[op, d] = sorted(seen, key=lambda t: (t.a, t.b))
    return groups


def score(model: GPTModel, groups: dict[tuple[str, int], list[Task]]) -> dict[str, float]:
    """Greedy accuracy per operator (mean of the digit groups) and overall."""
    tok = reasoning_tokenizer()
    model.to(get_device()).eval()
    acc: dict[tuple[str, int], float] = {}
    for key, tasks in groups.items():
        if tasks:
            done = sample_tasks(model, tok, tasks, 1, 100, 0.0)
            acc[key] = sum(accuracy_reward(s.task, s.texts[0]) for s in done) / len(tasks)
    per_op: dict[str, float] = {}
    for op in OPS:
        got = [a for (o, _), a in acc.items() if o == op]
        per_op[op] = sum(got) / len(got)
    return {**per_op, "mean": sum(acc.values()) / len(acc)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("ckpts", nargs="*", help="score these checkpoints")
    p.add_argument("--base")
    p.add_argument("--expert", action="append", default=[])
    p.add_argument("--scales", default="0.5,0.75,1.0,1.25")
    p.add_argument("--out")
    args = p.parse_args()
    groups = val_tasks()
    print("validation tasks:", {f"{o}{d}": len(t) for (o, d), t in groups.items()})
    for path in args.ckpts:
        print(path, score(load_model(path), groups))
    if args.base:
        base, experts = load_model(args.base), [load_model(e) for e in args.expert]
        rows = []
        for scale in (float(s) for s in args.scales.split(",")):
            merged = task_arithmetic(base, experts, scale)
            rows.append((score(merged, groups), scale, merged))
            print(f"L {scale}: {rows[-1][0]}")
        s, scale, best = max(rows, key=lambda r: r[0]["mean"])  # ties keep the first (smallest L)
        print(f"best L {scale} (validation mean {s['mean']:.4f})")
        save_checkpoint(args.out, best, None, step=0, val_loss=float("nan"))
        print(f"checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
