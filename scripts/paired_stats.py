"""Paired per-question bootstrap for a graded StudyBench condition pair.

Usage:
    .venv-dspy/bin/python scripts/paired_stats.py \
        grades/RUN_A/GRADE_ID/TASK grades/RUN_B/GRADE_ID/TASK [--reps 10000] [--seed 20260715]

Procedure (recorded so the CI is reproducible): resample the shared question
ids with replacement via random.Random(seed).choices, recompute each arm's
per-budget mean lenient / mean generated tokens on the resample, take the
Appendix-C 4-point WAUC of each arm, and report the empirical 2.5/97.5
percentiles of the B−A WAUC delta, plus the paired per-budget deltas.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studybench.grade import BUDGETS, weighted_auc  # noqa: E402


def load(grade_dir: Path) -> dict[tuple[str, str], list[tuple[int, int]]]:
    grades: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for path in sorted((grade_dir / "episodes").rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record["budget"], record["qid"])
        grades.setdefault(key, []).append((record["gen_tokens"], record["lenient"]))
    return grades


def wauc_of(grades, sample) -> float:
    points = []
    for budget in BUDGETS:
        episodes = [episode for qid in sample for episode in grades[(budget, qid)]]
        points.append((fmean(t for t, _ in episodes), fmean(s for _, s in episodes)))
    return weighted_auc(points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grade_a")
    parser.add_argument("grade_b")
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    a, b = load(Path(args.grade_a)), load(Path(args.grade_b))
    if set(a) != set(b):
        raise SystemExit("the two grade populations differ")
    qids = sorted({qid for _, qid in a})

    for budget in BUDGETS:
        deltas = [
            fmean(s for _, s in b[(budget, q)]) - fmean(s for _, s in a[(budget, q)])
            for q in qids
        ]
        wins = sum(d > 0 for d in deltas)
        losses = sum(d < 0 for d in deltas)
        print(
            f"{budget:6} paired delta mean={fmean(deltas):+6.2f} "
            f"W/L/T={wins}/{losses}/{len(deltas) - wins - losses}"
        )

    point = wauc_of(b, qids) - wauc_of(a, qids)
    rng = random.Random(args.seed)
    deltas = sorted(
        wauc_of(b, sample) - wauc_of(a, sample)
        for sample in (rng.choices(qids, k=len(qids)) for _ in range(args.reps))
    )
    low = deltas[round(0.025 * args.reps) - 1]
    high = deltas[round(0.975 * args.reps) - 1]
    positive = sum(d > 0 for d in deltas) / len(deltas)
    print(
        f"deltaE={point:+.2f}  95% CI [{low:+.2f}, {high:+.2f}]  "
        f"P(delta>0)={positive:.3f}  ({args.reps} reps, seed {args.seed})"
    )


if __name__ == "__main__":
    main()
