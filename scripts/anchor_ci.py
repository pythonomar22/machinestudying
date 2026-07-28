"""Per-anchor paired bootstrap for the 018 sweep claims.

Recomputes E(T0) from per-episode grade files (mean tokens + mean lenient
per budget, then the 4-point best-so-far weighted area with T0 varied),
verifies the 018 table, then bootstraps question-paired deltas per anchor.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import fmean

ROOT = Path("/matx/u/omarah/ms2")
BUDGETS = ("direct", "k5", "k20", "k20f")
ANCHORS = (250, 500, 1000, 2000, 3000, 6000)

ARMS = {
    "qwen no-study": "dspy-nostudy-20260722",
    "qwen cheatsheet": "dspy-cheatsheet-20260722",
    "qwen foldback v2": "dspy-qwenfoldback2-20260727",
    "mini no-study": "dspy-gptmini-nostudy-20260727",
    "mini cheatsheet": "dspy-gptmini-cheatsheet-20260727",
    "mini knowledge-only": "dspy-gptminikonly-20260727",
    "gpt51 no-study": "dspy-gpt51-nostudy-20260727",
    "gpt51 cheatsheet": "dspy-gpt51-cheatsheet-20260727",
    "sonnet no-study": "dspy-sonnet45-nostudy-20260725",
    "sonnet cheatsheet": "dspy-sonnet45-cheatsheet-20260725",
}


def wauc(points, t0):
    ordered = sorted(points)
    area = best = 0.0
    for i, (tokens, acc) in enumerate(ordered):
        best = max(best, acc)
        w = min(t0 / tokens, 1.0)
        nw = min(t0 / ordered[i + 1][0], 1.0) if i + 1 < len(ordered) else 0.0
        area += (w - nw) * best
    return area


def load(run_id):
    grades = {}
    for path in sorted((ROOT / "grades" / run_id / "gpt-5-4" / "dspy" / "episodes").rglob("*.json")):
        r = json.loads(path.read_text())
        grades.setdefault((r["budget"], r["qid"]), []).append((r["gen_tokens"], r["lenient"]))
    return grades


def e_of(grades, sample, t0):
    pts = []
    for b in BUDGETS:
        eps = [e for q in sample for e in grades[(b, q)]]
        pts.append((fmean(t for t, _ in eps), fmean(s for _, s in eps)))
    return wauc(pts, t0)


data = {name: load(run) for name, run in ARMS.items()}
qid_sets = {name: sorted({q for _, q in g}) for name, g in data.items()}
ref = qid_sets["mini knowledge-only"]
print("qid sets identical across arms:", all(v == ref for v in qid_sets.values()), "| n =", len(ref))
for name, g in data.items():
    rollouts = {len(v) for v in g.values()}
    print(f"  {name:22} rollouts-per-(budget,qid): {sorted(rollouts)}")

print("\n=== sweep table recompute (E(T0) x anchor) ===")
for name, g in data.items():
    qids = qid_sets[name]
    row = "  ".join(f"{e_of(g, qids, t0):6.2f}" for t0 in ANCHORS)
    print(f"{name:22} {row}")

PAIRS = [
    ("mini cheatsheet", "mini knowledge-only"),   # konly > cheatsheet at every anchor?
    ("mini no-study", "mini knowledge-only"),     # the inversion exhibit
    ("mini no-study", "mini cheatsheet"),
    ("gpt51 no-study", "gpt51 cheatsheet"),       # claim 1
    ("qwen cheatsheet", "qwen foldback v2"),      # claim 2 (tie)
    ("sonnet no-study", "sonnet cheatsheet"),     # claim 3 (null)
]

REPS = 10_000
for a_name, b_name in PAIRS:
    a, b = data[a_name], data[b_name]
    qids = qid_sets[a_name]
    assert qid_sets[b_name] == qids
    rng = random.Random(20260715)
    samples = [rng.choices(qids, k=len(qids)) for _ in range(REPS)]
    print(f"\n=== {b_name} minus {a_name} (paired bootstrap, {REPS} reps) ===")
    for t0 in ANCHORS:
        point = e_of(b, qids, t0) - e_of(a, qids, t0)
        deltas = sorted(e_of(b, s, t0) - e_of(a, s, t0) for s in samples)
        lo, hi = deltas[round(0.025 * REPS) - 1], deltas[round(0.975 * REPS) - 1]
        pos = sum(d > 0 for d in deltas) / REPS
        print(f"  T0={t0:5d}  delta={point:+6.2f}  95% CI [{lo:+6.2f}, {hi:+6.2f}]  P(>0)={pos:.3f}")
