"""Aggregate runs + grades into the paper's Table 1: per-budget accuracy, mean
generated tokens, and expertise (weighted AUC per Appendix C).

The expertise formula was verified against the paper: with x = log10(tokens/3000),
w(x) = ln(10)·10^(-x), the weight of the segment between consecutive budgets is
3000/tok_i - 3000/tok_{i+1}; performance is the best-score-so-far envelope; the region
below the first budget is floored to 0 and the last score carries the tail. This
reproduces the published 6.49 (DSPy base) and 7.64 (OpenClaw base) exactly.
"""

import argparse
import json
from statistics import mean

from .dataset import CORPORA, ROOT

BUDGET_ORDER = ["direct", "k5", "k20", "k20f"]

PAPER_BASE = {  # Table 1, Qwen3.5-9B (base), lenient: budget -> (acc %, tokens k)
    "dspy": {"direct": (3.3, 4.1), "k5": (8.6, 7.9), "k20": (9.6, 8.6), "k20f": (29.4, 34.6),
             "expertise": 6.49},
    "openclaw": {"direct": (2.3, 4.1), "k5": (6.9, 4.6), "k20": (15.8, 9.7), "k20f": (17.6, 24.3),
                 "expertise": 7.64},
}


def expertise(points: list[tuple[float, float]]) -> float:
    """Weighted AUC from (mean_tokens, accuracy) budget points; 3k-token anchor."""
    pts = sorted(p for p in points if p[0] > 0)
    e, best = 0.0, 0.0
    for i, (tok, acc) in enumerate(pts):
        best = max(best, acc)
        next_w = 3000 / pts[i + 1][0] if i + 1 < len(pts) else 0.0
        e += (min(3000 / tok, 1.0) - next_w) * best
    return e


def aggregate(task: str) -> dict:
    out = {}
    for budget in BUDGET_ORDER:
        gdir = ROOT / "grades" / task / budget
        grades = [json.loads(f.read_text()) for f in sorted(gdir.rglob("*.json"))] \
            if gdir.exists() else []
        if not grades:
            continue
        runs = [json.loads(f.read_text())
                for f in sorted((ROOT / "runs" / task / budget).rglob("*.json"))]
        out[budget] = {
            "n": len(grades),
            "lenient": mean(g["lenient"] for g in grades),
            "strict": mean(g["strict"] for g in grades),
            "compile_rate": mean(g["compile_check"]["compile_ok"] for g in grades),
            "needs_regrade": sum(bool(g.get("needs_regrade")) for g in grades),
            "tokens": mean(r["gen_tokens"] for r in runs),
            "bad_episodes": sum(r["status"] != "ok" for r in runs),
        }
    for kind in ("lenient", "strict"):
        pts = [(b["tokens"], b[kind]) for b in out.values()]
        out[f"expertise_{kind}"] = expertise(pts) if len(pts) == 4 else None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", default="dspy,openclaw")
    args = p.parse_args()

    for task in args.tasks.split(","):
        agg = aggregate(task)
        paper = PAPER_BASE[task]
        print(f"\n== {CORPORA[task].display} (Qwen3.5-9B base) ==")
        print(f"{'budget':8} {'n':>4} {'lenient':>8} {'strict':>7} {'tok(k)':>7} "
              f"{'compile':>8} {'regrade':>8} {'bad':>4}   paper-lenient  paper-tok(k)")
        for budget in BUDGET_ORDER:
            if budget not in agg:
                continue
            b = agg[budget]
            pa, pt = paper[budget]
            print(f"{budget:8} {b['n']:>4} {b['lenient']:>8.1f} {b['strict']:>7.1f} "
                  f"{b['tokens'] / 1000:>7.1f} {b['compile_rate']:>8.1%} "
                  f"{b['needs_regrade']:>8} {b['bad_episodes']:>4}   "
                  f"{pa:>13.1f} {pt:>12.1f}")
        if agg.get("expertise_lenient") is not None:
            print(f"expertise (lenient WAUC): {agg['expertise_lenient']:.2f} "
                  f"(paper: {paper['expertise']:.2f}); "
                  f"strict WAUC: {agg['expertise_strict']:.2f}")


if __name__ == "__main__":
    main()
