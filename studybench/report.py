"""Aggregate grades into the paper's Table 1: per-budget accuracy, mean generated
tokens, and expertise (weighted AUC per Appendix C). Each grade file embeds its
episode's gen_tokens and status, so scores and tokens come from one population.

Score definitions (see docs/jacob.md — the author's core-conjunctive rule is what
brings replicated scores into the paper's range; the raw rubric sum runs ~2-4x hot):
  lenient  = weighted claim sum if every core claim scored 1, else 0
             (the compilation check is omitted — paper: "lenient grading omits the
             compilation check"); this is the Table 1 comparison.
  rubric   = raw weighted claim sum, no gates (reported for transparency).
  strict   = lenient plus the compile-gate zero.

The expertise formula was verified against the paper: with x = log10(tokens/3000),
w(x) = ln(10)·10^(-x), the weight of the segment between consecutive budgets is
3000/tok_i - 3000/tok_{i+1}; performance is the best-score-so-far envelope; the region
below the first budget is floored to 0 and the last score carries the tail. It
reproduces the worked example (10.8) and DSPy base (6.49) exactly; the paper's own
Table 1 values for OpenClaw base give 7.66 vs the published 7.64, consistent with the
table's tokens being rounded to 0.1k.
"""

import argparse
import json
import random
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
        next_w = min(3000 / pts[i + 1][0], 1.0) if i + 1 < len(pts) else 0.0
        e += (min(3000 / tok, 1.0) - next_w) * best
    return e


def aggregate(task: str) -> dict:
    budgets = {}
    for budget in BUDGET_ORDER:
        gdir = ROOT / "grades" / task / budget
        grades = [json.loads(f.read_text()) for f in sorted(gdir.rglob("*.json"))] \
            if gdir.exists() else []
        if not grades:
            continue
        n_runs = len(list((ROOT / "runs" / task / budget).rglob("*.json")))
        if n_runs != len(grades):
            print(f"WARNING: {task}/{budget} has {n_runs} runs but {len(grades)} grades "
                  "— aggregating the graded subset only")
        budgets[budget] = {
            "n": len(grades),
            "lenient": mean(g["lenient"] if g["cores_ok"] else 0 for g in grades),
            "rubric": mean(g["lenient"] for g in grades),
            "strict": mean(g["strict"] for g in grades),
            "compile_rate": mean(g["compile_check"]["compile_ok"] for g in grades),
            "needs_regrade": sum(bool(g.get("needs_regrade")) for g in grades),
            "tokens": mean(g["gen_tokens"] for g in grades),
            "bad_episodes": sum(g["episode_status"] != "ok" for g in grades),
        }
    out = {"budgets": budgets}
    for kind in ("lenient", "strict"):
        pts = [(b["tokens"], b[kind]) for b in budgets.values()]
        out[f"expertise_{kind}"] = expertise(pts) if len(pts) == 4 else None
    return out


def bootstrap(task: str, n_boot: int, seed: int = 0) -> dict:
    """95% CIs via a two-stage cluster bootstrap: resample questions with
    replacement (the benchmark's sampling unit), then rollouts within each
    question. One question resample is shared across budgets, so each
    replicate's WAUC is computed from a coherent set of curve points."""
    data = {}  # budget -> qid -> [(lenient_cc, rubric, tokens)]
    for budget in BUDGET_ORDER:
        eps = {}
        for f in sorted((ROOT / "grades" / task / budget).rglob("*.json")):
            g = json.loads(f.read_text())
            eps.setdefault(g["qid"], []).append(
                (g["lenient"] if g["cores_ok"] else 0, g["lenient"], g["gen_tokens"]))
        data[budget] = eps
    qids = sorted(data[BUDGET_ORDER[0]])
    rng = random.Random(seed)
    stats = {b: [] for b in BUDGET_ORDER} | {"wauc": [], "wauc_rubric": []}
    for _ in range(n_boot):
        qs = rng.choices(qids, k=len(qids))
        pts_cc, pts_rub = [], []
        for b in BUDGET_ORDER:
            cc = rub = tok = n = 0
            for q in qs:
                pool = data[b][q]
                for ep in rng.choices(pool, k=len(pool)):
                    cc += ep[0]; rub += ep[1]; tok += ep[2]; n += 1
            stats[b].append(cc / n)
            pts_cc.append((tok / n, cc / n))
            pts_rub.append((tok / n, rub / n))
        stats["wauc"].append(expertise(pts_cc))
        stats["wauc_rubric"].append(expertise(pts_rub))

    def ci(xs):
        xs = sorted(xs)
        return xs[round(0.025 * (len(xs) - 1))], xs[round(0.975 * (len(xs) - 1))]

    return {k: (mean(v), *ci(v)) for k, v in stats.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", default="dspy,openclaw")
    p.add_argument("--ci", type=int, default=0, metavar="N",
                   help="add 95%% bootstrap CIs from N replicates (e.g. 10000)")
    args = p.parse_args()

    for task in args.tasks.split(","):
        agg = aggregate(task)
        paper = PAPER_BASE[task]
        print(f"\n== {CORPORA[task].display} (Qwen3.5-9B base) ==")
        print(f"{'budget':8} {'n':>4} {'lenient':>8} {'rubric':>7} {'strict':>7} {'tok(k)':>7} "
              f"{'compile':>8} {'regrade':>8} {'bad':>4}   paper-lenient  paper-tok(k)")
        for budget, b in agg["budgets"].items():
            pa, pt = paper[budget]
            print(f"{budget:8} {b['n']:>4} {b['lenient']:>8.1f} {b['rubric']:>7.1f} "
                  f"{b['strict']:>7.1f} {b['tokens'] / 1000:>7.1f} {b['compile_rate']:>8.1%} "
                  f"{b['needs_regrade']:>8} {b['bad_episodes']:>4}   "
                  f"{pa:>13.1f} {pt:>12.1f}")
        if agg.get("expertise_lenient") is not None:
            print(f"expertise (lenient WAUC): {agg['expertise_lenient']:.2f} "
                  f"(paper: {paper['expertise']:.2f}); "
                  f"strict WAUC: {agg['expertise_strict']:.2f}")
        if args.ci:
            b = bootstrap(task, args.ci)
            print(f"95% CIs ({args.ci} bootstrap replicates over questions×rollouts):")
            for budget in BUDGET_ORDER:
                m, lo, hi = b[budget]
                print(f"  {budget:8} lenient {m:5.1f} [{lo:5.1f}, {hi:5.1f}]")
            m, lo, hi = b["wauc"]
            print(f"  WAUC lenient {m:5.2f} [{lo:5.2f}, {hi:5.2f}] "
                  f"(paper: {paper['expertise']:.2f})")
            m, lo, hi = b["wauc_rubric"]
            print(f"  WAUC rubric  {m:5.2f} [{lo:5.2f}, {hi:5.2f}]")


if __name__ == "__main__":
    main()
