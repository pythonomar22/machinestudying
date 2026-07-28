"""Mini delivery-ablation figure in the fiveway house style (GPT-5.4 judge).

Four gpt-5.4-mini arms on the 30-question test set: no-study, own
cheatsheet, the full fold-back object (knowledge + behavior protocol +
lookup store), and the knowledge-only ablation (same verified entries,
behavior stripped). Reads grade reports; saves
docs/figures/miniablation-gpt54.{png,svg}.

Usage: uv run --with matplotlib python scripts/plot_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = ("direct", "k5", "k20", "k20f")

SERIES = [
    # run_id, label, color, linestyle, label y
    ("dspy-gptmini-nostudy-20260727", "no-study", "#7a7a7a", "dashed", 57.5),
    ("dspy-gptminikonly-20260727", "+ knowledge-only note (3 rollouts)", "#12a370", "solid", 62.0),
    ("dspy-gptmini-cheatsheet-20260727", "+ own cheatsheet", "#e2571f", "solid", 50.0),
    ("dspy-gptminifoldback-20260726", "+ fold-back object (note + protocol + store)", "#472f8f", "dashdot", 43.0),
]
STYLES = {"solid": "-", "dashed": (0, (5, 2.5)), "dashdot": (0, (6, 2, 1.5, 2))}
INK, INK_2 = "#1a1a1a", "#777777"
LABEL_X = 4_300


def load(run_id: str):
    report = json.loads(
        (ROOT / "grades" / run_id / "gpt-5-4" / "dspy" / "report.json").read_text())
    return ([report["budgets"][b]["mean_generated_tokens"] for b in BUDGETS],
            [report["budgets"][b]["mean_lenient"] for b in BUDGETS],
            report["expertise"])


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.3), dpi=300)
    ax.axvline(3000, color="#999999", linewidth=1, linestyle=(0, (1, 2.5)), zorder=1)

    for run_id, label, color, style, label_y in SERIES:
        tokens, lenient, expertise = load(run_id)
        ax.plot(tokens, lenient, color=color, linestyle=STYLES[style],
                linewidth=2.6 if style == "solid" else 2.1,
                marker="o", markersize=6, zorder=3)
        ax.text(LABEL_X, label_y, f"{label} · E {expertise:.1f}",
                color=color, fontsize=11.5, va="center")

    ax.set_xscale("log")
    ax.set_xticks([1000, 2000, 3000, 5000, 10_000])
    ax.set_xticklabels(["1k", "2k", "3k", "5k", "10k"])
    ax.minorticks_off()
    ax.set_xlim(750, 16_000)
    ax.set_ylim(0, 70)
    ax.set_yticks(range(0, 70, 10))
    ax.text(3080, 66.5, "3k anchor", color=INK_2, fontsize=10)

    ax.tick_params(colors=INK, labelsize=11, length=0)
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.9, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cccccc")

    ax.set_xlabel("generated tokens / question (log)", color=INK, fontsize=12.5)
    ax.set_ylabel("lenient score (%)", color=INK, fontsize=12.5)
    ax.set_title("gpt-5.4-mini study-object ablation — GPT-5.4 paper judge",
                 color=INK, fontsize=15.5, loc="left", pad=26)
    ax.text(0, 1.035, "30 test questions · same harness/seeds/judge · "
            "no-study/cheatsheet/fold-back 1 rollout, knowledge-only 3 rollouts",
            transform=ax.transAxes, color=INK_2, fontsize=10.5)

    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"miniablation-gpt54.{suffix}",
                    bbox_inches="tight", facecolor="white")
    print(f"saved {out}/miniablation-gpt54.png + .svg")


if __name__ == "__main__":
    main()
