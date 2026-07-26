"""Plot the citable StudyBench-DSPy budget curves (GPT-5.4 judge).

Reads mean lenient / mean generated tokens per budget straight from the
grade reports and saves docs/figures/dspy-budget-curves-gpt54.{png,svg}.

Usage: uv run --with matplotlib python scripts/plot_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = ("direct", "k5", "k20", "k20f")
SERIES = [
    ("Qwen3.5-9B", "no-study", "dspy-nostudy-20260722", "#2a78d6", "solid"),
    ("Qwen3.5-9B", "cheatsheet", "dspy-cheatsheet-20260722", "#2a78d6", (0, (4, 2))),
    ("codex gpt-5.4-mini", "no-study", "dspy-codexmini-nostudy-20260723", "#eb6834", "solid"),
    ("codex gpt-5.4-mini", "cheatsheet", "dspy-codexmini-cheatsheet-20260723", "#eb6834", (0, (4, 2))),
    ("Sonnet 4.5", "no-study", "dspy-sonnet45-nostudy-20260725", "#1baf7a", "solid"),
    ("Sonnet 4.5", "cheatsheet", "dspy-sonnet45-cheatsheet-20260725", "#1baf7a", (0, (4, 2))),
]
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"


def load(run_id: str) -> tuple[list[float], list[float], float]:
    report = json.loads(
        (ROOT / "grades" / run_id / "gpt-5-4" / "dspy" / "report.json").read_text()
    )
    tokens = [report["budgets"][b]["mean_generated_tokens"] for b in BUDGETS]
    lenient = [report["budgets"][b]["mean_lenient"] for b in BUDGETS]
    return tokens, lenient, report.get("expertise") or report["expertise_wauc"]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axvline(3000, color=INK_2, linewidth=1, linestyle=(0, (1, 3)), zorder=1)
    ax.text(3080, 1.5, "3k anchor", color=INK_2, fontsize=8.5, ha="left")

    ends = {}
    for agent, condition, run_id, color, style in SERIES:
        tokens, lenient, expertise = load(run_id)
        ax.plot(
            tokens, lenient, color=color, linestyle=style, linewidth=2,
            marker="o", markersize=5.5, markerfacecolor=color,
            markeredgecolor=SURFACE, markeredgewidth=1, zorder=3,
            label=f"{agent} {condition}  (E {expertise:.1f})",
        )
        ends[agent] = max(ends.get(agent, (0, 0)), (tokens[-1], lenient[-1]))

    for agent, (x, y) in ends.items():  # relief rule: direct label per agent
        color = next(c for a, _, _, c, _ in SERIES if a == agent)
        ax.annotate(agent, (x, y), xytext=(6, 4), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks([1500, 3000, 6000, 12000, 24000])
    ax.set_xticklabels(["1.5k", "3k", "6k", "12k", "24k"], color=INK)
    ax.minorticks_off()
    ax.set_ylim(0, 80)
    ax.set_xlim(1300, 42000)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#d8d7d2")

    ax.set_xlabel("Mean generated tokens per answer (log scale)", color=INK, fontsize=10)
    ax.set_ylabel("Mean lenient accuracy", color=INK, fontsize=10)
    ax.set_title(
        "StudyBench-DSPy budget curves — GPT-5.4 judge\n"
        "budgets direct / k5 / k20 / k20f per series; solid = no-study, dashed = cheatsheet",
        color=INK, fontsize=11, loc="left", pad=12,
    )
    legend = ax.legend(
        loc="upper left", fontsize=8.5, labelcolor=INK, frameon=True,
        facecolor=SURFACE, edgecolor="none", framealpha=1.0,
    )
    fig.text(
        0.01, 0.005,
        "30 questions; Qwen/codex 3 rollouts, Sonnet 4.5 one; Qwen & Sonnet share the "
        "dspy.ReAct harness, codex ran its own CLI (its 10k-token direct floor). "
        "E = 4-point WAUC, 3k-token anchor.",
        color=INK_2, fontsize=7.5,
    )

    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"dspy-budget-curves-gpt54.{suffix}",
                    bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {out}/dspy-budget-curves-gpt54.png + .svg")


if __name__ == "__main__":
    main()
