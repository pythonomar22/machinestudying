"""Nine-way budget curves: three models x three studying arms (GPT-5.4 judge).

House style contract (fiveway/eightway): hue = model; dashed light =
no-study, solid dark = + cheatsheet, dash-dot darkest = + fold-back
object; no legend box — colored "<series> · E <value>" labels; log-x
with the 3k anchor marked. Saves docs/figures/nineway-gpt54.{png,svg}.

Usage: uv run --with matplotlib python scripts/plot_nineway.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = ("direct", "k5", "k20", "k20f")

SERIES = [
    # run_id, label, color, linestyle, label position (x, y) in data coords
    ("dspy-gptmini-nostudy-20260727", "gpt-5.4-mini no-study", "#b3a1e6", "dashed", (900, 68.0)),
    ("dspy-gptmini-cheatsheet-20260727", "gpt-5.4-mini + cheatsheet", "#6d4fc2", "solid", (900, 63.5)),
    ("dspy-gptminifoldback-20260726", "gpt-5.4-mini + fold-back", "#472f8f", "dashdot", (900, 59.0)),
    ("dspy-gpt51-nostudy-20260727", "gpt-5.1 no-study", "#7fd0ae", "dashed", (950, 17.3)),
    ("dspy-gpt51-cheatsheet-20260727", "gpt-5.1 + cheatsheet", "#12a370", "solid", (11_800, 36.0)),
    ("dspy-gpt51foldback-20260727", "gpt-5.1 + fold-back", "#0a6e4a", "dashdot", (17_600, 43.0)),
    ("dspy-nostudy-20260722", "Qwen3.5-9B no-study", "#f2ab84", "dashed", (28_000, 22.0)),
    ("dspy-cheatsheet-20260722", "Qwen3.5-9B + cheatsheet", "#e2571f", "solid", (28_000, 30.5)),
    ("dspy-qwenfoldback2-20260727", "Qwen3.5-9B + fold-back", "#a03408", "dashdot", (28_000, 26.3)),
]
STYLES = {"solid": "-", "dashed": (0, (5, 2.5)), "dashdot": (0, (6, 2, 1.5, 2))}
INK, INK_2 = "#1a1a1a", "#777777"


def load(run_id: str) -> tuple[list[float], list[float], float]:
    report = json.loads(
        (ROOT / "grades" / run_id / "gpt-5-4" / "dspy" / "report.json").read_text()
    )
    tokens = [report["budgets"][b]["mean_generated_tokens"] for b in BUDGETS]
    lenient = [report["budgets"][b]["mean_lenient"] for b in BUDGETS]
    return tokens, lenient, report.get("expertise") or report["expertise_wauc"]


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.3), dpi=300)

    ax.axvline(3000, color="#999999", linewidth=1, linestyle=(0, (1, 2.5)), zorder=1)

    for run_id, label, color, style, (lx, ly) in SERIES:
        tokens, lenient, expertise = load(run_id)
        ax.plot(
            tokens, lenient, color=color, linestyle=STYLES[style],
            linewidth=2.6 if style == "solid" else 2.1,
            marker="o", markersize=6, zorder=3,
        )
        ax.text(lx, ly, f"{label} · E {expertise:.1f}",
                color=color, fontsize=11.5, fontweight="medium", va="center")

    ax.set_xscale("log")
    ticks = [1000, 2000, 3000, 5000, 10_000, 20_000, 40_000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1k", "2k", "3k", "5k", "10k", "20k", "40k"])
    ax.minorticks_off()
    ax.set_xlim(750, 130_000)
    ax.set_ylim(0, 75)
    ax.set_yticks(range(0, 71, 10))
    ax.text(3200, 72.0, "3k anchor", color=INK_2, fontsize=10)

    ax.tick_params(colors=INK, labelsize=11, length=0)
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.9, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cccccc")

    ax.set_xlabel("generated tokens / question (log)", color=INK, fontsize=12.5)
    ax.set_ylabel("lenient score (%)", color=INK, fontsize=12.5)
    ax.set_title("Score vs. inference compute — three models, three studying arms",
                 color=INK, fontsize=15.5, loc="left", pad=26)
    ax.text(0, 1.035, "dashed = no-study, solid = + cheatsheet, dash-dot = + fold-back object · "
            "GPT-5.4 paper judge · 30 test questions · Qwen 3 rollouts, mini/gpt-5.1 1",
            transform=ax.transAxes, color=INK_2, fontsize=10.5)

    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"nineway-gpt54.{suffix}", bbox_inches="tight", facecolor="white")
    print(f"saved {out}/nineway-gpt54.png + .svg")


if __name__ == "__main__":
    main()
