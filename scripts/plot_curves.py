"""Eight-way budget curves in the fiveway.png house style (GPT-5.4 judge).

Reads mean lenient / mean generated tokens per budget from the grade
reports and saves docs/figures/eightway-gpt54.{png,svg}. Style contract
(matches the original fiveway figure): cheatsheet = solid dark shade,
no-study = dashed light shade of the same agent hue; no legend box —
colored "<series> · E <value>" labels at line ends; log-x with the 3k
anchor marked.

Usage: uv run --with matplotlib python scripts/plot_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = ("direct", "k5", "k20", "k20f")

# agent hue pairs: (dark solid = cheatsheet, light dashed = no-study)
SERIES = [
    # run_id, label, color, linestyle, label position (x, y) in data coords
    ("dspy-cheatsheet-20260722", "Qwen3.5-9B + cheatsheet", "#e2571f", "solid", (27_000, 30.5)),
    ("dspy-nostudy-20260722", "Qwen3.5-9B no-study", "#f2ab84", "dashed", (27_000, 22.0)),
    ("dspy-codexmini-cheatsheet-20260723", "codex-mini + cheatsheet", "#2871d8", "solid", (22_500, 76.5)),
    ("dspy-codexmini-nostudy-20260723", "codex-mini no-study", "#85b3ea", "dashed", (27_000, 72.0)),
    ("dspy-sonnet45-cheatsheet-20260725", "Sonnet 4.5 + cheatsheet", "#12a370", "solid", (27_000, 60.5)),
    ("dspy-sonnet45-nostudy-20260725", "Sonnet 4.5 no-study", "#7fd0ae", "dashed", (27_000, 67.0)),
    ("dspy-gptmini-cheatsheet-20260727", "gpt-5.4-mini + cheatsheet", "#6d4fc2", "solid", (900, 63.5)),
    ("dspy-gptmini-nostudy-20260727", "gpt-5.4-mini no-study", "#b3a1e6", "dashed", (900, 68.0)),
    ("dspy-gptminifoldback-20260726", "gpt-5.4-mini + fold-back object", "#472f8f", "dashdot", (900, 59.0)),
]
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
            tokens, lenient, color=color,
            linestyle={"solid": "-", "dashed": (0, (5, 2.5)),
                       "dashdot": (0, (6, 2, 1.5, 2))}[style],
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
    ax.set_ylim(0, 85)
    ax.set_yticks(range(0, 81, 10))
    ax.text(3200, 81.5, "3k anchor", color=INK_2, fontsize=10)

    ax.tick_params(colors=INK, labelsize=11, length=0)
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.9, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cccccc")

    ax.set_xlabel("generated tokens / question (log)", color=INK, fontsize=12.5)
    ax.set_ylabel("lenient score (%)", color=INK, fontsize=12.5)
    ax.set_title("Score vs. inference compute — GPT-5.4 paper judge",
                 color=INK, fontsize=15.5, loc="left", pad=26)
    ax.text(0, 1.035, "solid = + cheatsheet, dashed = no-study, dash-dot = + fold-back object · 30 test questions"
            " · E = expertise (WAUC, 3k anchor)",
            transform=ax.transAxes, color=INK_2, fontsize=10.5)

    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"eightway-gpt54.{suffix}", bbox_inches="tight", facecolor="white")
    print(f"saved {out}/eightway-gpt54.png + .svg")


if __name__ == "__main__":
    main()
