"""Dev-slice evaluation driver for note variants of a fold-back run.

Evaluates the requested arms (bare and/or rendered variant notes from
study/variants/) on the held-out 30-question dev slice with the chosen
studier model, direct + k5, paper-contract GPT-5.4 grading. Idempotent
per episode/grade; the test set is untouched.

Usage:
    .venv-dspy/bin/python -m studying.foldback.devrun \
        --run-id dspy-gptminifoldback-20260726 --seed 20260715 \
        --model gptmini --arms bare,knowledge_only,both0_only [--rollouts 2]
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from studybench.artifacts import read_json, write_json
from studybench.dataset import CORPORA, ROOT
from studybench.react import (
    GPT51_MODEL,
    GPT_MODEL,
    GPT_SAMPLING,
    MODEL,
    MODEL_REVISION,
    SAMPLING,
    make_tools,
)
from studybench.tools import RepoTools

from .data import load_practice_questions
from .devval import evaluate_variant

MODELS = {
    "gptmini": (GPT_MODEL, None, GPT_SAMPLING),
    "gpt51": (GPT51_MODEL, None, GPT_SAMPLING),
    "qwen": (MODEL, MODEL_REVISION, SAMPLING),  # requires --base-urls
}
log = logging.getLogger("studying.foldback.devrun")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--arms", required=True)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-urls", help="local vLLM endpoints (qwen only)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required (studier and judge)")

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "logs" / f"{args.run_id}-devrun.log")],
    )
    corpus = CORPORA["dspy"]
    repository = RepoTools(corpus)
    tools = make_tools(repository)
    run_root = ROOT / "runs" / args.run_id / corpus.name
    by_id = {row["id"]: row for row in load_practice_questions()}
    dev_ids = read_json(run_root / "study" / "split.json")["dev_ids"]
    dev_rows = [by_id[qid] for qid in dev_ids][: args.limit or None]
    model, revision, sampling = MODELS[args.model]
    if (args.model == "qwen") != bool(args.base_urls):
        raise SystemExit("--base-urls is required for --model qwen and invalid otherwise")
    base_urls = args.base_urls.split(",") if args.base_urls else [None]

    summary = {}
    for arm in args.arms.split(","):
        if arm == "bare":
            note = ""
        else:
            note_path = run_root / "study" / "variants" / f"{arm}.md"
            if not note_path.is_file():
                raise SystemExit(f"unknown arm (no variant note): {note_path}")
            note = note_path.read_text(encoding="utf-8")
        report = evaluate_variant(
            variant=arm,
            note=note,
            rows=dev_rows,
            corpus=corpus,
            tools=tools,
            base_urls=base_urls,
            api_key=os.environ["OPENAI_API_KEY"],
            master_seed=args.seed,
            out_dir=run_root / "dev" / f"{args.model}-{arm}",
            budgets=("direct", "k5"),
            rollouts=args.rollouts,
            debug=args.debug,
            model=model,
            model_revision=revision,
            sampling=sampling,
        )
        summary[arm] = {
            budget: {"mean_lenient": round(r["mean_lenient"], 2),
                     "mean_generated_tokens": round(r["mean_generated_tokens"])}
            for budget, r in report["budgets"].items()
        }
        log.info("arm %s: %s", arm, json.dumps(summary[arm]))
    write_json(run_root / "dev" / f"summary-{args.model}.json", summary)
    log.info("dev summary: %s", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
