"""Fold-back study orchestrator: mine forced-k20 competence into a k=0 object.

Phases (each idempotent on disk, resumable by rerunning the same command):
  split -> gold sandbox runs -> attempts (study slice, forced k20, bare) ->
  mine (Qwen per question) -> assemble (Qwen per topic + protocol,
  deterministic merge) -> factcheck (GPT-5.4 codex; non-verified entries
  dropped) -> object + study.json -> dev evaluation (object vs baselines).

Usage (inside the Slurm allocation, vLLM replicas already serving):
    .venv-dspy/bin/python -m studying.foldback.run --run-id RUN --seed SEED \
        --base-urls http://127.0.0.1:PORT/v1[,...] [--smoke] [--debug]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from studybench.artifacts import read_json, sha256_json, sha256_text, stable_seed, write_json, write_text
from studybench.dataset import CORPORA, ROOT
from studybench.react import MODEL, MODEL_REVISION, SAMPLING, TOOL_CONFIG, make_tools, run_episode
from studybench.tools import RepoTools

from ..sandbox import extract_program, passed, run_program
from . import assemble, factcheck, mine
from .data import load_practice_questions, practice_dataset_sha256, split_practice
from .devval import evaluate_variant

ATTEMPT_ITERS = 20
CHEATSHEET_NOTE = ROOT / "runs" / "dspy-cheatsheet-20260722" / "dspy" / "cheatsheet.md"
log = logging.getLogger("studying.foldback")


def _source_state(smoke: bool) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    )
    if dirty and not smoke:
        raise SystemExit("full fold-back runs require a clean committed source tree")
    return commit, dirty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--base-urls", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("--run-id must contain only letters, digits, '-' and '_'")
    urls = args.base_urls.split(",")

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT / "logs" / f"{args.run_id}-foldback.log"),
        ],
    )
    source_commit, source_dirty = _source_state(args.smoke)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required (dev-evaluation judge)")

    corpus = CORPORA["dspy"]
    repository = RepoTools(corpus)
    tools = make_tools(repository)
    rows = load_practice_questions()
    by_id = {row["id"]: row for row in rows}

    run_root = ROOT / "runs" / args.run_id / corpus.name
    study_root = run_root / "study"
    study_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "kind": "foldback-study",
        "run_id": args.run_id,
        "task": corpus.name,
        "smoke": args.smoke,
        "debug": args.debug,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "corpus_commit": corpus.commit,
        "corpus_display": corpus.display,
        "corpus_file_count": len(repository.files),
        "corpus_snapshot_sha256": repository.snapshot_sha256,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "harness": "dspy.ReAct",
        "sampling": SAMPLING,
        "study_sampling": mine.STUDY_SAMPLING,
        "tools": {**TOOL_CONFIG, "corpus_roots": list(corpus.roots)},
        "master_seed": args.seed,
        "attempt_iters": ATTEMPT_ITERS,
        "practice_dataset_sha256": practice_dataset_sha256(),
        "split": "stratified 70 study / 30 dev, seeded",
        "mining": {"model": "studier", "max_items": mine.MAX_ITEMS,
                   "max_map_entries": mine.MAX_MAP_ENTRIES},
        "assembly": {"model": "studier", "topic_max_items": assemble.TOPIC_MAX_ITEMS,
                     "topic_max_map": assemble.TOPIC_MAX_MAP,
                     "protocol_max_items": assemble.PROTOCOL_MAX_ITEMS,
                     "mode": "single-pass map-reduce, deterministic merge"},
        "factcheck": {"model": factcheck.MODEL, "effort": factcheck.EFFORT,
                      "role": "verdict-only; non-verified entries dropped deterministically"},
        "dev_judge": {"model": "gpt-5.4", "contract": "paper"},
    }
    manifest_path = run_root / "foldback.json"
    volatile = {"source_commit", "source_dirty"}
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if {k: v for k, v in existing.items() if k not in volatile} != {
            k: v for k, v in manifest.items() if k not in volatile
        }:
            raise SystemExit(f"study configuration changed; use a new --run-id: {manifest_path}")
        manifest = existing
    else:
        write_json(manifest_path, manifest)

    # ---- phase 1: split -----------------------------------------------------
    split_path = study_root / "split.json"
    if not split_path.exists():
        write_json(split_path, split_practice(rows, args.seed))
    split = read_json(split_path)
    study_ids = split["study_ids"][: 3 if args.smoke else None]
    dev_ids = split["dev_ids"][: 2 if args.smoke else None]
    log.info("split: %d study / %d dev questions", len(study_ids), len(dev_ids))

    # ---- phase 2: gold sandbox runs (stdout for mining) ---------------------
    def gold_run(qid: str) -> None:
        out = study_root / "gold_runs" / qid
        if (out / "verdict.txt").exists():
            return
        program, _ = extract_program(by_id[qid]["gold_answer"])
        result = run_program(program, out)
        if not passed(result):
            log.warning("gold program did not pass in sandbox: %s (%s)",
                        qid, result["stderr"][-200:])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(gold_run, study_ids))

    # ---- phase 3: attempts (forced k20, bare) --------------------------------
    def attempt(index: int, qid: str) -> None:
        path = study_root / "attempts" / f"{qid}.json"
        if path.exists() and read_json(path).get("status") in {"ok", "no_answer", "gave_up"}:
            return
        episode = None
        for try_no in range(2):
            episode = run_episode(
                corpus=corpus,
                tools=tools,
                question={"id": qid, "question": by_id[qid]["question"]},
                condition="foldback-study",
                budget="k20f-attempt",
                rollout=0,
                seed=stable_seed(args.seed, "foldback-attempt", qid, try_no),
                base_url=urls[index % len(urls)],
                max_iters=2 if args.smoke else ATTEMPT_ITERS,
                forced=True,
                debug=args.debug,
            )
            if episode["status"] in {"ok", "no_answer"}:
                break
        if episode["status"] not in {"ok", "no_answer"}:
            episode["status"] = "gave_up"
            log.warning("attempt gave up: %s (%s)", qid, episode.get("error", ""))
        write_json(path, episode)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda item: attempt(*item), enumerate(study_ids)))
    statuses = [read_json(study_root / "attempts" / f"{qid}.json")["status"] for qid in study_ids]
    log.info("attempts: %d ok, %d no_answer, %d gave_up",
             statuses.count("ok"), statuses.count("no_answer"), statuses.count("gave_up"))

    # ---- phase 4: mine -------------------------------------------------------
    studier = OpenAI(api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
                     base_url=urls[0], timeout=mine.TIMEOUT, max_retries=1)

    def mine_one(qid: str) -> None:
        path = study_root / "mined" / f"{qid}.json"
        if path.exists():
            return
        episode = read_json(study_root / "attempts" / f"{qid}.json")
        answer = episode.get("answer", "")
        program, _ = extract_program(answer)
        sandbox_dir = study_root / "attempt_runs" / qid
        attempt_sandbox = None
        if program is not None:
            attempt_sandbox = (read_json(sandbox_dir / "result.json")
                               if (sandbox_dir / "result.json").exists() else None)
            if attempt_sandbox is None:
                attempt_sandbox = run_program(program, sandbox_dir)
                write_json(sandbox_dir / "result.json", attempt_sandbox)
        gold_stdout_path = study_root / "gold_runs" / qid / "stdout.txt"
        record, usage = mine.mine_question(
            studier,
            library=corpus.display,
            row=by_id[qid],
            episode=episode,
            attempt_sandbox=attempt_sandbox,
            gold_stdout=gold_stdout_path.read_text(encoding="utf-8")[:2000]
            if gold_stdout_path.exists() else "",
            seed=stable_seed(args.seed, "foldback-mine", qid),
        )
        write_json(path, {"qid": qid, "topic": by_id[qid]["topic"],
                          "attempt_status": episode["status"],
                          "attempt_sandbox_passed": bool(attempt_sandbox and passed(attempt_sandbox)),
                          "usage": usage, **record})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(mine_one, study_ids))
    mined = [read_json(study_root / "mined" / f"{qid}.json") for qid in study_ids]
    log.info("mined: %d records, %d items, %d map entries",
             len(mined), sum(len(r["items"]) for r in mined),
             sum(len(r["map_entries"]) for r in mined))

    # ---- phase 5: assemble ---------------------------------------------------
    raw_object_path = study_root / "object_raw.json"
    if not raw_object_path.exists():
        by_topic: dict[str, list[dict]] = {}
        for record in mined:
            by_topic.setdefault(record["topic"], []).append(record)
        topics = {}
        ledger = {}
        for topic, records in sorted(by_topic.items()):
            payload, usage, prompt = assemble.reduce_topic(
                studier, library=corpus.display, topic=topic, records=records,
                seed=stable_seed(args.seed, "foldback-reduce", topic),
            )
            write_text(study_root / "assemble" / f"prompt_{topic}.txt", prompt)
            topics[topic] = payload
            ledger[topic] = usage
        stats = {"n": len(mined), "ok": 0, "missing": 0, "syntax": 0, "crashed": 0}
        for record in mined:
            sandbox_path = study_root / "attempt_runs" / record["qid"] / "result.json"
            if not sandbox_path.exists():
                stats["missing"] += 1  # the answer contained no fenced program
                continue
            result = read_json(sandbox_path)
            if passed(result):
                stats["ok"] += 1
            elif not result["compiled"]:
                stats["syntax"] += 1
            else:
                stats["crashed"] += 1
        protocol, usage, prompt = assemble.reduce_protocol(
            studier, library=corpus.display, stats=stats,
            seed=stable_seed(args.seed, "foldback-protocol"),
        )
        write_text(study_root / "assemble" / "prompt_protocol.txt", prompt)
        ledger["protocol"] = usage
        obj = assemble.build_object(protocol["items"], topics)
        write_json(raw_object_path, obj)
        write_json(study_root / "assemble" / "ledger.json",
                   {"usages": ledger, "stats": stats})
    raw_object = read_json(raw_object_path)

    # ---- phase 6: fact-check ---------------------------------------------------
    clean_object_path = study_root / "object_clean.json"
    if not clean_object_path.exists():
        verdicts = factcheck.check_object(raw_object, study_root / "factcheck")
        clean, report = factcheck.apply_verdicts(raw_object, verdicts)
        write_json(study_root / "factcheck" / "report.json", report)
        write_json(clean_object_path, clean)
        log.info("factcheck: dropped %d/%d entries", report["dropped"], report["total"])
    clean_object = read_json(clean_object_path)

    note = assemble.render(clean_object, corpus.display)
    write_text(run_root / "cheatsheet.md", note)

    mine_tokens = sum(r["usage"].get("completion_tokens", 0) for r in mined)
    assemble_ledger = read_json(study_root / "assemble" / "ledger.json")
    assemble_tokens = sum(u.get("completion_tokens", 0)
                          for u in assemble_ledger["usages"].values())
    attempts_tokens = sum(
        read_json(study_root / "attempts" / f"{qid}.json").get("gen_tokens", 0)
        for qid in study_ids
    )
    study = {
        "kind": "foldback",
        "schema_version": 1,
        "config_sha256": sha256_json(manifest),
        "study_questions": len(study_ids),
        "attempt_iters": ATTEMPT_ITERS,
        "studier_generated_tokens": {
            "attempts": attempts_tokens,
            "mining": mine_tokens,
            "assembly": assemble_tokens,
        },
        "object": {
            "protocol_items": len(clean_object["protocol"]),
            "content_items": sum(len(v) for v in clean_object["topics"].values()),
            "map_entries": len(clean_object["map"]),
            "factcheck_dropped": read_json(study_root / "factcheck" / "report.json")["dropped"],
        },
        "final_cheatsheet_sha256": sha256_text(note),
        "final_cheatsheet_chars": len(note),
    }
    write_json(run_root / "study.json", study)
    log.info("fold-back object: %d chars (%d protocol, %d content, %d map entries)",
             len(note), study["object"]["protocol_items"],
             study["object"]["content_items"], study["object"]["map_entries"])

    # ---- phase 7: dev evaluation ----------------------------------------------
    dev_rows = [by_id[qid] for qid in dev_ids]
    dev_common = dict(
        rows=dev_rows,
        corpus=corpus,
        tools=tools,
        base_urls=urls,
        api_key=os.environ["OPENAI_API_KEY"],
        master_seed=args.seed,
        budgets=("direct",) if args.smoke else ("direct", "k5"),
        rollouts=1 if args.smoke else 2,
        debug=args.debug,
    )
    variants = {"object": note, "bare": ""}
    if CHEATSHEET_NOTE.exists():
        variants["cheatsheet008"] = CHEATSHEET_NOTE.read_text(encoding="utf-8")
    reports = {}
    for variant, variant_note in variants.items():
        reports[variant] = evaluate_variant(
            variant=variant, note=variant_note,
            out_dir=run_root / "dev" / variant, **dev_common,
        )
    summary = {
        variant: {budget: round(result["mean_lenient"], 2)
                  for budget, result in report["budgets"].items()}
        for variant, report in reports.items()
    }
    write_json(run_root / "dev" / "summary.json", summary)
    log.info("dev summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
