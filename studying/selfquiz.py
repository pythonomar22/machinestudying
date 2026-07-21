"""Self-quizzing study loop: evolve a cheatsheet by quiz → attempt → verify → distill.

Iteration 1 of the self-quizzing studying method on the SmallDSPy corpus.
Per round: the quizmaster (GPT-5.4 in Codex — a declared external-teacher
cheat) writes sandbox-verified practice questions in the validation set's
register; the studier (Qwen3.5-9B, paper ReAct harness, current cheatsheet
prepended) attempts them; every attempt's program runs in the pinned sandbox;
the verifier (GPT-5.4) grades attempts and extracts lessons; the studier
distills the findings into its cheatsheet. The fixed validation set is then
evaluated for reporting only. After the final round, the run directory holds
cheatsheet.md + study.json, ready for `studybench.react --condition selfquiz`.

Firewalls: the held-out test set (data/smalldspy.jsonl) is read only for
decontamination-by-dropping; validation results never reach the quizmaster,
verifier, or distiller.

Usage (inside the Slurm allocation, vLLM replicas already serving):
    .venv-dspy/bin/python -m studying.selfquiz --run-id RUN --seed SEED \
        --base-urls http://127.0.0.1:PORT/v1[,...] [--smoke] [--debug]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from studybench.artifacts import read_json, sha256_json, sha256_text, stable_seed, write_json, write_text
from studybench.dataset import CORPORA, NOTE_PREFIX, ROOT, load_questions
from studybench.react import MODEL, MODEL_REVISION, SAMPLING, TOOL_CONFIG, make_tools
from studybench.react import run_episode
from studybench.tools import RepoTools

from . import distill as distiller
from . import quizmaster, verifier
from .sandbox import extract_program, run_program
from .validate import load_validation_questions, run_validation, validation_dataset_sha256

ROUNDS = 5
QUESTIONS_PER_ROUND = 6
MIN_ACCEPT = 4
ATTEMPT_MAX_ITERS = 20
VALIDATION_BUDGETS = ("direct", "k5")
TEST_SIMILARITY_MAX = 0.35
log = logging.getLogger("studying.selfquiz")


def _shingles(text: str) -> set[str]:
    words = re.sub(r"\s+", " ", text.lower()).strip().split()
    if len(words) < 3:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}


def _jaccard(a: str, b: str) -> float:
    left, right = _shingles(a), _shingles(b)
    return len(left & right) / len(left | right) if left and right else 0.0


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
        raise SystemExit("full self-quizzing runs require a clean committed source tree")
    return commit, dirty


def decontaminate(rows, test_rows, path: Path) -> None:
    """Textual check of the fixed validation set against the held-out test set."""

    pairs = []
    for row in rows:
        for test in test_rows:
            score = _jaccard(row["question"], test["question"])
            pairs.append({"validation": row["id"], "test": test["id"],
                          "question_jaccard3": round(score, 4)})
    worst = max(pair["question_jaccard3"] for pair in pairs)
    write_json(path, {
        "method": "word 3-gram jaccard on question text",
        "threshold": TEST_SIMILARITY_MAX,
        "max_observed": worst,
        "pairs": pairs,
    })
    if worst > TEST_SIMILARITY_MAX:
        raise SystemExit(
            f"validation set is contaminated against the test set (jaccard {worst})"
        )


def _attempt_deterministic(answer: str, run_dir: Path) -> dict:
    program, fences = extract_program(answer)
    record = {"fenced_python_blocks": fences}
    if program is None:
        record["note"] = "no fenced python program in the answer"
    else:
        record["sandbox"] = run_program(program, run_dir)
    return record


def study_round(
    *, round_no: int, args, corpus, tools, urls, verify_api, round_dir: Path,
    note: str, exemplars, prior_questions, prior_verdicts, test_rows,
) -> tuple[list[dict], dict[str, dict]]:
    """Run one complete study round; every phase is idempotent on disk."""

    questions_path = round_dir / "questions.json"
    if questions_path.exists():
        questions = read_json(questions_path)["questions"]
    else:
        accepted = quizmaster.generate_questions(
            round_no=round_no,
            num_questions=args.questions_per_round,
            exemplars=exemplars,
            prior_questions=prior_questions,
            prior_verdicts=prior_verdicts,
            gen_dir=round_dir / "gen",
            min_accept=args.min_accept,
        )
        dropped = []
        questions = []
        for item in accepted:
            worst = max((_jaccard(item["question"], test["question"]) for test in test_rows), default=0.0)
            (dropped if worst > TEST_SIMILARITY_MAX else questions).append(
                {**item, "max_test_jaccard3": round(worst, 4)}
            )
        if len(questions) < args.min_accept:
            raise SystemExit(
                f"round {round_no}: only {len(questions)} questions survived "
                "test-set decontamination"
            )
        write_json(questions_path, {
            "round": round_no,
            "questions": questions,
            "dropped_test_similar": dropped,
        })
    log.info("round %d: %d training questions", round_no, len(questions))

    prefix = NOTE_PREFIX.format(library=corpus.display, note=note) if note else ""

    def attempt(index: int, question: dict) -> None:
        path = round_dir / "attempts" / f"{question['qid']}.json"
        if path.exists() and read_json(path).get("status") in {"ok", "no_answer", "gave_up"}:
            return
        episode = None
        for try_no in range(2):
            episode = run_episode(
                corpus=corpus,
                tools=tools,
                question={"id": question["qid"], "question": prefix + question["question"]},
                condition="selfquiz-study",
                budget="attempt",
                rollout=0,
                seed=stable_seed(args.seed, "selfquiz-attempt", round_no, question["qid"], try_no),
                base_url=urls[index % len(urls)],
                max_iters=args.attempt_max_iters,
                forced=False,
                debug=args.debug,
            )
            if episode["status"] in {"ok", "no_answer"}:
                break
        if episode["status"] not in {"ok", "no_answer"}:
            episode["status"] = "gave_up"
            log.warning("round %d attempt gave up: %s (%s)",
                        round_no, question["qid"], episode.get("error", ""))
        episode["round"] = round_no
        write_json(path, episode)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda item: attempt(*item), enumerate(questions)))

    verdicts_path = round_dir / "verdicts.json"
    if verdicts_path.exists():
        verdicts = read_json(verdicts_path)["verdicts"]
    else:
        def judge(question: dict) -> tuple[str, dict]:
            episode = read_json(round_dir / "attempts" / f"{question['qid']}.json")
            answer = episode.get("answer", "")
            deterministic = _attempt_deterministic(
                answer, round_dir / "attempt_runs" / question["qid"]
            )
            verdict = verifier.verify_attempt(
                verify_api,
                library=corpus.display,
                question=question,
                attempt_answer=answer,
                deterministic=deterministic,
            )
            return question["qid"], {
                **verdict,
                "mechanism": question["mechanism"],
                "deterministic": deterministic,
                "attempt_status": episode["status"],
                "attempt_gen_tokens": episode.get("gen_tokens", 0),
            }

        with ThreadPoolExecutor(max_workers=4) as pool:
            verdicts = dict(pool.map(judge, questions))
        write_json(verdicts_path, {"round": round_no, "verifier": verifier.MODEL,
                                   "verdicts": verdicts})
    counts = {"correct": 0, "partial": 0, "incorrect": 0}
    for verdict in verdicts.values():
        counts[verdict["verdict"]] += 1
    log.info("round %d verdicts: %s", round_no, counts)

    note_path = round_dir / "cheatsheet.md"
    if not note_path.exists():
        findings = distiller.build_findings(questions, verdicts)
        new_note, usage = distiller.distill(
            base_url=urls[0],
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            library=corpus.display,
            current=note,
            findings=findings,
            seed=stable_seed(args.seed, "selfquiz-distill", round_no),
        )
        write_json(round_dir / "distill.json", {"round": round_no, "usage": usage,
                                                "note_chars": len(new_note)})
        write_text(note_path, new_note)
    log.info("round %d cheatsheet: %d chars", round_no,
             len(note_path.read_text(encoding="utf-8")))
    return questions, verdicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--questions-per-round", type=int, default=QUESTIONS_PER_ROUND)
    parser.add_argument("--attempt-max-iters", type=int, default=ATTEMPT_MAX_ITERS)
    parser.add_argument("--base-urls", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not args.run_id.replace("-", "").replace("_", "").isalnum():
        parser.error("--run-id must contain only letters, digits, '-' and '_'")
    if args.smoke:
        args.rounds = 1
        args.questions_per_round = 2
        args.attempt_max_iters = 5
        args.min_accept = 1
        validation_budgets = ("direct",)
    else:
        args.min_accept = MIN_ACCEPT
        validation_budgets = VALIDATION_BUDGETS
    urls = args.base_urls.split(",")

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT / "logs" / f"{args.run_id}-selfquiz.log"),
        ],
    )
    source_commit, source_dirty = _source_state(args.smoke)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required (quizmaster verification and validation judge)")

    corpus = CORPORA["smalldspy"]
    repository = RepoTools(corpus)
    tools = make_tools(repository)
    validation_rows = load_validation_questions()
    test_rows = load_questions("smalldspy")  # decontamination-by-dropping only
    exemplars = [row["question"] for row in validation_rows]

    run_root = ROOT / "runs" / args.run_id / corpus.name
    study_root = run_root / "study"
    manifest = {
        "schema_version": 1,
        "kind": "selfquiz-study",
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
        "tools": {**TOOL_CONFIG, "corpus_roots": list(corpus.roots)},
        "master_seed": args.seed,
        "rounds": args.rounds,
        "questions_per_round": args.questions_per_round,
        "min_accept": args.min_accept,
        "attempt_max_iters": args.attempt_max_iters,
        "cheatsheet_cap": distiller.CHEATSHEET_CAP,
        "validation_dataset_sha256": validation_dataset_sha256(),
        "validation_budgets": list(validation_budgets),
        "validation_exemplars": "all validation question texts, every round",
        "test_similarity_max": TEST_SIMILARITY_MAX,
        "quizmaster": {"model": quizmaster.MODEL, "effort": quizmaster.EFFORT,
                       "harness": "codex exec, read-only, corpus checkout"},
        "verifier": {"model": verifier.MODEL},
        "validation_judge": {"model": "gpt-5.4", "contract": "paper"},
    }
    # The stored manifest is authoritative on resume; source_commit is
    # provenance of the launch, not part of the immutable study design.
    manifest_path = run_root / "selfquiz.json"
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

    decontaminate(validation_rows, test_rows, study_root / "decontamination.json")

    validation_common = dict(
        rows=validation_rows,
        corpus=corpus,
        tools=tools,
        base_urls=urls,
        api_key=os.environ["OPENAI_API_KEY"],
        master_seed=args.seed,
        budgets=validation_budgets,
        debug=args.debug,
    )
    reports = [run_validation(
        note="", round_no=0, out_dir=study_root / "round_0" / "validation",
        **validation_common,
    )]

    note = ""
    all_questions: list[dict] = []
    round_summaries = []
    verify_api = verifier.client()
    for round_no in range(1, args.rounds + 1):
        round_dir = study_root / f"round_{round_no}"
        prior_verdicts = []
        for earlier in range(1, round_no):
            earlier_verdicts = read_json(study_root / f"round_{earlier}" / "verdicts.json")
            prior_verdicts.extend(earlier_verdicts["verdicts"].values())
        questions, verdicts = study_round(
            round_no=round_no,
            args=args,
            corpus=corpus,
            tools=tools,
            urls=urls,
            verify_api=verify_api,
            round_dir=round_dir,
            note=note,
            exemplars=exemplars,
            prior_questions=[question["question"] for question in all_questions],
            prior_verdicts=prior_verdicts,
            test_rows=test_rows,
        )
        all_questions.extend(questions)
        note = (round_dir / "cheatsheet.md").read_text(encoding="utf-8")
        reports.append(run_validation(
            note=note, round_no=round_no, out_dir=round_dir / "validation",
            **validation_common,
        ))
        counts = {"correct": 0, "partial": 0, "incorrect": 0}
        for verdict in verdicts.values():
            counts[verdict["verdict"]] += 1
        round_summaries.append({
            "round": round_no,
            "questions": len(questions),
            "verdicts": counts,
            "attempt_gen_tokens": sum(v["attempt_gen_tokens"] for v in verdicts.values()),
            "note_chars": len(note),
        })

    write_text(run_root / "cheatsheet.md", note)
    distill_tokens = sum(
        entry.get("completion_tokens", 0)
        for round_no in range(1, args.rounds + 1)
        for entry in read_json(study_root / f"round_{round_no}" / "distill.json")["usage"]
    )
    study = {
        "kind": "selfquiz",
        "schema_version": 1,
        "config_sha256": sha256_json(manifest),
        "rounds": round_summaries,
        "studier_generated_tokens": {
            "attempts": sum(entry["attempt_gen_tokens"] for entry in round_summaries),
            "distillation": distill_tokens,
        },
        "validation_trajectory": [
            {"round": report["round"],
             "budgets": {budget: result["mean_lenient"]
                         for budget, result in report["budgets"].items()}}
            for report in reports
        ],
        "final_cheatsheet_sha256": sha256_text(note),
        "final_cheatsheet_chars": len(note),
    }
    write_json(run_root / "study.json", study)
    log.info("self-quizzing study complete: %s", run_root)
    log.info("validation trajectory: %s",
             json.dumps(study["validation_trajectory"], indent=None))


if __name__ == "__main__":
    main()
