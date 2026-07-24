"""Distill teacher trajectories into a study object (fold-back v1).

The distiller agent (codex gpt-5.4-mini — the studier itself) reads, per
practice question: its own closed-book attempt (and grade), its 5-command
attempt (tool_log + answer + grade), and the practice gold. It performs
delta attribution — what knowledge flipped which claims — under general
extraction discipline: quote-don't-paraphrase, diff-your-priors, nominate
load-bearing excerpts, tag reuse. Assembly is single-pass: deterministic
components (prior-diff cards, verbatim excerpt pack pulled from the
corpus by code, never from model memory) plus one model-written
fact/recipe sheet that is corpus fact-checked entry-by-entry (verdicts
only; non-verified entries dropped).

Usage:
    .venv-dspy/bin/python -m studying.foldback.teachermine mine \
        --run-id RUN --teacher-run dspy-teachermini-20260722 --seed SEED
    .venv-dspy/bin/python -m studying.foldback.teachermine build --run-id RUN
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from studybench.artifacts import read_json, sha256_text, write_json, write_text
from studybench.dataset import ROOT

from .data import load_practice_questions, split_practice

MODEL, EFFORT = "gpt-5.4-mini", "medium"
FACTCHECK_MODEL, FACTCHECK_EFFORT = "gpt-5.4", "xhigh"
CODEX_TIMEOUT = 5_400
CORPUS_REPO = ROOT / "corpora" / "dspy"
WORKERS = 8
MAX_EXCERPTS = 25
MAX_EXCERPT_LINES = 40
MAX_PRIOR_CARDS = 40
log = logging.getLogger("studying.foldback.teachermine")

MINE_PROMPT = """You are studying the {library} repository. Below are YOUR OWN two attempts at one practice question about it: a closed-book attempt (scored {direct_score}/100) and a 5-command exploration attempt (scored {k5_score}/100), with every command you ran, why, and what it revealed; the per-claim grades of both attempts; and the verified reference program.

Your task is not to summarize. It is delta attribution: for the rubric claims your exploration attempt got right and your closed-book attempt got wrong, identify EXACTLY what knowledge made the difference — then package that knowledge so your future self can answer such questions closed-book.

## Extraction discipline (hard rules)
- QUOTE, don't paraphrase: every fact carries a verbatim anchor (the exact signature, value, or line from a file you read), with the file path. If you cannot ground a fact in a trajectory discovery, the reference program, or the repository in front of you (you may re-verify — it is your working directory), do not extract it.
- DIFF YOUR PRIORS: where the closed-book attempt asserted something false, record the pair — what you believed, what is true, which file proves it. Address it to yourself; closed-book, you WILL believe it again.
- NOMINATE EXCERPTS: list the file regions (path, 1-indexed line range, at most {max_lines} lines) that were load-bearing — the reads that flipped claims. They will be included verbatim.
- TAG REUSE: for every fact — useful across many questions about this library (general), across this topic (topic), or only here (question_only)?
- NOMINATE EXEMPLARS: if your verified exploration program is a clean template for a whole family of offline proofs, say so and name the family.

## Question (topic: {topic})
{question}

## Your closed-book attempt (scored {direct_score}/100)
{direct_answer}

## Its per-claim grade
{direct_claims}

## Your 5-command exploration
{tool_log}

## Your exploration attempt (scored {k5_score}/100)
{k5_answer}

## Its per-claim grade
{k5_claims}

## Verified reference program
{gold}

Return JSON matching the provided schema and nothing else."""

MINE_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "anchor_quote": {"type": "string"},
                    "anchor_file": {"type": "string"},
                    "reuse": {"type": "string", "enum": ["general", "topic", "question_only"]},
                },
                "required": ["fact", "anchor_quote", "anchor_file", "reuse"],
                "additionalProperties": False,
            },
        },
        "prior_diffs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "you_will_believe": {"type": "string"},
                    "truth": {"type": "string"},
                    "file": {"type": "string"},
                },
                "required": ["you_will_believe", "truth", "file"],
                "additionalProperties": False,
            },
        },
        "excerpt_nominations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "why": {"type": "string"},
                },
                "required": ["path", "start_line", "end_line", "why"],
                "additionalProperties": False,
            },
        },
        "recipes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "pattern": {"type": "string", "description": "minimal runnable pattern, <=15 lines"},
                    "source_file": {"type": "string"},
                },
                "required": ["name", "pattern", "source_file"],
                "additionalProperties": False,
            },
        },
        "exemplar": {
            "type": "object",
            "properties": {
                "is_exemplar": {"type": "boolean"},
                "family": {"type": "string"},
            },
            "required": ["is_exemplar", "family"],
            "additionalProperties": False,
        },
    },
    "required": ["facts", "prior_diffs", "excerpt_nominations", "recipes", "exemplar"],
    "additionalProperties": False,
}

SHEET_PROMPT = """You are studying the {library} repository. Below are the lesson records you mined from your own {n} practice trajectories: facts with verbatim anchors, and offline proof recipes. Organize them into the fact/recipe sheet of your study notes — the reference your closed-book future self will read before answering questions.

Rules:
- Merge duplicates; when facts recur across records, keep the sharpest phrasing and union the anchors. Order sections by how often their content recurred (recurring beats rare).
- Choose your own section structure. Every item must keep a verbatim anchor (quote + file path) — items without anchors will be deleted downstream.
- Keep recipes as minimal runnable patterns (<=15 lines), one per distinct proof family.
- Compact and concrete; no filler prose. At most {max_items} items total.

## Mined records
{records_json}

Return JSON matching the provided schema and nothing else."""

SHEET_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "items": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "items"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["sections"],
    "additionalProperties": False,
}

FACTCHECK_PROMPT = """You are fact-checking study notes about this repository (the pinned checkout in your working directory). Each entry has an `id`. For EVERY entry decide: `verified` (accurate for THIS checkout — quotes real, claims true, patterns would run), `false` (contradicted), or `unverifiable`.

## Entries
{entries_json}

Return JSON matching the provided schema and nothing else."""


def run_codex(prompt: str, out_dir: Path, schema: dict, tag: str,
              model: str = MODEL, effort: str = EFFORT) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_text(out_dir / f"prompt_{tag}.txt", prompt)
    schema_path = out_dir / f"schema_{tag}.json"
    schema_path.write_text(json.dumps(schema, indent=1), encoding="utf-8")
    last_message = out_dir / f"last_message_{tag}.json"
    events_path = out_dir / f"events_{tag}.jsonl"
    command = [
        "codex", "exec", "-m", model, "-c", f"model_reasoning_effort={effort}",
        "-s", "read-only", "-C", str(CORPUS_REPO),
        "--output-schema", str(schema_path), "-o", str(last_message), "--json", "-",
    ]
    with open(events_path, "w", encoding="utf-8") as events:
        result = subprocess.run(command, input=prompt, stdout=events,
                                stderr=subprocess.PIPE, text=True, timeout=CODEX_TIMEOUT)
    if result.returncode != 0:
        raise RuntimeError(f"codex exec failed ({result.returncode}): {result.stderr[-2000:]}")
    return json.loads(last_message.read_text(encoding="utf-8"))


def _claims_summary(grade: dict) -> str:
    return "\n".join(
        f"- {c['claim_id']} (weight {w['weight']}): score {c['score']} — {c['rationale'][:160]}"
        for c, w in zip(grade["claims"], grade["rubric"])
    )


def cmd_mine(args) -> None:
    rows = {r["id"]: r for r in load_practice_questions()}
    split = split_practice(tuple(rows.values()), args.seed)
    study_ids = split["study_ids"][: args.limit or None]
    troot = ROOT / "runs" / args.teacher_run / "dspy"
    run_root = ROOT / "runs" / args.run_id / "dspy" / "study"
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "mine_manifest.json", {
        "kind": "foldback-teachermine",
        "teacher_run": args.teacher_run,
        "miner_model": MODEL, "miner_effort": EFFORT,
        "master_seed": args.seed,
        "inputs": "k5f tool_log+answer+grade, direct answer+grade, practice gold",
        "questions": list(study_ids),
        "mine_prompt_sha256": sha256_text(MINE_PROMPT),
    })

    def mine_one(qid: str) -> None:
        out = run_root / "mined" / f"{qid}.json"
        if out.exists():
            return
        row = rows[qid]
        k5 = read_json(troot / "k5f" / qid / "last_message.json")
        g_direct = read_json(troot / "grades" / "direct" / f"{qid}.json")
        g_k5 = read_json(troot / "grades" / "k5f" / f"{qid}.json")
        for g in (g_direct, g_k5):
            g["rubric"] = row["rubric"]
        tool_log = "\n".join(
            f"[{s['step']}] {s['command']}\n    why: {s['motivation']}\n    got: {s['discovery']}"
            for s in k5["tool_log"]
        )
        prompt = MINE_PROMPT.format(
            library="DSPy",
            topic=row["topic"],
            question=row["question"],
            direct_score=g_direct["lenient"],
            k5_score=g_k5["lenient"],
            direct_answer=(troot / "direct" / qid / "answer.md").read_text(encoding="utf-8")[:6000],
            direct_claims=_claims_summary(g_direct),
            tool_log=tool_log[:8000],
            k5_answer=k5["answer"][:6000],
            k5_claims=_claims_summary(g_k5),
            gold=row["gold_answer"][:6000],
            max_lines=MAX_EXCERPT_LINES,
        )
        record = run_codex(prompt, run_root / "mine_sessions" / qid, MINE_SCHEMA, "mine")
        record["qid"], record["topic"] = qid, row["topic"]
        write_json(out, record)
        log.info("mined %s: %d facts, %d priors, %d excerpts, %d recipes",
                 qid, len(record["facts"]), len(record["prior_diffs"]),
                 len(record["excerpt_nominations"]), len(record["recipes"]))

    failures = []

    def safe(qid):
        try:
            mine_one(qid)
        except Exception as error:
            failures.append((qid, str(error)[:200]))
            log.error("mine %s FAILED: %s", qid, str(error)[:200])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(safe, study_ids))
    if failures:
        raise SystemExit(f"{len(failures)} mining session(s) failed; rerun to retry: "
                         f"{[q for q, _ in failures]}")
    log.info("mining complete: %d records", len(study_ids))


def _relpath(path: str) -> str:
    prefix = str(CORPUS_REPO) + "/"
    return path[len(prefix):] if path.startswith(prefix) else path


def _excerpt_pack(records: list[dict]) -> tuple[str, list[dict]]:
    """Deterministic: dedupe nominated ranges, pull verbatim from the corpus."""

    votes: Counter = Counter()
    spans: dict[tuple, dict] = {}
    for record in records:
        for nom in record["excerpt_nominations"]:
            path = _relpath(nom["path"])
            if not (CORPUS_REPO / path).is_file():
                continue
            start = max(1, int(nom["start_line"]))
            end = min(int(nom["end_line"]), start + MAX_EXCERPT_LINES - 1)
            key = (path, start // 20)  # merge nominations pointing at the same region
            votes[key] += 1
            best = spans.get(key)
            if best is None or (end - start) > (best["end"] - best["start"]):
                spans[key] = {"path": path, "start": start, "end": end, "why": nom["why"]}
    chosen = [spans[key] for key, _ in votes.most_common(MAX_EXCERPTS)]
    parts, manifest = [], []
    for span in chosen:
        lines = (CORPUS_REPO / span["path"]).read_text(encoding="utf-8").splitlines()
        start, end = span["start"], min(span["end"], len(lines))
        body = "\n".join(f"{i:4}: {lines[i - 1]}" for i in range(start, end + 1))
        parts.append(f"$ sed -n '{start},{end}p' {span['path']}   # {span['why'][:90]}\n{body}")
        manifest.append({"path": span["path"], "start_line": start, "end_line": end,
                         "votes": votes[(span["path"], start // 20)]})
    return "\n\n".join(parts), manifest


def _prior_cards(records: list[dict]) -> list[str]:
    seen, cards = set(), []
    for record in records:
        for diff in record["prior_diffs"]:
            key = diff["truth"][:80].lower()
            if key in seen:
                continue
            seen.add(key)
            cards.append(f"- You will believe: {diff['you_will_believe']} — WRONG. "
                         f"Truth: {diff['truth']} (see `{_relpath(diff['file'])}`)")
    return cards[:MAX_PRIOR_CARDS]


def cmd_build(args) -> None:
    run_root = ROOT / "runs" / args.run_id / "dspy" / "study"
    manifest = read_json(run_root / "mine_manifest.json")
    records = [read_json(run_root / "mined" / f"{qid}.json") for qid in manifest["questions"]
               if (run_root / "mined" / f"{qid}.json").exists()]
    log.info("building object from %d mined records", len(records))

    # 1. deterministic components
    excerpts_text, excerpt_manifest = _excerpt_pack(records)
    cards = _prior_cards(records)

    # 2. model-written fact/recipe sheet (single pass over all records)
    sheet_path = run_root / "sheet.json"
    if not sheet_path.exists():
        slim = [{"topic": r["topic"], "facts": r["facts"], "recipes": r["recipes"]}
                for r in records]
        sheet = run_codex(
            SHEET_PROMPT.format(library="DSPy", n=len(records), max_items=80,
                                records_json=json.dumps(slim, ensure_ascii=False)),
            run_root / "assemble", SHEET_SCHEMA, "sheet",
        )
        write_json(sheet_path, sheet)
    sheet = read_json(sheet_path)

    # 3. fact-check the model-written sheet, entry by entry (verdict-only)
    checked_path = run_root / "sheet_checked.json"
    if not checked_path.exists():
        entries, ids = [], []
        for s_index, section in enumerate(sheet["sections"]):
            for i_index, item in enumerate(section["items"]):
                entry_id = f"s{s_index}i{i_index}"
                entries.append({"id": entry_id, "section": section["title"], "text": item})
                ids.append(entry_id)
        verdict_schema = {
            "type": "object",
            "properties": {"verdicts": {
                "type": "array", "minItems": len(ids), "maxItems": len(ids),
                "items": {"type": "object", "properties": {
                    "id": {"type": "string", "enum": ids},
                    "verdict": {"type": "string", "enum": ["verified", "false", "unverifiable"]},
                }, "required": ["id", "verdict"], "additionalProperties": False},
            }},
            "required": ["verdicts"], "additionalProperties": False,
        }
        result = run_codex(
            FACTCHECK_PROMPT.format(entries_json=json.dumps(entries, ensure_ascii=False, indent=1)),
            run_root / "factcheck", verdict_schema, "check",
            model=FACTCHECK_MODEL, effort=FACTCHECK_EFFORT,
        )
        verdicts = {v["id"]: v["verdict"] for v in result["verdicts"]}
        dropped = 0
        checked = {"sections": []}
        for s_index, section in enumerate(sheet["sections"]):
            kept = [item for i_index, item in enumerate(section["items"])
                    if verdicts.get(f"s{s_index}i{i_index}") == "verified"]
            dropped += len(section["items"]) - len(kept)
            if kept:
                checked["sections"].append({"title": section["title"], "items": kept})
        checked["factcheck_dropped"] = dropped
        write_json(checked_path, checked)
        log.info("factcheck: dropped %d entries", dropped)
    checked = read_json(checked_path)

    # 4. deterministic render
    parts = ["# DSPy Study Notes (self-distilled from practice trajectories)"]
    parts.append(
        "## How to answer\n"
        "Deliver exactly ONE fenced ```python program: self-contained, offline "
        "(use `dspy.utils.dummies.DummyLM` for any LM — it consumes a list of "
        "dicts mapping each OUTPUT FIELD NAME to a string, one dict per call), "
        "ending with the prints/assertions the question asks for. Prefer APIs "
        "and patterns confirmed in the notes below; where your memory and these "
        "notes disagree, the notes win — they were verified against the repository."
    )
    if cards:
        parts.append("## Corrections to your own priors\n" + "\n".join(cards))
    for section in checked["sections"]:
        parts.append(f"## {section['title']}\n" + "\n".join(f"- {item}" for item in section["items"]))
    if excerpts_text:
        parts.append("## Verified repository excerpts (as if you had run these reads)\n"
                     + excerpts_text)
    note = "\n\n".join(parts) + "\n"
    out_root = ROOT / "runs" / args.run_id / "dspy"
    write_text(out_root / "cheatsheet.md", note)
    write_json(out_root / "study.json", {
        "kind": "foldback-teachermine-v1",
        "records": len(records),
        "prior_cards": len(cards),
        "sheet_sections": len(checked["sections"]),
        "factcheck_dropped": checked.get("factcheck_dropped", 0),
        "excerpts": excerpt_manifest,
        "note_chars": len(note),
        "note_sha256": sha256_text(note),
    })
    log.info("study object written: %d chars -> %s", len(note), out_root / "cheatsheet.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("mine", "build"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--teacher-run", default="dspy-teachermini-20260722")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(ROOT / "logs" / f"{args.run_id}-teachermine.log")])
    if args.command == "mine":
        cmd_mine(args)
    else:
        cmd_build(args)


if __name__ == "__main__":
    main()
