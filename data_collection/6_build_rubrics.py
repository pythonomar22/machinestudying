# /// script
# requires-python = ">=3.12"
# ///
"""Stage 4 (paper): private rubric construction with GPT-5.4.

Paper, verbatim: "For each selected item, GPT-5.4 converts the gold answer
into atomic grading claims. Each claim is classified as core or supporting
and cited to specific line spans in the evidence files. Numbered file
dumps are provided to the rubric builder. Claim weights total 100, with
most weight assigned to core claims."

Ours: the same `codex exec` harness as Stages 3a/3b (gpt-5.4, xhigh,
read-only sandbox at the scope's checkout) runs the A.4 template verbatim,
one session per finalist, given the question, gold answer, evidence
references, and numbered dumps of the evidence files (format `0006: ...`,
matching the released bundles' excerpts). The builder returns claims and
span definitions; excerpts are then computed deterministically from the
checkout, never copied from model output. Output records carry the
released-bundle fields (id, topic, question, gold_answer, rubric,
evidence) plus our bookkeeping tags.

Usage:
    uv run data_collection/6_build_rubrics.py [fulldspy|smalldspy|all]

Idempotent: a finalist with a valid per-question output file is skipped;
per-item failures are reported and skipped so a rerun picks them up.
Output: artifacts/6_build_rubrics/<scope>/ with per-question prompts,
codex event logs, raw last-messages, and per-question records, plus the
merged 6_fulldspy_rubrics.json / 6_smalldspy_rubrics.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DC = Path(__file__).resolve().parent
ARTIFACTS = DC / "artifacts" / "6_build_rubrics"
FINALISTS_DIR = DC / "artifacts" / "5_critic_selection"


def load_stage3a():
    spec = importlib.util.spec_from_file_location("stage3a", DC / "4_generate_candidates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load_stage3a()

WORKERS = 4          # concurrent codex sessions per scope
MAX_SPAN_LINES = 300  # paper: "never exceed 300 lines"

# ---------------------------------------------------------------------------
# A.4 rubric-builder template, transcribed VERBATIM from the paper appendix.
# ---------------------------------------------------------------------------
RUBRIC_TEMPLATE = """You are building a private grading rubric for one {library_name} expert QA benchmark question.
Your output is confidential and will only be used by the evaluator.

## Goal
- Turn the gold answer into 2-8 atomic grading claims.
- Claims should be small enough to score independently.
- Together, the claims should capture what a strong code-grounded answer must say.

## Rules
- Use only the provided question, gold answer, evidence references, and evidence file contents.
- Make every claim judgeable from code and tests alone.
- Use `core` for essential mechanisms or facts that define correctness.
- Use `supporting` for narrower detail, nuance, edge cases, or examples.
- Claims should be minimally overlapping.
- The claim weights must sum to exactly 100.
- `core` claims should carry most of the total weight.
- Every claim must cite 1-3 evidence spans.
- Every evidence span must come from the provided files only.
- Use exact line numbers from the numbered file dumps.
- Keep spans focused. Prefer 1-40 lines when possible, and never exceed 300 lines.
- Reuse spans across claims when that is the cleanest grounding.
- Do not include any public-release wording, benchmarking commentary, or grading instructions in the claim text.

## Inputs
- Question ID: `{question_id}`
- Label: `{label}`
- Question: `{question}`
- Gold answer:
{gold_answer}

## Evidence references
{evidence_references_json}

## Full evidence files
{evidence_files_text}

Return JSON that matches the schema exactly."""

# The paper does not publish the rubric builder's output schema. Ours asks
# for exactly the fields the released bundles carry (rubric claims + span
# definitions); excerpts are computed by us from the checkout afterwards.
RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "rubric": {
            "type": "array",
            "minItems": 2,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "claim_type": {"type": "string", "enum": ["core", "supporting"]},
                    "weight": {"type": "integer"},
                    "statement": {"type": "string"},
                    "span_ids": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {"type": "string"},
                    },
                },
                "required": ["claim_id", "claim_type", "weight", "statement", "span_ids"],
                "additionalProperties": False,
            },
        },
        "evidence": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "span_id": {"type": "string"},
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["span_id", "path", "start_line", "end_line"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["rubric", "evidence"],
    "additionalProperties": False,
}


def question_id(question: str) -> str:
    return "dspy_" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]


def numbered_dump(path: str, repo: Path) -> str:
    lines = (repo / path).read_text(encoding="utf-8").splitlines()
    body = "\n".join(f"{i:04d}: {line}" for i, line in enumerate(lines, start=1))
    return f"### {path}\n{body}"


def evidence_paths(finalist: dict) -> list[str]:
    paths = []
    for evidence in finalist["code_evidence"]:
        if evidence["file"] not in paths:
            paths.append(evidence["file"])
    return paths


def rubric_violations(result: dict, paths: list[str], repo: Path) -> list[str]:
    problems = []
    rubric, spans = result["rubric"], result["evidence"]
    span_ids = [span["span_id"] for span in spans]
    if len(set(span_ids)) != len(span_ids):
        problems.append("duplicate span_id in evidence")
    claim_ids = [claim["claim_id"] for claim in rubric]
    if len(set(claim_ids)) != len(claim_ids):
        problems.append("duplicate claim_id in rubric")
    total = sum(claim["weight"] for claim in rubric)
    if total != 100:
        problems.append(f"claim weights sum to {total}, must be exactly 100")
    core = sum(claim["weight"] for claim in rubric if claim["claim_type"] == "core")
    if core <= 50:
        problems.append(f"core claims carry {core}/100 - they must carry most of the weight")
    defined = set(span_ids)
    for claim in rubric:
        missing = [sid for sid in claim["span_ids"] if sid not in defined]
        if missing:
            problems.append(f"claim {claim['claim_id']} cites undefined spans {missing}")
    line_counts = {
        path: len((repo / path).read_text(encoding="utf-8").splitlines())
        for path in paths
    }
    for span in spans:
        sid, path = span["span_id"], span["path"]
        if path not in line_counts:
            problems.append(f"span {sid}: path '{path}' is not one of the provided files")
            continue
        start, end = span["start_line"], span["end_line"]
        if not 1 <= start <= end <= line_counts[path]:
            problems.append(f"span {sid}: lines {start}-{end} out of range for {path} "
                            f"(1-{line_counts[path]})")
        elif end - start + 1 > MAX_SPAN_LINES:
            problems.append(f"span {sid}: {end - start + 1} lines exceeds {MAX_SPAN_LINES}")
    return problems


def normalize_ids(rubric: list[dict], evidence: list[dict]) -> tuple[list[dict], list[dict]]:
    """Rename claim/span IDs to the released bundles' c1../s1.. style."""
    span_map = {span["span_id"]: f"s{i}" for i, span in enumerate(evidence, start=1)}
    claims = [
        {**claim, "claim_id": f"c{i}",
         "span_ids": [span_map[sid] for sid in claim["span_ids"]]}
        for i, claim in enumerate(rubric, start=1)
    ]
    spans = [{**span, "span_id": span_map[span["span_id"]]} for span in evidence]
    return claims, spans


def run_codex(prompt: str, scope_dir: Path, repo: Path, qid: str, attempt: int) -> dict:
    last_message = scope_dir / f"last_message_{qid}_a{attempt}.json"
    events_path = scope_dir / f"events_{qid}_a{attempt}.jsonl"
    command = [
        "codex", "exec",
        "-m", gen.MODEL,
        "-c", f"model_reasoning_effort={gen.EFFORT}",
        "-s", "read-only",
        "-C", str(repo),
        "--output-schema", str(scope_dir / "rubric_schema.json"),
        "-o", str(last_message),
        "--json",
        "-",
    ]
    with open(events_path, "w", encoding="utf-8") as events:
        result = subprocess.run(
            command, input=prompt, stdout=events, stderr=subprocess.PIPE,
            text=True, timeout=gen.CODEX_TIMEOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"codex exec failed ({result.returncode}): {result.stderr[-2000:]}")
    return json.loads(last_message.read_text(encoding="utf-8"))


def build_rubric(finalist: dict, scope: str) -> dict:
    qid = question_id(finalist["question"])
    repo = gen.REPO_BY_SCOPE[scope]
    scope_dir = ARTIFACTS / scope
    output_path = scope_dir / f"6_q_{qid}.json"
    if output_path.exists():
        print(f"{scope}/{qid}: output exists, skipping")
        return json.loads(output_path.read_text(encoding="utf-8"))

    paths = evidence_paths(finalist)
    prompt = RUBRIC_TEMPLATE.format(
        library_name=gen.DSPY_VALUES["library_name"],
        question_id=qid,
        label=finalist["topic"],
        question=finalist["question"],
        gold_answer=finalist["answer"],
        evidence_references_json=json.dumps(finalist["code_evidence"],
                                            ensure_ascii=False, indent=2),
        evidence_files_text="\n\n".join(numbered_dump(path, repo) for path in paths),
    )
    (scope_dir / f"prompt_{qid}.txt").write_text(prompt, encoding="utf-8")

    result, problems = None, ["not run"]
    for archived in range(gen.MAX_RETRIES, -1, -1):  # latest archived attempt wins
        salvage = scope_dir / f"last_message_{qid}_a{archived}.json"
        if salvage.exists():
            result = json.loads(salvage.read_text(encoding="utf-8"))
            problems = rubric_violations(result, paths, repo)
            if not problems:
                print(f"{scope}/{qid}: salvaged valid archived attempt a{archived}")
                break
    for attempt in range(gen.MAX_RETRIES + 1):
        if not problems:
            break
        attempt_prompt = prompt if attempt == 0 else (
            prompt + "\n\n## Corrections required\nYour previous output had "
            "these problems; fix them and return the complete JSON again:\n- "
            + "\n- ".join(problems)
        )
        print(f"{scope}/{qid}: codex attempt {attempt + 1} ...", flush=True)
        result = run_codex(attempt_prompt, scope_dir, repo, qid, attempt)
        problems = rubric_violations(result, paths, repo)
        if problems:
            print(f"{scope}/{qid}: violations: {problems[:3]} ...")
    if problems:
        raise RuntimeError(f"{scope}/{qid}: unresolved violations after retries: {problems}")

    used = {sid for claim in result["rubric"] for sid in claim["span_ids"]}
    cited = [span for span in result["evidence"] if span["span_id"] in used]
    rubric, spans = normalize_ids(result["rubric"], cited)
    evidence = []
    for span in spans:
        lines = (repo / span["path"]).read_text(encoding="utf-8").splitlines()
        excerpt = "\n".join(
            f"{i:04d}: {lines[i - 1]}"
            for i in range(span["start_line"], span["end_line"] + 1)
        )
        evidence.append({**span, "excerpt": excerpt})

    record = {
        "id": qid,
        "topic": finalist["topic"],
        "question": finalist["question"],
        "gold_answer": finalist["answer"],
        "rubric": rubric,
        "evidence": evidence,
        "difficulty": finalist["difficulty"],
        "note": finalist["note"],
        "source_index": finalist["source_index"],
        "smalldspy_scope": finalist["smalldspy_scope"],
    }
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    core = sum(c["weight"] for c in record["rubric"] if c["claim_type"] == "core")
    print(f"{scope}/{qid}: {len(record['rubric'])} claims, core weight {core}, "
          f"{len(evidence)} spans")
    return record


def run_scope(scope: str) -> None:
    repo = gen.REPO_BY_SCOPE[scope]
    if subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip() != gen.PINNED_COMMIT:
        raise RuntimeError(f"{repo} is not at the pinned commit")
    scope_dir = ARTIFACTS / scope
    scope_dir.mkdir(parents=True, exist_ok=True)
    (scope_dir / "rubric_schema.json").write_text(
        json.dumps(RUBRIC_SCHEMA, indent=1), encoding="utf-8"
    )
    finalists = json.loads(
        (FINALISTS_DIR / f"5_{scope}_finalists.json").read_text(encoding="utf-8")
    )["finalists"]

    records, failures = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(build_rubric, f, scope): f for f in finalists}
        for future, finalist in futures.items():
            try:
                records.append(future.result())
            except Exception as error:
                failures.append((question_id(finalist["question"]), str(error)))
                print(f"{scope}/{question_id(finalist['question'])}: FAILED: "
                      f"{str(error)[:300]}", flush=True)
    if failures:
        raise RuntimeError(f"{scope}: {len(failures)} rubric(s) failed - rerun to "
                           f"retry them: {[qid for qid, _ in failures]}")

    order = {question_id(f["question"]): i for i, f in enumerate(finalists)}
    records.sort(key=lambda r: order[r["id"]])
    merged = {
        "scope": scope,
        "harness": f"codex exec (codex-cli), model {gen.MODEL}, effort {gen.EFFORT}, "
                   "read-only sandbox",
        "repository": str(repo),
        "commit": gen.PINNED_COMMIT,
        "num_questions": len(records),
        "questions": records,
    }
    merged_path = ARTIFACTS / f"6_{scope}_rubrics.json"
    merged_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"{scope}: wrote {len(records)} rubric bundles to {merged_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", nargs="?", default="all",
                        choices=["fulldspy", "smalldspy", "all"])
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for scope in (("fulldspy", "smalldspy") if args.scope == "all" else (args.scope,)):
        run_scope(scope)


if __name__ == "__main__":
    main()
