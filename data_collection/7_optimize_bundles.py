# /// script
# requires-python = ">=3.12"
# ///
"""Stage 5 (paper): bundle optimization with checker, sandbox, and self-grading.

Paper, verbatim: "A Codex agent uses a deterministic syntax checker and a
sandbox to run and verify answers. The questions, answers, and rubrics are
optimized together. Human experts confirm that the questions and rubrics
are fair and that the answers are correct. Answers and rubrics are further
refined using the syntax checker and sandbox as debugging tools, iterating
until the reference answers achieve full scores under the private rubric."

Ours, per bundle: (1) deterministic syntax check - every fenced python
block in the gold answer must compile; (2) sandbox - the reference program
(the longest block) must run to completion offline in the pinned-corpus
venv; (3) self-grade - GPT-5.4 grades the gold answer under the private
rubric with the A.5 template verbatim, and the bundle passes only at score
100 with no regrade flag. Failures go to a revision agent (same codex
harness, read-only at the scope checkout) that revises answer, rubric, and
question jointly; revisions are re-validated (Stage-4 rubric rules),
excerpts recomputed from the checkout, and the loop repeats up to
MAX_ROUNDS. Bundles still failing are dropped from the final set and
reported. The paper's human-expert review is SKIPPED - not substituted.

Usage:
    uv run data_collection/7_optimize_bundles.py [fulldspy|smalldspy|all]

Idempotent per bundle. Output: artifacts/7_optimize_bundles/<scope>/ with
per-bundle prompts, event logs, sandbox run dirs, and per-bundle records,
plus the final validation sets 7_fulldspy_validation.json/.jsonl and
7_smalldspy_validation.json/.jsonl (canonical released-bundle fields in
the .jsonl; full metadata in the .json).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DC = Path(__file__).resolve().parent
ROOT = DC.parent
ARTIFACTS = DC / "artifacts" / "7_optimize_bundles"
RUBRICS_DIR = DC / "artifacts" / "6_build_rubrics"
VENV_PY = ROOT / ".venv-dspy" / "bin" / "python"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, DC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load_module("stage3a", "4_generate_candidates.py")
s6 = load_module("stage4", "6_build_rubrics.py")

WORKERS = 4
MAX_ROUNDS = 3        # revision rounds before a bundle is dropped
RUN_TIMEOUT = 180     # seconds per sandbox execution
CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)

# ---------------------------------------------------------------------------
# A.5 grader template, transcribed VERBATIM from the paper appendix.
# ---------------------------------------------------------------------------
GRADER_TEMPLATE = """You are grading one model answer for a private {library_name} expert QA benchmark.

## Scoring model
- The question gets one final continuous 0-100 score.
- Claims are only the internal rubric used to compute that question's score.
- Score each claim as:
- `0` = wrong or missing
- `0.5` = partially correct but incomplete, vague, or only partly grounded
- `1` = fully correct
- Do not give extra credit for material outside the rubric.
- If an answer is polished but misses essential content, score the missing claims low.
- Use the evidence spans and gold answer to resolve ambiguity.

## Output rules
- Score every rubric claim exactly once.
- `question_score` must equal the weighted sum of the claim scores.
- Set `needs_regrade` to `true` only if the rubric or evidence is genuinely insufficient to judge the answer confidently.
- Keep rationales concise and specific.

## Inputs
- Question ID: `{question_id}`
- Label: `{label}`
- Question: `{question}`
- Model answer:
{model_answer}

## Gold answer
{gold_answer}

## Claim rubric
{claim_rubric_json}

## Evidence spans
{evidence_spans_json}

## Whole evidence files
{whole_evidence_text}

Return JSON that matches the schema exactly."""

# ---------------------------------------------------------------------------
# Revision prompt: OURS (the paper publishes no optimizer prompt). It binds
# the reviser to the A.2 naming discipline, the A.4 rubric rules, and the
# runnable-offline requirement, and demands minimal joint edits.
# ---------------------------------------------------------------------------
REVISE_TEMPLATE = """You are optimizing one {library_name} expert QA benchmark bundle (question, gold answer, private rubric, evidence spans) inside the official {library_name} repository. The bundle failed verification; revise it so it passes.

## Verification the bundle must pass
1. Deterministic syntax check: every fenced ```python block in the gold answer compiles.
2. Sandbox run: the reference program (the longest fenced python block) runs to completion offline - no API key, no network; whenever a language model is needed it must use the offline stub the library ships for its own tests (`dspy.utils.dummies.DummyLM`).
3. Self-grading: the gold answer, graded against the private rubric by an independent grader, must earn score 1 on every claim.

## Rules
- Revise the answer, the rubric, and (only if unavoidable) the question TOGETHER; fix root causes, and change as little as possible.
- Preserve the question's intent and difficulty. Do not weaken the question or the rubric merely to make grading easy.
- The question must keep the naming discipline: it may name brand-level user-facing concepts, but never the internal class/method/helper/file that is the answer - those belong only in the gold answer and evidence.
- The rubric must still satisfy: 2-8 atomic, minimally overlapping claims; each `core` or `supporting`; weights sum to exactly 100 with core claims carrying most of the weight; every claim cites 1-3 evidence spans; every span is a real `*.py` file under the code roots ({code_roots_inline}) with exact 1-indexed line numbers; spans stay focused (prefer 1-40 lines, never exceed 300); claims judgeable from code and tests alone.
- The repository is available read-only in your working directory - verify behavior in the source before asserting it. If the gold answer contradicts the code, the code wins.

## Current bundle
{bundle_json}

## Verification failures
{failure_report}

Return JSON that matches the provided schema and nothing else - the complete revised bundle (all fields, including parts you did not change)."""

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "claim_scores": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "score": {"type": "string", "enum": ["0", "0.5", "1"]},
                    "rationale": {"type": "string"},
                },
                "required": ["claim_id", "score", "rationale"],
                "additionalProperties": False,
            },
        },
        "question_score": {"type": "number"},
        "needs_regrade": {"type": "boolean"},
    },
    "required": ["claim_scores", "question_score", "needs_regrade"],
    "additionalProperties": False,
}

REVISE_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "rubric": s6.RUBRIC_SCHEMA["properties"]["rubric"],
        "evidence": s6.RUBRIC_SCHEMA["properties"]["evidence"],
        "revision_note": {"type": "string"},
    },
    "required": ["question", "answer", "rubric", "evidence", "revision_note"],
    "additionalProperties": False,
}


def sandbox_env() -> dict:
    blocked = ("API_KEY", "TOKEN", "SECRET", "_PAT")
    return {k: v for k, v in os.environ.items()
            if not any(word in k.upper() for word in blocked)}


def deterministic_report(bundle: dict, run_dir: Path) -> list[str]:
    """Syntax-check every python block; run the longest one in the sandbox."""
    problems = []
    blocks = CODE_BLOCK.findall(bundle["gold_answer"])
    if not blocks:
        return ["gold answer contains no fenced ```python block; it must include "
                "a self-contained runnable reference program"]
    for i, block in enumerate(blocks):
        try:
            compile(block, f"<block {i}>", "exec")
        except SyntaxError as error:
            problems.append(f"syntax check failed on python block {i}: {error}")
    if problems:
        return problems
    run_dir.mkdir(parents=True, exist_ok=True)
    program = max(blocks, key=len)
    (run_dir / "program.py").write_text(program, encoding="utf-8")
    env = sandbox_env()
    env["DSPY_CACHEDIR"] = str(run_dir / "cache")
    try:
        result = subprocess.run(
            [str(VENV_PY), "program.py"], cwd=run_dir, env=env,
            capture_output=True, text=True, timeout=RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        (run_dir / "verdict.txt").write_text("timeout\n", encoding="utf-8")
        return [f"sandbox run timed out after {RUN_TIMEOUT}s"]
    (run_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    (run_dir / "verdict.txt").write_text(f"returncode {result.returncode}\n",
                                         encoding="utf-8")
    if result.returncode != 0:
        problems.append(
            f"sandbox run failed (exit {result.returncode}); stderr tail:\n"
            + result.stderr[-3000:]
        )
    return problems


def run_codex(prompt: str, scope_dir: Path, repo: Path, tag: str,
              schema_name: str) -> dict:
    last_message = scope_dir / f"last_message_{tag}.json"
    events_path = scope_dir / f"events_{tag}.jsonl"
    command = [
        "codex", "exec",
        "-m", gen.MODEL,
        "-c", f"model_reasoning_effort={gen.EFFORT}",
        "-s", "read-only",
        "-C", str(repo),
        "--output-schema", str(scope_dir / schema_name),
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


def run_validated(prompt: str, scope_dir: Path, repo: Path, tag: str,
                  schema_name: str, validate) -> dict:
    result, problems = None, ["not run"]
    for attempt in range(gen.MAX_RETRIES + 1):
        attempt_prompt = prompt if attempt == 0 else (
            prompt + "\n\n## Corrections required\nYour previous output had "
            "these problems; fix them and return the complete JSON again:\n- "
            + "\n- ".join(problems)
        )
        result = run_codex(attempt_prompt, scope_dir, repo, f"{tag}_a{attempt}",
                           schema_name)
        problems = validate(result)
        if not problems:
            return result
    raise RuntimeError(f"{tag}: unresolved violations after retries: {problems}")


def grade_violations(grade: dict, rubric: list[dict]) -> list[str]:
    problems = []
    expected = [claim["claim_id"] for claim in rubric]
    scored = [entry["claim_id"] for entry in grade["claim_scores"]]
    if sorted(scored) != sorted(expected):
        problems.append(f"claims scored {scored} do not match rubric {expected}")
        return problems
    weights = {claim["claim_id"]: claim["weight"] for claim in rubric}
    computed = sum(weights[e["claim_id"]] * float(e["score"])
                   for e in grade["claim_scores"])
    if abs(grade["question_score"] - computed) > 0.01:
        problems.append(f"question_score {grade['question_score']} != weighted "
                        f"sum {computed}")
    return problems


def self_grade(bundle: dict, repo: Path, scope_dir: Path, tag: str) -> dict:
    paths = []
    for span in bundle["evidence"]:
        if span["path"] not in paths:
            paths.append(span["path"])
    prompt = GRADER_TEMPLATE.format(
        library_name=gen.DSPY_VALUES["library_name"],
        question_id=bundle["id"],
        label=bundle["topic"],
        question=bundle["question"],
        model_answer=bundle["gold_answer"],
        gold_answer=bundle["gold_answer"],
        claim_rubric_json=json.dumps(bundle["rubric"], ensure_ascii=False, indent=2),
        evidence_spans_json=json.dumps(bundle["evidence"], ensure_ascii=False, indent=2),
        whole_evidence_text="\n\n".join(s6.numbered_dump(p, repo) for p in paths),
    )
    (scope_dir / f"prompt_{tag}.txt").write_text(prompt, encoding="utf-8")
    return run_validated(prompt, scope_dir, repo, tag, "grade_schema.json",
                         lambda g: grade_violations(g, bundle["rubric"]))


def revision_violations(result: dict, repo: Path) -> list[str]:
    problems = []
    if problem := gen.question_violation(result["question"]):
        problems.append(problem)
    if problem := gen.program_violation(result["answer"]):
        problems.append(problem)
    paths = []
    for span in result["evidence"]:
        path = span["path"]
        if path not in paths:
            paths.append(path)
        if not path.endswith(".py"):
            problems.append(f"span {span['span_id']}: '{path}' does not match *.py")
        elif path.split("/")[0] not in ("dspy", "tests"):
            problems.append(f"span {span['span_id']}: '{path}' outside dspy/ and tests/")
        elif not (repo / path).is_file():
            problems.append(f"span {span['span_id']}: '{path}' is not a file in the repository")
    if problems:
        return problems
    return s6.rubric_violations(result, paths, repo)


def revise(bundle: dict, failures: list[str], repo: Path, scope_dir: Path,
           tag: str, scope: str) -> dict:
    values = gen.values_for_scope(scope)
    prompt = REVISE_TEMPLATE.format(
        library_name=values["library_name"],
        code_roots_inline=values["code_roots_inline"],
        bundle_json=json.dumps(
            {k: bundle[k] for k in ("id", "topic", "question", "gold_answer",
                                    "rubric", "evidence")},
            ensure_ascii=False, indent=2),
        failure_report="\n\n".join(f"- {failure}" for failure in failures),
    )
    (scope_dir / f"prompt_{tag}.txt").write_text(prompt, encoding="utf-8")
    result = run_validated(prompt, scope_dir, repo, tag, "revise_schema.json",
                           lambda r: revision_violations(r, repo))
    used = {sid for claim in result["rubric"] for sid in claim["span_ids"]}
    cited = [span for span in result["evidence"] if span["span_id"] in used]
    rubric, spans = s6.normalize_ids(result["rubric"], cited)
    evidence = []
    for span in spans:
        lines = (repo / span["path"]).read_text(encoding="utf-8").splitlines()
        excerpt = "\n".join(
            f"{i:04d}: {lines[i - 1]}"
            for i in range(span["start_line"], span["end_line"] + 1)
        )
        evidence.append({**span, "excerpt": excerpt})
    return {
        **bundle,
        "question": result["question"],
        "gold_answer": result["answer"],
        "rubric": rubric,
        "evidence": evidence,
        "revision_note": result["revision_note"],
    }


def optimize_bundle(bundle: dict, scope: str) -> dict:
    qid = bundle["id"]
    repo = gen.REPO_BY_SCOPE[scope]
    scope_dir = ARTIFACTS / scope
    output_path = scope_dir / f"7_q_{qid}.json"
    if output_path.exists():
        print(f"{scope}/{qid}: output exists, skipping")
        return json.loads(output_path.read_text(encoding="utf-8"))

    history = []
    for round_no in range(MAX_ROUNDS + 1):
        failures = deterministic_report(
            bundle, scope_dir / "runs" / f"{qid}_r{round_no}")
        grade = None
        if not failures:
            grade = self_grade(bundle, repo, scope_dir, f"{qid}_grade_r{round_no}")
            low = [e for e in grade["claim_scores"] if float(e["score"]) < 1]
            if grade["needs_regrade"]:
                failures.append("grader set needs_regrade=true: the rubric or "
                                "evidence is insufficient to judge the answer")
            for entry in low:
                failures.append(f"self-grade: claim {entry['claim_id']} scored "
                                f"{entry['score']}: {entry['rationale']}")
        history.append({
            "round": round_no,
            "failures": failures,
            "self_score": grade["question_score"] if grade else None,
        })
        if not failures:
            record = {
                **{k: bundle[k] for k in ("id", "topic", "question", "gold_answer",
                                          "rubric", "evidence")},
                "difficulty": bundle["difficulty"],
                "note": bundle["note"],
                "source_index": bundle["source_index"],
                "smalldspy_scope": all(
                    span["path"] in s6.gen.smalldspy_files()
                    for span in bundle["evidence"]
                ),
                "optimization": {
                    "rounds": round_no,
                    "revised": round_no > 0,
                    "self_score": grade["question_score"],
                    "history": history,
                },
            }
            output_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8")
            print(f"{scope}/{qid}: PASS after {round_no} revision(s), "
                  f"self-score {grade['question_score']}")
            return record
        if round_no == MAX_ROUNDS:
            break
        print(f"{scope}/{qid}: round {round_no} failures "
              f"({len(failures)}), revising ...", flush=True)
        bundle = revise(bundle, failures, repo, scope_dir,
                        f"{qid}_revise_r{round_no}", scope)

    dropped = {"id": qid, "dropped": True, "history": history}
    output_path.write_text(
        json.dumps(dropped, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{scope}/{qid}: DROPPED after {MAX_ROUNDS} revision rounds")
    return dropped


def run_scope(scope: str) -> None:
    repo = gen.REPO_BY_SCOPE[scope]
    if subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip() != gen.PINNED_COMMIT:
        raise RuntimeError(f"{repo} is not at the pinned commit")
    scope_dir = ARTIFACTS / scope
    scope_dir.mkdir(parents=True, exist_ok=True)
    (scope_dir / "grade_schema.json").write_text(
        json.dumps(GRADE_SCHEMA, indent=1), encoding="utf-8")
    (scope_dir / "revise_schema.json").write_text(
        json.dumps(REVISE_SCHEMA, indent=1), encoding="utf-8")
    bundles = json.loads(
        (RUBRICS_DIR / f"6_{scope}_rubrics.json").read_text(encoding="utf-8")
    )["questions"]

    records, failures = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(optimize_bundle, b, scope): b for b in bundles}
        for future, bundle in futures.items():
            try:
                records.append(future.result())
            except Exception as error:
                failures.append((bundle["id"], str(error)))
                print(f"{scope}/{bundle['id']}: FAILED: {str(error)[:300]}",
                      flush=True)
    if failures:
        raise RuntimeError(f"{scope}: {len(failures)} bundle(s) errored - rerun "
                           f"to retry them: {[qid for qid, _ in failures]}")

    order = {b["id"]: i for i, b in enumerate(bundles)}
    records.sort(key=lambda r: order[r["id"]])
    final = [r for r in records if not r.get("dropped")]
    dropped = [r["id"] for r in records if r.get("dropped")]
    merged = {
        "scope": scope,
        "harness": f"codex exec (codex-cli), model {gen.MODEL}, effort {gen.EFFORT}, "
                   "read-only sandbox",
        "repository": str(repo),
        "commit": gen.PINNED_COMMIT,
        "sandbox_python": str(VENV_PY),
        "max_revision_rounds": MAX_ROUNDS,
        "human_review": "skipped (no substitute) - see FIDELITY.md",
        "num_questions": len(final),
        "dropped_ids": dropped,
        "questions": final,
    }
    merged_path = ARTIFACTS / f"7_{scope}_validation.json"
    merged_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    canonical = ["id", "topic", "question", "gold_answer", "rubric", "evidence"]
    jsonl_path = ARTIFACTS / f"7_{scope}_validation.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps({k: r[k] for k in canonical}, ensure_ascii=False) + "\n"
                for r in final),
        encoding="utf-8")
    revised = sum(r["optimization"]["revised"] for r in final)
    print(f"{scope}: {len(final)} questions in the final validation set "
          f"({revised} revised, {len(dropped)} dropped) -> {merged_path.name}, "
          f"{jsonl_path.name}")


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
