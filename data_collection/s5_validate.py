"""s5: paper Stage 5 — deterministic checking, sandbox verification,
self-grading, revision, and decontamination.

Paper: a Codex agent uses a deterministic syntax checker and a sandbox to
run and verify answers; questions, answers, and rubrics are optimized
together, iterating until the reference answers achieve full scores under
the private rubric, with human experts reviewing failures.

Per bundle, in order:
1. sandbox control (once): the 5 released SmallDSPy gold answers must run
   cleanly in .venv-dspy (dspy checked out at the pinned corpus commit) —
   proves the sandbox, not the bundles;
2. deterministic checks: dataset schema, rubric/evidence integrity,
   byte-exact excerpts, exactly one fenced runnable ```python block in the
   gold answer, no corpus file path in the question;
3. syntax + offline sandbox run of the gold program (API keys stripped);
4. GPT-5.4 self-grade of the gold answer against its own rubric with the
   exact paper judge contract from studybench.grade — must score 100;
5. failures go to a GPT-5.4 revision call (REVISE_TEMPLATE, ours) and the
   checks rerun, up to 3 rounds; bundles that never pass are dropped;
6. decontamination: exact 3-word-shingle Jaccard against every held-out
   test question must stay below the paper's 0.7 near-dup threshold, and
   surviving bundles are near-deduped against each other;
7. the final rows (six dataset fields only) are written to
   data/smalldspy_ourvalidationset.jsonl and re-validated from disk;
   dotted-name warnings and per-question nearest test question go to
   artifacts/review/ for the human pass.

Usage: uv run --frozen python data_collection/s5_validate.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studybench.dataset import CORPORA, load_questions, read_corpus_file
from studybench.grade import build_prompt, response_schema, score_verdict

from common import ARTIFACTS, OPENAI_MODEL, REASONING_EFFORT, ROOT, chat, jaccard, load_env, read_json, shingles, write_json
from prompts import DSPY_VALUES, REVISE_TEMPLATE
from s4_rubrics import RUBRIC_SCHEMA, materialize_evidence, numbered_dump, rubric_violations

TASK = "smalldspy"
MAX_REVISIONS = 3
SANDBOX_TIMEOUT = 240
DECONTAMINATION_THRESHOLD = 0.7  # the paper's near-dup threshold
ROW_FIELDS = ("id", "topic", "question", "gold_answer", "rubric", "evidence")
FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
PATH_IN_QUESTION = re.compile(r"\b(?:dspy|tests)/[A-Za-z0-9_/.]+\.py\b")
DOTTED_NAME = re.compile(r"\b[A-Za-z_][\w.]*\.[a-z_][\w]*\b")

_inner = RUBRIC_SCHEMA["json_schema"]["schema"]["properties"]
REVISE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "revised_bundle",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "gold_answer": {"type": "string"},
                "claims": _inner["claims"],
                "spans": _inner["spans"],
            },
            "required": ["question", "gold_answer", "claims", "spans"],
            "additionalProperties": False,
        },
    },
}


def extract_program(gold_answer: str) -> tuple[str | None, str | None]:
    blocks = FENCE.findall(gold_answer)
    if len(blocks) != 1:
        return None, f"gold answer has {len(blocks)} fenced ```python blocks; need exactly 1"
    return blocks[0], None


def run_sandbox(program: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="sbx-") as scratch:
        script = Path(scratch) / "gold.py"
        script.write_text(program, encoding="utf-8")
        try:
            result = subprocess.run(
                [str(ROOT / ".venv-dspy/bin/python"), str(script)],
                capture_output=True, text=True, timeout=SANDBOX_TIMEOUT,
                cwd=scratch,
                env={"PATH": "/usr/bin:/bin", "HOME": scratch,
                     "PYTHONDONTWRITEBYTECODE": "1", "OPENAI_API_KEY": ""},
            )
        except subprocess.TimeoutExpired:
            return False, f"sandbox run exceeded {SANDBOX_TIMEOUT}s"
        if result.returncode != 0:
            return False, (f"sandbox exit code {result.returncode}\n"
                           f"stderr tail:\n{result.stderr[-3000:]}")
        return True, result.stdout[-1000:]


def deterministic_failures(bundle: dict) -> list[str]:
    problems = []
    question = bundle["question"]
    if PATH_IN_QUESTION.search(question):
        problems.append(
            f"question names a repository file path: "
            f"{PATH_IN_QUESTION.findall(question)} — describe the behavior instead"
        )
    evidence_files = list(dict.fromkeys(span["path"] for span in bundle["evidence"]))
    spans = [{k: span[k] for k in ("span_id", "path", "start_line", "end_line")}
             for span in bundle["evidence"]]
    problems += rubric_violations(
        {"claims": bundle["rubric"], "spans": spans}, evidence_files
    )
    for span in bundle["evidence"]:
        source = read_corpus_file(CORPORA[TASK], span["path"]).splitlines()
        expected = "\n".join(
            f"{number:04d}: {source[number - 1]}"
            for number in range(span["start_line"], span["end_line"] + 1)
        )
        if span["excerpt"] != expected:
            problems.append(f"span {span['span_id']} excerpt is not byte-exact")
    program, error = extract_program(bundle["gold_answer"])
    if error:
        problems.append(error)
    else:
        try:
            compile(program, "<gold>", "exec")
        except SyntaxError as syntax_error:
            problems.append(f"gold program has a syntax error: {syntax_error}")
    return problems


def self_grade(client, row: dict) -> tuple[int, list[dict]]:
    """Grade the gold answer against its own rubric with the paper contract."""
    prompt = build_prompt(TASK, row, row["gold_answer"], "paper")
    response = chat(client, "s5_selfgrade", {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": response_schema(row, "paper"),
    })
    verdict = json.loads(response.choices[0].message.content)
    claims, score, _ = score_verdict(row, verdict, "paper")
    return score, claims


def check_bundle(client, bundle: dict) -> tuple[bool, list[str], dict]:
    """Run checks 2-4; returns (passed, failures, detail)."""
    detail = {}
    problems = deterministic_failures(bundle)
    if problems:
        return False, problems, detail
    program, _ = extract_program(bundle["gold_answer"])
    ok, output = run_sandbox(program)
    detail["sandbox_output"] = output
    if not ok:
        return False, [f"gold program failed in the offline sandbox:\n{output}"], detail
    score, claims = self_grade(client, bundle)
    detail["self_grade"] = score
    detail["self_grade_claims"] = claims
    if score != 100:
        missed = [claim for claim in claims if claim["score"] != 1]
        return False, [
            "gold answer scored "
            f"{score}/100 against its own rubric; missed claims:\n"
            + json.dumps(missed, ensure_ascii=False, indent=2)
        ], detail
    return True, [], detail


def revise(client, bundle: dict, failures: list[str]) -> dict:
    evidence_files = list(dict.fromkeys(span["path"] for span in bundle["evidence"]))
    spans = [{k: span[k] for k in ("span_id", "path", "start_line", "end_line")}
             for span in bundle["evidence"]]
    prompt = REVISE_TEMPLATE.format(
        library_name=DSPY_VALUES["library_name"],
        bundle_json=json.dumps(
            {"question": bundle["question"], "gold_answer": bundle["gold_answer"],
             "claims": bundle["rubric"], "spans": spans},
            ensure_ascii=False, indent=2,
        ),
        failure_report="\n".join(f"- {failure}" for failure in failures),
        evidence_files_text="\n\n".join(
            f"### {relative}\n{numbered_dump(relative)}" for relative in evidence_files
        ),
    )
    response = chat(client, "s5_revise", {
        "model": OPENAI_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": REVISE_SCHEMA,
    })
    revision = json.loads(response.choices[0].message.content)
    revised = dict(bundle)
    revised["question"] = revision["question"]
    revised["gold_answer"] = revision["gold_answer"]
    revised["rubric"] = revision["claims"]
    span_problems = rubric_violations(revision, evidence_files)
    structural = [p for p in span_problems if "span" in p]
    if structural:  # cannot materialize invalid spans; keep old evidence and re-fail
        return revised
    revised["evidence"] = materialize_evidence(revision["spans"])
    return revised


def sandbox_control() -> None:
    for row in load_questions(TASK):
        program, error = extract_program(row["gold_answer"])
        if error:
            raise RuntimeError(f"control {row['id']}: {error}")
        ok, output = run_sandbox(program)
        if not ok:
            raise RuntimeError(
                f"sandbox control failed on released gold {row['id']}:\n{output}"
            )
        print(f"control {row['id']}: released gold runs clean")


def main() -> None:
    load_env()
    client = OpenAI(timeout=3600, max_retries=2)
    bundles = read_json(ARTIFACTS / "bundles.json")["bundles"]
    tests = {row["id"]: row["question"] for row in load_questions(TASK)}
    test_shingles = {qid: shingles(question) for qid, question in tests.items()}

    sandbox_control()

    report, survivors = [], []
    for bundle in bundles:
        entry = {"id": bundle["id"], "rounds": []}
        current = bundle
        passed, failures, detail = check_bundle(client, current)
        entry["rounds"].append({"failures": failures, **detail})
        rounds = 0
        while not passed and rounds < MAX_REVISIONS:
            rounds += 1
            print(f"{bundle['id']}: revision round {rounds}: {failures[0][:200]}")
            current = revise(client, current, failures)
            passed, failures, detail = check_bundle(client, current)
            entry["rounds"].append({"failures": failures, **detail})
        entry["passed"] = passed
        if not passed:
            entry["dropped_reason"] = "checks failed after max revisions"
            report.append(entry)
            print(f"{bundle['id']}: DROPPED after {rounds} revisions")
            continue

        similarity = {
            qid: round(jaccard(shingles(current["question"]), test_shingles[qid]), 4)
            for qid in tests
        }
        entry["test_similarity"] = similarity
        worst = max(similarity, key=similarity.get)
        if similarity[worst] >= DECONTAMINATION_THRESHOLD:
            entry["passed"] = False
            entry["dropped_reason"] = f"near-duplicate of test question {worst}"
            report.append(entry)
            print(f"{bundle['id']}: DROPPED — too similar to {worst}")
            continue
        for kept in survivors:
            overlap = jaccard(shingles(current["question"]), shingles(kept["question"]))
            if overlap >= DECONTAMINATION_THRESHOLD:
                entry["passed"] = False
                entry["dropped_reason"] = f"near-duplicate of {kept['id']}"
                break
        if not entry["passed"]:
            report.append(entry)
            print(f"{bundle['id']}: DROPPED — {entry['dropped_reason']}")
            continue
        entry["revisions_used"] = rounds
        report.append(entry)
        survivors.append(current)
        print(f"{bundle['id']}: PASSED (revisions={rounds}, "
              f"max test similarity={similarity[worst]:.3f} vs {worst})")

    if not survivors:
        raise RuntimeError("no bundle survived validation")
    rows = [{field: bundle[field] for field in ROW_FIELDS} for bundle in survivors]
    output = ROOT / "data" / "smalldspy_ourvalidationset.jsonl"
    with open(output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Re-validate the final file from disk with dataset-identical rules.
    for row in (json.loads(line) for line in open(output, encoding="utf-8")):
        if set(row) != set(ROW_FIELDS):
            raise RuntimeError(f"bad fields in final row {row.get('id')}")
        if deterministic_failures(row):
            raise RuntimeError(f"final row fails deterministic checks: {row['id']}")

    review_lines = ["# Human review: smalldspy_ourvalidationset", ""]
    for bundle in survivors:
        entry = next(item for item in report if item["id"] == bundle["id"])
        worst = max(entry["test_similarity"], key=entry["test_similarity"].get)
        dotted = sorted(set(DOTTED_NAME.findall(bundle["question"])))
        review_lines += [
            f"## {bundle['id']} ({bundle['difficulty']})",
            "",
            bundle["question"],
            "",
            f"- nearest test question: {worst} "
            f"(jaccard {entry['test_similarity'][worst]:.3f})",
            f"- dotted names in question (verify they are user-misconception or "
            f"brand-level, not the answer's locator): {dotted or 'none'}",
            f"- generator note: {bundle['note']}",
            "",
        ]
    (ARTIFACTS / "review").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "review" / "questions.md").write_text(
        "\n".join(review_lines), encoding="utf-8"
    )
    write_json(ARTIFACTS / "validation_report.json", {
        "judge_model": OPENAI_MODEL,
        "decontamination_threshold": DECONTAMINATION_THRESHOLD,
        "bundles_in": len(bundles),
        "bundles_out": len(survivors),
        "report": report,
    })
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
