"""s4: paper Stage 4 — private rubric construction.

For each finalist, GPT-5.4 (A.4 verbatim template) converts the gold answer
into 2-8 atomic claims (core/supporting, integer weights summing to 100),
each citing 1-3 evidence spans with exact line numbers taken from numbered
dumps of the cited corpus files. Excerpts are then materialized byte-exact
from the pinned corpus in the released dataset's format. Deterministic
validation failures are sent back to the model (up to 3 attempts); a
finalist whose rubric never validates is recorded as dropped.

Usage: uv run --frozen python data_collection/s4_rubrics.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studybench.dataset import CORPORA, read_corpus_file

from common import ARTIFACTS, OPENAI_MODEL, REASONING_EFFORT, chat, load_env, read_json, write_json
from prompts import DSPY_VALUES, RUBRIC_TEMPLATE, TARGET_LABEL

TASK = "smalldspy"
ID_PREFIX = "dspyval"
MAX_ATTEMPTS = 3
MAX_SPAN_LINES = 300  # paper rule

RUBRIC_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rubric_bundle",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_id": {"type": "string", "description": "c1, c2, ..."},
                            "claim_type": {"type": "string", "enum": ["core", "supporting"]},
                            "weight": {"type": "integer"},
                            "statement": {"type": "string"},
                            "span_ids": {"type": "array", "minItems": 1, "maxItems": 3,
                                         "items": {"type": "string"}},
                        },
                        "required": ["claim_id", "claim_type", "weight", "statement", "span_ids"],
                        "additionalProperties": False,
                    },
                },
                "spans": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "span_id": {"type": "string", "description": "s1, s2, ..."},
                            "path": {"type": "string"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                        },
                        "required": ["span_id", "path", "start_line", "end_line"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["claims", "spans"],
            "additionalProperties": False,
        },
    },
}


def question_id(question: str) -> str:
    return f"{ID_PREFIX}_{hashlib.sha256(question.encode('utf-8')).hexdigest()[:12]}"


def numbered_dump(relative: str) -> str:
    source = read_corpus_file(CORPORA[TASK], relative)
    return "\n".join(
        f"{number:04d}: {line}" for number, line in enumerate(source.splitlines(), 1)
    )


def file_line_count(relative: str) -> int:
    return len(read_corpus_file(CORPORA[TASK], relative).splitlines())


def rubric_violations(payload: dict, evidence_files: list[str]) -> list[str]:
    problems = []
    claims, spans = payload["claims"], payload["spans"]
    span_ids = [span["span_id"] for span in spans]
    claim_ids = [claim["claim_id"] for claim in claims]
    if len(set(span_ids)) != len(span_ids):
        problems.append("duplicate span_id")
    if len(set(claim_ids)) != len(claim_ids):
        problems.append("duplicate claim_id")
    total = sum(claim["weight"] for claim in claims)
    if total != 100:
        problems.append(f"claim weights sum to {total}, must be exactly 100")
    if any(claim["weight"] < 1 for claim in claims):
        problems.append("every claim weight must be a positive integer")
    core = sum(claim["weight"] for claim in claims if claim["claim_type"] == "core")
    if core <= 50:
        problems.append(f"core claims carry {core}/100; they must carry most of the weight")
    for claim in claims:
        unknown = set(claim["span_ids"]) - set(span_ids)
        if unknown:
            problems.append(f"claim {claim['claim_id']} cites unknown spans {sorted(unknown)}")
    used = {span_id for claim in claims for span_id in claim["span_ids"]}
    unused = set(span_ids) - used
    if unused:
        problems.append(f"spans {sorted(unused)} are cited by no claim; remove or use them")
    for span in spans:
        if span["path"] not in evidence_files:
            problems.append(
                f"span {span['span_id']} path '{span['path']}' is not one of the "
                f"provided evidence files {evidence_files}"
            )
            continue
        lines = file_line_count(span["path"])
        if not 1 <= span["start_line"] <= span["end_line"] <= lines:
            problems.append(
                f"span {span['span_id']} range {span['start_line']}-{span['end_line']} "
                f"is invalid for {span['path']} ({lines} lines)"
            )
        elif span["end_line"] - span["start_line"] + 1 > MAX_SPAN_LINES:
            problems.append(f"span {span['span_id']} exceeds {MAX_SPAN_LINES} lines")
    return problems


def materialize_evidence(spans: list[dict]) -> list[dict]:
    evidence = []
    for span in spans:
        source = read_corpus_file(CORPORA[TASK], span["path"]).splitlines()
        excerpt = "\n".join(
            f"{number:04d}: {source[number - 1]}"
            for number in range(span["start_line"], span["end_line"] + 1)
        )
        evidence.append({**span, "excerpt": excerpt})
    return evidence


def main() -> None:
    load_env()
    finalists = read_json(ARTIFACTS / "finalists.json")["finalists"]
    client = OpenAI(timeout=3600, max_retries=2)
    bundles, dropped = [], []
    for finalist in finalists:
        qid = question_id(finalist["question"])
        evidence_files = list(dict.fromkeys(
            item["file"] for item in finalist["code_evidence"]
        ))
        prompt = RUBRIC_TEMPLATE.format(
            library_name=DSPY_VALUES["library_name"],
            question_id=qid,
            label=TARGET_LABEL,
            question=finalist["question"],
            gold_answer=finalist["answer"],
            evidence_references_json=json.dumps(
                finalist["code_evidence"], ensure_ascii=False, indent=2
            ),
            evidence_files_text="\n\n".join(
                f"### {relative}\n{numbered_dump(relative)}" for relative in evidence_files
            ),
        )
        messages = [{"role": "user", "content": prompt}]
        payload = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = chat(client, "s4_rubric", {
                "model": OPENAI_MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "messages": messages,
                "response_format": RUBRIC_SCHEMA,
            })
            candidate = json.loads(response.choices[0].message.content)
            problems = rubric_violations(candidate, evidence_files)
            if not problems:
                payload = candidate
                break
            print(f"{qid} attempt {attempt}: {problems}")
            messages += [
                {"role": "assistant", "content": response.choices[0].message.content},
                {"role": "user", "content":
                    "Your rubric violates the rules below. Fix them and return the "
                    "complete JSON again, same schema:\n- " + "\n- ".join(problems)},
            ]
        if payload is None:
            dropped.append({"id": qid, "question": finalist["question"],
                            "reason": "rubric never validated"})
            continue
        bundles.append({
            "id": qid,
            "topic": TARGET_LABEL,
            "question": finalist["question"],
            "gold_answer": finalist["answer"],
            "rubric": payload["claims"],
            "evidence": materialize_evidence(payload["spans"]),
            "difficulty": finalist["difficulty"],
            "note": finalist["note"],
            "code_evidence": finalist["code_evidence"],
            "rubric_attempts": attempt,
        })
        core = sum(c["weight"] for c in payload["claims"] if c["claim_type"] == "core")
        print(f"{qid}: {len(payload['claims'])} claims "
              f"(core weight {core}), {len(payload['spans'])} spans")
    if not bundles:
        raise RuntimeError("no finalist produced a valid rubric")
    write_json(ARTIFACTS / "bundles.json", {
        "model": OPENAI_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "bundles": bundles,
        "dropped": dropped,
    })
    print(f"wrote {len(bundles)} bundles ({len(dropped)} dropped)")


if __name__ == "__main__":
    main()
