"""Verifier for self-quizzing attempts: GPT-5.4 with structured feedback.

The verifier judges each studier attempt against the sandbox-verified gold
program and the attempt's own sandbox outcome, then extracts the durable,
corpus-grounded lessons the studier should retain. Its output feeds the
distiller and the next round's quizmaster; it never sees validation or test
questions.
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

MODEL = "gpt-5.4"
TIMEOUT = 600
ATTEMPT_CHARS = 30_000

PROMPT = """You are the verifier in a self-quizzing study loop for {library}. A studier model attempted a practice question about the {library} repository. Judge the attempt against the corpus-verified gold program and the sandbox evidence, then extract the lessons the studier should retain in its cheatsheet.

## Question (mechanism: {mechanism})
{question}

## Gold reference program (verified: runs in the pinned sandbox, exit 0)
{gold}

Gold program output:
{gold_stdout}

## Studier attempt
{attempt}

## Deterministic evidence for the attempt
{deterministic}

## Rules
- The gold program and the sandbox results are ground truth; judge substance, not style. An attempt that reaches the same behavior through different valid {library} APIs can still be correct.
- `verdict`: `correct` (right mechanism and working behavior), `partial` (right direction with material gaps), or `incorrect`.
- `studier_mistakes`: concrete and specific — name hallucinated or stale APIs exactly, wrong fields or data flow, missing mechanisms, programs that do not run. Empty only if the attempt is clean. Classify every mistake:
  - `answer_format`: no single fenced runnable program, prose where a program was asked for, code that does not compile.
  - `stale_or_hallucinated_api`: imports or symbols that do not exist in this repository (name the wrong one and the right one).
  - `offline_harness_misuse`: wrong offline-LM setup (hand-rolled mocks where the shipped test stub was needed, broken LM wiring).
  - `mechanism_misunderstanding`: wrong model of how the library mechanism actually behaves.
  - `other`: anything else.
- `lessons`: the durable {library} facts that would have prevented the mistakes — each self-contained, actionable, at most 2 sentences, citing the relevant repository files. Extract lessons even from correct attempts when they reveal something worth retaining (a lucky guess, a fragile pattern, a cleaner idiom the gold uses).

Return JSON that matches the schema exactly."""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "selfquiz_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
                "summary": {"type": "string"},
                "studier_mistakes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "mistake": {"type": "string"},
                            "mistake_class": {
                                "type": "string",
                                "enum": [
                                    "answer_format",
                                    "stale_or_hallucinated_api",
                                    "offline_harness_misuse",
                                    "mechanism_misunderstanding",
                                    "other",
                                ],
                            },
                        },
                        "required": ["mistake", "mistake_class"],
                        "additionalProperties": False,
                    },
                },
                "lessons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "lesson": {"type": "string"},
                            "evidence_files": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["lesson", "evidence_files"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["verdict", "summary", "studier_mistakes", "lessons"],
            "additionalProperties": False,
        },
    },
}


def client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for the selfquiz verifier")
    return OpenAI(api_key=key, timeout=TIMEOUT, max_retries=2)


def verify_attempt(
    api: OpenAI, *, library: str, question: dict, attempt_answer: str, deterministic: dict
) -> dict:
    """One verdict; the caller persists the result beside the attempt."""

    if not attempt_answer.strip():
        return {
            "verdict": "incorrect",
            "summary": "The studier produced no answer.",
            "studier_mistakes": [
                {"mistake": "produced no answer at all", "mistake_class": "answer_format"}
            ],
            "lessons": [],
            "judge_response": None,
        }
    prompt = PROMPT.format(
        library=library,
        mechanism=question["mechanism"],
        question=question["question"],
        gold=question["answer"],
        gold_stdout=question["gold_stdout"] or "(no output)",
        attempt=attempt_answer[:ATTEMPT_CHARS],
        deterministic=json.dumps(deterministic, ensure_ascii=False, indent=1),
    )
    response = api.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=SCHEMA,
    )
    verdict = json.loads(response.choices[0].message.content)
    verdict["judge_response"] = {
        "id": response.id,
        "model": response.model,
        "finish_reason": response.choices[0].finish_reason,
        "usage": response.usage.model_dump(exclude_none=True) if response.usage else None,
    }
    return verdict
