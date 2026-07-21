"""Training-question generation for self-quizzing: GPT-5.4 in Codex.

The quizmaster is the external examiner of the study loop (a declared
teacher-model cheat, in the same spirit as the paper's DeepSeek-written SFT
questions). Each round it writes fresh practice questions in the register of
our validation set, targeting mechanisms the studier fumbled in earlier
rounds. Every gold answer is a single fenced offline program, deterministically
validated and executed in the pinned sandbox before the question is accepted.

Prompt constants (library description, deliverable contract, SmallDSPy corpus
truth, register bounds) are imported from the frozen data_collection pipeline
so there is a single source of truth.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

from studybench.dataset import ROOT

from .sandbox import extract_program, run_program, passed


def _load_stage3a():
    path = ROOT / "data_collection" / "4_generate_candidates.py"
    spec = importlib.util.spec_from_file_location("dc_stage3a", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage3a = _load_stage3a()
VALUES = stage3a.values_for_scope("smalldspy")
MODEL, EFFORT = stage3a.MODEL, stage3a.EFFORT
CODEX_TIMEOUT = stage3a.CODEX_TIMEOUT
CORPUS_REPO = ROOT / "corpora" / "smalldspy"
MAX_SIMILARITY = 0.5
GOLD_STDOUT_CHARS = 2_000

QUIZ_TEMPLATE = """You are the quizmaster in a self-quizzing study loop for {library_name}. A smaller studier model is learning this repository. Each round you write fresh practice questions; the studier attempts them with repository tools (grep, glob, read_file), a verifier grades the attempts against your gold answers, and the studier distills its mistakes into a growing cheatsheet. Your questions decide what the studier gets to learn, so probe what real users need from THIS codebase.

## About {library_name}
{library_description}

## Corpus reality
{code_roots_bullets}

## Mission
- Produce exactly {num_questions} new practice questions with gold answers.
- Tag each with a short snake_case `mechanism` slug naming the library mechanism it probes.
- Difficulty must be `hard` or `very_hard`; prefer a mix.
- `code_evidence` must cite at least 2 real `*.py` files under the code roots that ground the answer. List the directories before citing; cite only files you verified exist in THIS checkout.
- Verify every gold program against the actual source before returning it: read the code, run the logic in your head, and make sure the printed/asserted proof is what the library really does.

## Execution contract for gold programs
Every gold program is executed in an EMPTY working directory with the pinned {library_name} package installed: `import dspy` works, nothing else is present. The program must NOT read repository files, search for a repo root, inspect source paths, or depend on the working directory in any way — it proves the behavior purely through the imported library (DummyLM-driven calls, prints, assertions). A program that tries to locate or open the repository fails verification automatically.

## Register exemplars (match the style, never the content)
These questions define the register and distribution to match: support-thread framing over 2-4 short paragraphs, behavioral symptoms, locator-hard wording, and a closing ask for a runnable offline program. Do NOT duplicate, paraphrase, or trivially perturb any of them — write questions about clearly different mechanisms or clearly distinct behaviors.

{exemplar_questions}

## Questions already used in earlier rounds (write nothing similar)
{prior_questions}

## Studier performance so far
{feedback}

## Naming discipline (the locator is the challenge)
The studier has grep/glob/read over the repo. If the question text names the internal class, method, helper, or file that IS the answer, the question collapses into trivia. Name behaviors, symptoms, and brand-level public concepts (as in the exemplars); keep internal symbols for the gold answer and code_evidence. Rule of thumb: if grepping a token from the question lands within a few files of the answer, the token belongs in the answer, not the question.

Return JSON that matches the provided schema and nothing else."""

FEEDBACK_EMPTY = (
    "No rounds completed yet. Cover a broad, diverse set of mechanisms across "
    "the corpus."
)
FEEDBACK_HEADER = (
    "Per-question outcomes from earlier rounds (mechanism — verdict — key "
    "mistake). Target the studier's weaknesses from NEW angles: probe the "
    "mechanisms it got wrong or partial through different behaviors and edge "
    "cases, and move on from mechanisms it answers correctly.\n"
)


def question_id(round_no: int, question: str) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]
    return f"r{round_no}_{digest}"


def _schema(num_questions: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": num_questions,
                "maxItems": num_questions,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "mechanism": {"type": "string"},
                        "difficulty": {"type": "string", "enum": ["hard", "very_hard"]},
                        "code_evidence": {
                            "type": "array",
                            "minItems": 2,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file": {"type": "string"},
                                    "symbol": {"type": "string"},
                                },
                                "required": ["file", "symbol"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["question", "answer", "mechanism", "difficulty", "code_evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["questions"],
        "additionalProperties": False,
    }


def _shingles(text: str) -> set[str]:
    words = re.sub(r"\s+", " ", text.lower()).strip().split()
    if len(words) < 3:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}


def _too_similar(question: str, others: list[str]) -> bool:
    mine = _shingles(question)
    for other in others:
        theirs = _shingles(other)
        if mine and theirs and len(mine & theirs) / len(mine | theirs) > MAX_SIMILARITY:
            return True
    return False


def _violations(items: list[dict], avoid: list[str]) -> list[str]:
    problems = []
    texts = [item["question"].strip() for item in items]
    if len(set(texts)) != len(texts):
        problems.append("duplicate question text within the batch")
    for position, item in enumerate(items):
        if problem := stage3a.question_violation(item["question"]):
            problems.append(f"question {position}: {problem}")
        if problem := stage3a.program_violation(item["answer"]):
            problems.append(f"question {position}: {problem}")
        if _too_similar(item["question"], avoid):
            problems.append(
                f"question {position}: too similar to an exemplar or an "
                "earlier round's question — probe a different mechanism or "
                "clearly distinct behavior"
            )
        for evidence in item["code_evidence"]:
            file = evidence["file"]
            if not file.endswith(".py"):
                problems.append(f"question {position}: evidence '{file}' does not match *.py")
            elif file.split("/")[0] not in ("dspy", "tests"):
                problems.append(f"question {position}: evidence '{file}' is outside dspy/ and tests/")
            elif not (CORPUS_REPO / file).is_file():
                problems.append(f"question {position}: evidence '{file}' does not exist in this corpus")
    return problems


def _run_codex(prompt: str, gen_dir: Path, num_questions: int, attempt: int) -> dict:
    schema_path = gen_dir / "question_schema.json"
    schema_path.write_text(json.dumps(_schema(num_questions), indent=1), encoding="utf-8")
    last_message = gen_dir / f"last_message_a{attempt}.json"
    events_path = gen_dir / f"events_a{attempt}.jsonl"
    command = [
        "codex", "exec",
        "-m", MODEL,
        "-c", f"model_reasoning_effort={EFFORT}",
        "-s", "read-only",
        "-C", str(CORPUS_REPO),
        "--output-schema", str(schema_path),
        "-o", str(last_message),
        "--json",
        "-",
    ]
    with open(events_path, "w", encoding="utf-8") as events:
        result = subprocess.run(
            command, input=prompt, stdout=events, stderr=subprocess.PIPE,
            text=True, timeout=CODEX_TIMEOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"codex exec failed ({result.returncode}): {result.stderr[-2000:]}")
    return json.loads(last_message.read_text(encoding="utf-8"))


def _feedback_block(prior_verdicts: list[dict]) -> str:
    if not prior_verdicts:
        return FEEDBACK_EMPTY
    lines = []
    for verdict in prior_verdicts:
        mistake = (verdict["studier_mistakes"] or ["clean"])[0]
        lines.append(f"- [{verdict['mechanism']}] {verdict['verdict']} — {mistake}")
    return FEEDBACK_HEADER + "\n".join(lines)


def generate_questions(
    *,
    round_no: int,
    num_questions: int,
    exemplars: list[str],
    prior_questions: list[str],
    prior_verdicts: list[dict],
    gen_dir: Path,
    min_accept: int,
) -> list[dict]:
    """One quizmaster round: generate, validate, sandbox-verify, accept."""

    gen_dir.mkdir(parents=True, exist_ok=True)
    prompt = QUIZ_TEMPLATE.format(
        library_name=VALUES["library_name"],
        library_description=VALUES["library_description"],
        code_roots_bullets=VALUES["code_roots_bullets"],
        num_questions=num_questions,
        exemplar_questions="\n\n".join(
            f"### Exemplar {i + 1}\n{text}" for i, text in enumerate(exemplars)
        ),
        prior_questions="\n\n".join(
            f"### Used question {i + 1}\n{text}" for i, text in enumerate(prior_questions)
        ) or "(none yet)",
        feedback=_feedback_block(prior_verdicts),
    )
    (gen_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    avoid = exemplars + prior_questions
    accepted, rejected, problems = [], [], ["not run"]
    for attempt in range(2):
        attempt_prompt = prompt if attempt == 0 else (
            prompt + "\n\n## Corrections required\nYour previous output had "
            "these problems; fix them and return the complete JSON again:\n- "
            + "\n- ".join(problems)
        )
        items = _run_codex(attempt_prompt, gen_dir, num_questions, attempt)["questions"]
        accepted, rejected, problems = [], [], []
        for position, item in enumerate(items):
            item_problems = _violations([item], avoid)
            if any(item["question"].strip() == kept["question"].strip() for kept in accepted):
                item_problems.append("duplicate of another question in this batch")
            sandbox_result = None
            if not item_problems:
                qid = question_id(round_no, item["question"])
                program, _ = extract_program(item["answer"])
                sandbox_result = run_program(
                    program, gen_dir / "gold_runs" / f"a{attempt}_{qid}"
                )
                if not passed(sandbox_result):
                    item_problems.append(
                        "gold program failed in the sandbox: "
                        + (sandbox_result["stderr"][-600:] or "timeout or nonzero exit")
                    )
            if item_problems:
                problems.extend(f"question {position}: {text}" for text in item_problems)
                rejected.append({"position": position, "problems": item_problems,
                                 "sandbox": sandbox_result, **item})
            else:
                accepted.append({
                    "qid": question_id(round_no, item["question"]),
                    **item,
                    "gold_stdout": sandbox_result["stdout"][:GOLD_STDOUT_CHARS],
                })
        if not problems:
            break
    (gen_dir / "rejected.json").write_text(
        json.dumps(rejected, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    if len(accepted) < min_accept:
        raise RuntimeError(
            f"round {round_no}: only {len(accepted)}/{num_questions} questions "
            f"survived verification (need {min_accept}); inspect {gen_dir}"
        )
    return accepted
