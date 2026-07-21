# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy<2.3"]
# ///
"""Stage 3a (paper): candidate generation with GPT-5.4 in Codex.

Paper, verbatim: "For each label, GPT-5.4 at reasoning effort xhigh,
running within Codex with access to the full repository and documentation,
generates 12 candidate (question, gold answer, code_evidence) triples. The
generator is conditioned on 20 sampled seed sessions, the label
description, and the library description. The complete template is
provided in A.2."

Ours: the harness is literally `codex exec` (ChatGPT-account auth) with
model gpt-5.4 at xhigh, read-only sandbox rooted at the full DSPy checkout
(source + tests + docs) at the pinned corpus commit. Per topic we condition
on 10 seed sessions (the members nearest the cluster centroid in the
original embedding space; the paper's sampling method for its 20 is
unpublished) and ask for 20 candidates instead of 12. The A.2 template is
verbatim; its {placeholder} values are our reconstructions (see
FIDELITY.md).

Two sets are generated, identical in seeds, template, and harness, and
differing ONLY in the repository the generator can read:
- fulldspy:  the full checkout (source + tests + docs) at the pinned commit
- smalldspy: the 66-file SmallDSPy corpus checkout (no docs on disk); the
  read-only sandbox itself enforces scope - out-of-scope evidence files do
  not exist there

Usage:
    uv run data_collection/4_generate_candidates.py [fulldspy|smalldspy|all]

Idempotent: a topic with a valid per-topic output file is skipped. Output:
artifacts/4_generate_candidates/<scope>/ with per-topic prompts, codex
event logs, raw last-messages, and per-topic candidate files, plus the
merged 4_fulldspy_candidates.json / 4_smalldspy_candidates.json (topics x
20 triples, each tagged with SmallDSPy-scope compatibility).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

DC = Path(__file__).resolve().parent
ROOT = DC.parent
ARTIFACTS = DC / "artifacts" / "4_generate_candidates"
TOPICS_FILE = DC / "artifacts" / "3_label_topics" / "3_seed_sessions_topic.json"
EMBED_DIR = DC / "artifacts" / "3_label_topics"
SMALL_CORPUS = ROOT / "corpora" / "smalldspy"  # 66-file scope, for tagging
REPO_BY_SCOPE = {
    "fulldspy": ROOT / "corpora" / "dspy",  # full checkout, incl. docs
    "smalldspy": SMALL_CORPUS,              # 66-file sparse checkout
}
PINNED_COMMIT = "9cdb0aac28b2a04b064e40697ccd301872cf6a43"

MODEL, EFFORT = "gpt-5.4", "xhigh"            # paper
NUM_SEEDS = 10                                 # ours (paper: 20, method unpublished)
NUM_CANDIDATES = 20                            # ours (paper: 12)
SEED_QUESTION_CHARS, SEED_ANSWERS, SEED_ANSWER_CHARS = 3_000, 3, 2_000
CODEX_TIMEOUT = 5_400
MAX_RETRIES = 1

# ---------------------------------------------------------------------------
# A.2 generator template, transcribed VERBATIM from the paper appendix.
# ---------------------------------------------------------------------------
GENERATOR_TEMPLATE = """You are generating benchmark-grade {library_name} expert QA inside the official {library_name} repository.

## About {library_name}
{library_description}

## Mission
- Target primary label: `{label}`
- Label description: `{label_description}`
- Produce exactly `{num_candidates}` candidate QA pairs.
- The final benchmark will evaluate an answering agent that has access to the {library_name}.
- You may use both code and docs right now to understand {library_name} deeply, but every final answer must be recoverable from the code roots listed below alone (no docs at answer time).

## Available context
- {library_name} code roots the answering agent will also see:
{code_roots_bullets}
- Documentation under `{repo_docs_subpath}/` for privileged generation-time orientation only (the answering agent might or might not have these)
- Sampled community QA sessions below, which represent realistic user-question distribution for the target label

## How to use the sampled community QA
- The sampled community sessions are the **distribution we want to match**, not just a tone reference. Real {library_name} users hit real friction --- that is the question gold mine. Anchor each generated question in what a user in the seeds was actually trying to do or observing.
- Real community questions are often vague, mis-framed, or mixed with multiple issues. Your job is to **sharpen, not imitate**: keep the user framing ("I'm trying to X and I see Y"), drop the noise, and commit to one crisp locator-hard question.
- **Do not trust the community answer as ground truth.** The answers in the seed sessions were written by a weaker assistant and are frequently wrong, incomplete, or out of date. Treat them only as hints about what the user was confused about. Your gold `answer` must be re-derived from the actual {library_name} source and docs --- read the code, verify the behavior, and cite `code_evidence` by file and symbol. If the community answer contradicts the code, the code wins.
- Do **not** copy or lightly paraphrase a sampled question; upscale by sharpening the behavioral framing.

## Naming discipline (critical --- the locator is the challenge)
The answering agent has grep/glob/read over the full {library_name} repo and tests. If the question text names the specific class, method, file, or internal helper that **is** the answer, you've given away the page number and turned this into a trivia question. The whole point of the benchmark is that *locating* the right code is half the work.

**OK to name (brand-level, user-facing concepts that a real user would type):**
{ok_to_name_bullets}
- Anything the community user in the seed session typed first, at the same granularity they typed it.

**Not OK to name (these attach the question to a specific implementation and leak the locator):**
- The method or attribute on a branded class that contains the answer --- name the behavior, not the method.
- Internal adapter / handler / parser / helper classes.
- Internal helper functions, file paths, test-file names, private config flags, snake_case function names with dot-paths.
- **Do not refer to "the X example / tutorial / notebook / walkthrough / demo / README / guide".** These phrasings are awkward and underspecified --- they point at an artifact as if a shared referent exists ("in the repo's multihop RAG example, ..."). A strong question stands on its own: describe the *scenario* or *setup* itself ("in a multi-hop retrieval pipeline where the model refines its query across hops, ..."), not the artifact that demonstrates it.
- Examples for this codebase:
{not_ok_examples_bullets}

Rule of thumb: if a reader can `grep -R "<token>"` and land within a few files of the answer, the token belongs in the gold `answer` and `code_evidence`, not in the question.

**Bad (names the attachment point):**
{bad_examples_block}

**Good (forces the agent to locate):**
{good_examples_block}

Walk this line carefully: a good question is **specific enough** that a careful reader of the repo converges on one well-defined answer, and **general enough** in wording that no symbol name gives the answer away. If the question could match a dozen unrelated places in the repo, it's too generic; tighten by adding behavioral constraints, not by naming the class.

## Required quality bar
- Questions should read like a thoughtful senior user describing **what they observed or what they want to accomplish**, not like an exam asking about a specific symbol.
- Questions must be difficult enough that they require synthesis across files, abstractions, behavior, tests, edge cases, or design tradeoffs.
- Prefer questions whose answers require reading implementation and tests together.
- Gold answers should be concise but precise, and must be supported by `code_evidence` pointing into actual {library_name} source (not just paraphrased from the seed's community answer).
- `code_evidence` must cite only real files under the code roots ({code_roots_inline}), and each cited filename must match the pattern `{file_glob}` (files with other extensions are out of scope and will be rejected).
- Provide at least two evidence items per candidate.
- Difficulty must be either `hard` or `very_hard`.

## Hard bans
- No documentation-only questions.
- No questions about exact wording from docs, tutorials, README, notebooks, or guides.
- No "in the X example / tutorial / notebook / walkthrough / demo / README / guide" phrasings. Describe the scenario itself.
- No trivial "does {library_name} have X?" or single-symbol existence questions.
- No one-grep questions with an obvious single-line answer.
- No ambiguous or underspecified questions.
- No questions whose answers depend on privileged docs rather than code/tests.
- No questions that violate the naming discipline above (no internal class/method/helper names, no file paths, no `Class.method` attachment points).
- No questions whose gold answer rests on "the seed said so" rather than on verified code behavior.

## Sampling anchors
The JSON block below contains the sampled community sessions for the target label. Use it to preserve the real-world distribution while raising the quality bar sharply.

{sampled_sources_json}

Return JSON that matches the provided schema and nothing else."""

# ---------------------------------------------------------------------------
# RECONSTRUCTED placeholder values (the paper publishes none). Rationale for
# each is in FIDELITY.md; nothing here references the held-out test set.
# ---------------------------------------------------------------------------
DSPY_VALUES = {
    "library_name": "DSPy",
    "library_description": (
        "DSPy is a declarative framework for programming --- rather than "
        "prompting --- language models. Users compose typed modules (such as "
        "`dspy.Predict`, `dspy.ChainOfThought`, and `dspy.ReAct`) whose "
        "input/output behavior is declared by signatures; adapters translate "
        "signatures into provider-specific prompts and parse completions back "
        "into typed fields; optimizers tune prompts and demonstrations "
        "against metrics.\n\n"
        "Deliverable contract (hard requirement for every candidate):\n"
        "- Every question must END by explicitly asking for a small, "
        "self-contained, RUNNABLE Python program (a repro, harness, working "
        "example, or module) as its deliverable, phrased the way a real user "
        "asks ('Give me a complete, runnable program that ...', 'Write the "
        "metric plus a tiny harness that ...', 'Show me the idiomatic DSPy "
        "program that ...'), and must ask for printed output or assertions "
        "that PROVE the behavior in question.\n"
        "- Questions should read like a real support thread: 2-4 short "
        "paragraphs --- the setup and goal, what was tried and what was "
        "observed (symptoms, error messages as the user saw them), then the "
        "deliverable ask.\n"
        "- Every gold `answer` must BE that program: exactly one fenced "
        "```python block and NOTHING outside it, with any needed explanation "
        "as inline comments, ending with the prints/assertions the question "
        "demands.\n"
        "- Programs must run offline with no API key --- whenever a language "
        "model is needed, use the offline LM stub the library itself ships "
        "for its own tests (`dspy.utils.dummies.DummyLM`) --- so that every "
        "reference answer can be executed and verified in a sandbox."
    ),
    "code_roots_bullets": (
        "- `dspy/` --- the DSPy library source\n"
        "- `tests/` --- its test suite"
    ),
    "code_roots_inline": "`dspy/`, `tests/`",
    "repo_docs_subpath": "docs",
    "file_glob": "*.py",
    "ok_to_name_bullets": (
        "- Public branded module and concept names a real user would type: "
        "`dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`, `dspy.Module`, "
        "`dspy.Signature`, `dspy.Example`, `dspy.Evaluate`, `dspy.LM`, "
        "`dspy.configure` / `dspy.context`, `DummyLM`.\n"
        "- User-facing concepts at brand granularity: signatures, "
        "input/output fields, demos, adapters (as a concept), optimizers / "
        "teleprompters (as a concept), metrics, tools, trajectories, "
        "conversation history, streaming, async, caching, saving/loading "
        "compiled programs.\n"
        "- User-visible exception or error names exactly as they appear in a "
        "traceback or error message the user would paste."
    ),
    "not_ok_examples_bullets": (
        "- Internal adapter/parser/formatter classes or their methods --- "
        "name the formatting or parsing *behavior* instead.\n"
        "- Any `Class.method` attachment point (for example a branded class "
        "name with a `.method` suffix that is where the answer lives).\n"
        "- Internal helpers or dotted snake_case paths, and file paths such "
        "as `dspy/some/module.py` or `tests/some/test_file.py`.\n"
        "- Private flags or keyword arguments that only appear inside the "
        "implementation."
    ),
    "bad_examples_block": (
        "- \"What does `Predict._forward_preprocess` do when the demo list "
        "is empty?\" (names the internal helper --- the locator is given "
        "away)\n"
        "- \"In `dspy/teleprompt/bootstrap.py`, how does the retry loop "
        "select candidate demonstrations?\" (names the file path)"
    ),
    "good_examples_block": (
        "- \"I compiled a program with an optimizer, saved it, and loaded it "
        "in a fresh process --- my demos came back, but a piece of my "
        "configuration silently did not. Give me a complete, runnable "
        "round-trip program (save, then load in a fresh instance, offline "
        "with `DummyLM`, no API key) that reproduces this and prints the "
        "loaded program's state at the end, proving exactly what survives "
        "the round-trip and what resets.\" (behavioral symptom; the agent "
        "must locate the persistence logic; the deliverable is a program "
        "that proves the answer)\n"
        "- \"Two of my output fields have similar names, and one sometimes "
        "comes back empty even though the raw completion clearly contains "
        "the text. Write a minimal offline repro --- a small signature plus "
        "`DummyLM`-driven calls --- that triggers the failure, then prints "
        "the parsed fields so the empty-field behavior is undeniable.\" "
        "(describes observed behavior; forces locating the parsing "
        "mechanism; demands a runnable proof)"
    ),
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "minItems": NUM_CANDIDATES,
            "maxItems": NUM_CANDIDATES,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
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
                    "note": {"type": "string"},
                },
                "required": ["question", "answer", "difficulty", "code_evidence", "note"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def load_topics() -> tuple[list[dict], dict[int, dict]]:
    data = json.loads(TOPICS_FILE.read_text(encoding="utf-8"))
    topics = [
        cluster for cluster in data["clusters"]
        if cluster["coherent"] and cluster["corpus_groundable"]
    ]
    sessions = {row["number"]: row for row in data["sessions"]}
    return topics, sessions


def nearest_centroid_members(topic: dict, sessions: dict[int, dict]) -> list[int]:
    import numpy as np

    index = json.loads((EMBED_DIR / "3_embeddings_index.json").read_text(encoding="utf-8"))
    matrix = np.load(EMBED_DIR / "3_embeddings.npy")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    position = {number: i for i, number in enumerate(index["numbers"])}
    members = [position[number] for number in topic["member_numbers"]]
    centroid = matrix[members].mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    order = sorted(range(len(members)), key=lambda i: -(matrix[members[i]] @ centroid))
    return [topic["member_numbers"][i] for i in order[:NUM_SEEDS]]


def seed_payload(numbers: list[int], sessions: dict[int, dict]) -> list[dict]:
    payload = []
    for number in numbers:
        row = sessions[number]
        answers = [
            comment["body"][:SEED_ANSWER_CHARS]
            for comment in row["comments"]
            if comment["author_type"] == "User" and comment["author"] != row["author"]
        ][:SEED_ANSWERS]
        payload.append({
            "number": number,
            "question": row["question_text"][:SEED_QUESTION_CHARS],
            "community_answers": answers,
        })
    return payload


def smalldspy_files() -> frozenset[str]:
    return frozenset(
        str(path.relative_to(SMALL_CORPUS))
        for root in ("dspy", "tests")
        for path in (SMALL_CORPUS / root).rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )


PROGRAM_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def program_violation(answer: str) -> str | None:
    """The deliverable contract: the answer IS one fenced, compiling program."""
    blocks = PROGRAM_FENCE.findall(answer)
    if len(blocks) != 1:
        return (f"answer has {len(blocks)} fenced ```python blocks; it must be "
                "exactly one runnable program in a single fence")
    if PROGRAM_FENCE.sub("", answer).strip():
        return ("answer has prose outside the fenced program; explanation "
                "belongs in inline comments inside the single fence")
    try:
        compile(blocks[0], "<answer>", "exec")
    except SyntaxError as error:
        return f"answer program does not compile: {error}"
    return None


def violations(candidates: list[dict], repo: Path) -> list[str]:
    problems = []
    questions = [candidate["question"].strip() for candidate in candidates]
    if len(set(questions)) != len(questions):
        problems.append("duplicate question text across candidates")
    for position, candidate in enumerate(candidates):
        if problem := program_violation(candidate["answer"]):
            problems.append(f"candidate {position}: {problem}")
        for evidence in candidate["code_evidence"]:
            file = evidence["file"]
            if not file.endswith(".py"):
                problems.append(f"candidate {position}: evidence '{file}' does not match *.py")
            elif file.split("/")[0] not in ("dspy", "tests"):
                problems.append(f"candidate {position}: evidence '{file}' is outside dspy/ and tests/")
            elif not (repo / file).is_file():
                problems.append(f"candidate {position}: evidence '{file}' is not a file in the repository")
    return problems


def run_codex(prompt: str, scope_dir: Path, repo: Path, slug: str, attempt: int) -> dict:
    schema_path = scope_dir / "output_schema.json"
    schema_path.write_text(json.dumps(OUTPUT_SCHEMA, indent=1), encoding="utf-8")
    last_message = scope_dir / f"last_message_{slug}_a{attempt}.json"
    events_path = scope_dir / f"events_{slug}_a{attempt}.jsonl"
    command = [
        "codex", "exec",
        "-m", MODEL,
        "-c", f"model_reasoning_effort={EFFORT}",
        "-s", "read-only",
        "-C", str(repo),
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


def generate_topic(topic: dict, sessions: dict[int, dict], scope: str) -> dict:
    slug = topic["label"]
    repo = REPO_BY_SCOPE[scope]
    scope_dir = ARTIFACTS / scope
    scope_dir.mkdir(parents=True, exist_ok=True)
    output_path = scope_dir / f"4_topic_{topic['cluster_id']}_{slug}.json"
    if output_path.exists():
        print(f"{scope}/{slug}: output exists, skipping")
        return json.loads(output_path.read_text(encoding="utf-8"))

    seed_numbers = nearest_centroid_members(topic, sessions)
    payload = seed_payload(seed_numbers, sessions)
    prompt = GENERATOR_TEMPLATE.format(
        **DSPY_VALUES,
        label=slug,
        label_description=topic["description"],
        num_candidates=NUM_CANDIDATES,
        sampled_sources_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    (scope_dir / f"prompt_{slug}.txt").write_text(prompt, encoding="utf-8")

    candidates, problems = None, ["not run"]
    salvage = scope_dir / f"last_message_{slug}_a0.json"
    if salvage.exists():  # a completed archived attempt survives a crash/rerun
        candidates = json.loads(salvage.read_text(encoding="utf-8"))["candidates"]
        problems = violations(candidates, repo)
        if not problems:
            print(f"{scope}/{slug}: salvaged valid archived attempt a0")
    for attempt in range(MAX_RETRIES + 1):
        if not problems:
            break
        attempt_prompt = prompt if attempt == 0 else (
            prompt + "\n\n## Corrections required\nYour previous output had "
            "these problems; fix them and return the complete JSON again:\n- "
            + "\n- ".join(problems)
        )
        print(f"{scope}/{slug}: codex attempt {attempt + 1} ...", flush=True)
        result = run_codex(attempt_prompt, scope_dir, repo, slug, attempt)
        candidates = result["candidates"]
        problems = violations(candidates, repo)
        if problems:
            print(f"{scope}/{slug}: violations: {problems[:3]} ...")
    if problems:
        raise RuntimeError(f"{scope}/{slug}: unresolved violations after retries: {problems}")

    scope_files = smalldspy_files()
    record = {
        "cluster_id": topic["cluster_id"],
        "label": slug,
        "label_description": topic["description"],
        "seed_numbers": seed_numbers,
        "candidates": [
            {
                **candidate,
                "topic": slug,
                "smalldspy_scope": all(
                    evidence["file"] in scope_files
                    for evidence in candidate["code_evidence"]
                ),
            }
            for candidate in candidates
        ],
    }
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    in_scope = sum(candidate["smalldspy_scope"] for candidate in record["candidates"])
    print(f"{scope}/{slug}: {len(candidates)} candidates ({in_scope} in SmallDSPy scope)")
    return record


def run_scope(scope: str, topics: list[dict], sessions: dict[int, dict]) -> None:
    repo = REPO_BY_SCOPE[scope]
    if subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip() != PINNED_COMMIT:
        raise RuntimeError(f"{repo} is not at the pinned commit")
    records = [generate_topic(topic, sessions, scope) for topic in topics]
    merged = {
        "scope": scope,
        "harness": f"codex exec (codex-cli), model {MODEL}, effort {EFFORT}, "
                   "read-only sandbox",
        "repository": str(repo),
        "commit": PINNED_COMMIT,
        "num_seeds_per_topic": NUM_SEEDS,
        "seed_selection": "nearest cluster centroid, original embedding space",
        "num_candidates_per_topic": NUM_CANDIDATES,
        "topics": [
            {k: record[k] for k in ("cluster_id", "label", "seed_numbers")}
            for record in records
        ],
        "candidates": [c for record in records for c in record["candidates"]],
    }
    merged_path = ARTIFACTS / f"4_{scope}_candidates.json"
    merged_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    total = len(merged["candidates"])
    in_scope = sum(candidate["smalldspy_scope"] for candidate in merged["candidates"])
    print(f"{scope}: wrote {total} candidates ({in_scope} in SmallDSPy scope) "
          f"to {merged_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", nargs="?", default="all",
                        choices=["fulldspy", "smalldspy", "all"])
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    topics, sessions = load_topics()
    print(f"{len(topics)} coherent+groundable topics")
    for scope in (("fulldspy", "smalldspy") if args.scope == "all" else (args.scope,)):
        run_scope(scope, topics, sessions)


if __name__ == "__main__":
    main()
