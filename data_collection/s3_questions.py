"""s3: paper Stage 3a/3b — GPT-5.4 candidate generation and critic selection.

Stage 3a (A.2 verbatim template): GPT-5.4 at xhigh effort, in an agentic
harness with the same three repository tools the answering agent gets
(grep/glob/read_file, here over the pinned 66-file SmallDSPy corpus) plus
privileged read access to the repo's docs/ tree at the same commit, generates
12 candidate (question, answer, code_evidence) triples conditioned on 20 seed
sessions sampled from the selected cluster.

Stage 3b (A.3 verbatim template): the same model and harness act as critic
and select up to 5 finalists.

Divergences from the paper, recorded in the README: the paper ran GPT-5.4
inside Codex over the full DSPy repository; we expose the identical tool
surface over the corpus the studier actually sees (which is the honest scope
for questions meant to be answerable from that corpus), and our simple
tool loop caps observations at 15k characters and 80 tool calls per session.

Usage: uv run --frozen python data_collection/s3_questions.py
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studybench.dataset import CORPORA
from studybench.tools import RepoTools

from common import ARTIFACTS, MASTER_SEED, OPENAI_MODEL, REASONING_EFFORT, load_env, read_json, read_jsonl, respond, write_json
from prompts import CRITIC_TEMPLATE, DSPY_VALUES, GENERATOR_TEMPLATE

NUM_SEEDS = 20          # paper: 20 sampled seed sessions per label
NUM_CANDIDATES = 12     # paper: 12 candidates per label
NUM_FINAL = 5           # paper: 5 finalists per label
MAX_TOOL_CALLS = 80
MAX_OBSERVATION_CHARS = 15_000
MAX_CORRECTIONS = 2
SEED_QUESTION_CHARS, SEED_ANSWERS, SEED_ANSWER_CHARS = 3_000, 3, 2_000
DOCS_SUFFIXES, DOCS_MAX_CHARS = (".md", ".mdx", ".rst", ".txt"), 200_000

# Responses-API function tools (chat.completions rejects tools + reasoning).
TOOL_DEFS = [
    {"type": "function",
     "name": "grep",
     "description": "Search repository files with a case-sensitive regular "
                    "expression; returns matching lines as path:line:content.",
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": "string"},
         "path": {"type": "string", "description": "Optional directory or file prefix."},
     }, "required": ["pattern"]}},
    {"type": "function",
     "name": "glob",
     "description": "List repository files matching a glob pattern (supports **).",
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": "string"},
     }, "required": ["pattern"]}},
    {"type": "function",
     "name": "read_file",
     "description": "Read a line range of a repository file (at most 200 lines per call).",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"},
         "start_line": {"type": "integer"},
         "end_line": {"type": "integer"},
     }, "required": ["path"]}},
]

CANDIDATE_ITEM = {
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
                "properties": {"file": {"type": "string"}, "symbol": {"type": "string"}},
                "required": ["file", "symbol"],
                "additionalProperties": False,
            },
        },
        "note": {"type": "string"},
    },
    "required": ["question", "answer", "difficulty", "code_evidence", "note"],
    "additionalProperties": False,
}


def text_format(name: str, properties: dict, required: list[str]) -> dict:
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }
    }


GENERATOR_FORMAT = text_format(
    "generator_candidates",
    {"candidates": {"type": "array", "minItems": NUM_CANDIDATES,
                    "maxItems": NUM_CANDIDATES, "items": CANDIDATE_ITEM}},
    ["candidates"],
)
CRITIC_FORMAT = text_format(
    "critic_finalists",
    {"finalists": {"type": "array", "minItems": 1, "maxItems": NUM_FINAL,
                   "items": CANDIDATE_ITEM},
     "selection_notes": {"type": "string"}},
    ["finalists", "selection_notes"],
)


def docs_snapshot(repo: Path) -> dict[str, str]:
    """docs/ tree at the pinned commit (privileged, generation-time only)."""
    listing = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD", "docs"],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.splitlines()
    snapshot = {}
    for path in listing:
        if path.endswith(DOCS_SUFFIXES):
            content = subprocess.run(
                ["git", "-C", str(repo), "show", f"HEAD:{path}"],
                check=True, capture_output=True, text=True,
                errors="replace", timeout=30,
            ).stdout
            snapshot[path] = content[:DOCS_MAX_CHARS]
    return snapshot


def build_tools() -> tuple[RepoTools, frozenset[str]]:
    corpus = CORPORA["smalldspy"]
    tools = RepoTools(corpus)
    corpus_files = frozenset(tools.files)
    docs = docs_snapshot(corpus.repo)
    # Widen the generator's view with privileged docs; the studier never sees these.
    tools.text.update(docs)
    tools.files = tuple(tools.text)
    tools._line_starts.update({
        path: [0, *(match.end() for match in re.finditer("\n", text))]
        for path, text in docs.items()
    })
    print(f"tool view: {len(corpus_files)} corpus files + {len(docs)} docs files")
    return tools, corpus_files


def dispatch(tools: RepoTools, name: str, arguments: dict) -> str:
    try:
        if name == "grep":
            return tools.grep(arguments.get("pattern", ""), arguments.get("path", ""))
        if name == "glob":
            return tools.glob(arguments.get("pattern", ""))
        if name == "read_file":
            return tools.read_file(
                arguments.get("path", ""),
                arguments.get("start_line", 1),
                arguments.get("end_line", 0),
            )
    except Exception as error:  # tool errors go back to the model, not up
        return f"Error: {error}"
    return f"Error: unknown tool '{name}'"


class AgenticSession:
    """One Responses-API tool-loop session (state chained server-side)."""

    def __init__(self, client, stage, tools, output_format):
        self.client, self.stage, self.tools = client, stage, tools
        self.output_format = output_format
        self.previous_id = None
        self.calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def turn(self, input_items: list[dict]) -> dict:
        """Send input, run tool calls to quiescence, return the final JSON."""
        while True:
            request = {
                "model": OPENAI_MODEL,
                "reasoning": {"effort": REASONING_EFFORT},
                "input": input_items,
                "tools": TOOL_DEFS,
                "text": self.output_format,
            }
            if self.previous_id:
                request["previous_response_id"] = self.previous_id
            if self.calls >= MAX_TOOL_CALLS:
                request["tool_choice"] = "none"
            response = respond(self.client, self.stage, request)
            self.previous_id = response.id
            for field in self.usage:
                self.usage[field] += getattr(response.usage, field, 0) or 0
            function_calls = [
                item for item in response.output if item.type == "function_call"
            ]
            if not function_calls:
                return json.loads(response.output_text)
            input_items = []
            for call in function_calls:
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                observation = dispatch(self.tools, call.name, arguments)
                self.calls += 1
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": observation[:MAX_OBSERVATION_CHARS],
                })


def violations(items: list[dict], corpus_files: frozenset[str]) -> list[str]:
    problems = []
    questions = [item["question"].strip() for item in items]
    if len(set(questions)) != len(questions):
        problems.append("duplicate question text across items")
    for position, item in enumerate(items):
        for evidence in item["code_evidence"]:
            file = evidence["file"]
            if file not in corpus_files:
                problems.append(
                    f"item {position}: evidence file '{file}' is not a file in the "
                    f"code roots ({DSPY_VALUES['code_roots_inline']}); docs are not "
                    "allowed as evidence"
                )
            elif not file.endswith(".py"):
                problems.append(f"item {position}: evidence file '{file}' does not match *.py")
    return problems


def corrected(client, stage, tools, corpus_files, prompt, output_format, key):
    session = AgenticSession(client, stage, tools, output_format)
    payload = session.turn([{"role": "user", "content": prompt}])
    for round_number in range(MAX_CORRECTIONS + 1):
        problems = violations(payload[key], corpus_files)
        if not problems:
            return payload, session.calls, session.usage, round_number
        if round_number == MAX_CORRECTIONS:
            raise RuntimeError(f"{stage}: unresolved violations after corrections: {problems}")
        print(f"{stage}: correction round {round_number + 1}: {problems}")
        payload = session.turn([{
            "role": "user",
            "content": "Your JSON has problems that must be fixed. Correct them and "
                       "return the complete JSON again, same schema:\n- "
                       + "\n- ".join(problems),
        }])
    raise AssertionError


def main() -> None:
    load_env()
    selected = read_json(ARTIFACTS / "clusters.json")["selection"]
    seeds = {row["number"]: row for row in read_jsonl(ARTIFACTS / "seed_pool.jsonl")}
    members = sorted(selected["member_numbers"])
    sampled = sorted(random.Random(MASTER_SEED).sample(members, min(NUM_SEEDS, len(members))))
    sessions = [
        {
            "number": number,
            "question": seeds[number]["question_text"][:SEED_QUESTION_CHARS],
            "community_answers": [
                comment["body"][:SEED_ANSWER_CHARS]
                for comment in seeds[number]["comments"][:SEED_ANSWERS]
            ],
        }
        for number in sampled
    ]
    write_json(ARTIFACTS / "seeds_sampled.json", {
        "selection_mode": selected["mode"],
        "cluster_id": selected["cluster_id"],
        "label": selected["label"],
        "label_description": selected["description"],
        "sample_seed": MASTER_SEED,
        "sessions": sessions,
    })

    tools, corpus_files = build_tools()
    sessions_json = json.dumps(sessions, ensure_ascii=False, indent=2)
    client = OpenAI(timeout=3600, max_retries=2)

    generator_prompt = GENERATOR_TEMPLATE.format(
        **DSPY_VALUES,
        label=selected["label"],
        label_description=selected["description"],
        num_candidates=NUM_CANDIDATES,
        sampled_sources_json=sessions_json,
    )
    payload, calls, usage, rounds = corrected(
        client, "s3_generator", tools, corpus_files,
        generator_prompt, GENERATOR_FORMAT, "candidates",
    )
    candidates = payload["candidates"]
    write_json(ARTIFACTS / "candidates.json", {
        "model": OPENAI_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "label": selected["label"],
        "tool_calls": calls,
        "usage": usage,
        "correction_rounds": rounds,
        "candidates": candidates,
    })
    print(f"generator: {len(candidates)} candidates, {calls} tool calls, usage {usage}")

    critic_prompt = CRITIC_TEMPLATE.format(
        **DSPY_VALUES,
        label=selected["label"],
        label_description=selected["description"],
        num_final=NUM_FINAL,
        sampled_sources_json=sessions_json,
        candidate_json=json.dumps(candidates, ensure_ascii=False, indent=2),
    )
    payload, calls, usage, rounds = corrected(
        client, "s3_critic", tools, corpus_files,
        critic_prompt, CRITIC_FORMAT, "finalists",
    )
    finalists = payload["finalists"]
    write_json(ARTIFACTS / "finalists.json", {
        "model": OPENAI_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "label": selected["label"],
        "tool_calls": calls,
        "usage": usage,
        "correction_rounds": rounds,
        "selection_notes": payload["selection_notes"],
        "finalists": finalists,
    })
    print(f"critic: kept {len(finalists)} finalists, {calls} tool calls, usage {usage}")
    print(f"notes: {payload['selection_notes'][:400]}")


if __name__ == "__main__":
    main()
