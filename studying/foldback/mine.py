"""Per-question mining: Qwen turns one high-budget trajectory into lessons.

The studier reads its own forced-k20 trajectory (every thought, tool call,
and observation), its answer with the sandbox outcome, and the verified
gold program with its sandbox stdout — then extracts the durable items a
k=0 answer would have needed: content items (idioms, API facts,
behaviors, pitfalls) and navigational map entries (mechanism → file /
symbol). These records are the raw material for assembly; no model ever
rewrites them afterwards.
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from studybench.react import MODEL

log = logging.getLogger("studying.foldback.mine")
STUDY_SAMPLING = {"temperature": 0.2, "top_p": 0.95, "max_tokens": 8_192}
MAX_TOKENS_CEILING = 32_768
TIMEOUT = 1_800
ITEM_KINDS = ("idiom", "api_fact", "behavior", "pitfall")
MAX_ITEMS = 8
MAX_MAP_ENTRIES = 6
TURN_REASONING_CHARS = 300
TURN_OBSERVATION_CHARS = 400
ANSWER_CHARS = 6_000
GOLD_CHARS = 6_000
STDERR_CHARS = 1_000

PROMPT = """You are studying the {library} repository. You just attempted a practice question with repository tools at a large search budget (forced {iters} tool iterations). Below are your full trajectory, your final answer with its sandbox outcome, and the verified reference program with its output.

Extract what this episode teaches, so that a future you could answer such questions correctly WITHOUT any tool calls. Be selective: only items you can ground in the trajectory observations or the reference program — never from memory alone.

## Question (topic: {topic})
{question}

## Your trajectory ({iters} iterations)
{trajectory}

## Your final answer
{answer}

## Sandbox outcome of your answer's program
{attempt_sandbox}

## Verified reference program (runs in the pinned sandbox, exit 0)
{gold}

## Reference program output
{gold_stdout}

## What to extract
- `items` (at most {max_items}): the {library} facts a direct answer needed. Each has:
  - `kind`: `idiom` (a minimal working code pattern), `api_fact` (a real import/signature/default), `behavior` (what the library actually does), or `pitfall` (a mistake you made or nearly made, with the correction).
  - `claim`: at most 2 sentences, self-contained, precise.
  - `code`: optional minimal snippet (at most 12 lines) proving or using the claim; empty string if not needed.
  - `files`: repository files that ground the claim (from your observations or the reference's evidence).
- `map_entries` (at most {max_map}): where things live, for faster future search. Each has `mechanism` (snake_case), `file` (repo-relative path you actually saw), `symbol` (class/function), `note` (one line: what is there).
- Prefer items that generalize beyond this exact question. Skip anything you cannot ground.

Return JSON matching the schema exactly."""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "maxItems": MAX_ITEMS,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(ITEM_KINDS)},
                    "claim": {"type": "string"},
                    "code": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["kind", "claim", "code", "files"],
                "additionalProperties": False,
            },
        },
        "map_entries": {
            "type": "array",
            "maxItems": MAX_MAP_ENTRIES,
            "items": {
                "type": "object",
                "properties": {
                    "mechanism": {"type": "string"},
                    "file": {"type": "string"},
                    "symbol": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["mechanism", "file", "symbol", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items", "map_entries"],
    "additionalProperties": False,
}


def call_structured(
    api: OpenAI, *, prompt: str, schema: dict, name: str, seed: int, required_key: str
) -> tuple[dict, dict]:
    """One schema-constrained studier call, resilient to thinking-budget
    exhaustion: an empty or key-less payload retries once with a doubled
    token budget, then fails loudly with the finish reason."""

    sampling = dict(STUDY_SAMPLING)
    finish = None
    for attempt in range(2):
        response = api.chat.completions.create(
            model=MODEL.removeprefix("openai/"),
            messages=[{"role": "user", "content": prompt}],
            seed=seed + attempt,
            **sampling,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        )
        content = response.choices[0].message.content or ""
        usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
        finish = response.choices[0].finish_reason
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and required_key in payload:
            return payload, usage
        log.warning("%s: unusable structured payload (finish_reason=%s, "
                    "content_chars=%d); retrying with a larger budget",
                    name, finish, len(content))
        sampling["max_tokens"] = min(sampling["max_tokens"] * 4, MAX_TOKENS_CEILING)
    raise RuntimeError(
        f"{name}: structured call returned no usable payload after retry "
        f"(finish_reason={finish})"
    )


def compact_trajectory(turns: list[dict]) -> str:
    lines = []
    for index, turn in enumerate(turns):
        reasoning = " ".join(str(turn.get("reasoning") or "").split())[:TURN_REASONING_CHARS]
        observation = " ".join(str(turn.get("observation") or "").split())[:TURN_OBSERVATION_CHARS]
        arguments = json.dumps(turn.get("arguments"), ensure_ascii=False)[:200]
        lines.append(
            f"[{index}] thought: {reasoning}\n"
            f"    call: {turn.get('tool')}({arguments})\n"
            f"    saw: {observation}"
        )
    return "\n".join(lines) or "(no tool calls)"


def _sandbox_block(result: dict | None) -> str:
    if result is None:
        return "no fenced python program in the answer"
    status = ("ran to exit 0" if result["returncode"] == 0 and result["compiled"]
              else "timeout" if result["timeout"]
              else "did not compile" if not result["compiled"]
              else f"exit {result['returncode']}")
    tail = result["stderr"][-STDERR_CHARS:]
    return f"{status}\nstderr tail: {tail or '(empty)'}"


def mine_question(
    api: OpenAI,
    *,
    library: str,
    row: dict,
    episode: dict,
    attempt_sandbox: dict | None,
    gold_stdout: str,
    seed: int,
) -> tuple[dict, dict]:
    """One mining call; returns (record, usage)."""

    prompt = PROMPT.format(
        library=library,
        iters=episode["react_iterations"],
        topic=row["topic"],
        question=row["question"],
        trajectory=compact_trajectory(episode["turns"]),
        answer=(episode.get("answer") or "(no answer)")[:ANSWER_CHARS],
        attempt_sandbox=_sandbox_block(attempt_sandbox),
        gold=row["gold_answer"][:GOLD_CHARS],
        gold_stdout=gold_stdout or "(no output)",
        max_items=MAX_ITEMS,
        max_map=MAX_MAP_ENTRIES,
    )
    return call_structured(
        api, prompt=prompt, schema=SCHEMA, name="foldback_mine",
        seed=seed, required_key="items",
    )
