"""Assembly: single-pass map-reduce from lesson records to the study object.

Per topic, Qwen dedups and ranks the mined items and map entries; one
further call writes the answering-protocol section from aggregate sandbox
statistics. The final object is merged DETERMINISTICALLY — no model ever
rewrites the assembled artifact, so the iterative-destruction failure mode
of experiments/007 cannot occur. Rendering to markdown happens after
fact-checking drops unverified entries.
"""

from __future__ import annotations

import json

from openai import OpenAI

from .mine import ITEM_KINDS, call_structured

TOPIC_MAX_ITEMS = 15
TOPIC_MAX_MAP = 20
PROTOCOL_MAX_ITEMS = 8
CODE_MAX_LINES = 12

TOPIC_PROMPT = """You are studying the {library} repository and building one section of your study object: a reference prepended to every future question, so a correct answer needs no tool calls.

Below are lesson items and map entries you mined from {n_records} practice trajectories on the topic `{topic}`. Merge them into the strongest section:
- Deduplicate aggressively: one item per distinct fact; merge overlapping claims into the sharpest version and union their `files`.
- Keep at most {max_items} items, ranked most-generally-useful first. Prefer items that answer whole families of questions (core APIs, defaults, offline-testing patterns) over one-question trivia.
- Keep code snippets minimal (at most {code_lines} lines) and only where they carry the fact.
- Keep at most {max_map} map entries; merge duplicates (same file+symbol), sharpen notes to one line.
- Drop anything vague, redundant, or ungrounded (no `files`).
- Do not invent new facts: every output item must trace to an input item.

## Mined items
{items_json}

## Mined map entries
{map_json}

Return JSON matching the schema exactly."""

PROTOCOL_PROMPT = """You are studying the {library} repository. From the practice statistics below, write the answering-protocol items for your study object: how you should ANSWER questions about {library} (format, structure, verification), independent of topic.

## Practice statistics ({n} attempts at forced 20-iteration search)
- attempts whose final program ran to exit 0: {ok}/{n}
- attempts with no fenced program in the answer: {missing}/{n}
- attempts whose program failed to compile: {syntax}/{n}
- attempts whose program crashed or timed out: {crashed}/{n}

## The register every question uses
Questions ask for ONE complete, runnable, offline Python program (DummyLM for any LM calls) that prints/asserts proof of the behavior, as the entire answer.

Produce at most {max_items} protocol items (kind `pitfall` or `idiom`): concrete rules that would have converted your failed attempts into passing ones — answer shape, offline harness discipline, verification habits. Ground them in the statistics; no topic-specific facts.

Return JSON matching the schema exactly."""


def _items_schema(max_items: int, with_map: bool, max_map: int = 0) -> dict:
    properties: dict = {
        "items": {
            "type": "array",
            "maxItems": max_items,
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
        }
    }
    required = ["items"]
    if with_map:
        properties["map_entries"] = {
            "type": "array",
            "maxItems": max_map,
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
        }
        required.append("map_entries")
    return {"type": "object", "properties": properties,
            "required": required, "additionalProperties": False}


def _call(api: OpenAI, prompt: str, schema: dict, seed: int) -> tuple[dict, dict]:
    return call_structured(
        api, prompt=prompt, schema=schema, name="foldback_assemble",
        seed=seed, required_key="items",
    )


def reduce_topic(
    api: OpenAI, *, library: str, topic: str, records: list[dict], seed: int
) -> tuple[dict, dict, str]:
    """Returns (payload, usage, prompt)."""

    items = [item for record in records for item in record["items"]]
    entries = [entry for record in records for entry in record["map_entries"]]
    prompt = TOPIC_PROMPT.format(
        library=library,
        topic=topic,
        n_records=len(records),
        max_items=TOPIC_MAX_ITEMS,
        max_map=TOPIC_MAX_MAP,
        code_lines=CODE_MAX_LINES,
        items_json=json.dumps(items, ensure_ascii=False, indent=1),
        map_json=json.dumps(entries, ensure_ascii=False, indent=1),
    )
    payload, usage = _call(api, prompt, _items_schema(TOPIC_MAX_ITEMS, True, TOPIC_MAX_MAP), seed)
    return payload, usage, prompt


def reduce_protocol(
    api: OpenAI, *, library: str, stats: dict, seed: int
) -> tuple[dict, dict, str]:
    """Returns (payload, usage, prompt)."""

    prompt = PROTOCOL_PROMPT.format(
        library=library, max_items=PROTOCOL_MAX_ITEMS, **stats
    )
    payload, usage = _call(api, prompt, _items_schema(PROTOCOL_MAX_ITEMS, False), seed)
    return payload, usage, prompt


def build_object(protocol: list[dict], topics: dict[str, dict]) -> dict:
    """Deterministic merge with stable entry IDs for fact-checking."""

    obj = {"protocol": [], "topics": {}, "map": []}
    for index, item in enumerate(protocol):
        obj["protocol"].append({"id": f"p{index}", **item})
    for t_index, (topic, payload) in enumerate(sorted(topics.items())):
        obj["topics"][topic] = [
            {"id": f"t{t_index}i{i}", **item} for i, item in enumerate(payload["items"])
        ]
        for e_index, entry in enumerate(payload["map_entries"]):
            obj["map"].append({"id": f"t{t_index}m{e_index}", "topic": topic, **entry})
    return obj


def render(obj: dict, library: str) -> str:
    """The prepended note: playbook sections then the navigation map."""

    parts = [f"# {library} Study Notes"]
    parts.append("## Answering protocol")
    parts.extend(_render_item(item) for item in obj["protocol"])
    for topic, items in sorted(obj["topics"].items()):
        parts.append(f"## {topic.replace('_', ' ').title()}")
        parts.extend(_render_item(item) for item in items)
    if obj["map"]:
        parts.append("## Where things live (mechanism -> file: symbol)")
        parts.extend(
            f"- {entry['mechanism']}: `{entry['file']}` — {entry['symbol']} — {entry['note']}"
            for entry in obj["map"]
        )
    return "\n\n".join(part for part in parts if part.strip()) + "\n"


def _render_item(item: dict) -> str:
    files = ", ".join(f"`{f}`" for f in item["files"]) if item["files"] else ""
    text = f"- **[{item['kind']}]** {item['claim'].strip()}"
    if files:
        text += f" ({files})"
    code = item["code"].strip()
    if code:
        text += f"\n```python\n{code}\n```"
    return text
