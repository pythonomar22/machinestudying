"""Fact-check the assembled object against the corpus: GPT-5.4 in Codex.

The teacher's only role (ledgered, bounded): mark each entry `verified`,
`false`, or `unverifiable` against the pinned fulldspy checkout. Entries
that are not verified are DROPPED deterministically — the teacher never
writes or rewrites content, so the object's author remains the studier.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from studybench.dataset import ROOT

MODEL, EFFORT = "gpt-5.4", "xhigh"
CODEX_TIMEOUT = 5_400
CORPUS_REPO = ROOT / "corpora" / "dspy"
VERDICTS = ("verified", "false", "unverifiable")

PROMPT = """You are fact-checking a study object about this repository (the pinned DSPy checkout in your working directory). Each entry below has an `id`. For EVERY entry, read the relevant source and decide:

- `verified`: the claim/code/location is accurate for THIS checkout (minor wording latitude is fine; code must reflect real APIs and semantics; map entries must point at a real file containing the named symbol, and the note must be true).
- `false`: the entry asserts something this checkout contradicts (wrong API, wrong default, wrong behavior, wrong file/symbol, code that could not work).
- `unverifiable`: you cannot ground the entry in this checkout either way.

Check code snippets against real signatures and imports. Do not grade style. Judge every entry independently; return a verdict for every id.

## Entries
{entries_json}

Return JSON matching the provided schema and nothing else."""


def _schema(entry_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "minItems": len(entry_ids),
                "maxItems": len(entry_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": entry_ids},
                        "verdict": {"type": "string", "enum": list(VERDICTS)},
                        "note": {"type": "string"},
                    },
                    "required": ["id", "verdict", "note"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    }


def _entries(obj: dict) -> list[dict]:
    entries = []
    for item in obj["protocol"]:
        entries.append({"id": item["id"], "type": "protocol_item",
                        "kind": item["kind"], "claim": item["claim"],
                        "code": item["code"], "files": item["files"]})
    for topic, items in obj["topics"].items():
        for item in items:
            entries.append({"id": item["id"], "type": "content_item", "topic": topic,
                            "kind": item["kind"], "claim": item["claim"],
                            "code": item["code"], "files": item["files"]})
    for entry in obj["map"]:
        entries.append({"id": entry["id"], "type": "map_entry",
                        "mechanism": entry["mechanism"], "file": entry["file"],
                        "symbol": entry["symbol"], "note": entry["note"]})
    return entries


def check_object(obj: dict, out_dir: Path) -> dict:
    """One codex session; returns {id: verdict-record}. Artifacts under out_dir."""

    out_dir.mkdir(parents=True, exist_ok=True)
    entries = _entries(obj)
    ids = [entry["id"] for entry in entries]
    schema_path = out_dir / "schema.json"
    schema_path.write_text(json.dumps(_schema(ids), indent=1), encoding="utf-8")
    prompt = PROMPT.format(entries_json=json.dumps(entries, ensure_ascii=False, indent=1))
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    last_message = out_dir / "last_message.json"
    events_path = out_dir / "events.jsonl"
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
    verdicts = json.loads(last_message.read_text(encoding="utf-8"))["verdicts"]
    by_id = {verdict["id"]: verdict for verdict in verdicts}
    missing = [entry_id for entry_id in ids if entry_id not in by_id]
    if missing:
        raise RuntimeError(f"fact-check returned no verdict for: {missing[:10]}")
    return by_id


def apply_verdicts(obj: dict, verdicts: dict[str, dict]) -> tuple[dict, dict]:
    """Drop every non-verified entry; return (clean object, drop report)."""

    def keep(item: dict) -> bool:
        return verdicts[item["id"]]["verdict"] == "verified"

    dropped = [
        {**verdicts[item["id"]], "entry": item}
        for group in ([obj["protocol"]], obj["topics"].values(), [obj["map"]])
        for items in group
        for item in items
        if not keep(item)
    ]
    clean = {
        "protocol": [item for item in obj["protocol"] if keep(item)],
        "topics": {
            topic: [item for item in items if keep(item)]
            for topic, items in obj["topics"].items()
        },
        "map": [entry for entry in obj["map"] if keep(entry)],
    }
    report = {
        "total": sum(len(v) for v in (obj["protocol"], obj["map"])) +
                 sum(len(v) for v in obj["topics"].values()),
        "dropped": len(dropped),
        "dropped_entries": dropped,
    }
    return clean, report
