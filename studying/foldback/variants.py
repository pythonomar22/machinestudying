"""Delivery-ablation variants of an already-built fold-back object.

Re-renders the verified entries of a completed build into notes that
isolate the object's KNOWLEDGE from its BEHAVIOR: no answer-protocol
capsule, no lookup store — the studier's entries and section order are
preserved untouched (drop-only selection, no rewriting).

  knowledge_only  every kept entry (store entries dissolved into their
                  sections), protocol stripped.
  both0_only      only the gold-derived exception-bank section — the
                  content tool search failed to surface — protocol
                  stripped; tests the attention-dilution hypothesis.

Usage:
    .venv-dspy/bin/python -m studying.foldback.variants \
        --run-id dspy-gptminifoldback-20260726
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from studybench.dataset import CORPORA, ROOT
from studybench.tools import RepoTools

from .build import entry_id

BOTH0_SECTION_MARKERS = ("both0", "exception bank")


def load_kept_entries(build_root: Path, corpus_files: set[str]) -> tuple[list[dict], dict]:
    plan = json.loads((build_root / "plan.json").read_text())["payload"]
    entries = []
    for path in sorted(build_root.glob("assembled_*.json")):
        for entry in json.loads(path.read_text())["payload"]["entries"]:
            entries.append({"id": entry_id(entry), **entry})
    verdicts: dict[str, str] = {}
    for shard in (build_root / "verdicts").glob("*.json"):
        verdicts.update(json.loads(shard.read_text())["verdicts"])
    kept = []
    for entry in entries:
        entry["grounding_paths"] = [p for p in entry["grounding_paths"] if p in corpus_files]
        if entry["grounding_paths"] and verdicts.get(entry["id"]) == "verified":
            kept.append(entry)
    return kept, plan


def render(plan: dict, rows: list[dict], library: str) -> str:
    by_section: dict[str, list[dict]] = defaultdict(list)
    for entry in rows:
        by_section[entry["section"]].append(entry)
    parts = [f"# {library} study notes (self-authored)"]
    for section in plan["sections"]:
        section_rows = by_section.get(section["name"], [])
        if not section_rows:
            continue
        parts.append(f"## {section['name']}")
        for entry in section_rows:  # assembly order preserved
            block = f"- {entry['text'].strip()}"
            if entry["code"].strip():
                block += f"\n```python\n{entry['code'].strip()}\n```"
            parts.append(block)
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    corpus = CORPORA["dspy"]
    repository = RepoTools(corpus)
    study_root = ROOT / "runs" / args.run_id / corpus.name / "study"
    kept, plan = load_kept_entries(study_root / "build", set(repository.files))

    both0_sections = [
        s["name"] for s in plan["sections"]
        if any(m in s["name"].lower() for m in BOTH0_SECTION_MARKERS)
    ]
    if not both0_sections:
        raise SystemExit(f"no BOTH0 section found in plan sections: "
                         f"{[s['name'] for s in plan['sections']]}")

    out = study_root / "variants"
    out.mkdir(exist_ok=True)
    variants = {
        "knowledge_only": kept,
        "both0_only": [e for e in kept if e["section"] in both0_sections],
    }
    manifest = {"source_build_plan_form": plan["form"], "both0_sections": both0_sections}
    for name, rows in variants.items():
        note = render(plan, rows, corpus.display)
        (out / f"{name}.md").write_text(note, encoding="utf-8")
        manifest[name] = {"entries": len(rows), "chars": len(note)}
        print(f"{name}: {len(rows)} entries, {len(note)} chars -> {out / f'{name}.md'}")
    (out / "variants.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
