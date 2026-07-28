"""Delivery/size variants of an already-built fold-back object.

Re-renders the verified entries of a completed build without rewriting a
word (drop-only selection; the studier's section order and within-section
entry order are preserved untouched):

  knowledge_only  every kept entry (store entries dissolved into their
                  sections), protocol stripped.
  both0_only      only the gold-derived exception-bank section — the
                  content tool search failed to surface — protocol
                  stripped; tests the attention-dilution hypothesis.
  compact         the note-length-tax variant: protocol kept exactly as
                  the studier wrote it, entries ranked by
                  (reusable first, claim-weight served desc, assembly
                  order) and the maximal rank-prefix whose render fits
                  --target-chars is kept.

Usage:
    .venv-dspy/bin/python -m studying.foldback.variants \
        --run-id dspy-gptminifoldback-20260726 [--variants compact] \
        [--target-chars 18000]
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


def render(plan: dict, rows: list[dict], library: str, *, with_protocol: bool = False) -> str:
    by_section: dict[str, list[dict]] = defaultdict(list)
    for entry in rows:
        by_section[entry["section"]].append(entry)
    parts = [f"# {library} study notes (self-authored)"]
    protocol = plan["answer_protocol"]
    if with_protocol and protocol["include"] and protocol["text"].strip():
        parts.append("## Answer protocol\n" + protocol["text"].strip())
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


def compact_rows(plan: dict, kept: list[dict], library: str, target_chars: int) -> list[dict]:
    """Maximal rank-prefix of the kept entries whose render fits the target."""

    ranked = sorted(
        range(len(kept)),
        key=lambda i: (kept[i]["generality"] != "reusable", -kept[i]["weight_hint"], i),
    )
    selected: set[int] = set()
    for i in ranked:
        rows = [kept[j] for j in sorted(selected | {i})]
        if len(render(plan, rows, library, with_protocol=True)) > target_chars:
            break
        selected.add(i)
    return [kept[j] for j in sorted(selected)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variants", default="knowledge_only,both0_only")
    parser.add_argument("--target-chars", type=int, default=18_000)
    args = parser.parse_args()
    names = args.variants.split(",")
    unknown = set(names) - {"knowledge_only", "both0_only", "compact"}
    if unknown:
        parser.error(f"unknown variants: {sorted(unknown)}")
    corpus = CORPORA["dspy"]
    repository = RepoTools(corpus)
    study_root = ROOT / "runs" / args.run_id / corpus.name / "study"
    kept, plan = load_kept_entries(study_root / "build", set(repository.files))

    variants: dict[str, tuple[list[dict], bool]] = {}
    for name in names:
        if name == "knowledge_only":
            variants[name] = (kept, False)
        elif name == "both0_only":
            both0_sections = [
                s["name"] for s in plan["sections"]
                if any(m in s["name"].lower() for m in BOTH0_SECTION_MARKERS)
            ]
            if not both0_sections:
                raise SystemExit(f"no BOTH0 section found in plan sections: "
                                 f"{[s['name'] for s in plan['sections']]}")
            variants[name] = ([e for e in kept if e["section"] in both0_sections], False)
        else:
            variants[name] = (
                compact_rows(plan, kept, corpus.display, args.target_chars), True)

    out = study_root / "variants"
    out.mkdir(exist_ok=True)
    manifest_path = out / "variants.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else {})
    manifest["source_build_plan_form"] = plan["form"]
    for name, (rows, with_protocol) in variants.items():
        note = render(plan, rows, corpus.display, with_protocol=with_protocol)
        (out / f"{name}.md").write_text(note, encoding="utf-8")
        manifest[name] = {"entries": len(rows), "kept_total": len(kept),
                          "chars": len(note), "protocol": with_protocol}
        if name == "compact":
            manifest[name]["target_chars"] = args.target_chars
            manifest[name]["rank"] = "reusable first, weight_hint desc, assembly order"
        print(f"{name}: {len(rows)}/{len(kept)} entries, {len(note)} chars "
              f"-> {out / f'{name}.md'}")
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
