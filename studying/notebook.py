"""Structured cheatsheet: titled sections with stable IDs and deterministic edits.

Iteration 2 replaces full-note rewrites (which destroyed content — see
experiments/007, D2) with schema-constrained edit operations applied
deterministically. The rendered markdown is what evaluation prepends; the
section structure exists only at study time.
"""

from __future__ import annotations

import re

HEADING = re.compile(r"^## +(.+?)\s*$")


def parse(markdown: str, title: str = "SmallDSPy Cheatsheet") -> dict:
    """Split a markdown note into sections on `##` headings.

    Content before the first `##` heading becomes a preamble section.
    """

    sections, current_title, buffer = [], None, []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if current_title is None and not content:
            return
        sections.append({"title": current_title or "Overview", "content": content})

    for line in markdown.splitlines():
        if line.startswith("# ") and current_title is None and not buffer:
            continue  # the document title line, re-added at render time
        match = HEADING.match(line)
        if match:
            flush()
            current_title, buffer = match.group(1), []
        else:
            buffer.append(line)
    flush()
    return {
        "title": title,
        "sections": [
            {"id": f"s{i}", **section} for i, section in enumerate(sections, start=1)
        ],
        "next_id": len(sections) + 1,
    }


def render(notebook: dict) -> str:
    parts = [f"# {notebook['title']}"]
    for section in notebook["sections"]:
        parts.append(f"## {section['title']}\n{section['content']}".rstrip())
    return "\n\n".join(parts) + "\n"


def render_with_ids(notebook: dict) -> str:
    """The distiller's view: every section labeled with its stable ID."""

    parts = []
    for section in notebook["sections"]:
        parts.append(
            f"[{section['id']}] ## {section['title']}\n{section['content']}".rstrip()
        )
    return "\n\n".join(parts)


def ops_schema(notebook: dict, max_ops: int = 12) -> dict:
    """JSON schema for one round of edit operations against this notebook."""

    ids = [section["id"] for section in notebook["sections"]] or ["s0"]
    return {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_ops,
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["add_section", "replace_section", "append_to_section", "delete_section"],
                        },
                        "section_id": {"type": "string", "enum": ids + ["new"]},
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["op", "section_id", "title", "content"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["operations"],
        "additionalProperties": False,
    }


def apply_ops(notebook: dict, operations: list[dict]) -> tuple[dict, list[str]]:
    """Apply edits deterministically; invalid operations are skipped and reported."""

    sections = [dict(section) for section in notebook["sections"]]
    next_id = notebook["next_id"]
    skipped = []
    index = {section["id"]: position for position, section in enumerate(sections)}
    for operation in operations:
        op, sid = operation["op"], operation["section_id"]
        title = operation["title"].strip()
        content = operation["content"].strip()
        if op == "add_section":
            if not title or not content:
                skipped.append(f"add_section without title/content")
                continue
            sections.append({"id": f"s{next_id}", "title": title, "content": content})
            index[f"s{next_id}"] = len(sections) - 1
            next_id += 1
        elif sid not in index:
            skipped.append(f"{op} on unknown section {sid}")
        elif op == "replace_section":
            if not content:
                skipped.append(f"replace_section {sid} with empty content")
                continue
            sections[index[sid]]["content"] = content
            if title:
                sections[index[sid]]["title"] = title
        elif op == "append_to_section":
            if not content:
                skipped.append(f"append_to_section {sid} with empty content")
                continue
            sections[index[sid]]["content"] = (
                sections[index[sid]]["content"].rstrip() + "\n" + content
            ).strip()
        elif op == "delete_section":
            position = index[sid]
            sections.pop(position)
            index = {section["id"]: i for i, section in enumerate(sections)}
    return {**notebook, "sections": sections, "next_id": next_id}, skipped
