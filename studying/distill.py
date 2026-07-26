"""Distillation: the studier (Qwen3.5-9B) edits its own cheatsheet.

Iteration 2: instead of rewriting the whole note (which destroyed content —
experiments/007 D2), the studier emits schema-constrained edit operations
against the structured notebook. Foundations-class lessons (answer format,
stale imports, offline-harness misuse) are routed into standing sections
rather than new per-mechanism sections (experiments/007 C4). The distiller
remains the studier so the study object has the same author as the baseline
cheatsheet.
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from studybench.react import MODEL, SAMPLING

from . import notebook as nb

CHEATSHEET_CAP = 12_000
GOLD_PROGRAM_CHARS = 3_000
TIMEOUT = 1_800
MAX_OPS = 12
log = logging.getLogger("studying.distill")

PROMPT = """You are studying the {library} repository. You maintain a cheatsheet: a reference note prepended to every future question you answer about {library}. Repository tools (grep, glob, read_file) remain available when you answer, so the cheatsheet should make your answers correct and direct, and your searches targeted.

You just completed a round of practice questions; a verifier graded your attempts against sandbox-verified gold programs and classified each mistake. Edit your cheatsheet using this round's findings.

## Your current cheatsheet (sections labeled with their IDs)
{current}

## This round's findings
{findings}

## How to edit
Return edit operations (not a rewritten note). Allowed operations:
- `add_section` (section_id "new"): a new titled section.
- `replace_section`: rewrite one existing section's content (title optional).
- `append_to_section`: add lines to the end of one existing section.
- `delete_section`: remove a section that is wrong or worthless.

## Editing priorities
1. Route foundations-class findings into STANDING sections, creating them once if missing and appending afterwards:
   - mistakes classified `answer_format` -> a section titled "Answering style" (how to answer: exactly one fenced runnable ```python program, offline with DummyLM, ending in prints/assertions; be direct and concise).
   - `stale_or_hallucinated_api` -> a section titled "Verified imports and API surface" (correct import lines and real API names you verified, with the wrong name you used noted alongside).
   - `offline_harness_misuse` -> a section titled "Offline testing idiom" (the minimal DummyLM harness pattern that runs).
2. For `mechanism_misunderstanding` findings, prefer appending a short verified idiom or warning to an existing related section over creating a new one; create a new section only for a genuinely new topic.
3. Fix or delete existing content the findings contradict. Keep everything else — your accumulated sections are your knowledge; do not discard them.
4. Keep each edit short: minimal runnable idioms, one-line warnings naming wrong vs right API, repository pointers. The whole note must stay useful under {cap} characters.

Return JSON matching the schema exactly."""

COMPRESS_PROMPT = """Your cheatsheet is {length} characters; the limit is {cap}. Compress it with edit operations: replace the largest or most verbose sections with tighter versions, merge overlapping ones (replace one, delete the other), and delete the least valuable content. Preserve every distinct verified fact, idiom, and warning you can.

## Your current cheatsheet (sections labeled with their IDs)
{current}

Return JSON matching the schema exactly."""


def build_findings(questions: list[dict], verdicts: dict[str, dict]) -> str:
    entries = []
    for question in questions:
        verdict = verdicts[question["qid"]]
        entry = {
            "mechanism": question["mechanism"],
            "difficulty": question["difficulty"],
            "verdict": verdict["verdict"],
            "your_mistakes": [
                {"mistake": item["mistake"], "class": item["mistake_class"]}
                for item in verdict["studier_mistakes"]
            ],
            "lessons": [
                {"lesson": item["lesson"], "files": item["evidence_files"]}
                for item in verdict["lessons"]
            ],
        }
        if verdict["verdict"] != "correct":
            entry["verified_gold_program"] = question["answer"][:GOLD_PROGRAM_CHARS]
        entries.append(entry)
    return json.dumps(entries, ensure_ascii=False, indent=1)


def _call_ops(api: OpenAI, prompt: str, current: dict, seed: int) -> tuple[list[dict], dict]:
    response = api.chat.completions.create(
        model=MODEL.removeprefix("openai/"),
        messages=[{"role": "user", "content": prompt}],
        seed=seed,
        temperature=SAMPLING["temperature"],
        top_p=SAMPLING["top_p"],
        max_tokens=SAMPLING["max_tokens"],
        presence_penalty=SAMPLING["presence_penalty"],
        extra_body=SAMPLING["extra_body"],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "notebook_edits",
                "strict": True,
                "schema": nb.ops_schema(current, MAX_OPS),
            },
        },
    )
    content = response.choices[0].message.content or ""
    usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
    return json.loads(content)["operations"], usage


def distill(
    *, base_url: str, api_key: str, library: str, current: dict,
    findings: str, seed: int,
) -> tuple[dict, dict]:
    """One round of notebook edits; returns (new notebook, ledger)."""

    api = OpenAI(api_key=api_key, base_url=base_url, timeout=TIMEOUT, max_retries=1)
    prompt = PROMPT.format(
        library=library,
        current=nb.render_with_ids(current),
        findings=findings,
        cap=CHEATSHEET_CAP,
    )
    operations, usage = _call_ops(api, prompt, current, seed)
    updated, skipped = nb.apply_ops(current, operations)
    ledger = {"edit": {"operations": operations, "skipped": skipped, "usage": usage}}

    for attempt in range(2):
        if len(nb.render(updated)) <= CHEATSHEET_CAP:
            break
        compress = COMPRESS_PROMPT.format(
            length=len(nb.render(updated)), cap=CHEATSHEET_CAP,
            current=nb.render_with_ids(updated),
        )
        operations, usage = _call_ops(api, compress, updated, seed + 1 + attempt)
        updated, skipped = nb.apply_ops(updated, operations)
        ledger[f"compress_{attempt}"] = {
            "operations": operations, "skipped": skipped, "usage": usage,
        }
    rendered = nb.render(updated)
    if len(rendered) > CHEATSHEET_CAP:
        # Last resort: drop the largest sections until under the cap; loud, recorded.
        sections = sorted(updated["sections"], key=lambda s: len(s["content"]))
        while sections and len(nb.render({**updated, "sections": sections})) > CHEATSHEET_CAP:
            dropped = sections.pop()
            log.warning("cheatsheet cap: dropped largest section %s (%s)",
                        dropped["id"], dropped["title"])
        updated = {**updated, "sections": sections}
        ledger["cap_dropped"] = True
    if not updated["sections"]:
        raise RuntimeError("distillation produced an empty notebook")
    return updated, ledger
