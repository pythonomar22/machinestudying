"""Distillation: the studier (Qwen3.5-9B) revises its own cheatsheet.

The studier — not the teacher models — writes the study object, so the
comparison against the exploration-written baseline cheatsheet stays a
comparison between study procedures, not between authors. The distiller sees
this round's verified findings (verdicts, mistakes, lessons, and gold programs
for the questions it missed) and outputs the complete updated cheatsheet.
"""

from __future__ import annotations

import json

from openai import OpenAI

from studybench.react import MODEL, SAMPLING

CHEATSHEET_CAP = 12_000
MIN_CHEATSHEET = 200
GOLD_PROGRAM_CHARS = 3_000
TIMEOUT = 1_800

PROMPT = """You are studying the {library} repository. You maintain a cheatsheet: a reference note that will be prepended to every future question you answer about {library}. Repository tools (grep, glob, read_file) remain available when you answer, so the cheatsheet should make your future searches targeted, not replace them.

You just completed a round of practice questions; a verifier graded your attempts against sandbox-verified gold programs. Revise your cheatsheet using this round's findings.

## Current cheatsheet
{current}

## This round's findings
{findings}

## Revision rules
- Merge the new lessons into the cheatsheet; reorganize freely and group by mechanism.
- Prefer, in order: correct minimal API idioms (short code snippets you now know run), warnings about mistakes you actually made (name the wrong API and the right one), and pointers to where mechanisms live in the repository.
- Keep existing entries you still consider valuable; drop or rewrite anything these findings contradict.
- The complete cheatsheet must stay under {cap} characters. Compress rather than truncate.
- Output ONLY the complete updated cheatsheet as markdown — no preamble, no commentary, no code fence around the whole document.
"""

RETRY_LONG = (
    "Your cheatsheet was {length} characters, over the {cap}-character limit. "
    "Rewrite it under the limit: compress aggressively, keep every distinct "
    "fact, idiom, and warning. Output only the cheatsheet."
)


def build_findings(questions: list[dict], verdicts: dict[str, dict]) -> str:
    entries = []
    for question in questions:
        verdict = verdicts[question["qid"]]
        entry = {
            "mechanism": question["mechanism"],
            "difficulty": question["difficulty"],
            "verdict": verdict["verdict"],
            "your_mistakes": verdict["studier_mistakes"],
            "lessons": [
                {"lesson": item["lesson"], "files": item["evidence_files"]}
                for item in verdict["lessons"]
            ],
        }
        if verdict["verdict"] != "correct":
            entry["verified_gold_program"] = question["answer"][:GOLD_PROGRAM_CHARS]
        entries.append(entry)
    return json.dumps(entries, ensure_ascii=False, indent=1)


def _strip_outer_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        body = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        return body.rsplit("```", 1)[0].strip()
    return stripped


def distill(
    *, base_url: str, api_key: str, library: str, current: str,
    findings: str, seed: int,
) -> tuple[str, list[dict]]:
    """One distillation; returns the new cheatsheet and the LM usage ledger."""

    api = OpenAI(api_key=api_key, base_url=base_url, timeout=TIMEOUT, max_retries=1)
    prompt = PROMPT.format(
        library=library,
        current=current if current.strip() else "(empty — this is your first round)",
        findings=findings,
        cap=CHEATSHEET_CAP,
    )
    messages = [{"role": "user", "content": prompt}]
    usage = []
    note = ""
    for attempt in range(3):
        response = api.chat.completions.create(
            model=MODEL.removeprefix("openai/"),
            messages=messages,
            seed=seed + attempt,
            temperature=SAMPLING["temperature"],
            top_p=SAMPLING["top_p"],
            max_tokens=SAMPLING["max_tokens"],
            presence_penalty=SAMPLING["presence_penalty"],
            extra_body=SAMPLING["extra_body"],
        )
        content = response.choices[0].message.content or ""
        usage.append({
            "attempt": attempt,
            "finish_reason": response.choices[0].finish_reason,
            **(response.usage.model_dump(exclude_none=True) if response.usage else {}),
        })
        note = _strip_outer_fence(content)
        if len(note) < MIN_CHEATSHEET:
            continue
        if len(note) <= CHEATSHEET_CAP:
            return note, usage
        messages = messages[:1] + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": RETRY_LONG.format(length=len(note), cap=CHEATSHEET_CAP)},
        ]
    if len(note) > CHEATSHEET_CAP:
        return note[:CHEATSHEET_CAP], usage
    raise RuntimeError(f"distillation produced no usable cheatsheet ({len(note)} chars)")
