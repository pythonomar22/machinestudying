# /// script
# requires-python = ">=3.12"
# ///
"""Stage 3b (paper): critic selection with GPT-5.4 in Codex.

Paper, verbatim: "In a second pass, the same model and harness act as a
critic. It reviews the 12 candidates and the seed sessions, then selects
five finalists per label."

Ours: the same `codex exec` harness as Stage 3a (gpt-5.4, xhigh,
read-only sandbox rooted at the scope's repository checkout) reviews each
topic's 20 candidates alongside the exact seed sessions used during
generation (reconstructed from the archived `seed_numbers`), and selects
8 finalists — the paper's 5/12 selection ratio carried to our pool size
(round(20 * 5/12) = 8). The A.3 template is verbatim; its {placeholder}
values are imported from Stage 3a so both stages see identical
reconstructions (see FIDELITY.md).

The critic runs once per scope (fulldspy / smalldspy), reading the same
repository its candidates were generated against.

Usage:
    uv run data_collection/5_critic_selection.py [fulldspy|smalldspy|all]

Idempotent: a topic with a valid per-topic output file is skipped. Output:
artifacts/5_critic_selection/<scope>/ with per-topic prompts, codex event
logs, raw last-messages, and per-topic finalist files, plus the merged
5_fulldspy_finalists.json / 5_smalldspy_finalists.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DC = Path(__file__).resolve().parent
ARTIFACTS = DC / "artifacts" / "5_critic_selection"
CANDIDATES_DIR = DC / "artifacts" / "4_generate_candidates"


def load_stage3a():
    spec = importlib.util.spec_from_file_location("stage3a", DC / "4_generate_candidates.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load_stage3a()

# Paper: 5 finalists from 12 candidates; we keep the same proportion of our 20.
NUM_FINAL = round(gen.NUM_CANDIDATES * 5 / 12)

# ---------------------------------------------------------------------------
# A.3 critic template, transcribed VERBATIM from the paper appendix.
# ---------------------------------------------------------------------------
CRITIC_TEMPLATE = """You are the final critic and selector for benchmark-grade {library_name} expert QA.

## About {library_name}
{library_description}

## Benchmark reality
- Target primary label: `{label}`
- Label description: `{label_description}`
- You are selecting the final `{num_final}` items from a larger candidate set.
- Treat docs as potentially unavailable at answer time even though you may have seen them during generation.

## Selection criteria
- Keep only candidates that are clearly answerable from the code roots alone ({code_roots_inline}) --- no docs at answer time.
- Reject anything based on exact wording from documentation.
- Reject questions that are too easy, one-grep, or single-symbol lookups.
- **Reject questions that give away the locator.** The benchmark tests an agent with grep/glob/read over the repo --- locating the right code is half the challenge. If the question names a method/attribute on a class, an internal handler/adapter/parser/helper class, an internal helper, a file path, a test-file name, or a `snake_case` dotted function, it's a closed-book question. Rewrite to describe the behavior / symptom / user goal, keeping the specific symbol only in the gold `answer` and `code_evidence`. If rewriting would require fabricating a question unsupported by the seed or code, reject outright.
- **Reject "in the X example / tutorial / notebook / walkthrough / demo / README / guide" phrasings.** These are awkward and underspecified --- they point at an artifact as if a shared referent exists ("in the repo's multihop RAG example, ..."). A strong question stands on its own: rewrite to describe the *scenario* or *setup* itself (e.g., "in a multi-hop retrieval pipeline where the model refines its query across hops, ..."), or reject.
- **OK to keep**: branded user-facing concept names at the granularity a user would type:
{ok_to_name_bullets}
  The rule is "named the concept, not the attachment point." A branded class named as a concept is fine; the same class with a `.method` suffix is not.
- **Reject questions that are too generic to have one locator** (e.g., "how does {library_name} handle errors?" or "how does {library_name} do retries?"). A valid question is one where, after reading the repo, a careful expert would converge on the same specific file/symbol as the answer. Tighten generic questions by adding behavioral constraints, not by naming the class.
- **Reject questions whose gold answer rests on the seed's community answer as truth.** The community answers come from a weaker assistant and are frequently wrong. The gold answer must be supported by `code_evidence` pointing into actual {library_name} source. If the only support is "the seed said so," reject.
- Reject questions that copy or closely paraphrase sampled community questions.
- Reject anything outside the target label or overly similar to another candidate.
- Prefer diversity across subtopics within the label.
- You may rewrite the question, answer, difficulty, evidence, and note to improve quality.
- Keep final answers concise and well-grounded.
- `code_evidence` must contain real repo files under one of the code roots ({code_roots_inline}), and each filename must match the pattern `{file_glob}`; reject candidates that cite files with other extensions.

## Sampled community anchors
These are the same sampled sessions used during generation. They are for distribution anchoring only.

{sampled_sources_json}

## Candidate set to review
{candidate_json}

Return JSON that matches the provided schema and nothing else. If fewer than `{num_final}` candidates truly qualify, return fewer and explain the shortage in `selection_notes`."""

# The paper does not publish the critic's output schema. Ours mirrors the
# Stage-3a candidate shape (the critic may rewrite any field) and adds
# `source_index` linking each finalist back to the reviewed candidate, plus
# the `selection_notes` field that A.3 itself names.
CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "selections": {
            "type": "array",
            "minItems": 1,
            "maxItems": NUM_FINAL,
            "items": {
                "type": "object",
                "properties": {
                    "source_index": {"type": "integer"},
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
                "required": ["source_index", "question", "answer", "difficulty",
                             "code_evidence", "note"],
                "additionalProperties": False,
            },
        },
        "selection_notes": {"type": "string"},
    },
    "required": ["selections", "selection_notes"],
    "additionalProperties": False,
}


def critic_violations(selections: list[dict], repo: Path) -> list[str]:
    problems = gen.violations(selections, repo)
    indices = [item["source_index"] for item in selections]
    if len(set(indices)) != len(indices):
        problems.append("duplicate source_index across selections")
    out_of_range = [i for i in indices if not 0 <= i < gen.NUM_CANDIDATES]
    if out_of_range:
        problems.append(f"source_index out of range: {out_of_range}")
    return problems


def run_codex(prompt: str, scope_dir: Path, repo: Path, slug: str, attempt: int) -> dict:
    schema_path = scope_dir / "critic_schema.json"
    if not schema_path.exists():
        schema_path.write_text(json.dumps(CRITIC_SCHEMA, indent=1), encoding="utf-8")
    last_message = scope_dir / f"last_message_{slug}_a{attempt}.json"
    events_path = scope_dir / f"events_{slug}_a{attempt}.jsonl"
    command = [
        "codex", "exec",
        "-m", gen.MODEL,
        "-c", f"model_reasoning_effort={gen.EFFORT}",
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
            text=True, timeout=gen.CODEX_TIMEOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"codex exec failed ({result.returncode}): {result.stderr[-2000:]}")
    return json.loads(last_message.read_text(encoding="utf-8"))


def critic_prompt(record: dict, sessions: dict[int, dict]) -> str:
    seeds = gen.seed_payload(record["seed_numbers"], sessions)
    candidates = [
        {"index": i, **{k: c[k] for k in ("question", "answer", "difficulty",
                                          "code_evidence", "note")}}
        for i, c in enumerate(record["candidates"])
    ]
    return CRITIC_TEMPLATE.format(
        **gen.DSPY_VALUES,
        label=record["label"],
        label_description=record["label_description"],
        num_final=NUM_FINAL,
        sampled_sources_json=json.dumps(seeds, ensure_ascii=False, indent=2),
        candidate_json=json.dumps(candidates, ensure_ascii=False, indent=2),
    )


def select_topic(record: dict, sessions: dict[int, dict], scope: str) -> dict:
    slug = record["label"]
    repo = gen.REPO_BY_SCOPE[scope]
    scope_dir = ARTIFACTS / scope
    scope_dir.mkdir(parents=True, exist_ok=True)
    output_path = scope_dir / f"5_topic_{record['cluster_id']}_{slug}.json"
    if output_path.exists():
        print(f"{scope}/{slug}: output exists, skipping")
        return json.loads(output_path.read_text(encoding="utf-8"))

    prompt = critic_prompt(record, sessions)
    (scope_dir / f"prompt_{slug}.txt").write_text(prompt, encoding="utf-8")

    result, problems = None, ["not run"]
    salvage = scope_dir / f"last_message_{slug}_a0.json"
    if salvage.exists():  # a completed archived attempt survives a crash/rerun
        result = json.loads(salvage.read_text(encoding="utf-8"))
        problems = critic_violations(result["selections"], repo)
        if not problems:
            print(f"{scope}/{slug}: salvaged valid archived attempt a0")
    for attempt in range(gen.MAX_RETRIES + 1):
        if not problems:
            break
        attempt_prompt = prompt if attempt == 0 else (
            prompt + "\n\n## Corrections required\nYour previous output had "
            "these problems; fix them and return the complete JSON again:\n- "
            + "\n- ".join(problems)
        )
        print(f"{scope}/{slug}: codex attempt {attempt + 1} ...", flush=True)
        result = run_codex(attempt_prompt, scope_dir, repo, slug, attempt)
        problems = critic_violations(result["selections"], repo)
        if problems:
            print(f"{scope}/{slug}: violations: {problems[:3]} ...")
    if problems:
        raise RuntimeError(f"{scope}/{slug}: unresolved violations after retries: {problems}")

    scope_files = gen.smalldspy_files()
    kept = sorted(item["source_index"] for item in result["selections"])
    out = {
        "cluster_id": record["cluster_id"],
        "label": slug,
        "label_description": record["label_description"],
        "seed_numbers": record["seed_numbers"],
        "num_candidates_reviewed": len(record["candidates"]),
        "kept_indices": kept,
        "rejected_indices": sorted(set(range(len(record["candidates"]))) - set(kept)),
        "selection_notes": result["selection_notes"],
        "finalists": [
            {
                **item,
                "topic": slug,
                "smalldspy_scope": all(
                    evidence["file"] in scope_files
                    for evidence in item["code_evidence"]
                ),
            }
            for item in result["selections"]
        ],
    }
    output_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"{scope}/{slug}: kept {len(kept)}/{len(record['candidates'])} "
          f"(indices {kept})")
    return out


def run_scope(scope: str, sessions: dict[int, dict]) -> None:
    repo = gen.REPO_BY_SCOPE[scope]
    if subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30,
    ).stdout.strip() != gen.PINNED_COMMIT:
        raise RuntimeError(f"{repo} is not at the pinned commit")
    scope_dir = ARTIFACTS / scope
    scope_dir.mkdir(parents=True, exist_ok=True)
    (scope_dir / "critic_schema.json").write_text(
        json.dumps(CRITIC_SCHEMA, indent=1), encoding="utf-8")
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((CANDIDATES_DIR / scope).glob("4_topic_*.json"))
    ]

    results, failures = [], []
    with ThreadPoolExecutor(max_workers=len(records)) as pool:
        futures = {pool.submit(select_topic, r, sessions, scope): r for r in records}
        for future, record in futures.items():
            try:
                results.append(future.result())
            except Exception as error:
                failures.append((record["label"], str(error)))
                print(f"{scope}/{record['label']}: FAILED: {str(error)[:300]}",
                      flush=True)
    if failures:
        raise RuntimeError(f"{scope}: {len(failures)} topic(s) failed - rerun to "
                           f"retry them: {[label for label, _ in failures]}")
    results.sort(key=lambda result: result["cluster_id"])
    merged = {
        "scope": scope,
        "harness": f"codex exec (codex-cli), model {gen.MODEL}, effort {gen.EFFORT}, "
                   "read-only sandbox",
        "repository": str(repo),
        "commit": gen.PINNED_COMMIT,
        "num_candidates_per_topic": gen.NUM_CANDIDATES,
        "num_final_per_topic": NUM_FINAL,
        "selection_ratio_note": "paper keeps 5/12; we keep round(20 * 5/12) = 8 of 20",
        "topics": [
            {k: result[k] for k in ("cluster_id", "label", "kept_indices",
                                    "rejected_indices", "selection_notes")}
            for result in results
        ],
        "finalists": [f for result in results for f in result["finalists"]],
    }
    merged_path = ARTIFACTS / f"5_{scope}_finalists.json"
    merged_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    total = len(merged["finalists"])
    in_scope = sum(finalist["smalldspy_scope"] for finalist in merged["finalists"])
    print(f"{scope}: wrote {total} finalists ({in_scope} in SmallDSPy scope) "
          f"to {merged_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", nargs="?", default="all",
                        choices=["fulldspy", "smalldspy", "all"])
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    sessions = gen.load_topics()[1]
    for scope in (("fulldspy", "smalldspy") if args.scope == "all" else (args.scope,)):
        run_scope(scope, sessions)


if __name__ == "__main__":
    main()
