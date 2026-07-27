"""Fold-back analysis: what exactly did forced-k20 compute buy over direct?

Consumes the stage-0 artifacts (direct + k20f episodes, both grade sets,
the practice rubrics/golds/evidence) and produces one machine-readable
analysis record per study question plus a corpus summary. Two layers:

  1. Deterministic: per-claim flip buckets (FLIP+ 0->1, BOTH0, FLIP- 1->0,
     BOTH1), judge rationales for every miss, evidence paths per claim,
     and the k20f trajectory turns that touched each claim's evidence.
  2. Studier (gpt-5.4-mini, the model under study — authorship stays with
     it): for each *recoverable* claim (FLIP+ or BOTH0), the exact
     knowledge that would have earned the claim with zero tool calls,
     grounded in trajectory observations or the reference answer, tagged
     with source and generality; plus corrections for the direct answer's
     actual errors and navigational map entries. Questions whose first
     pass leaves recoverable claims without a lesson get one targeted
     follow-up call; still-uncovered claims are recorded, never invented.

The output schema is the builder's input contract (build.py); nothing
here presupposes the study object's eventual form.

Usage:
    .venv-dspy/bin/python -m studying.foldback.analyze \
        --run-id dspy-gptminifoldback-20260726 --seed 20260715 \
        [--limit N] [--concurrency 8] [--debug]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from studybench.artifacts import read_json, sha256_json, stable_seed, write_json
from studybench.dataset import CORPORA, ROOT

from .data import load_practice_questions, practice_dataset_sha256

ANALYST_MODEL = "gpt-5.4-mini"
# The gpt-5.x surface pins temperature to 1; the 010 temp-0.2 study-time
# convention is unreachable for this studier (ledgered in the summary).
ANALYST_SAMPLING = {"temperature": 1.0, "max_completion_tokens": 8_192}
# Studier profiles. gptmini's config sha must stay byte-identical to the
# pre-parameterization constant, so its profile adds nothing to the payload.
ANALYST_MODELS = {
    "gptmini": {
        "model": ANALYST_MODEL,
        "sampling": ANALYST_SAMPLING,
        "cap_key": "max_completion_tokens",
        "key_env": "OPENAI_API_KEY",
        "local": False,
        "note": "temperature pinned to 1.0 by the gpt-5.x surface "
                "(010 study-time convention was 0.2)",
    },
    "qwen": {
        "model": "Qwen/Qwen3.5-9B",
        # thinking bounded so rumination cannot eat the JSON budget (the
        # local-judge convention); 010 temp-0.2 study-time convention kept.
        "sampling": {"temperature": 0.2, "top_p": 0.95, "max_tokens": 16_384,
                     "extra_body": {"thinking_token_budget": 6_000}},
        "cap_key": "max_tokens",
        "key_env": "VLLM_API_KEY",
        "local": True,
        "note": "010 temp-0.2 study-time convention (local vLLM, "
                "thinking_token_budget 6000)",
    },
}
MAX_COMPLETION_CEILING = 32_768
TIMEOUT = 600
KINDS = ("api_fact", "idiom", "behavior", "pitfall", "concept", "other")
MAX_LESSONS = 12
MAX_FIXES = 6
MAX_MAP = 6
REASONING_CHARS = 600
OBS_CHARS = 2_000
TOUCHED_OBS_CHARS = 4_000
ANSWER_CHARS = 12_000
GOLD_CHARS = 7_000
log = logging.getLogger("studying.foldback.analyze")

BUCKETS = {(0, 1): "FLIP+", (1, 0): "FLIP-", (0, 0): "BOTH0", (1, 1): "BOTH1"}
RECOVERABLE = {"FLIP+", "BOTH0"}

PROMPT = """You are {studier} studying the {library} repository. You answered one practice question twice, and a judge graded both answers against a weighted claim rubric:

- `direct`: zero tool calls, from your own knowledge. Score {direct_score}/100.
- `k20f`: 20 forced repository-tool iterations. Score {k20f_score}/100.

Your job now is to extract, claim by claim, the knowledge that would let a future zero-tool you earn every recoverable claim. Later, all extractions across {n_questions} practice questions get compiled into a study artifact you will carry into an exam with DIFFERENT, unseen questions about {library} drawn from the same generator distribution — so tag honestly which knowledge generalizes and which is specific to this question.

## Question (topic: {topic})
{question}

## Reference answer (verified correct)
{gold}

## Claim rubric with grading outcomes
Each claim: weight, bucket (FLIP+ = only k20f earned it; BOTH0 = both missed it — the reference answer above is your source for these; BOTH1 = you already knew it at direct; FLIP- = only direct earned it), the judge's rationale for any miss, the repository files its evidence lives in, and which of your k20f trajectory turns touched those files.
{claim_table}

## Your direct answer (what you believe with no tools)
{direct_answer}

## Your k20f trajectory ({iters} iterations; observations at turns that touched claim evidence are shown at full length)
{trajectory}

## Your k20f answer
{k20f_answer}

## Extract
- `lessons`: for EVERY claim in bucket FLIP+ or BOTH0 (skip BOTH1 — you already know those; skip FLIP-), the exact knowledge that earns it at zero tools:
  - `claim_id`: the rubric claim this earns.
  - `kind`: one of {kinds}.
  - `lesson`: 1-3 self-contained sentences, precise enough to act on without the repository.
  - `code`: minimal snippet proving/using it (at most 14 lines), or "".
  - `source`: `trajectory` (grounded in an observation), `gold` (from the reference answer), `both`, or `prior` (you actually knew it but failed to produce it — say so honestly).
  - `source_turns`: trajectory turn indices that ground it ([] unless source involves trajectory).
  - `grounding_paths`: repository files that back the lesson (use the claim's evidence files unless you saw better ones).
  - `generality`: `reusable` (helps on unseen {library} questions) or `question_specific` (only this exact question).
- `misconception_fixes`: concrete errors in your direct answer (see the judge rationales), each as `wrong_belief` (what you wrongly said/assumed), `correction` (the truth), `claim_id`.
- `map_entries`: at most {max_map} navigational facts worth remembering for future tool-assisted search: `mechanism` (snake_case), `file` (a path you actually saw), `symbol`, `note` (one line).

Ground everything; if you cannot ground a lesson in the trajectory, the reference answer, or a repository file you saw, either mark source `prior` or omit it. Return JSON matching the schema exactly."""

FOLLOWUP_PROMPT = """You are {studier}, continuing the extraction you just did for the practice question below. Your first pass produced no lesson for these RECOVERABLE claims (bucket FLIP+ or BOTH0), which together carry {missing_weight} points of rubric weight:

{missing_claims}

Using the same materials (question, reference answer, your trajectory, your answers — all repeated below), extract a lesson for each of these claims specifically, with the same fields and grounding discipline as before. If a claim genuinely cannot be grounded in the trajectory, the reference answer, or a repository file you saw, omit it — do not invent.

{body}

Return JSON matching the schema exactly (lessons only for the listed claims; misconception_fixes and map_entries may be empty)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "lessons": {
            "type": "array",
            "maxItems": MAX_LESSONS,
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "lesson": {"type": "string"},
                    "code": {"type": "string"},
                    "source": {"type": "string", "enum": ["trajectory", "gold", "both", "prior"]},
                    "source_turns": {"type": "array", "items": {"type": "integer"}},
                    "grounding_paths": {"type": "array", "items": {"type": "string"}},
                    "generality": {"type": "string", "enum": ["reusable", "question_specific"]},
                },
                "required": ["claim_id", "kind", "lesson", "code", "source",
                             "source_turns", "grounding_paths", "generality"],
                "additionalProperties": False,
            },
        },
        "misconception_fixes": {
            "type": "array",
            "maxItems": MAX_FIXES,
            "items": {
                "type": "object",
                "properties": {
                    "wrong_belief": {"type": "string"},
                    "correction": {"type": "string"},
                    "claim_id": {"type": "string"},
                },
                "required": ["wrong_belief", "correction", "claim_id"],
                "additionalProperties": False,
            },
        },
        "map_entries": {
            "type": "array",
            "maxItems": MAX_MAP,
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
    "required": ["lessons", "misconception_fixes", "map_entries"],
    "additionalProperties": False,
}

def config_sha256(model_key: str) -> str:
    profile = ANALYST_MODELS[model_key]
    payload = {
        "prompt": PROMPT,
        "followup": FOLLOWUP_PROMPT,
        "schema": SCHEMA,
        "sampling": profile["sampling"],
        "caps": [REASONING_CHARS, OBS_CHARS, TOUCHED_OBS_CHARS, ANSWER_CHARS, GOLD_CHARS],
    }
    if model_key != "gptmini":  # gptmini sha predates parameterization
        payload["studier_model"] = profile["model"]
    return sha256_json(payload)


CONFIG_SHA256 = config_sha256("gptmini")


def call_structured(api: OpenAI, *, prompt: str, schema: dict, name: str, seed: int,
                    model: str = ANALYST_MODEL, sampling: dict | None = None,
                    cap_key: str = "max_completion_tokens") -> tuple[dict, dict]:
    sampling = dict(sampling if sampling is not None else ANALYST_SAMPLING)
    finish = None
    for attempt in range(3):  # resample protocol; final attempts run at the cap ceiling
        response = api.chat.completions.create(
            model=model,
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
        if isinstance(payload, dict) and finish == "stop":
            return payload, usage
        log.warning("%s: unusable payload (finish=%s, chars=%d); retrying larger",
                    name, finish, len(content))
        sampling[cap_key] = min(sampling[cap_key] * 4, MAX_COMPLETION_CEILING)
    raise RuntimeError(f"{name}: no usable structured payload after retry (finish={finish})")


def claim_flip_table(row: dict, direct_grade: dict, k20f_grade: dict, episode: dict) -> list[dict]:
    """Deterministic per-claim record: bucket, rationales, evidence, touched turns."""

    d_claims = {c["claim_id"]: c for c in direct_grade["claims"]}
    k_claims = {c["claim_id"]: c for c in k20f_grade["claims"]}
    spans = {s["span_id"]: s for s in row["evidence"]}
    turn_blobs = [
        f"{turn.get('arguments')} {str(turn.get('observation'))}"
        for turn in episode["turns"]
    ]
    table = []
    for claim in row["rubric"]:
        cid = claim["claim_id"]
        d, k = d_claims[cid]["score"], k_claims[cid]["score"]
        paths = sorted({spans[sid]["path"] for sid in claim["span_ids"]})
        touched = [i for i, blob in enumerate(turn_blobs) if any(p in blob for p in paths)]
        table.append({
            "claim_id": cid,
            "text": claim["statement"],
            "weight": claim["weight"],
            "bucket": BUCKETS[(d, k)],
            "direct_rationale": d_claims[cid]["rationale"] if d == 0 else "",
            "k20f_rationale": k_claims[cid]["rationale"] if k == 0 else "",
            "evidence_paths": paths,
            "touched_turns": touched,
        })
    return table


def render_claim_table(table: list[dict]) -> str:
    lines = []
    for c in table:
        lines.append(
            f"- {c['claim_id']} (w={c['weight']}, {c['bucket']}): {c['text']}\n"
            f"    evidence: {', '.join(c['evidence_paths'])}"
            f" | trajectory turns touching evidence: {c['touched_turns'] or 'none'}"
        )
        if c["direct_rationale"]:
            lines.append(f"    judge on your direct miss: {c['direct_rationale']}")
        if c["k20f_rationale"]:
            lines.append(f"    judge on your k20f miss: {c['k20f_rationale']}")
    return "\n".join(lines)


def render_trajectory(turns: list[dict], touched: set[int]) -> str:
    """Analysis-grade trajectory: expanded observations, extra length on turns
    that touched claim evidence (mine.py's 400-char cap starves extraction)."""

    lines = []
    for index, turn in enumerate(turns):
        reasoning = " ".join(str(turn.get("reasoning") or "").split())[:REASONING_CHARS]
        cap = TOUCHED_OBS_CHARS if index in touched else OBS_CHARS
        observation = " ".join(str(turn.get("observation") or "").split())[:cap]
        arguments = json.dumps(turn.get("arguments"), ensure_ascii=False)[:200]
        lines.append(
            f"[{index}] thought: {reasoning}\n"
            f"    call: {turn.get('tool')}({arguments})\n"
            f"    saw: {observation}"
        )
    return "\n".join(lines) or "(no tool calls)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model", default="gptmini", choices=sorted(ANALYST_MODELS))
    parser.add_argument("--base-urls", help="local vLLM endpoints (local studiers only)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    profile = ANALYST_MODELS[args.model]
    if profile["local"] != bool(args.base_urls):
        raise SystemExit("--base-urls is required for local studiers and invalid otherwise")
    api_key = os.environ.get(profile["key_env"]) or ("EMPTY" if profile["local"] else None)
    if not api_key:
        raise SystemExit(f"{profile['key_env']} is required (studier analyst)")
    config_sha = config_sha256(args.model)

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "logs" / f"{args.run_id}-analyze.log")],
    )
    corpus = CORPORA["dspy"]
    study_root = ROOT / "runs" / args.run_id / corpus.name / "study"
    out_root = study_root / "analysis"
    rows = {row["id"]: row for row in load_practice_questions()}
    split = read_json(study_root / "split.json")
    if split["dataset_sha256"] != practice_dataset_sha256():
        raise SystemExit("practice dataset changed since the split was made")
    full_study_ids = split["study_ids"]
    study_ids = full_study_ids[: args.limit or None]
    urls = args.base_urls.split(",") if args.base_urls else [None]
    clients = [OpenAI(api_key=api_key, base_url=url, timeout=TIMEOUT, max_retries=2)
               for url in urls]

    def analyze_one(index: int, qid: str) -> dict:
        api = clients[index % len(clients)]
        path = out_root / f"{qid}.json"
        if path.exists():
            cached = read_json(path)
            if cached.get("config_sha256") == config_sha:
                return cached
            log.warning("%s: cached record from a different analyze config; regenerating", qid)
        row = rows[qid]
        episode = read_json(study_root / "attempts" / f"{qid}.json")
        direct_ep = read_json(study_root / "direct" / f"{qid}.json")
        if (episode["status"] not in {"ok", "no_answer"}
                or direct_ep["status"] not in {"ok", "no_answer"}):
            raise SystemExit(f"{qid}: stage-0 episode not usable; rerun stage 0 first")
        direct_grade = read_json(study_root / "grades" / "direct" / f"{qid}.json")
        k20f_grade = read_json(study_root / "grades" / "k20f" / f"{qid}.json")
        table = claim_flip_table(row, direct_grade, k20f_grade, episode)
        touched_all = {i for c in table if c["bucket"] in RECOVERABLE for i in c["touched_turns"]}

        record = {
            "qid": qid,
            "topic": row["topic"],
            "config_sha256": config_sha,
            "direct_lenient": direct_grade["lenient"],
            "k20f_lenient": k20f_grade["lenient"],
            "claims": table,
            "lessons": [],
            "misconception_fixes": [],
            "map_entries": [],
            "uncovered_recoverable": [],
            "analyst_usage": [],
        }
        recoverable_ids = {c["claim_id"] for c in table if c["bucket"] in RECOVERABLE}
        if recoverable_ids:
            body = PROMPT.format(
                studier=profile["model"],
                library=corpus.display,
                direct_score=direct_grade["lenient"],
                k20f_score=k20f_grade["lenient"],
                n_questions=len(full_study_ids),
                topic=row["topic"],
                question=row["question"],
                gold=row["gold_answer"][:GOLD_CHARS],
                claim_table=render_claim_table(table),
                direct_answer=(direct_ep.get("answer") or "(no answer)")[:ANSWER_CHARS],
                iters=episode["react_iterations"],
                trajectory=render_trajectory(episode["turns"], touched_all),
                k20f_answer=(episode.get("answer") or "(no answer)")[:ANSWER_CHARS],
                kinds=", ".join(KINDS),
                max_map=MAX_MAP,
            )
            payload, usage = call_structured(
                api, prompt=body, schema=SCHEMA, name="foldback_analyze",
                seed=stable_seed(args.seed, "fb-analyze", qid),
                model=profile["model"], sampling=profile["sampling"],
                cap_key=profile["cap_key"],
            )
            record["analyst_usage"].append(usage)
            valid_ids = {c["claim_id"] for c in table}
            lessons = [l for l in payload["lessons"] if l["claim_id"] in recoverable_ids]
            dropped = len(payload["lessons"]) - len(lessons)
            if dropped:
                log.warning("%s: dropped %d lessons aimed at non-recoverable/unknown claims",
                            qid, dropped)
            missing = recoverable_ids - {l["claim_id"] for l in lessons}
            if missing:
                by_id = {c["claim_id"]: c for c in table}
                missing_lines = "\n".join(
                    f"- {cid} (w={by_id[cid]['weight']}, {by_id[cid]['bucket']}): "
                    f"{by_id[cid]['text']}" for cid in sorted(missing))
                followup, usage2 = call_structured(
                    api,
                    prompt=FOLLOWUP_PROMPT.format(
                        studier=profile["model"],
                        missing_weight=sum(by_id[cid]["weight"] for cid in missing),
                        missing_claims=missing_lines,
                        body=body),
                    schema=SCHEMA, name="foldback_analyze_followup",
                    seed=stable_seed(args.seed, "fb-analyze2", qid),
                    model=profile["model"], sampling=profile["sampling"],
                    cap_key=profile["cap_key"],
                )
                record["analyst_usage"].append(usage2)
                lessons += [l for l in followup["lessons"]
                            if l["claim_id"] in missing]
            record["lessons"] = lessons
            record["uncovered_recoverable"] = sorted(
                recoverable_ids - {l["claim_id"] for l in lessons})
            record["misconception_fixes"] = [
                f for f in payload["misconception_fixes"] if f["claim_id"] in valid_ids
            ]
            record["map_entries"] = payload["map_entries"]
        write_json(path, record)
        log.info("%s: %d lessons (%d uncovered recoverable), %d fixes, %d map entries",
                 qid, len(record["lessons"]), len(record["uncovered_recoverable"]),
                 len(record["misconception_fixes"]), len(record["map_entries"]))
        return record

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        records = list(pool.map(lambda item: analyze_one(*item), enumerate(study_ids)))

    weight = {"FLIP+": 0, "FLIP-": 0, "BOTH0": 0, "BOTH1": 0}
    covered = {"FLIP+": 0, "BOTH0": 0}
    uncovered_weight = 0
    by_topic: dict[str, dict] = {}
    for record in records:
        lesson_ids = {l["claim_id"] for l in record["lessons"]}
        topic_stats = by_topic.setdefault(
            record["topic"],
            {"questions": 0, "lessons": 0, "recoverable_weight": 0,
             "lesson_chars": 0, "code_chars": 0})
        topic_stats["questions"] += 1
        topic_stats["lessons"] += len(record["lessons"])
        topic_stats["lesson_chars"] += sum(len(l["lesson"]) for l in record["lessons"])
        topic_stats["code_chars"] += sum(len(l["code"]) for l in record["lessons"])
        for c in record["claims"]:
            weight[c["bucket"]] += c["weight"]
            if c["bucket"] in RECOVERABLE:
                topic_stats["recoverable_weight"] += c["weight"]
                if c["claim_id"] in lesson_ids:
                    covered[c["bucket"]] += c["weight"]
                else:
                    uncovered_weight += c["weight"]

    kind_counts: dict[str, int] = {}
    for record in records:
        for lesson in record["lessons"]:
            kind_counts[lesson["kind"]] = kind_counts.get(lesson["kind"], 0) + 1

    summary = {
        "kind": "foldback-analysis",
        "run_id": args.run_id,
        "analyst": {"model": profile["model"], "sampling": profile["sampling"],
                    "note": profile["note"]},
        "master_seed": args.seed,
        "questions": len(records),
        "bucket_weight": weight,
        "recoverable_weight_covered_by_lessons": covered,
        "uncovered_recoverable_weight": uncovered_weight,
        "lessons": sum(len(r["lessons"]) for r in records),
        "reusable_lessons": sum(
            sum(l["generality"] == "reusable" for l in r["lessons"]) for r in records),
        "lesson_kind_counts": kind_counts,
        "gold_sourced_lessons": sum(
            sum(l["source"] in {"gold", "both"} for l in r["lessons"]) for r in records),
        "total_lesson_chars": sum(
            sum(len(l["lesson"]) + len(l["code"]) for l in r["lessons"]) for r in records),
        "misconception_fixes": sum(len(r["misconception_fixes"]) for r in records),
        "map_entries": sum(len(r["map_entries"]) for r in records),
        "by_topic": by_topic,
        "analyst_completion_tokens": sum(
            u.get("completion_tokens", 0) for r in records for u in r["analyst_usage"]),
        "config_sha256": config_sha,
    }
    if len(study_ids) == len(full_study_ids):
        write_json(out_root / "summary.json", summary)
    else:
        write_json(out_root / "summary_partial.json", summary)
        log.warning("partial run (%d/%d questions): wrote summary_partial.json only",
                    len(study_ids), len(full_study_ids))
    log.info("analysis complete: %d lessons (%d reusable, %d gold-sourced), covered "
             "weight FLIP+ %d/%d, BOTH0 %d/%d, uncovered %d",
             summary["lessons"], summary["reusable_lessons"],
             summary["gold_sourced_lessons"],
             covered["FLIP+"], weight["FLIP+"], covered["BOTH0"], weight["BOTH0"],
             uncovered_weight)


if __name__ == "__main__":
    main()
