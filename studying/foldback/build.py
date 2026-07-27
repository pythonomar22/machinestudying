"""Fold-back study-object builder: the studier designs and writes its own object.

Consumes analyze.py's records. Four phases, each idempotent on disk under
study/build/ (every cached artifact stores {payload, usage} plus the build
config hash; a config change refuses to reuse stale caches):

  A. Form decision — the studier (gpt-5.4-mini) receives the true metric
     mechanics (computed from its own stage-0 artifacts, including the
     marginal weight of each budget point and scenario math), drafts
     three independent designs, then critiques them and commits to one.
     All three delivery forms are real: prompt note (prepended at every
     budget), study_lookup store (a keyed tool available at tool budgets
     — implemented in studybench/react.py), or hybrid with per-entry
     routing.
  B. Assembly — per-topic merge of lessons/fixes/map entries into
     discrete entries in the committed design's organization, with the
     claim statements, buckets, and an already-known (BOTH1) digest in
     view. All content authored by the studier; code only concatenates.
  C. Verification, drop-only — deterministic path pruning (invalid cited
     paths removed; an entry is dropped only if nothing valid remains),
     then GPT-5.4 verdict-only fact-check against the union of each
     entry's cited sources; non-verified entries are dropped and
     ledgered, never rewritten.
  D. Render — cheatsheet.md + study.json at the run root in the exact
     form the studier chose (note entries only in the note; store
     entries in study/build/lookup_store.json; pure-store designs get a
     protocol-plus-key-index note), preserving the studier's section and
     entry order.

Usage:
    .venv-dspy/bin/python -m studying.foldback.build \
        --run-id dspy-gptminifoldback-20260726 --seed 20260715 [--debug]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from studybench.artifacts import read_json, sha256_json, sha256_text, stable_seed, write_json, write_text
from studybench.dataset import CORPORA, ROOT, read_corpus_file
from studybench.tools import RepoTools

from .analyze import ANALYST_MODELS, call_structured, config_sha256

VERIFIER_MODEL = "gpt-5.4"
VERIFY_FILE_CHARS = 40_000
VERIFY_GROUP_FILES = 3
NOTE_CHAR_CEILING = 150_000
N_PROPOSALS = 3
log = logging.getLogger("studying.foldback.build")


def two_point_wauc(points: list[tuple[float, float]]) -> float:
    pts = sorted(points)
    area = best = 0.0
    for i, (tokens, score) in enumerate(pts):
        best = max(best, score)
        w = min(3000 / tokens, 1.0)
        nw = min(3000 / pts[i + 1][0], 1.0) if i + 1 < len(pts) else 0.0
        area += (w - nw) * best
    return area


def mechanics_block(report: dict, summary: dict, library: str) -> str:
    d = report["budgets"]["direct"]
    k = report["budgets"]["k20f"]
    dt, ds = d["mean_generated_tokens"], d["mean_lenient"]
    kt, ks = k["mean_generated_tokens"], k["mean_lenient"]
    w_d, w_k = min(3000 / dt, 1.0), min(3000 / kt, 1.0)
    base = two_point_wauc([(dt, ds), (kt, ks)])
    bucket = summary["bucket_weight"]
    total = sum(bucket.values())
    pct = {b: 100 * v / total for b, v in bucket.items()}
    scenarios = [
        ("perfect direct fix (direct score -> k20f level, same tokens)",
         [(dt, ks), (kt, ks)]),
        ("k20f score +15 (better knowledge at tool budgets too)",
         [(dt, ds), (kt, ks + 15)]),
        ("both: direct 45, k20f +10, tokens unchanged",
         [(dt, 45.0), (kt, ks + 10)]),
        ("direct at k20f level but answers inflate to 6k tokens",
         [(6000, ks), (kt, ks)]),
    ]
    scenario_lines = "\n".join(
        f"  - {name}: {two_point_wauc(pts):.1f} ({two_point_wauc(pts) - base:+.1f})"
        for name, pts in scenarios)
    return f"""## The exam mechanics (all true, measured facts — design for them)

- The exam: 30 UNSEEN questions about {library} drawn from the same generator distribution as your practice set, each answered at four budgets: `direct` (zero tool iterations of any kind), `k5`/`k20` (up to 5/20 tool iterations, VOLUNTARY early stop — you may call finish on iteration 1), `k20f` (exactly 20 forced; a finish attempt is rejected with "You cannot finish yet — keep searching" and consumes the iteration). A judge scores each answer against a hidden weighted claim rubric, 0-100.
- The expertise metric sorts the four budget points by mean GENERATED tokens and computes a best-so-far area: each point contributes (its weight minus the next point's weight) x the best score so far, with weight = min(3000/tokens, 1). Only tokens you generate count; everything in your prompt — including your entire study artifact if delivered as a note — is free.
- Your measured practice profile: direct {ds:.1f}/100 at {dt:.0f} generated tokens (weight {w_d:.2f}), forced-k20 {ks:.1f}/100 at {kt:.0f} tokens (weight {w_k:.2f}). On these two points the metric is {base:.1f}, decomposing as {1 - w_k:.2f} x direct + {w_k:.2f} x best-of-both: **your k20f point carries ~{w_k:.0%} of the area, so raising accuracy at the tool budgets matters MORE for you than fixing direct alone** — and note-delivered knowledge raises both, since the note rides every budget's prompt. Scenario math (practice slice, 2-point):
{scenario_lines}
- Token headroom, not brevity, is the real constraint: direct answers at {dt:.0f} tokens have ~{3000 - dt:.0f} generated tokens of free headroom before the 3000-token anchor starts discounting; spend it on coverage if it earns claims. Inflating past ~{kt:.0f} tokens starts shrinking weights outright. A sibling model's study object once raised direct accuracy +14 while inflating answers ~2k tokens in an already-over-budget harness and lost the metric gain — the mechanism to respect is the weight formula, per budget.
- Your claim-weight breakdown across {summary['questions']} practice questions: {pct['FLIP+']:.1f}% earned only with tools (FLIP+), {pct['BOTH0']:.1f}% missed even with tools but present in the verified reference answers you studied (BOTH0 — gold-derived knowledge, lifts BOTH direct and tool budgets), {pct['BOTH1']:.1f}% you already know at direct (do not waste attention re-teaching), {pct['FLIP-']:.1f}% tool-run noise.
- Delivery options (all three are implemented; whatever you choose is exactly what gets built):
  1. `prompt_note`: a text artifact prepended to EVERY question at EVERY budget, including direct. Any internal structure you want. Ceiling {NOTE_CHAR_CEILING} characters; prompt space is free, so the cost of an entry is your own attention at answer time, not tokens — prefer compress-and-organize over drop.
  2. `lookup_tool`: a keyed store exposed as a `study_lookup(key)` tool during tool budgets (exact key or substring match; each call consumes one tool iteration plus the generated tokens of making it; the returned content arrives prompt-side, free). It CANNOT help at `direct` (zero iterations by definition). If you choose this pure form, your note is reduced to your answer protocol plus a mechanical index of store keys.
  3. `hybrid`: you route each entry to the note or the store (`delivery` field at assembly).
- Question-specific practice content is free to keep in a note (space-wise) but competes for your attention and helps only if a similar mechanism recurs; your analysis tagged generality per lesson — use it to organize, not necessarily to discard."""


PROPOSAL_PROMPT = """You are {studier}. You studied the {library} repository on {n_questions} practice questions and your extracted knowledge is summarized below. You will now DESIGN the study artifact you carry into the exam. Design for yourself: you know how you fail and what you already know.

{mechanics}

## Your analysis summary
{summary}

## Design task
Propose ONE complete design for your study artifact. Be concrete and opinionated; optimize expected exam score under the mechanics above, nothing else. Return JSON matching the schema exactly:
- `form`: prompt_note | lookup_tool | hybrid
- `sections`: ordered plan; each has `name`, `purpose` (what exam failure mode it prevents), `organization` (how entries inside it are arranged and why that ordering helps you find/absorb them at answer time), `budget_chars` (rough size).
- `answer_protocol`: `include` (bool) and `text` (the exact self-instructions about answering style, structure, token spend per budget, and early stopping to embed, or "").
- `rationale`: why this form and structure beat the alternatives FOR YOU, referencing the mechanics numerically.
- `failure_modes`: the ways this design could fail on the unseen exam."""

COMMIT_PROMPT = """You are {studier}, choosing the final design of your own study artifact. Below are {n} designs you drafted independently, plus the mechanics and your analysis summary. Critique each against the mechanics (the best-so-far area metric and each budget's marginal weight, attention dilution, per-budget token headroom, unseen-question transfer), then commit to a final design — one of the drafts or a merge of their best ideas. Return the same schema as the drafts (this is the binding plan).

{mechanics}

## Your analysis summary
{summary}

## Your draft designs
{proposals}"""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "form": {"type": "string", "enum": ["prompt_note", "lookup_tool", "hybrid"]},
        "sections": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "purpose": {"type": "string"},
                    "organization": {"type": "string"},
                    "budget_chars": {"type": "integer"},
                },
                "required": ["name", "purpose", "organization", "budget_chars"],
                "additionalProperties": False,
            },
        },
        "answer_protocol": {
            "type": "object",
            "properties": {"include": {"type": "boolean"}, "text": {"type": "string"}},
            "required": ["include", "text"],
            "additionalProperties": False,
        },
        "rationale": {"type": "string"},
        "failure_modes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["form", "sections", "answer_protocol", "rationale", "failure_modes"],
    "additionalProperties": False,
}

ASSEMBLY_PROMPT = """You are {studier} writing part of your own study artifact, following the design you committed to. Merge the raw extracted material below into final entries: deduplicate aggressively, keep every distinct load-bearing fact, and write so your exam self can act on each entry without the repository. The claim statements show exactly what graded weight each piece of material served; the already-known list shows what NOT to re-teach.

{mechanics}

## Your committed design
{plan}

## Already reliably known at direct for this topic (BOTH1) — do not spend entries re-teaching these
{known}

## Topic: {topic} — raw extracted material ({n_lessons} lessons, {n_fixes} misconception fixes, {n_map} navigation facts across {n_questions} practice questions)
{material}

## Output
`entries`: the final, deduplicated entries for this topic, in the order they should appear. Each has:
- `section`: one of the design's section names.
- `text`: the entry, self-contained, 1-4 sentences.
- `code`: minimal snippet or "".
- `grounding_paths`: repository files backing it (copy from the material; every factual entry must cite at least one — uncited entries are dropped by verification).
- `weight_hint`: summed claim weight this entry serves (integer, from the material).
- `generality`: reusable | question_specific.
- `source_qids`: practice question ids this entry came from.
- `delivery`: note | store (only meaningful for a hybrid design; use "note" otherwise).
- `lookup_key`: short snake_case key for the store ("" if delivery is note).
Return JSON matching the schema exactly."""

ASSEMBLY_SCHEMA = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "maxItems": 80,
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "text": {"type": "string"},
                    "code": {"type": "string"},
                    "grounding_paths": {"type": "array", "items": {"type": "string"}},
                    "weight_hint": {"type": "integer"},
                    "generality": {"type": "string",
                                   "enum": ["reusable", "question_specific"]},
                    "source_qids": {"type": "array", "items": {"type": "string"}},
                    "delivery": {"type": "string", "enum": ["note", "store"]},
                    "lookup_key": {"type": "string"},
                },
                "required": ["section", "text", "code", "grounding_paths", "weight_hint",
                             "generality", "source_qids", "delivery", "lookup_key"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entries"],
    "additionalProperties": False,
}

VERIFY_PROMPT = """You are fact-checking entries of a study document about the {library} repository against the repository sources below. For each entry, judge ONLY whether its factual content about {library} is supported by these sources (or is a self-evident consequence of them). Do not rewrite anything.

Verdicts: `verified` (supported), `unverified` (cannot confirm from these sources), `contradicted` (the sources show it is wrong).

{sources}

## Entries
{entries}

Return JSON matching the schema exactly: a verdict for every entry id."""

def build_config_sha256(model_key: str) -> str:
    payload = {
        "proposal": PROPOSAL_PROMPT, "commit": COMMIT_PROMPT, "plan_schema": PLAN_SCHEMA,
        "assembly": ASSEMBLY_PROMPT, "assembly_schema": ASSEMBLY_SCHEMA,
        "verify": VERIFY_PROMPT, "verifier": VERIFIER_MODEL,
        "verify_caps": [VERIFY_FILE_CHARS, VERIFY_GROUP_FILES],
        "note_ceiling": NOTE_CHAR_CEILING, "n_proposals": N_PROPOSALS,
        "analyze_config": config_sha256(model_key),
    }
    if model_key != "gptmini":  # gptmini sha predates parameterization
        payload["studier_model"] = ANALYST_MODELS[model_key]["model"]
    return sha256_json(payload)


def verify_schema(ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": ids},
                        "verdict": {"type": "string",
                                    "enum": ["verified", "unverified", "contradicted"]},
                    },
                    "required": ["id", "verdict"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    }


def entry_id(entry: dict) -> str:
    return sha256_text(
        f"{entry['text']}|{entry['code']}|{','.join(sorted(entry['grounding_paths']))}"
    )[:12]


def _source_state() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        check=True, capture_output=True, text=True).stdout.strip())
    return commit, dirty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model", default="gptmini", choices=sorted(ANALYST_MODELS))
    parser.add_argument("--base-urls", help="local vLLM endpoints (local studiers only)")
    parser.add_argument("--dirty-ok", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    profile = ANALYST_MODELS[args.model]
    if profile["local"] != bool(args.base_urls):
        raise SystemExit("--base-urls is required for local studiers and invalid otherwise")
    studier_key = os.environ.get(profile["key_env"]) or ("EMPTY" if profile["local"] else None)
    if not studier_key:
        raise SystemExit(f"{profile['key_env']} is required (studier)")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required (verifier)")
    BUILD_CONFIG_SHA256 = build_config_sha256(args.model)
    ANALYZE_CONFIG_SHA256 = config_sha256(args.model)

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "logs" / f"{args.run_id}-build.log")],
    )
    source_commit, source_dirty = _source_state()
    if source_dirty and not args.dirty_ok:
        raise SystemExit("build spends studier/verifier calls: commit first or pass --dirty-ok")

    corpus = CORPORA["dspy"]
    repository = RepoTools(corpus)
    run_root = ROOT / "runs" / args.run_id / corpus.name
    study_root = run_root / "study"
    build_root = study_root / "build"
    build_root.mkdir(parents=True, exist_ok=True)

    # completeness + config guards -------------------------------------------
    split = read_json(study_root / "split.json")
    study_ids = split["study_ids"]
    summary_path = study_root / "analysis" / "summary.json"
    if not summary_path.exists():
        raise SystemExit("analysis summary missing — run a FULL analyze first")
    summary = read_json(summary_path)
    if summary["config_sha256"] != ANALYZE_CONFIG_SHA256:
        raise SystemExit("analysis summary was produced by a different analyze config")
    records = []
    for qid in study_ids:
        path = study_root / "analysis" / f"{qid}.json"
        if not path.exists():
            raise SystemExit(f"analysis record missing for {qid} — run a FULL analyze")
        record = read_json(path)
        if record.get("config_sha256") != ANALYZE_CONFIG_SHA256:
            raise SystemExit(f"stale analysis record (different config): {qid}")
        records.append(record)
    if summary["questions"] != len(records):
        raise SystemExit("summary/records mismatch — rerun analyze")
    manifest_path = build_root / "manifest.json"
    manifest = {"build_config_sha256": BUILD_CONFIG_SHA256, "source_commit": source_commit,
                "source_dirty": source_dirty, "master_seed": args.seed}
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing["build_config_sha256"] != BUILD_CONFIG_SHA256:
            raise SystemExit("build config changed since cached artifacts were made; "
                             "delete study/build/ (or move it aside) and rerun")
    else:
        write_json(manifest_path, manifest)

    api = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=600, max_retries=2)
    studier_api = (OpenAI(api_key=studier_key, base_url=args.base_urls.split(",")[0],
                          timeout=600, max_retries=2)
                   if profile["local"] else api)
    report = read_json(study_root / "report.json")
    mechanics = mechanics_block(report, summary, corpus.display)
    summary_text = json.dumps(
        {k: summary[k] for k in (
            "questions", "bucket_weight", "recoverable_weight_covered_by_lessons",
            "uncovered_recoverable_weight", "lessons", "reusable_lessons",
            "gold_sourced_lessons", "lesson_kind_counts", "total_lesson_chars",
            "misconception_fixes", "map_entries", "by_topic")},
        indent=1)

    def studier(path: Path, prompt: str, schema: dict, name: str, seed: int) -> dict:
        if path.exists():
            return read_json(path)["payload"]
        payload, usage = call_structured(studier_api, prompt=prompt, schema=schema,
                                         name=name, seed=seed,
                                         model=profile["model"],
                                         sampling=profile["sampling"],
                                         cap_key=profile["cap_key"])
        write_json(path, {"payload": payload, "usage": usage})
        return payload

    # ---- phase A: form decision ---------------------------------------------
    proposals = [
        studier(build_root / f"proposal_{i}.json",
                PROPOSAL_PROMPT.format(studier=profile["model"], library=corpus.display,
                                       n_questions=summary["questions"],
                                       mechanics=mechanics, summary=summary_text),
                PLAN_SCHEMA, "fb_proposal", stable_seed(args.seed, "fb-proposal", i))
        for i in range(N_PROPOSALS)
    ]
    plan = studier(build_root / "plan.json",
                   COMMIT_PROMPT.format(studier=profile["model"], n=N_PROPOSALS, mechanics=mechanics,
                                        summary=summary_text,
                                        proposals=json.dumps(proposals, indent=1)),
                   PLAN_SCHEMA, "fb_commit", stable_seed(args.seed, "fb-commit"))
    planned_chars = sum(s["budget_chars"] for s in plan["sections"])
    if planned_chars > NOTE_CHAR_CEILING:
        log.warning("plan budgets %d chars exceed the %d ceiling; render will trim",
                    planned_chars, NOTE_CHAR_CEILING)
    log.info("committed form: %s | sections: %s | protocol: %s",
             plan["form"], [s["name"] for s in plan["sections"]],
             plan["answer_protocol"]["include"])

    # ---- phase B: assembly ---------------------------------------------------
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_topic[record["topic"]].append(record)
    section_names = {s["name"].lower(): s["name"] for s in plan["sections"]}
    entries: list[dict] = []
    for topic, topic_records in sorted(by_topic.items()):
        material, known = [], []
        for record in topic_records:
            claims = {c["claim_id"]: c for c in record["claims"]}
            for c in record["claims"]:
                if c["bucket"] == "BOTH1":
                    known.append(c["text"])
            for lesson in record["lessons"]:
                claim = claims.get(lesson["claim_id"], {})
                material.append({
                    "qid": record["qid"],
                    "claim_statement": claim.get("text", ""),
                    "claim_weight": claim.get("weight", 0),
                    "bucket": claim.get("bucket", ""),
                    **{k: lesson[k] for k in ("kind", "lesson", "code", "source",
                                              "grounding_paths", "generality")}})
            for fix in record["misconception_fixes"]:
                claim = claims.get(fix["claim_id"], {})
                material.append({
                    "qid": record["qid"],
                    "claim_statement": claim.get("text", ""),
                    "claim_weight": claim.get("weight", 0),
                    "bucket": claim.get("bucket", ""),
                    "kind": "misconception_fix",
                    "lesson": f"WRONG: {fix['wrong_belief']} CORRECTION: {fix['correction']}",
                    "code": "", "source": "self-diagnosis",
                    "grounding_paths": claim.get("evidence_paths", []),
                    "generality": "reusable"})
            for m in record["map_entries"]:
                material.append({
                    "qid": record["qid"], "claim_statement": "", "claim_weight": 0,
                    "bucket": "", "kind": "map_entry",
                    "lesson": f"{m['mechanism']}: {m['file']} — {m['symbol']} ({m['note']})",
                    "code": "", "source": "trajectory",
                    "grounding_paths": [m["file"]], "generality": "reusable"})
        payload = studier(
            build_root / f"assembled_{topic}.json",
            ASSEMBLY_PROMPT.format(
                studier=profile["model"],
                mechanics=mechanics, plan=json.dumps(plan, indent=1), topic=topic,
                known="\n".join(f"- {k}" for k in known) or "(none)",
                n_lessons=sum(len(r["lessons"]) for r in topic_records),
                n_fixes=sum(len(r["misconception_fixes"]) for r in topic_records),
                n_map=sum(len(r["map_entries"]) for r in topic_records),
                n_questions=len(topic_records),
                material=json.dumps(material, indent=1)),
            ASSEMBLY_SCHEMA, "fb_assemble", stable_seed(args.seed, "fb-assemble", topic))
        for entry in payload["entries"]:
            canonical = section_names.get(entry["section"].lower())
            if canonical is None:
                log.warning("entry rehomed from unknown section %r to %r",
                            entry["section"], plan["sections"][0]["name"])
                canonical = plan["sections"][0]["name"]
            entry["section"] = canonical
            entries.append({"id": entry_id(entry), "topic": topic, **entry})
    log.info("assembled %d entries across %d topics", len(entries), len(by_topic))

    # ---- phase C: verification (drop-only) ----------------------------------
    dropped_paths: list[str] = []
    for entry in entries:
        valid = [p for p in entry["grounding_paths"] if p in set(repository.files)]
        if len(valid) != len(entry["grounding_paths"]):
            log.warning("%s: pruned %d invalid cited paths", entry["id"],
                        len(entry["grounding_paths"]) - len(valid))
        entry["grounding_paths"] = valid
    verifiable = [e for e in entries if e["grounding_paths"]]
    dropped_uncited = [e["id"] for e in entries if not e["grounding_paths"]]
    if dropped_uncited:
        log.warning("dropping %d entries with no valid citations", len(dropped_uncited))

    verdict_dir = build_root / "verdicts"
    verdict_dir.mkdir(exist_ok=True)
    verifier_usage = {"completion_tokens": 0, "prompt_tokens": 0}
    verdicts: dict[str, str] = {}
    groups: dict[frozenset, list[dict]] = defaultdict(list)
    for entry in verifiable:
        groups[frozenset(entry["grounding_paths"][:VERIFY_GROUP_FILES])].append(entry)
    for paths, group in sorted(groups.items(), key=lambda kv: sorted(kv[0])):
        shard = verdict_dir / f"{sha256_text('|'.join(sorted(paths)))[:16]}.json"
        if shard.exists():
            cached = read_json(shard)
            if set(cached["ids"]) == {e["id"] for e in group}:
                verdicts.update(cached["verdicts"])
                continue
        sources = "\n\n".join(
            f"## Repository source: {p}\n{read_corpus_file(corpus, p)[:VERIFY_FILE_CHARS]}"
            for p in sorted(paths))
        listing = "\n".join(
            f"[{e['id']}] {e['text']}" + (f"\n{e['code']}" if e["code"] else "")
            for e in group)
        ids = [e["id"] for e in group]
        sampling = {"max_completion_tokens": 8_192}
        payload = None
        for attempt in range(2):
            response = api.chat.completions.create(
                model=VERIFIER_MODEL,
                messages=[{"role": "user", "content": VERIFY_PROMPT.format(
                    library=corpus.display, sources=sources, entries=listing)}],
                **sampling,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "fb_verify", "strict": True, "schema": verify_schema(ids)}},
            )
            if response.usage:
                verifier_usage["completion_tokens"] += response.usage.completion_tokens
                verifier_usage["prompt_tokens"] += response.usage.prompt_tokens
            content = response.choices[0].message.content or ""
            if response.choices[0].finish_reason == "stop" and content:
                try:
                    payload = json.loads(content)
                    break
                except json.JSONDecodeError:
                    pass
            sampling["max_completion_tokens"] *= 2
            log.warning("verify retry for %s", sorted(paths))
        if payload is None:
            raise RuntimeError(f"verifier failed twice for {sorted(paths)}")
        got = {v["id"]: v["verdict"] for v in payload["verdicts"]}
        for missing in set(ids) - set(got):
            log.warning("verifier omitted %s; treating as unverified", missing)
            got[missing] = "unverified"
        verdicts.update(got)
        write_json(shard, {"ids": ids, "verdicts": got})
        log.info("verified %s: %s", sorted(paths),
                 {v: sum(1 for i in ids if got.get(i) == v)
                  for v in ("verified", "unverified", "contradicted")})

    kept = [e for e in verifiable if verdicts.get(e["id"]) == "verified"]
    dropped_verify = {
        v: [e["id"] for e in verifiable if verdicts.get(e["id"]) == v]
        for v in ("unverified", "contradicted")
    }
    log.info("factcheck: kept %d / %d entries (dropped %d unverified, %d contradicted, "
             "%d uncited)", len(kept), len(entries), len(dropped_verify["unverified"]),
             len(dropped_verify["contradicted"]), len(dropped_uncited))

    # ---- phase D: render -----------------------------------------------------
    pure_store = plan["form"] == "lookup_tool"
    note_entries = [e for e in kept
                    if not pure_store and (plan["form"] == "prompt_note"
                                           or e["delivery"] == "note")]
    store_entries = [e for e in kept
                     if plan["form"] in {"lookup_tool", "hybrid"}
                     and (pure_store or e["delivery"] == "store")]

    trimmed: list[str] = []

    def render(rows: list[dict]) -> str:
        by_section: dict[str, list[dict]] = defaultdict(list)
        for entry in rows:
            by_section[entry["section"]].append(entry)
        parts = [f"# {corpus.display} study notes (self-authored)"]
        if plan["answer_protocol"]["include"] and plan["answer_protocol"]["text"].strip():
            parts.append("## Answer protocol\n" + plan["answer_protocol"]["text"].strip())
        for section in plan["sections"]:
            rows_here = by_section.get(section["name"], [])
            if not rows_here:
                continue
            parts.append(f"## {section['name']}")
            for entry in rows_here:  # assembly order is the studier's order
                block = f"- {entry['text'].strip()}"
                if entry["code"].strip():
                    block += f"\n```python\n{entry['code'].strip()}\n```"
                parts.append(block)
        if store_entries:
            parts.append("## study_lookup key index (tool budgets)\n" + "\n".join(
                f"- `{e['lookup_key'] or e['id']}`" for e in store_entries))
        return "\n\n".join(parts) + "\n"

    note = render(note_entries)
    while len(note) > NOTE_CHAR_CEILING and note_entries:
        # drop-only trim: question-specific lightest first, then lightest overall
        note_entries.sort(key=lambda e: (e["generality"] == "reusable", e["weight_hint"]))
        trimmed.append(note_entries.pop(0)["id"])
        note = render(note_entries)
    if trimmed:
        log.warning("ceiling trim dropped %d entries", len(trimmed))
    write_text(run_root / "cheatsheet.md", note)

    if store_entries:
        store: dict[str, dict] = {}
        for e in store_entries:
            key = e["lookup_key"] or e["id"]
            if key in store:
                key = f"{key}__{e['id']}"
                log.warning("lookup key collision; stored as %s", key)
            store[key] = {"text": e["text"], "code": e["code"],
                          "paths": e["grounding_paths"]}
        write_json(build_root / "lookup_store.json", store)
        log.info("lookup store: %d keys (served by react.py study_lookup)", len(store))

    def disk_usage_tokens() -> dict:
        spent = {}
        for path in sorted(build_root.glob("*.json")):
            if path.name in {"manifest.json", "lookup_store.json"}:
                continue
            data = read_json(path)
            if isinstance(data, dict) and "usage" in data:
                spent[path.stem] = data["usage"].get("completion_tokens", 0)
        return spent

    study = {
        "kind": "foldback",
        "schema_version": 2,
        "pipeline": "stage0 -> analyze -> build (self-designed object)",
        "form": plan["form"],
        "plan_sha256": sha256_json(plan),
        "build_config_sha256": BUILD_CONFIG_SHA256,
        "analyze_config_sha256": ANALYZE_CONFIG_SHA256,
        "source_commit": source_commit,
        "study_questions": summary["questions"],
        "studier": {"model": profile["model"], "sampling": profile["sampling"],
                    "sampling_note": profile["note"]},
        "verifier": {"model": VERIFIER_MODEL, "role": "verdict-only, drop-only",
                     "usage": verifier_usage},
        "studier_generated_tokens": {
            "attempts": sum(read_json(p).get("gen_tokens", 0)
                            for p in (study_root / "attempts").glob("*.json")),
            "direct_probe": sum(read_json(p).get("gen_tokens", 0)
                                for p in (study_root / "direct").glob("*.json")),
            "analysis": summary["analyst_completion_tokens"],
            "build": disk_usage_tokens(),
        },
        "object": {
            "entries_assembled": len(entries),
            "entries_kept": len(kept),
            "note_entries": len(note_entries),
            "store_entries": len(store_entries),
            "dropped_uncited": len(dropped_uncited),
            "dropped_unverified": len(dropped_verify["unverified"]),
            "dropped_contradicted": len(dropped_verify["contradicted"]),
            "ceiling_trimmed": len(trimmed),
            "question_specific_kept": sum(
                e["generality"] == "question_specific" for e in kept),
            "note_chars": len(note),
        },
        "final_cheatsheet_sha256": sha256_text(note),
        "final_cheatsheet_chars": len(note),
    }
    write_json(run_root / "study.json", study)
    log.info("study object: form=%s, %d note + %d store entries, %d chars -> %s",
             plan["form"], len(note_entries), len(store_entries), len(note),
             run_root / "cheatsheet.md")


if __name__ == "__main__":
    main()
