"""Offline grading: sandbox compilation check, then GPT-5.4 rubric judge.

The judge prompt is the paper's Appendix A.5 grader prompt verbatim, with one change
confirmed by the first author: claims are scored 0 or 1 only (the 0.5 partial-credit
level was removed because it increased variance). Scores:

  lenient = weighted sum of claim scores (what Table 1 reports)
  strict  = 0 unless the compilation check passes AND every core claim scores 1;
            otherwise equal to the weighted sum

Writes grades/{task}/{budget}/r{rollout}/{qid}.json next to each episode in runs/.
"""

import argparse
import asyncio
import json
import logging
import os

from openai import AsyncOpenAI

from . import sandbox
from .dataset import CORPORA, ROOT, load_questions

JUDGE_MODEL = "gpt-5.4"

GRADER_PROMPT = """You are grading one model answer for a private {library_name} expert QA benchmark.

## Scoring model
- The question gets one final continuous 0-100 score.
- Claims are only the internal rubric used to compute that question's score.
- Score each claim as:
  - `0` = wrong or missing
  - `1` = fully correct
- Do not give extra credit for material outside the rubric.
- If an answer is polished but misses essential content, score the missing claims low.
- Use the evidence spans and gold answer to resolve ambiguity.

## Output rules
- Score every rubric claim exactly once.
- `question_score` must equal the weighted sum of the claim scores.
- Set `needs_regrade` to `true` only if the rubric or evidence is genuinely insufficient to judge the answer confidently.
- Keep rationales concise and specific.

## Inputs
- Question ID: `{question_id}`
- Label: `{label}`
- Question: `{question}`
- Model answer:
{model_answer}

## Gold answer
{gold_answer}

## Claim rubric
{claim_rubric_json}

## Evidence spans
{evidence_spans_json}

## Whole evidence files
{whole_evidence_text}

Return JSON that matches the schema exactly."""

def judge_schema(row: dict) -> dict:
    """Structured-output schema, pinned to this question's claim ids and count."""
    ids = [c["claim_id"] for c in row["rubric"]]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "grading",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "minItems": len(ids),
                        "maxItems": len(ids),
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim_id": {"type": "string", "enum": ids},
                                "score": {"type": "integer", "enum": [0, 1]},
                                "rationale": {"type": "string"},
                            },
                            "required": ["claim_id", "score", "rationale"],
                            "additionalProperties": False,
                        },
                    },
                    "question_score": {"type": "number"},
                    "needs_regrade": {"type": "boolean"},
                },
                "required": ["claims", "question_score", "needs_regrade"],
                "additionalProperties": False,
            },
        },
    }

log = logging.getLogger("grade")


def build_prompt(corpus, row: dict, model_answer: str, whole_files: bool = False) -> str:
    if whole_files:
        # A.5-faithful: spans = the dataset's excerpts; whole files = full numbered
        # dumps of every evidence file from the pinned checkout
        spans_meta = row["evidence"]
        paths = list(dict.fromkeys(e["path"] for e in row["evidence"]))
        whole_text = "\n\n".join(
            f"### {p}\n" + "\n".join(
                f"{i:04d}: {line}" for i, line in enumerate(
                    (corpus.repo / p).read_text().splitlines(), 1)
            ) for p in paths
        )
    else:
        # dataset-README variant: the excerpts are the only code context
        spans_meta = [
            {k: e[k] for k in ("span_id", "path", "start_line", "end_line")}
            for e in row["evidence"]
        ]
        whole_text = "\n\n".join(
            f"### {e['path']} lines {e['start_line']}-{e['end_line']} ({e['span_id']})\n{e['excerpt']}"
            for e in row["evidence"]
        )
    return GRADER_PROMPT.format(
        library_name=corpus.display,
        question_id=row["id"],
        label=row["topic"],
        question=row["question"],
        model_answer=model_answer,
        gold_answer=row["gold_answer"],
        claim_rubric_json=json.dumps(row["rubric"], indent=2),
        evidence_spans_json=json.dumps(spans_meta, indent=2),
        whole_evidence_text=whole_text,
    )


def score_from_claims(row: dict, claim_scores: dict[str, int], compile_ok: bool) -> dict:
    lenient = sum(c["weight"] * claim_scores.get(c["claim_id"], 0) for c in row["rubric"])
    cores_ok = all(
        claim_scores.get(c["claim_id"], 0) == 1
        for c in row["rubric"] if c["claim_type"] == "core"
    )
    strict = lenient if (compile_ok and cores_ok) else 0
    return {"lenient": lenient, "strict": strict, "cores_ok": cores_ok}


async def grade_episode(client: AsyncOpenAI, corpus, row: dict, ep: dict,
                        whole_files: bool = False, effort: str = "") -> dict:
    grade = {
        "task": ep["task"], "qid": ep["qid"], "budget": ep["budget"], "rollout": ep["rollout"],
        "judge_model": JUDGE_MODEL, "episode_status": ep["status"],
        "gen_tokens": ep["gen_tokens"],
    }
    answer = ep.get("answer", "")
    if not answer.strip():
        grade.update(compile_check={"compile_ok": False, "detail": "empty answer"},
                     claims=[], needs_regrade=False, judge_question_score=0,
                     lenient=0, strict=0, cores_ok=False)
        return grade

    grade["compile_check"] = await asyncio.to_thread(sandbox.check, answer, corpus.language)

    rubric_ids = {c["claim_id"] for c in row["rubric"]}
    for attempt in range(2):  # retry once if the judge duplicates/omits claim ids
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user",
                       "content": build_prompt(corpus, row, answer, whole_files)}],
            response_format=judge_schema(row),
            **({"reasoning_effort": effort} if effort else {}),
        )
        verdict = json.loads(resp.choices[0].message.content)
        claim_scores = {c["claim_id"]: c["score"] for c in verdict["claims"]}
        if set(claim_scores) == rubric_ids:
            break
        log.warning("%s/%s/r%d attempt %d: judge returned claim ids %s, want %s",
                    ep["budget"], ep["qid"], ep["rollout"], attempt,
                    sorted(claim_scores), sorted(rubric_ids))
    grade.update(
        claims=verdict["claims"],
        needs_regrade=verdict["needs_regrade"],
        judge_question_score=verdict["question_score"],
        **score_from_claims(row, claim_scores, grade["compile_check"]["compile_ok"]),
    )
    return grade


async def main_async(args):
    corpus = CORPORA[args.task]
    rows = {q["id"]: q for q in load_questions(args.task)}
    client = AsyncOpenAI(timeout=600)
    runs_root = ROOT / ("runs" if not args.variant else f"runs-{args.variant}")
    out_root = ROOT / ("grades" + (f"-{args.variant}" if args.variant else "")
                       + ("-wholefiles" if args.whole_files else "")
                       + (f"-effort-{args.judge_effort}" if args.judge_effort else ""))

    run_files = sorted((runs_root / args.task).rglob("*.json"))
    pending = []
    for rf in run_files:
        gf = out_root / rf.relative_to(runs_root)
        # (re-)grade if ungraded, or if the episode was regenerated since grading
        if not gf.exists() or gf.stat().st_mtime < rf.stat().st_mtime:
            pending.append((rf, gf))
    log.info("%d episodes to grade (task=%s)", len(pending), args.task)

    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def one(rf, gf):
        nonlocal done
        async with sem:
            try:
                ep = json.loads(rf.read_text())
                grade = await grade_episode(client, corpus, rows[ep["qid"]], ep,
                                            args.whole_files, args.judge_effort)
                gf.parent.mkdir(parents=True, exist_ok=True)
                tmp = gf.with_suffix(".tmp")
                tmp.write_text(json.dumps(grade, indent=2))
                tmp.rename(gf)
            except Exception:
                log.exception("grading %s failed (no grade written; rerun to retry)", rf)
                return
            done += 1
            log.info("[%d/%d] %s/%s/r%d lenient=%.0f strict=%.0f compile_ok=%s%s",
                     done, len(pending), grade["budget"], grade["qid"], grade["rollout"],
                     grade["lenient"], grade["strict"],
                     grade["compile_check"]["compile_ok"],
                     " NEEDS_REGRADE" if grade.get("needs_regrade") else "")
            if args.debug:
                log.debug("claims for %s/r%d: %s", grade["qid"], grade["rollout"],
                          json.dumps(grade["claims"], indent=2))

    await asyncio.gather(*(one(*p) for p in pending))
    log.info("all done")


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=list(CORPORA))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--whole-files", action="store_true",
                   help="give the judge full evidence files (paper A.5) instead of "
                        "the dataset's span excerpts; writes to grades-wholefiles/")
    p.add_argument("--variant", default="",
                   help="grade runs-<variant>/ into grades-<variant>/")
    p.add_argument("--judge-effort", default="",
                   choices=["", "low", "medium", "high", "xhigh"],
                   help="judge reasoning effort (default: API default)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    load_env()
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "logs" / f"grade-{args.task}.log")],
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
