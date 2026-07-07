"""SELF-QUIZZING study procedure (experiments/005 v1.2): the agent studies a
codebase by quizzing itself; verified errors become its note.

Per round r (chapters advance through a fixed size-ordered syllabus, wrapping):
  QUIZ     one ReAct episode per chapter writes M questions (anchored, deduped;
           1 of M held out to the accumulating dev exam)
  ATTEMPT  closed book (dspy.Predict): note_{r-1} + question -> committed answer
  VERIFY   Phase A derives the answer blind (ReAct + run_python probe; never
           sees the attempt or the note); Phase B diffs attempt vs derivation
  DISTILL  wrong/partial only -> {belief, correction, quote, file, line};
           a model-free gate string-matches the quote at file:line before the
           entry is admitted to the note
  RETEST   (r>=2) ~20% of slots re-run previous items against the current note

Everything is logged per item (study-selfquiz/{task}/r{r}/items.jsonl) and the
note is the markdown rendering of the admitted entries plus a code-generated
repo map. Study tokens are recorded and stay off the eval token axis.
Corpus-agnostic by construction: inputs are the repo, the three tools, and a
sandbox where the language runs (Axiom 0).
"""

import argparse
import json
import logging
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import dspy
import pydantic

from .dataset import CORPORA, ROOT
from .react import MODEL_ID, READ_MAX_LINES, SAMPLING, make_tools
from .sandbox import PYTHON_BIN
from .tools import RepoTools

K_CHAPTERS = 4
M_QUESTIONS = 5      # per chapter; 1 of these is held out to the dev exam
RETEST_FRAC = 0.2
QUIZ_MAX_ITERS = 15
DERIVE_MAX_ITERS = 15
NOTE_CAP_TOKENS = 4000  # soft cap; compaction is manual for now (design §DISTILL)
DEDUP_JACCARD = 0.5

log = logging.getLogger("selfquiz")


# ---------------------------------------------------------------- structures

class QuizQ(pydantic.BaseModel):
    question: str
    qtype: Literal["usage", "behavior", "location", "pitfall"]
    anchors: list[str]
    writer_sketch: str = ""


class Evidence(pydantic.BaseModel):
    file: str
    line: int
    quote: str


class QuizSig(dspy.Signature):
    """You are studying one module of a code repository to become an expert on
    the whole repository. Explore the module with your tools (read its code and
    its tests). Then write quiz questions that test whether someone who has NOT
    just read this code could use it correctly — usage ("write code that ..."),
    behavior ("what happens when ..."), location ("where/how is ... implemented"),
    or pitfall ("what breaks if ...") questions. Each question must be answerable
    from the repository alone, must NOT contain its own answer, and must cite the
    files that motivated it in `anchors`. In `writer_sketch` note in one line what
    you believe the answer is (this is not trusted; it is audit metadata)."""

    chapter: str = dspy.InputField(desc="the module (directory) to study")
    num_questions: int = dspy.InputField()
    questions: list[QuizQ] = dspy.OutputField()


class DeriveSig(dspy.Signature):
    """Answer this question about the code repository with certainty. Explore the
    repository with your tools; when the question concerns executable behavior and
    a run_python tool is available, write a short probe script and run it to
    confirm the behavior. Cite the decisive lines of source in `evidence` — file
    path, 1-indexed line number, and a short verbatim quote of that line."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="the correct answer, precise and complete")
    evidence: list[Evidence] = dspy.OutputField()


class AdjudicateSig(dspy.Signature):
    """Compare a student's attempt against a reference answer that was derived
    directly from the source code with cited evidence. Judge agreement on the
    substantive claims, not the wording. verdict: `correct` = the attempt makes
    the reference's substantive claims; `partial` = right direction, missing or
    muddling something essential; `wrong` = contradicts the reference or invents
    behavior; `unresolved` = the reference itself does not decisively settle the
    question. In `delta`, state precisely what the attempt got wrong or missed."""

    question: str = dspy.InputField()
    reference_answer: str = dspy.InputField()
    reference_evidence: str = dspy.InputField()
    attempt: str = dspy.InputField()
    verdict: Literal["correct", "partial", "wrong", "unresolved"] = dspy.OutputField()
    delta: str = dspy.OutputField()


class DistillSig(dspy.Signature):
    """Write one note entry correcting a mistaken belief. `belief` = the specific
    wrong belief revealed by the attempt, stated in second person ("you believe
    ..."). `correction` = the actual behavior per the reference answer, precise
    enough to act on. `quote`/`file`/`line` = one decisive verbatim source line
    from the reference evidence (copy it exactly; it will be checked against the
    file)."""

    question: str = dspy.InputField()
    attempt: str = dspy.InputField()
    reference_answer: str = dspy.InputField()
    reference_evidence: str = dspy.InputField()
    belief: str = dspy.OutputField()
    correction: str = dspy.OutputField()
    quote: str = dspy.OutputField()
    file: str = dspy.OutputField()
    line: int = dspy.OutputField()


# ---------------------------------------------------------------- corpus bits

def chapters(rt: RepoTools) -> list[str]:
    """The syllabus: first-level directories under the corpus roots, ordered by
    lines of code (descending). Test directories are evidence, not chapters."""
    loc = defaultdict(int)
    for f in rt.files:
        parts = f.split("/")
        if parts[0].lower() in ("tests", "test", "spec", "specs"):
            continue
        chap = "/".join(parts[:2]) if len(parts) > 2 else parts[0]
        loc[chap] += rt.text[f].count("\n") + 1
    return [c for c, _ in sorted(loc.items(), key=lambda kv: -kv[1])]


def _chapter_of(f: str) -> str:
    parts = f.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]


def repo_map(rt: RepoTools, chaps: list[str]) -> str:
    """Model-free orientation section: chapters with their largest files."""
    lines = ["## Repo map"]
    for c in chaps:
        fs = sorted((f for f in rt.files if _chapter_of(f) == c),
                    key=lambda f: -len(rt.text[f]))[:3]
        lines.append(f"- `{c}/`: " + ", ".join(f.rsplit('/', 1)[-1] for f in fs))
    return "\n".join(lines)


def make_run_python():
    def run_python(code: str) -> str:
        """Execute a short Python script against the pinned repository install and
        return its exit code and output (120s timeout). Use it to probe actual
        behavior: construct objects, call functions, print results."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "probe.py"
            p.write_text(code)
            try:
                proc = subprocess.run(
                    [str(PYTHON_BIN), "-I", str(p)], cwd=td,
                    env={"PATH": "/usr/bin:/bin", "HOME": td, "TMPDIR": td},
                    capture_output=True, text=True, timeout=120)
                out = (proc.stdout + "\n" + proc.stderr)[-3000:]
                return f"exit={proc.returncode}\n{out}"
            except subprocess.TimeoutExpired:
                return "exit=timeout(120s)"
    return run_python


# ---------------------------------------------------------------- gates

def quote_gate(rt: RepoTools, file: str, line: int, quote: str) -> bool:
    """Model-free integrity check: `quote` must appear at file:line (+/-2).
    The quote is normalized first (first non-empty line, markdown wrapping
    stripped) — models often quote real code across lines or inside backticks;
    normalization forgives formatting while still requiring the exact source
    text at the cited location."""
    text = rt.text.get(file.strip("/"))
    if text is None:
        return False
    q = next((ln.strip() for ln in quote.splitlines() if ln.strip()), "")
    q = q.strip("`>").strip()
    if len(q) < 6:
        return False
    lines = text.splitlines()
    lo, hi = max(0, line - 3), min(len(lines), line + 2)
    return any(q in ln for ln in lines[lo:hi])


def dedup(question: str, seen: list[str]) -> bool:
    toks = set(re.findall(r"[a-z0-9_]+", question.lower()))
    for s in seen:
        st = set(re.findall(r"[a-z0-9_]+", s.lower()))
        if toks and st and len(toks & st) / len(toks | st) > DEDUP_JACCARD:
            return True
    return False


# ---------------------------------------------------------------- LM plumbing

def fresh_lm(base_url: str) -> dspy.LM:
    return dspy.LM(MODEL_ID, api_base=base_url, api_key="EMPTY", model_type="chat",
                   cache=False, num_retries=3, **SAMPLING)


def spent(lm: dspy.LM) -> int:
    return sum((h.get("usage") or {}).get("completion_tokens") or 0 for h in lm.history)


# ---------------------------------------------------------------- pipeline

def run_quiz(chapter: str, tools_fns, url: str, n: int, seen: list[str]) -> list[dict]:
    lm = fresh_lm(url)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        try:
            pred = dspy.ReAct(QuizSig, tools=list(tools_fns), max_iters=QUIZ_MAX_ITERS)(
                chapter=chapter, num_questions=n)
            qs = pred.questions
        except Exception as e:
            log.warning("quiz episode failed for %s: %s", chapter, str(e)[:200])
            return []
    out = []
    for q in qs:
        rec = q.model_dump() | {"chapter": chapter, "quiz_tokens": 0}
        if dedup(q.question, seen):
            log.info("  DROP(dup) %s :: %.80s", chapter, q.question)
            continue
        seen.append(q.question)
        out.append(rec)
    if out:
        out[0]["quiz_tokens"] = spent(lm)  # episode cost, attributed once
    log.info("QUIZ %s: %d questions kept (%d tokens)", chapter, len(out), spent(lm))
    return out


def run_item(item: dict, note: str, tools_fns, run_py, url: str,
             ensemble: int, rt: RepoTools) -> dict:
    """ATTEMPT -> VERIFY(A blind derive xN, B adjudicate) -> DISTILL(+gate)."""
    rec = dict(item)
    lm = fresh_lm(url)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        try:
            rec["attempt"] = dspy.Predict("note, question -> answer")(
                note=note or "(no notes yet)", question=item["question"]).answer
        except Exception as e:
            rec.update(status="attempt_error", error=str(e)[:300], tokens=spent(lm))
            return rec

        derivations = []
        d_tools = list(tools_fns) + ([run_py] if run_py else [])
        for _ in range(ensemble):
            try:
                d = dspy.ReAct(DeriveSig, tools=d_tools, max_iters=DERIVE_MAX_ITERS)(
                    question=item["question"])
                derivations.append({"answer": d.answer,
                                    "evidence": [e.model_dump() for e in d.evidence]})
            except Exception as e:
                derivations.append({"answer": "", "evidence": [],
                                    "error": str(e)[:200]})
        rec["derivations"] = derivations
        ref = max(derivations, key=lambda d: len(d.get("evidence", [])))
        if not ref["answer"].strip():
            rec.update(status="derive_error", tokens=spent(lm))
            return rec

        try:
            adj = dspy.Predict(AdjudicateSig)(
                question=item["question"], reference_answer=ref["answer"],
                reference_evidence=json.dumps(ref["evidence"]),
                attempt=rec["attempt"])
            rec["verdict"], rec["delta"] = adj.verdict, adj.delta
        except Exception as e:
            rec.update(status="adjudicate_error", error=str(e)[:300], tokens=spent(lm))
            return rec

        # ensemble agreement rule (OpenClaw / dev items): a wrong/partial verdict
        # may only distill if a second derivation supports the same reference
        if ensemble > 1 and rec["verdict"] in ("wrong", "partial"):
            others = [d for d in derivations if d is not ref and d["answer"].strip()]
            agree = False
            for o in others:
                try:
                    a2 = dspy.Predict(AdjudicateSig)(
                        question=item["question"], reference_answer=o["answer"],
                        reference_evidence=json.dumps(o["evidence"]),
                        attempt=rec["attempt"])
                    agree |= a2.verdict in ("wrong", "partial")
                except Exception:
                    pass
            if not agree:
                rec["verdict"] = "unresolved"
                rec["delta"] += " [downgraded: derivations disagree]"

        rec["entry"] = None
        if rec["verdict"] in ("wrong", "partial") and not item.get("dev"):
            try:
                d = dspy.Predict(DistillSig)(
                    question=item["question"], attempt=rec["attempt"],
                    reference_answer=ref["answer"],
                    reference_evidence=json.dumps(ref["evidence"]))
                entry = {"belief": d.belief, "correction": d.correction,
                         "quote": d.quote, "file": d.file, "line": int(d.line),
                         "chapter": item["chapter"]}
                if quote_gate(rt, entry["file"], entry["line"], entry["quote"]):
                    rec["entry"] = entry
                else:
                    rec["entry_bounced"] = entry
            except Exception as e:
                rec.update(distill_error=str(e)[:300])
    rec["status"] = "ok"
    rec["tokens"] = spent(lm)
    return rec


def render_note(rt: RepoTools, chaps: list[str], entries: list[dict],
                display: str) -> str:
    parts = [f"# {display} — corrections from studying (your beliefs vs. this repository)",
             "", repo_map(rt, chaps[:20]), ""]  # map capped: openclaw has 169 chapters
    by_ch = defaultdict(list)
    for e in entries:
        by_ch[e["chapter"]].append(e)
    for ch in chaps:
        if not by_ch.get(ch):
            continue
        parts.append(f"## {ch}")
        for e in by_ch[ch]:
            parts.append(f"- **{e['belief'].strip()}** {e['correction'].strip()}\n"
                         f"  > `{e['file']}:{e['line']}`: `{e['quote'].strip()}`")
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------- round driver

def run_round(args):
    corpus = CORPORA[args.task]
    rt = RepoTools(corpus, read_max_lines=READ_MAX_LINES)
    tools_fns = make_tools(rt)
    run_py = make_run_python() if corpus.language == "python" else None
    ensemble = 1 if corpus.language == "python" else 2  # execution grounds python
    urls = args.base_urls.split(",")
    sdir = ROOT / "study-selfquiz" / args.task
    rdir = sdir / f"r{args.round}"
    rdir.mkdir(parents=True, exist_ok=True)

    chaps = chapters(rt)
    k = min(args.chapters, len(chaps))
    start = (args.round - 1) * k
    todo = [chaps[(start + i) % len(chaps)] for i in range(k)]
    log.info("ROUND %d %s: chapters=%s (syllabus has %d)", args.round, args.task, todo, len(chaps))

    # prior state: note entries + all previous questions (dedup + retest pool)
    entries, prev_items, seen_q = [], [], []
    for r in range(1, args.round):
        pf = sdir / f"r{r}" / "items.jsonl"
        if pf.exists():
            for line in pf.read_text().splitlines():
                it = json.loads(line)
                prev_items.append(it)
                seen_q.append(it["question"])
                if it.get("entry"):
                    entries.append(it["entry"])
    note = render_note(rt, chaps, entries, corpus.display) if entries else ""

    # QUIZ (parallel across chapters), resumable via questions.jsonl
    qfile = rdir / "questions.jsonl"
    if qfile.exists():
        items = [json.loads(l) for l in qfile.read_text().splitlines()]
        log.info("resuming: %d questions from %s", len(items), qfile)
    else:
        n = args.questions if not args.smoke else 3
        with ThreadPoolExecutor(max_workers=k) as pool:
            batches = list(pool.map(
                lambda i: run_quiz(todo[i], tools_fns, urls[i % len(urls)], n, seen_q),
                range(len(todo) if not args.smoke else 1)))
        items = [q for b in batches for q in b]
        for ch_items in batches:  # 1 dev holdout per chapter
            if len(ch_items) > 1:
                ch_items[-1]["dev"] = True
        # RETEST slots from round 2 on
        if args.round > 1 and prev_items:
            import random
            rng = random.Random(args.round)
            n_retest = max(1, int(len(items) * RETEST_FRAC))
            for it in rng.sample(prev_items, min(n_retest, len(prev_items))):
                items.append({k2: it[k2] for k2 in ("question", "qtype", "anchors", "chapter")}
                             | {"retest": True})
        qfile.write_text("\n".join(json.dumps(i) for i in items))
    log.info("round has %d items (%d dev, %d retest)", len(items),
             sum(1 for i in items if i.get("dev")), sum(1 for i in items if i.get("retest")))

    # ATTEMPT/VERIFY/DISTILL (parallel across items), resumable via items.jsonl
    ifile = rdir / "items.jsonl"
    done_q = set()
    if ifile.exists():
        done_q = {json.loads(l)["question"] for l in ifile.read_text().splitlines()}
    pending = [i for i in items if i["question"] not in done_q]
    log.info("%d items pending", len(pending))

    lock = __import__("threading").Lock()

    def one(idx, item):
        rec = run_item(item, note, tools_fns, run_py, urls[idx % len(urls)], ensemble, rt)
        with lock:
            with open(ifile, "a") as f:
                f.write(json.dumps(rec) + "\n")
        log.info("[%s%s] %s :: verdict=%s entry=%s tokens=%d",
                 "dev " if item.get("dev") else "", "retest" if item.get("retest") else "item",
                 item["question"][:70], rec.get("verdict", rec["status"]),
                 "ADMITTED" if rec.get("entry") else
                 ("BOUNCED" if rec.get("entry_bounced") else "-"), rec.get("tokens", 0))
        if args.debug:
            log.debug("attempt: %.500s\nderived: %.500s\ndelta: %.300s",
                      rec.get("attempt"), (rec.get("derivations") or [{}])[0].get("answer"),
                      rec.get("delta"))

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(lambda t: one(*t), list(enumerate(pending))))

    if not ifile.exists():
        raise SystemExit("no items were produced — every quiz episode failed; see log")

    # round summary + note snapshot
    recs = [json.loads(l) for l in ifile.read_text().splitlines()]
    train = [r for r in recs if not r.get("dev") and not r.get("retest")]
    dev = [r for r in recs if r.get("dev")]
    new_entries = [r["entry"] for r in recs if r.get("entry")]
    bounced = sum(1 for r in recs if r.get("entry_bounced"))
    verdicts = defaultdict(int)
    for r in recs:
        verdicts[r.get("verdict", r["status"])] += 1
    all_entries = entries + new_entries
    note_text = render_note(rt, chaps, all_entries, corpus.display)
    (sdir / f"note-r{args.round}.md").write_text(note_text)
    summary = {
        "round": args.round, "task": args.task, "chapters": todo,
        "items": len(recs), "verdicts": dict(verdicts),
        "train_error_rate": (sum(1 for r in train if r.get("verdict") in ("wrong", "partial"))
                             / max(1, len(train))),
        "dev_error_rate": (sum(1 for r in dev if r.get("verdict") in ("wrong", "partial"))
                           / max(1, len(dev))),
        "entries_admitted": len(new_entries), "entries_bounced": bounced,
        "note_entries_total": len(all_entries), "note_chars": len(note_text),
        "study_tokens": sum(r.get("tokens", 0) for r in recs)
                        + sum(i.get("quiz_tokens", 0) for i in items),
    }
    (rdir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("ROUND %d SUMMARY %s", args.round, json.dumps(summary))


class CompactSig(dspy.Signature):
    """Rewrite these note entries about one module to be tighter: merge entries
    that correct the same or overlapping beliefs, drop redundancy, keep every
    distinct correction. Each output entry must keep `quote`, `file`, and `line`
    copied EXACTLY from one of the input entries (they are verified against the
    source); only the belief/correction wording may change."""

    chapter: str = dspy.InputField()
    entries_json: str = dspy.InputField()
    compacted: list[dict] = dspy.OutputField(
        desc="entries with keys belief, correction, quote, file, line")


def compact(args):
    """Cap-triggered note compaction with a regression guard (design §DISTILL).
    Compacts per chapter; re-gates every quote; then re-ATTEMPTs a sample of
    entry-backed questions with old vs new note — keeps the compacted note only
    if closed-book performance does not regress."""
    corpus = CORPORA[args.task]
    rt = RepoTools(corpus, read_max_lines=READ_MAX_LINES)
    chaps = chapters(rt)
    sdir = ROOT / "study-selfquiz" / args.task
    recs = []
    for r in range(1, args.round + 1):
        p = sdir / f"r{r}" / "items.jsonl"
        if p.exists():
            recs += [json.loads(l) for l in p.read_text().splitlines()]
    entries = [x["entry"] for x in recs if x.get("entry")]
    old_note = render_note(rt, chaps, entries, corpus.display)
    lm = fresh_lm(args.base_urls.split(",")[0])

    by_ch = defaultdict(list)
    for e in entries:
        by_ch[e["chapter"]].append(e)
    new_entries, dropped = [], 0
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        for ch, es in by_ch.items():
            if len(es) < 2:
                new_entries += es
                continue
            try:
                out = dspy.Predict(CompactSig)(
                    chapter=ch, entries_json=json.dumps(es)).compacted
            except Exception as e2:
                log.warning("compaction failed for %s (%s); keeping originals", ch, e2)
                new_entries += es
                continue
            kept = [dict(o, chapter=ch) for o in out
                    if all(k in o for k in ("belief", "correction", "quote", "file", "line"))
                    and quote_gate(rt, o["file"], int(o["line"]), o["quote"])]
            if kept:
                new_entries += kept
                dropped += len(es) - len(kept)
            else:  # everything failed the re-gate: refuse this chapter's compaction
                new_entries += es
    new_note = render_note(rt, chaps, new_entries, corpus.display)
    log.info("compaction: %d -> %d entries, %d -> %d chars",
             len(entries), len(new_entries), len(old_note), len(new_note))

    # regression guard: closed-book re-attempts on entry-backed questions
    backed = [x for x in recs if x.get("entry")][: args.guard_n]
    def score(note):
        ok = 0
        with dspy.context(lm=fresh_lm(args.base_urls.split(",")[0]),
                          adapter=dspy.ChatAdapter()):
            for x in backed:
                try:
                    a = dspy.Predict("note, question -> answer")(
                        note=note, question=x["question"]).answer
                    ref = max(x["derivations"], key=lambda d: len(d.get("evidence", [])))
                    v = dspy.Predict(AdjudicateSig)(
                        question=x["question"], reference_answer=ref["answer"],
                        reference_evidence=json.dumps(ref["evidence"]), attempt=a).verdict
                    ok += v in ("correct", "partial")
                except Exception:
                    pass
        return ok
    old_ok, new_ok = score(old_note), score(new_note)
    log.info("regression guard on %d entry-backed questions: old=%d new=%d",
             len(backed), old_ok, new_ok)
    if new_ok >= old_ok:
        (sdir / f"note-r{args.round}.md").write_text(new_note)
        log.info("compacted note ACCEPTED -> note-r%d.md", args.round)
    else:
        (sdir / f"note-r{args.round}.md").write_text(old_note)
        log.info("compacted note REJECTED (regressed); kept full note")


def select_note(args):
    """Iteration 2 (artifact-cited: the r1-r4 milestone curve peaks at small
    notes and declines as entries accumulate): build a hard-capped note by
    deterministic selection over the full entry pool. Scoring favors entries
    from wrong (vs partial) verdicts and executed evidence, then round-robins
    chapters for diversity. No new model calls."""
    corpus = CORPORA[args.task]
    rt = RepoTools(corpus, read_max_lines=READ_MAX_LINES)
    chaps = chapters(rt)
    sdir = ROOT / "study-selfquiz" / args.task
    pool = []
    for r in range(1, args.round + 1):
        p = sdir / f"r{r}" / "items.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                x = json.loads(line)
                if x.get("entry"):
                    e = dict(x["entry"])
                    e["_score"] = (2 if x.get("verdict") == "wrong" else 1)
                    pool.append(e)
    by_ch = defaultdict(list)
    for e in sorted(pool, key=lambda e: -e["_score"]):
        by_ch[e["chapter"]].append(e)
    picked, i = [], 0
    order = [c for c in chaps if by_ch.get(c)]
    while len(picked) < args.select and any(by_ch.values()):
        c = order[i % len(order)]
        if by_ch[c]:
            picked.append(by_ch[c].pop(0))
        i += 1
    for e in picked:
        e.pop("_score", None)
    note = render_note(rt, chaps, picked, corpus.display)
    (sdir / "note-select.md").write_text(note)
    log.info("select-note: %d/%d entries kept, %d chars -> note-select.md",
             len(picked), len(pool), len(note))


class SnippetSig(dspy.Signature):
    """Write one minimal, self-contained code snippet (<=12 lines) that
    demonstrates the CORRECT usage described by this note entry — the snippet a
    developer would copy to avoid the mistaken belief. For Python it must run
    as-is against the installed library (prints something small to prove it
    ran); prefer offline constructs (e.g. dummy models) over network calls."""

    belief: str = dspy.InputField()
    correction: str = dspy.InputField()
    evidence_quote: str = dspy.InputField()
    language: str = dspy.InputField()
    snippet: str = dspy.OutputField()


def usage_note(args):
    """Iteration 3 (artifact-cited: the direct-column gap vs the cheatsheet is
    the entire remaining deficit, and the cheatsheet's direct advantage comes
    from code-shaped content): attach an execution-gated usage snippet to each
    selected entry. DSPy snippets must exit 0 in the pinned sandbox; OpenClaw
    snippets must tree-sitter-parse. Entries whose snippets fail keep prose."""
    from . import sandbox as sb
    corpus = CORPORA[args.task]
    rt = RepoTools(corpus, read_max_lines=READ_MAX_LINES)
    chaps = chapters(rt)
    sdir = ROOT / "study-selfquiz" / args.task
    # entries = the select-12 note's entries (iteration 2 pool)
    pool = []
    for r in range(1, args.round + 1):
        p = sdir / f"r{r}" / "items.jsonl"
        if p.exists():
            pool += [json.loads(l)["entry"] for l in p.read_text().splitlines()
                     if json.loads(l).get("entry")]
    select_txt = (sdir / "note-select.md").read_text()
    entries = [e for e in pool if e["belief"].strip()[:60] in select_txt]
    log.info("usage-note: %d selected entries to snippet", len(entries))
    lm = fresh_lm(args.base_urls.split(",")[0])
    ok = fail = 0
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        for e in entries:
            try:
                snip = dspy.Predict(SnippetSig)(
                    belief=e["belief"], correction=e["correction"],
                    evidence_quote=f"{e['file']}:{e['line']}: {e['quote']}",
                    language=corpus.language).snippet
                snip = snip.strip().strip("`")
                if snip.startswith(("python", "typescript", "ts")):
                    snip = snip.split("\n", 1)[1] if "\n" in snip else ""
                gate = (sb._check_python(snip) if corpus.language == "python"
                        else sb._check_typescript(snip))
                if snip and gate["compile_ok"]:
                    e["snippet"] = snip
                    ok += 1
                else:
                    fail += 1
            except Exception as ex:
                fail += 1
                log.warning("snippet failed: %s", str(ex)[:150])
    log.info("usage-note: %d snippets execution-gated in, %d failed (prose kept)", ok, fail)

    parts = [f"# {corpus.display} — corrections from studying (verified usage)",
             "", repo_map(rt, chaps[:20]), ""]
    by_ch = defaultdict(list)
    for e in entries:
        by_ch[e["chapter"]].append(e)
    for ch in chaps:
        if not by_ch.get(ch):
            continue
        parts.append(f"## {ch}")
        for e in by_ch[ch]:
            parts.append(f"- **{e['belief'].strip()}** {e['correction'].strip()}\n"
                         f"  > `{e['file']}:{e['line']}`: `{e['quote'].strip()}`")
            if e.get("snippet"):
                lang = "python" if corpus.language == "python" else "ts"
                parts.append(f"```{lang}\n{e['snippet']}\n```")
        parts.append("")
    note = "\n".join(parts)
    (sdir / "note-usage.md").write_text(note)
    log.info("wrote note-usage.md (%d chars)", len(note))


class ChapterSummarySig(dspy.Signature):
    """Write a compact reference summary (250-400 words) of this module for a
    developer who will answer questions about the library without reading it.
    Use ONLY the supplied evidence (derived answers and cited source lines from
    prior study of this module) — do not add API details the evidence does not
    support. Prefer concrete signatures, argument names, defaults, and behaviors
    over prose."""

    chapter: str = dspy.InputField()
    evidence: str = dspy.InputField(desc="accumulated grounded findings for this module")
    summary: str = dspy.OutputField()


def studied_note(args):
    """Iteration 5 (declared 2026-07-06): breadth from the study loop's own
    accumulated grounded reading — per-chapter summaries generated from all
    Phase-A derivations/evidence across rounds, plus the select corrections."""
    corpus = CORPORA[args.task]
    rt = RepoTools(corpus, read_max_lines=READ_MAX_LINES)
    chaps = chapters(rt)
    sdir = ROOT / "study-selfquiz" / args.task
    by_ch = defaultdict(list)
    for r in range(1, args.round + 1):
        p = sdir / f"r{r}" / "items.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            x = json.loads(line)
            d = max(x.get("derivations") or [{}],
                    key=lambda dd: len(dd.get("evidence", [])))
            if d.get("answer"):
                by_ch[x["chapter"]].append(
                    {"q": x["question"], "finding": d["answer"][:800],
                     "evidence": d.get("evidence", [])[:3]})
    lm = fresh_lm(args.base_urls.split(",")[0])
    summaries = {}
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        for ch, recs in by_ch.items():
            try:
                summaries[ch] = dspy.Predict(ChapterSummarySig)(
                    chapter=ch, evidence=json.dumps(recs)).summary
            except Exception as e:
                log.warning("summary failed for %s: %s", ch, str(e)[:150])
    log.info("studied summaries: %d chapters", len(summaries))

    sel = (sdir / "note-select.md").read_text()
    corrections = sel.split("\n## ", 1)[1] if "\n## " in sel else ""
    parts = [f"# {corpus.display} — studied reference (grounded in prior study)",
             "", repo_map(rt, chaps[:20]), ""]
    for ch in chaps:
        if ch in summaries:
            parts += [f"## {ch}", summaries[ch].strip(), ""]
    parts += ["---", "", "# Verified corrections (trust these over the summaries)",
              "", "## " + corrections]
    note = "\n".join(parts)
    (sdir / "note-studied.md").write_text(note)
    log.info("wrote note-studied.md (%d chars)", len(note))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=list(CORPORA))
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--chapters", type=int, default=K_CHAPTERS)
    p.add_argument("--questions", type=int, default=M_QUESTIONS)
    p.add_argument("--base-urls", default="http://localhost:8100/v1")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--smoke", action="store_true", help="1 chapter, 3 questions")
    p.add_argument("--compact", action="store_true",
                   help="compact the cumulative note through --round (cap-triggered; "
                        "regression-guarded) instead of running a study round")
    p.add_argument("--guard-n", type=int, default=12)
    p.add_argument("--select", type=int, default=0,
                   help="build a hard-capped note of N selected entries (no LM calls)")
    p.add_argument("--usage", action="store_true",
                   help="attach execution-gated usage snippets to the select note")
    p.add_argument("--studied", action="store_true",
                   help="build the studied-summary note (iteration 5)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(ROOT / "logs" / f"selfquiz-{args.task}.log")])
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.studied:
        studied_note(args)
    elif args.usage:
        usage_note(args)
    elif args.select:
        select_note(args)
    elif args.compact:
        compact(args)
    else:
        run_round(args)


if __name__ == "__main__":
    main()
