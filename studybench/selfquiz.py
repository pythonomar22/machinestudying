"""SELF-QUIZZING study procedure (experiments/005 v1.2): the agent studies a
codebase by quizzing itself; independently screened, source-cited candidate
corrections enter its note and remain non-claim-ready until blinded human audit.

Per round r (chapters advance through a fixed size-ordered syllabus, wrapping):
  QUIZ     one ReAct episode per chapter writes M questions (anchored, deduped;
           1 of M held out to the accumulating dev exam)
  ATTEMPT  closed book (dspy.Predict): note_{r-1} + question -> committed answer
  VERIFY   Phase A derives the answer blind (never sees the attempt or note),
           retaining exact source-line citations;
           Phase B diffs attempt vs an independently agreed reference
  DISTILL  wrong/partial only -> {belief, correction, quote, file, line};
           a model-free gate exactly matches the source line before the
           entry is admitted to the note
  RETEST   (r>=2) ~20% of slots re-run previous items against the current note

Everything is logged per item under
study-selfquiz/studies/{study_id}/{task}/r{r}/, and the
note is the markdown rendering of the admitted entries plus a code-generated
repo map. Every provider-reported prompt, generated, and total token count is
recorded by phase and stays off the eval token axis.
Corpus-agnostic by construction: inputs are the repository and its read tools.
This protocol does not expose a generated-code execution tool.
"""

import argparse
from contextlib import contextmanager
import json
import logging
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import dspy
import pydantic

from .dataset import CORPORA, ROOT
from .integrity import (
    canonical_json_bytes,
    exclusive_process_lock,
    load_json_artifact,
    read_artifact_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_text,
    stable_seed,
    strict_json_loads,
    write_immutable_json,
    write_immutable_text,
)
from .human_audit import (
    HUMAN_AUDIT_SCHEMA_VERSION,
    HumanAuditError,
    validate_human_audit_protocol,
    validate_human_audit_result,
)
from .provenance import (
    corpus_record,
    environment_contract_is_valid,
    environment_contract_record,
    environment_is_claim_ready,
    environment_record,
    environments_compatible,
    source_record,
    validate_id,
    validate_environment_snapshot,
    validate_local_server_urls,
    write_environment_snapshot,
)
from .react import MODEL_ID, MODEL_REVISION, READ_MAX_LINES, SAMPLING, make_tools
from .tools import RepoTools

K_CHAPTERS = 4
M_QUESTIONS = 5      # per chapter; 1 of these is held out to the dev exam
RETEST_FRAC = 0.2
QUIZ_MAX_ITERS = 15
DERIVE_MAX_ITERS = 15
DEDUP_JACCARD = 0.5
FRESHNESS_NEAR_JACCARD = 0.8
MAX_FRESHNESS_NEAR_RATE = 0.1
SCHEMA_VERSION = 2
TRAIN_ENSEMBLE = 2
DEV_ENSEMBLE = 2

log = logging.getLogger("selfquiz")


# ---------------------------------------------------------------- structures

class QuizQ(pydantic.BaseModel):
    question: str
    qtype: Literal["usage", "behavior", "location", "pitfall"]
    anchors: list[str]
    writer_sketch: str = ""


class Evidence(pydantic.BaseModel):
    file: str
    line: pydantic.StrictInt
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
    repository with the read-only repository tools. Cite the decisive lines of
    source in `evidence` — file path, 1-indexed line number, and a short verbatim
    quote of that line."""

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
    line: pydantic.StrictInt = dspy.OutputField()


class ReferenceSupportSig(dspy.Signature):
    """Check that the cited source lines substantively support the derived
    answer to the question. A citation that merely exists but is irrelevant is
    unsupported."""

    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    evidence: str = dspy.InputField()
    supported: bool = dspy.OutputField()
    rationale: str = dspy.OutputField()


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


# ---------------------------------------------------------------- gates

def _record_id(prefix: str, *parts: object) -> str:
    return f"{prefix}-{sha256_json(parts)[:20]}"


def _safe_artifact_id(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[a-z][a-z0-9-]{2,79}", value))


def _canonical_file(path: object) -> str | None:
    """Return a strict corpus-relative POSIX path or ``None``.

    Strictness is deliberate: provenance should never silently repair an
    absolute path, traversal, backslash, or surrounding whitespace.
    """
    if not isinstance(path, str) or not path or path != path.strip():
        return None
    if path.startswith("/") or "\\" in path:
        return None
    parsed = PurePosixPath(path)
    if any(part in ("", ".", "..") for part in parsed.parts):
        return None
    canonical = str(parsed)
    return canonical if canonical == path else None


def validate_anchor(rt: RepoTools, path: object) -> str | None:
    """Validate one quiz anchor as an exact readable corpus file."""
    canonical = _canonical_file(path)
    return canonical if canonical is not None and canonical in rt.text else None


def validate_anchors(rt: RepoTools, anchors: object) -> list[str] | None:
    """Validate all anchors; one malformed anchor rejects the question."""
    if not isinstance(anchors, list) or not anchors:
        return None
    valid = [validate_anchor(rt, anchor) for anchor in anchors]
    if any(anchor is None for anchor in valid):
        return None
    return list(dict.fromkeys(anchor for anchor in valid if anchor is not None))


def _quoted_line(quote: object) -> str | None:
    if not isinstance(quote, str):
        return None
    lines = quote.strip().splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        lines = lines[1:-1]
    nonempty = [line.strip() for line in lines if line.strip()]
    if len(nonempty) != 1:
        return None
    line = nonempty[0]
    if line.startswith(">"):
        line = line[1:].lstrip()
    if len(line) >= 2 and line.startswith("`") and line.endswith("`"):
        line = line[1:-1].strip()
    return line or None


def validate_evidence(rt: RepoTools, evidence: object, tolerance: int = 2) -> dict | None:
    """Return a canonical citation only for an exact source-line quote.

    The requested line may be off by at most ``tolerance``. The returned line
    is the actual match, so rendered notes never preserve an inaccurate locator.
    """
    if isinstance(evidence, pydantic.BaseModel):
        evidence = evidence.model_dump()
    if not isinstance(evidence, dict):
        return None
    file = validate_anchor(rt, evidence.get("file"))
    line = evidence.get("line")
    quote = _quoted_line(evidence.get("quote"))
    if file is None or isinstance(line, bool) or not isinstance(line, int) or quote is None:
        return None
    lines = rt.text[file].splitlines()
    if line < 1 or line > len(lines):
        return None
    candidates = range(max(1, line - tolerance), min(len(lines), line + tolerance) + 1)
    matches = [actual for actual in candidates if lines[actual - 1].strip() == quote]
    if not matches:
        return None
    actual = min(matches, key=lambda value: (abs(value - line), value))
    return {"file": file, "line": actual, "quote": lines[actual - 1].strip()}


def quote_gate(rt: RepoTools, file: str, line: int, quote: str) -> bool:
    """Compatibility wrapper around the strict canonical evidence gate."""
    return validate_evidence(rt, {"file": file, "line": line, "quote": quote}) is not None


def dedup(question: str, seen: list[str]) -> bool:
    toks = set(re.findall(r"[a-z0-9_]+", question.lower()))
    for s in seen:
        st = set(re.findall(r"[a-z0-9_]+", s.lower()))
        if toks and st and len(toks & st) / len(toks | st) > DEDUP_JACCARD:
            return True
    return False


def _question_tokens(question: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9_]+", question.casefold()))


def freshness_audit(records: list[dict], *, task: str, study_dir: Path,
                    root: Path = ROOT, snapshot_dir: Path | None = None) -> dict:
    """Compare this round's new questions with every other stored curriculum.

    Exact normalized reuse is never fresh. Near reuse is pre-registered as a
    token-set Jaccard score of at least ``FRESHNESS_NEAR_JACCARD`` and may cover
    at most ``MAX_FRESHNESS_NEAR_RATE`` of the round.
    """
    current = [record for record in records if record.get("kind") == "quiz"]
    current_rounds = {
        record.get("round") for record in current
        if isinstance(record.get("round"), int) and not isinstance(record.get("round"), bool)
    }
    paths: set[Path] = set()
    discovery_errors = []
    for study_root in root.glob("study-selfquiz*"):
        if not study_root.is_dir():
            continue
        if study_root.is_symlink():
            discovery_errors.append({
                "path": str(study_root),
                "error": "freshness study root must not be a symlink",
            })
            continue
        paths.update(study_root.glob(f"{task}/r*/questions.jsonl"))
        paths.update(study_root.glob(f"studies/*/{task}/r*/questions.jsonl"))
    current_path = None
    round_identity_complete = len(current_rounds) == 1 and all(
        record.get("round") in current_rounds for record in current)
    if round_identity_complete:
        current_round = next(iter(current_rounds))
        current_path = (study_dir / f"r{current_round}" / "questions.jsonl").resolve()
    else:
        discovery_errors.append({
            "path": str(study_dir),
            "error": "current quiz records do not identify exactly one round",
        })
    comparison, sources, errors = [], [], list(discovery_errors)
    for path in sorted(paths):
        try:
            relative_from_root = PurePosixPath(path.relative_to(root).as_posix())
            if _has_symlink_component(root, relative_from_root):
                raise ValueError("freshness input path traverses a symlink")
            resolved = path.resolve()
            if current_path is not None and resolved == current_path:
                continue
            raw = read_artifact_bytes(path).decode("utf-8")
            source_questions = []
            for line_number, line in enumerate(raw.splitlines(), 1):
                if not line.strip():
                    continue
                value = strict_json_loads(
                    line, label=f"freshness source {path}:{line_number}"
                )
                question = value.get("question") if isinstance(value, dict) else None
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f"line {line_number} has no question")
                source_questions.append(question)
                comparison.append((str(path.relative_to(root)), question))
            digest = sha256_text(raw)
            source_record = {
                "path": str(path.relative_to(root)),
                "sha256": digest,
                "questions": len(source_questions),
            }
            if snapshot_dir is not None:
                snapshot = Path("freshness-sources") / f"{digest}.jsonl"
                write_immutable_text(snapshot_dir / f"{digest}.jsonl", raw)
                source_record["snapshot"] = str(snapshot)
            sources.append(source_record)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
            errors.append({"path": str(path), "error": str(error)})

    matches = []
    for record in current:
        tokens = _question_tokens(record["question"])
        token_set = set(tokens)
        normalized = " ".join(tokens)
        best = None
        for path, prior_question in comparison:
            prior_tokens = _question_tokens(prior_question)
            prior_set = set(prior_tokens)
            union = token_set | prior_set
            score = len(token_set & prior_set) / len(union) if union else 1.0
            exact = normalized == " ".join(prior_tokens)
            candidate = (exact, score, path, prior_question)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is not None and (best[0] or best[1] >= FRESHNESS_NEAR_JACCARD):
            matches.append({
                "item_id": record["item_id"],
                "question": record["question"],
                "exact": best[0],
                "jaccard": best[1],
                "prior_path": best[2],
                "prior_question": best[3],
            })
    exact_count = sum(match["exact"] for match in matches)
    near_rate = len(matches) / len(current) if current else None
    complete = bool(current) and round_identity_complete and not errors
    fresh = bool(
        complete and exact_count == 0 and near_rate is not None
        and near_rate <= MAX_FRESHNESS_NEAR_RATE
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "normalized exact plus token-set Jaccard",
        "near_jaccard_threshold": FRESHNESS_NEAR_JACCARD,
        "max_near_overlap_rate": MAX_FRESHNESS_NEAR_RATE,
        "current_questions": len(current),
        "comparison_questions": len(comparison),
        "comparison_sources": sources,
        "comparison_bundle_sha256": sha256_json(sources),
        "errors": errors,
        "exact_overlaps": exact_count,
        "near_overlaps_including_exact": len(matches),
        "near_overlap_rate": near_rate,
        "matches": matches,
        "audit_complete": complete,
        "fresh": fresh,
    }


def freshness_sources_complete(rdir: Path, freshness: dict) -> bool:
    """Validate the snapshotted external curricula used by a freshness decision."""
    sources = freshness.get("comparison_sources")
    if not isinstance(sources, list):
        return False
    if freshness.get("comparison_bundle_sha256") != sha256_json(sources):
        return False
    for source in sources:
        if not isinstance(source, dict):
            return False
        question_count = source.get("questions")
        if (not isinstance(question_count, int) or isinstance(question_count, bool)
                or question_count < 0):
            return False
        relative = _canonical_file(source.get("snapshot"))
        if relative is None or not relative.startswith("freshness-sources/"):
            return False
        path = rdir.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file() \
                or sha256_file(path) != source.get("sha256"):
            return False
        try:
            values = _read_jsonl(path)
        except (OSError, UnicodeError, ValueError):
            return False
        if (question_count != len(values)
                or any(not isinstance(value, dict)
                       or not isinstance(value.get("question"), str)
                       or not value["question"].strip() for value in values)):
            return False
    return sum(source.get("questions", 0) for source in sources) \
        == freshness.get("comparison_questions")


def _positive_json_integer(value: object) -> bool:
    """Return true only for a positive JSON integer (never for a boolean)."""

    return type(value) is int and value >= 1


def _is_original_quiz(item: object, *, split: str) -> bool:
    """Validate the immutable identity shared by original train/dev quizzes."""

    if not isinstance(item, dict):
        return False
    required = (
        "schema_version", "item_id", "origin_item_id", "origin_round",
        "round", "kind", "split",
    )
    if any(key not in item for key in required):
        return False
    return (
        type(item["schema_version"]) is int
        and item["schema_version"] == SCHEMA_VERSION
        and isinstance(item["item_id"], str)
        and bool(item["item_id"])
        and isinstance(item["origin_item_id"], str)
        and bool(item["origin_item_id"])
        and _positive_json_integer(item["round"])
        and _positive_json_integer(item["origin_round"])
        and item["kind"] == "quiz"
        and item["split"] == split
        and item["item_id"] == item["origin_item_id"]
        and item["round"] == item["origin_round"]
        and item.get("retest_of") is None
    )


def is_distillable_item(item: dict) -> bool:
    """Only an original training quiz may ever update the note."""

    return _is_original_quiz(item, split="train")


def eligible_retest(item: dict) -> bool:
    """A retest samples only resolved original train items worth measuring."""
    return (
        is_distillable_item(item)
        and item.get("status") == "ok"
        and (item.get("entry") is not None or item.get("verdict") == "correct")
    )


def collect_note_entries(items: list[dict]) -> list[dict]:
    """Collect admitted entries without permitting dev/retest ancestry."""
    entries: dict[str, dict] = {}
    for item in items:
        if not is_distillable_item(item) or not item.get("entry"):
            continue
        entry = dict(item["entry"])
        entry_id = entry["entry_id"]
        if entry_id in entries and entries[entry_id] != entry:
            raise ValueError(f"conflicting records for entry {entry_id}")
        entries[entry_id] = entry
    return [entries[entry_id] for entry_id in sorted(entries)]


def collect_dev_questions(question_sets: list[list[dict]]) -> list[dict]:
    """Build the cumulative, unique dev pool from immutable question records."""
    dev: dict[str, dict] = {}
    for questions in question_sets:
        for item in questions:
            if _is_original_quiz(item, split="dev"):
                item_id = item["item_id"]
                candidate = dict(item)
                if item_id in dev and dev[item_id] != candidate:
                    raise ValueError(f"conflicting records for dev question {item_id}")
                dev[item_id] = candidate
    return [dev[item_id] for item_id in sorted(dev)]


def make_retest_item(origin: dict, *, task: str, study_id: str, round_number: int) -> dict:
    if not eligible_retest(origin):
        raise ValueError("retests require an eligible original training item")
    if (not _positive_json_integer(round_number)
            or round_number <= origin["origin_round"]):
        raise ValueError("retest round must be a later positive integer round")
    item_id = _record_id("retest", study_id, task, round_number, origin["item_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "item_id": item_id,
        "origin_item_id": origin["item_id"],
        "origin_round": origin["origin_round"],
        "round": round_number,
        "kind": "retest",
        "split": "train",
        "retest_of": origin["item_id"],
        **{key: origin[key] for key in
           ("question", "qtype", "anchors", "chapter", "writer_sketch")},
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pydantic.BaseModel):
        return _jsonable(value.model_dump())
    return repr(value)


def serialize_trajectory(trajectory: object) -> dict:
    return _jsonable(trajectory) if isinstance(trajectory, dict) else {}


def _trajectory_hash_valid(record: object) -> bool:
    return (
        isinstance(record, dict)
        and isinstance(record.get("trajectory"), dict)
        and record.get("trajectory_sha256") == sha256_json(record["trajectory"])
    )


# ---------------------------------------------------------------- LM plumbing

def fresh_lm(base_url: str, seed: int) -> dspy.LM:
    api_key = os.environ.get("SB_VLLM_API_KEY")
    if not api_key:
        raise RuntimeError("authenticated local server key is unavailable")
    return dspy.LM(MODEL_ID, api_base=base_url, api_key=api_key, model_type="chat",
                   cache=False, num_retries=0, **{**SAMPLING, "seed": seed})


def _server_url(urls: list[str], master_seed: int, owner_id: str) -> str:
    """Choose a server deterministically so partial resumption cannot reassign work."""
    if not urls:
        raise ValueError("at least one model server is required")
    return urls[stable_seed(master_seed, owner_id, "server") % len(urls)]


def _response_value(response: object, field: str) -> object:
    return response.get(field) if isinstance(response, dict) else getattr(response, field, None)


def usage_records(lm: dspy.LM, *, phase: str, owner_id: str, seed: int) -> list[dict]:
    records = []
    for index, history in enumerate(lm.history):
        raw = _jsonable(history.get("usage") or {})
        raw = raw if isinstance(raw, dict) else {}
        response = history.get("response")
        messages = _jsonable(history.get("messages"))
        outputs = _jsonable(history.get("outputs"))
        prompt_raw = raw.get("prompt_tokens")
        completion_raw = raw.get("completion_tokens")
        usage_reported = all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (prompt_raw, completion_raw)
        )
        prompt = prompt_raw if usage_reported else 0
        completion = completion_raw if usage_reported else 0
        records.append({
            "call_id": _record_id("call", owner_id, phase, seed, index),
            "owner_id": owner_id,
            "phase": phase,
            "seed": seed,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "response_model": history.get("response_model")
            or _response_value(response, "model"),
            "response_id": _response_value(response, "id"),
            "system_fingerprint": _response_value(response, "system_fingerprint"),
            "request_messages_sha256": sha256_json(messages),
            "request_messages_available": history.get("messages") is not None,
            "outputs_sha256": sha256_json(outputs),
            "outputs_available": history.get("outputs") is not None,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "usage_reported": usage_reported,
            "provider_usage": raw,
        })
    return records


def usage_totals(records: list[dict]) -> dict[str, int]:
    return {
        "calls": len(records),
        "prompt_tokens": sum(int(record.get("prompt_tokens") or 0) for record in records),
        "generated_tokens": sum(int(record.get("completion_tokens") or 0) for record in records),
        "total_tokens": sum(int(record.get("total_tokens") or 0) for record in records),
    }


def usage_by_phase(records: list[dict]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("phase", ""))].append(record)
    return {phase: usage_totals(grouped[phase]) for phase in sorted(grouped)}


def artifact_usage_consistent(artifacts: list[dict]) -> bool:
    return all(
        isinstance(artifact.get("calls"), list)
        and artifact.get("usage") == usage_totals(artifact["calls"])
        for artifact in artifacts
    )


def usage_ledger_audit(expected: list[dict], ledger: list[dict]) -> dict:
    """Verify exact, unique, provider-reported accounting for every model call."""
    errors = []

    def index(records: list[dict], label: str) -> dict[str, dict]:
        indexed = {}
        response_ids: set[str] = set()
        for offset, record in enumerate(records):
            call_id = record.get("call_id") if isinstance(record, dict) else None
            if not isinstance(call_id, str) or not call_id:
                errors.append(f"{label}[{offset}] has no call_id")
                continue
            if call_id in indexed:
                errors.append(f"{label} repeats call_id {call_id}")
            indexed[call_id] = record
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = record.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    errors.append(f"{call_id} has invalid {field}")
            prompt = record.get("prompt_tokens")
            completion = record.get("completion_tokens")
            total = record.get("total_tokens")
            if all(isinstance(value, int) and not isinstance(value, bool)
                   for value in (prompt, completion, total)) and total != prompt + completion:
                errors.append(f"{call_id} total_tokens is inconsistent")
            if record.get("usage_reported") is not True:
                errors.append(f"{call_id} lacks provider-reported usage")
            raw = record.get("provider_usage")
            if not isinstance(raw, dict) or raw.get("prompt_tokens") != prompt \
                    or raw.get("completion_tokens") != completion:
                errors.append(f"{call_id} provider usage disagrees with canonical counts")
            elif raw.get("total_tokens") is not None and raw.get("total_tokens") != total:
                errors.append(f"{call_id} provider total disagrees with canonical total")
            if record.get("model") != MODEL_ID or record.get("model_revision") != MODEL_REVISION:
                errors.append(f"{call_id} model identity drifted")
            if not isinstance(record.get("response_model"), str) \
                    or not record.get("response_model"):
                errors.append(f"{call_id} has no resolved response model")
            response_id = record.get("response_id")
            if not isinstance(response_id, str) or not response_id:
                errors.append(f"{call_id} has no provider response ID")
            elif response_id in response_ids:
                errors.append(f"provider response ID is reused: {response_id}")
            else:
                response_ids.add(response_id)
            fingerprint = record.get("system_fingerprint")
            if fingerprint is not None and (not isinstance(fingerprint, str)
                                            or not fingerprint):
                errors.append(f"{call_id} has an invalid system fingerprint")
            for hash_field, available_field in (
                    ("request_messages_sha256", "request_messages_available"),
                    ("outputs_sha256", "outputs_available")):
                if not isinstance(record.get(hash_field), str) or not re.fullmatch(
                        r"[0-9a-f]{64}", record[hash_field]):
                    errors.append(f"{call_id} has invalid {hash_field}")
                if record.get(available_field) is not True:
                    errors.append(f"{call_id} has no hashable provider {available_field}")
            if not isinstance(record.get("seed"), int) or isinstance(record.get("seed"), bool):
                errors.append(f"{call_id} has invalid seed")
            if not isinstance(record.get("phase"), str) or not record.get("phase"):
                errors.append(f"{call_id} has invalid phase")
            if not isinstance(record.get("owner_id"), str) or not record.get("owner_id"):
                errors.append(f"{call_id} has invalid owner_id")
        return indexed

    expected_by_id = index(expected, "artifacts")
    ledger_by_id = index(ledger, "ledger")
    missing = sorted(expected_by_id.keys() - ledger_by_id.keys())
    extra = sorted(ledger_by_id.keys() - expected_by_id.keys())
    drifted = sorted(
        call_id for call_id in expected_by_id.keys() & ledger_by_id.keys()
        if expected_by_id[call_id] != ledger_by_id[call_id]
    )
    if missing:
        errors.append(f"ledger missing call IDs: {missing}")
    if extra:
        errors.append(f"ledger has extra call IDs: {extra}")
    if drifted:
        errors.append(f"ledger call records drifted: {drifted}")
    return {
        "complete": not errors,
        "errors": errors,
        "artifact_calls": len(expected),
        "ledger_calls": len(ledger),
        "unique_call_ids": len(ledger_by_id),
    }


def _is_construction_call(call: dict) -> bool:
    owner_id = str(call.get("owner_id", ""))
    return not owner_id.startswith(("dev-exam-", "dev-reference-"))


# ---------------------------------------------------------------- pipeline

def run_quiz(chapter: str, tools_fns, url: str, n: int, *, seed: int,
             owner_id: str) -> dict:
    """Run one quiz episode and retain its complete auditable output."""
    lm = fresh_lm(url, seed)
    episode = {"owner_id": owner_id, "chapter": chapter, "seed": seed,
               "status": "ok", "questions": [], "trajectory": {}}
    try:
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
            pred = dspy.ReAct(QuizSig, tools=list(tools_fns), max_iters=QUIZ_MAX_ITERS)(
                chapter=chapter, num_questions=n)
        episode["questions"] = [question.model_dump() for question in pred.questions]
        episode["trajectory"] = serialize_trajectory(pred.trajectory)
    except Exception as error:
        episode.update(status="error", error=f"{type(error).__name__}: {str(error)[:300]}")
        log.warning("quiz episode failed for %s: %s", chapter, str(error)[:200])
    episode["calls"] = usage_records(lm, phase="quiz", owner_id=owner_id, seed=seed)
    episode["trajectory_sha256"] = sha256_json(episode["trajectory"])
    episode["usage"] = usage_totals(episode["calls"])
    return episode


def _attempt(question: str, note: str, url: str, *, seed: int,
             owner_id: str, phase: str) -> tuple[str, list[dict], str | None]:
    lm = fresh_lm(url, seed)
    answer = ""
    error = None
    try:
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
            answer = dspy.Predict("note, question -> answer")(
                note=note or "(no study note provided)", question=question).answer or ""
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    if error is None and not answer.strip():
        error = "empty model answer"
    return answer, usage_records(lm, phase=phase, owner_id=owner_id, seed=seed), error


def _adjudicate(question: str, reference: dict, attempt: str, url: str, *,
                seed: int, owner_id: str, phase: str) -> tuple[dict, list[dict]]:
    lm = fresh_lm(url, seed)
    result = {"status": "ok", "seed": seed, "verdict": "unresolved", "delta": ""}
    try:
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
            pred = dspy.Predict(AdjudicateSig)(
                question=question,
                reference_answer=reference["answer"],
                reference_evidence=json.dumps(reference["evidence"], sort_keys=True),
                attempt=attempt,
            )
        result.update(verdict=pred.verdict, delta=pred.delta)
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
    return result, usage_records(lm, phase=phase, owner_id=owner_id, seed=seed)


def _derive(item: dict, tools_fns, url: str, *, seed: int,
            owner_id: str, index: int) -> tuple[dict, list[dict]]:
    lm = fresh_lm(url, seed)
    result = {
        "derivation_id": _record_id("derivation", owner_id, index, seed),
        "seed": seed,
        "status": "ok",
        "answer": "",
        "raw_evidence": [],
        "evidence": [],
        "rejected_evidence": [],
        "trajectory": {},
        "reference_support": None,
        "evidence_class": "quote-only",
    }
    try:
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
            pred = dspy.ReAct(
                DeriveSig,
                tools=list(tools_fns),
                max_iters=DERIVE_MAX_ITERS,
            )(question=item["question"])
        result["answer"] = pred.answer or ""
        result["trajectory"] = serialize_trajectory(pred.trajectory)
        raw_evidence = [evidence.model_dump() for evidence in pred.evidence]
        result["raw_evidence"] = raw_evidence
        for raw in raw_evidence:
            valid = validate_evidence(item["_rt"], raw)
            if valid is None:
                result["rejected_evidence"].append(raw)
            elif valid not in result["evidence"]:
                result["evidence"].append(valid)
        if not result["answer"].strip() or not result["evidence"]:
            result["status"] = "invalid"
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
    calls = usage_records(lm, phase=f"derive-{index}", owner_id=owner_id, seed=seed)
    result["trajectory_sha256"] = sha256_json(result["trajectory"])

    if result["status"] == "ok":
        support_seed = stable_seed(seed, "reference-support")
        support_lm = fresh_lm(url, support_seed)
        support = {"status": "ok", "seed": support_seed, "supported": False,
                   "rationale": ""}
        try:
            with dspy.context(lm=support_lm, adapter=dspy.ChatAdapter()):
                pred = dspy.Predict(ReferenceSupportSig)(
                    question=item["question"], answer=result["answer"],
                    evidence=json.dumps(result["evidence"], sort_keys=True))
            if type(pred.supported) is not bool:
                raise TypeError("reference support output must be an exact boolean")
            support.update(supported=pred.supported, rationale=pred.rationale)
        except Exception as exc:
            support.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        result["reference_support"] = support
        calls += usage_records(support_lm, phase=f"reference-support-{index}",
                               owner_id=owner_id, seed=support_seed)
        if support["status"] != "ok" or not support["supported"]:
            result["status"] = "invalid"

    return result, calls


def _derive_consensus(item: dict, tools_fns, url: str, *,
                      master_seed: int, owner_id: str,
                      ensemble: int) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    derivations, calls = [], []
    for index in range(ensemble):
        seed = stable_seed(master_seed, owner_id, "derive", index)
        derivation, new_calls = _derive(item, tools_fns, url, seed=seed,
                                        owner_id=owner_id, index=index)
        derivations.append(derivation)
        calls += new_calls
    valid = [derivation for derivation in derivations if derivation["status"] == "ok"]
    if len(valid) < 2:
        return derivations, [], [], calls

    # Two references agree only if each is a substantively correct answer under
    # the other. Merely rejecting the same student attempt is not agreement.
    checks = []
    for left, right, label in ((valid[0], valid[1], "a-to-b"),
                               (valid[1], valid[0], "b-to-a")):
        seed = stable_seed(master_seed, owner_id, "reference-consensus", label)
        check, new_calls = _adjudicate(
            item["question"], left, right["answer"], url, seed=seed,
            owner_id=owner_id, phase=f"reference-consensus-{label}")
        check.update(reference_id=left["derivation_id"], candidate_id=right["derivation_id"])
        checks.append(check)
        calls += new_calls
    consensus = valid[:2] if all(check["status"] == "ok" and check["verdict"] == "correct"
                                 for check in checks) else []
    return derivations, consensus, checks, calls


def _judge_attempt(item: dict, attempt: str, references: list[dict], url: str, *,
                   master_seed: int, owner_id: str,
                   phase: str, seed_namespace: str | None = None,
                   seed_phase: str | None = None,
                   ) -> tuple[str, str, list[dict], list[dict]]:
    checks, calls = [], []
    for index, reference in enumerate(references):
        effective_seed_phase = seed_phase or phase
        seed = stable_seed(
            master_seed, seed_namespace or owner_id, effective_seed_phase, index)
        audit_phase = f"{phase}-{index}"
        check, new_calls = _adjudicate(
            item["question"], reference, attempt, url, seed=seed,
            owner_id=owner_id, phase=audit_phase)
        check.update(
            reference_id=reference["derivation_id"],
            audit_phase=audit_phase,
            seed_phase=effective_seed_phase,
        )
        checks.append(check)
        calls += new_calls
    verdict, delta = _adjudication_result(checks, len(references))
    return verdict, delta, checks, calls


def _distill(item: dict, attempt: str, references: list[dict], url: str, *,
             master_seed: int, owner_id: str) -> tuple[dict | None, dict | None, list[dict]]:
    reference = sorted(references, key=lambda ref: ref["derivation_id"])[0]
    seed = stable_seed(master_seed, owner_id, "distill")
    lm = fresh_lm(url, seed)
    raw: dict = {}
    error = None
    try:
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
            pred = dspy.Predict(DistillSig)(
                question=item["question"], attempt=attempt,
                reference_answer=reference["answer"],
                reference_evidence=json.dumps(reference["evidence"], sort_keys=True))
        raw = {"belief": pred.belief, "correction": pred.correction,
               "quote": pred.quote, "file": pred.file, "line": int(pred.line)}
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
    calls = usage_records(lm, phase="distill", owner_id=owner_id, seed=seed)
    bounced = {"raw_entry": raw, "reasons": []}
    if error:
        bounced["reasons"].append(f"distill_error: {error}")
        return None, bounced, calls
    if not str(raw.get("belief", "")).strip():
        bounced["reasons"].append("belief is empty")
    if not str(raw.get("correction", "")).strip():
        bounced["reasons"].append("correction is empty")
    citation = validate_evidence(item["_rt"], raw)
    if citation is None:
        bounced["reasons"].append("citation failed exact source validation")
    elif citation not in reference["evidence"]:
        bounced["reasons"].append("citation is not evidence from the selected reference")

    support_checks = []
    for index, candidate in enumerate(references):
        support_seed = stable_seed(master_seed, owner_id, "correction-support", index)
        check, new_calls = _adjudicate(
            item["question"], candidate, raw.get("correction", ""), url,
            seed=support_seed, owner_id=owner_id, phase=f"correction-support-{index}")
        check["reference_id"] = candidate["derivation_id"]
        support_checks.append(check)
        calls += new_calls
    if not all(check["status"] == "ok" and check["verdict"] == "correct"
               for check in support_checks):
        bounced["reasons"].append("correction is not supported by every consensus reference")
    bounced["support_checks"] = support_checks
    if bounced["reasons"]:
        return None, bounced, calls

    entry = {
        **raw,
        **citation,
        "entry_id": _record_id("entry", owner_id, raw["belief"], raw["correction"], citation),
        "origin_item_id": item["origin_item_id"],
        "origin_round": item["origin_round"],
        "chapter": item["chapter"],
        "verdict": item.get("_verdict"),
        "reference_ids": [reference["derivation_id"] for reference in references],
        "evidence_class": reference["evidence_class"],
        "correction_support": support_checks,
    }
    return entry, None, calls


def _schema_error(item: dict) -> str | None:
    required = ("schema_version", "item_id", "origin_item_id", "origin_round",
                "round", "kind", "split", "question", "qtype", "anchors", "chapter")
    missing = [key for key in required if key not in item]
    if missing:
        return f"missing required fields: {', '.join(missing)}"
    if type(item["schema_version"]) is not int \
            or item["schema_version"] != SCHEMA_VERSION:
        return f"unsupported schema_version {item['schema_version']!r}"
    if (not isinstance(item["item_id"], str) or not item["item_id"]
            or not isinstance(item["origin_item_id"], str)
            or not item["origin_item_id"]
            or not _positive_json_integer(item["origin_round"])
            or not _positive_json_integer(item["round"])):
        return "item IDs and round lineage must be nonempty strings/positive integers"
    if item["kind"] not in ("quiz", "retest") or item["split"] != "train":
        return "training runner accepts only train quiz/retest items"
    if item["kind"] == "quiz" and not is_distillable_item(item):
        return "original train quiz has inconsistent origin lineage"
    if item["kind"] == "retest" and (
            item.get("retest_of") != item.get("origin_item_id")
            or item.get("item_id") == item.get("origin_item_id")
            or item["round"] <= item["origin_round"]):
        return "retest has inconsistent origin lineage"
    return None


def _question_record_error(item: dict, rt: RepoTools) -> str | None:
    if item.get("split") == "train":
        error = _schema_error(item)
    else:
        required = ("schema_version", "item_id", "origin_item_id", "origin_round",
                    "round", "kind", "split", "question", "qtype", "anchors", "chapter")
        missing = [key for key in required if key not in item]
        error = f"missing required fields: {', '.join(missing)}" if missing else None
        if not error and not _is_original_quiz(item, split="dev"):
            error = "dev quiz has inconsistent split or origin lineage"
    if error:
        return error
    if item["kind"] == "quiz" and (
            not isinstance(item.get("quiz_episode_id"), str)
            or not item["quiz_episode_id"]
            or not isinstance(item.get("quiz_ordinal"), int)
            or isinstance(item.get("quiz_ordinal"), bool)
            or item["quiz_ordinal"] < 0):
        return "quiz question has no valid episode/ordinal lineage"
    anchors = validate_anchors(rt, item.get("anchors"))
    if anchors is None or anchors != item["anchors"]:
        return "question anchors are not canonical exact corpus files"
    return None


def _validate_question_provenance(args, records: list[dict], episodes: list[dict],
                                  rt: RepoTools, *,
                                  expected_chapters: list[str],
                                  expected_question_count: int) -> None:
    if (not isinstance(expected_chapters, list) or not expected_chapters
            or any(not isinstance(chapter, str) or not chapter
                   for chapter in expected_chapters)
            or len(set(expected_chapters)) != len(expected_chapters)):
        raise SystemExit("quiz provenance requires an exact, unique chapter plan")
    if (not isinstance(records, list) or not isinstance(episodes, list)
            or any(not isinstance(record, dict)
                   or not isinstance(record.get("item_id"), str)
                   or not isinstance(record.get("chapter"), str)
                   or not isinstance(record.get("question"), str)
                   or not record["question"].strip()
                   or record.get("qtype")
                   not in {"usage", "behavior", "location", "pitfall"}
                   or not isinstance(record.get("writer_sketch", ""), str)
                   for record in records)
            or any(not isinstance(episode, dict)
                   or not isinstance(episode.get("owner_id"), str)
                   for episode in episodes)):
        raise SystemExit("question provenance contains malformed records")
    episodes_by_id = {episode["owner_id"]: episode for episode in episodes}
    if len(episodes_by_id) != len(episodes):
        raise SystemExit("quiz episode owner IDs are missing or duplicated")
    quiz_records = [record for record in records if record.get("kind") == "quiz"]
    record_ids = [record["item_id"] for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise SystemExit("question records contain duplicate item IDs")
    expected_episode_ids = {
        _record_id("quiz", args.study_id, args.task, args.round, chapter)
        for chapter in expected_chapters
    }
    if set(episodes_by_id) != expected_episode_ids:
        raise SystemExit("quiz episodes do not match the exact chapter plan")
    if type(expected_question_count) is not int or expected_question_count < 2:
        raise SystemExit("quiz provenance has an invalid question-count contract")
    by_chapter: dict[str, list[dict]] = defaultdict(list)
    for record in quiz_records:
        by_chapter[record.get("chapter")].append(record)
    if set(by_chapter) != set(expected_chapters):
        raise SystemExit("quiz questions do not match the exact chapter plan")

    for chapter in expected_chapters:
        owner_id = _record_id(
            "quiz", args.study_id, args.task, args.round, chapter)
        episode = episodes_by_id[owner_id]
        expected_seed = stable_seed(
            args.seed, args.study_id, args.task, args.round, "quiz", chapter)
        raw_questions = episode.get("questions")
        if (episode.get("owner_id") != owner_id
                or episode.get("chapter") != chapter
                or not _json_identity_equal(episode.get("seed"), expected_seed)
                or episode.get("status") != "ok"
                or not isinstance(episode.get("trajectory"), dict)
                or episode.get("trajectory_sha256")
                != sha256_json(episode.get("trajectory"))
                or not isinstance(raw_questions, list)
                or len(raw_questions) != expected_question_count):
            raise SystemExit(f"quiz episode identity drifted: {owner_id}")
        episode_phases = _artifact_call_phases(
            episode, owner_id=owner_id, label=f"quiz episode {owner_id}")
        _consume_call_phase(
            episode_phases, owner_id=owner_id, phase="quiz", seed=expected_seed,
            label=f"quiz episode {owner_id}", required=True)
        _finish_call_graph(episode_phases, label=f"quiz episode {owner_id}")

        chapter_records = by_chapter[chapter]
        ordinals = [record.get("quiz_ordinal") for record in chapter_records]
        if (len(chapter_records) != expected_question_count
                or any(type(ordinal) is not int for ordinal in ordinals)
                or set(ordinals) != set(range(expected_question_count))):
            raise SystemExit(
                f"quiz chapter {chapter} does not preserve raw ordinals 0.."
                f"{expected_question_count - 1}"
            )
        splits = [record.get("split") for record in chapter_records]
        if splits.count("dev") != 1 or splits.count("train") != expected_question_count - 1:
            raise SystemExit(
                "each quiz chapter must retain exactly one dev item and all other train items"
            )
        for record in chapter_records:
            ordinal = record["quiz_ordinal"]
            expected_item_id = _record_id(
                "question", args.study_id, args.task, args.round, chapter,
                ordinal, record["question"])
            if (not _json_identity_equal(record.get("round"), args.round)
                    or record.get("quiz_episode_id") != owner_id
                    or record.get("item_id") != expected_item_id):
                raise SystemExit(f"quiz question identity drifted: {record.get('item_id')}")
            raw = raw_questions[ordinal]
            if (not isinstance(raw, dict)
                    or raw.get("question") != record["question"]
                    or raw.get("qtype") != record["qtype"]
                    or raw.get("writer_sketch", "") != record.get("writer_sketch", "")
                    or validate_anchors(rt, raw.get("anchors")) != record["anchors"]):
                raise SystemExit(
                    f"quiz question drifted from its raw episode: {record['item_id']}"
                )
    for record in records:
        if record.get("kind") == "retest":
            expected_id = _record_id(
                "retest", args.study_id, args.task, record["round"], record["origin_item_id"])
            if (not _json_identity_equal(record["round"], args.round)
                    or record["item_id"] != expected_id):
                raise SystemExit(f"retest identity drifted: {record['item_id']}")


def run_item(item: dict, note: str, tools_fns, url: str,
             ensemble: int, rt: RepoTools, *, master_seed: int) -> dict:
    """ATTEMPT -> blind consensus VERIFY -> supported DISTILL.

    Schema and lineage are fail-closed. In particular, dev and retest items can
    never distill even if a caller tampers with legacy boolean flags.
    """
    rec = dict(item)
    rec["entry"] = None
    rec["calls"] = []
    if error := _schema_error(item):
        rec.update(status="schema_error", error=error, usage=usage_totals([]))
        return rec
    owner_id = item["item_id"]
    attempt_seed = stable_seed(master_seed, owner_id, "attempt")
    attempt, calls, error = _attempt(item["question"], note, url, seed=attempt_seed,
                                     owner_id=owner_id, phase="attempt")
    rec["calls"] += calls
    rec["attempt"] = attempt
    rec["attempt_seed"] = attempt_seed
    rec["input_note_sha256"] = sha256_text(note)
    if error:
        rec.update(status="attempt_error", error=error, usage=usage_totals(rec["calls"]))
        return rec

    internal = dict(item, _rt=rt)
    derivations, references, consensus, calls = _derive_consensus(
        internal, tools_fns, url, master_seed=master_seed,
        owner_id=owner_id, ensemble=ensemble)
    rec["calls"] += calls
    rec["derivations"] = derivations
    rec["reference_consensus"] = consensus
    rec["reference_ids"] = [reference["derivation_id"] for reference in references]
    if not references:
        rec.update(status="ok", verdict="unresolved",
                   delta="independent derivations did not substantively agree",
                   usage=usage_totals(rec["calls"]))
        return rec

    verdict, delta, checks, calls = _judge_attempt(
        internal, attempt, references, url, master_seed=master_seed,
        owner_id=owner_id, phase="attempt-adjudication")
    rec["calls"] += calls
    rec.update(verdict=verdict, delta=delta, adjudications=checks)
    internal["_verdict"] = verdict
    if verdict in ("wrong", "partial") and is_distillable_item(item):
        entry, bounced, calls = _distill(
            internal, attempt, references, url, master_seed=master_seed, owner_id=owner_id)
        rec["calls"] += calls
        rec["entry"] = entry
        if bounced:
            rec["entry_bounced"] = bounced
    rec["status"] = "ok"
    rec["usage"] = usage_totals(rec["calls"])
    return rec


def build_dev_reference(item: dict, tools_fns, url: str, *, master_seed: int,
                        created_round: int, rt: RepoTools, study_id: str,
                        task: str) -> dict:
    """Derive one blind reference artifact, independent of every dev attempt."""
    reference_id = _record_id("dev-reference", study_id, task, item["item_id"])
    internal = dict(item, _rt=rt)
    derivations, references, checks, calls = _derive_consensus(
        internal, tools_fns, url, master_seed=master_seed,
        owner_id=reference_id, ensemble=DEV_ENSEMBLE)
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_id": reference_id,
        "origin_item_id": item["item_id"],
        "origin_round": item["origin_round"],
        "created_round": created_round,
        "question": item["question"],
        "qtype": item["qtype"],
        "anchors": item["anchors"],
        "chapter": item["chapter"],
        "status": "ok" if references else "unresolved",
        "derivations": derivations,
        "references": references,
        "reference_consensus": checks,
        "reference_ids": [reference["derivation_id"] for reference in references],
        "calls": calls,
        "usage": usage_totals(calls),
    }


def run_dev_item(item: dict, note: str, reference: dict, url: str, *,
                 master_seed: int, exam_round: int) -> dict:
    """Evaluate one cumulative dev item in paired note/bare arms.

    The two attempts use the identical DSPy signature and sampling seed, and
    differ only in the note value. Both are judged against one immutable blind
    reference created independently of every attempt and note.
    """
    if (not _is_original_quiz(item, split="dev")
            or not _positive_json_integer(exam_round)):
        return {"status": "schema_error", "error": "invalid dev lineage", "calls": [],
                "usage": usage_totals([])}
    owner_id = _record_id("dev-exam", item["item_id"], exam_round)
    rec = {
        **item,
        "item_id": owner_id,
        "origin_item_id": item["item_id"],
        "kind": "dev_exam",
        "split": "dev",
        "round": exam_round,
        "exam_round": exam_round,
        "input_note_sha256": sha256_text(note),
        "reference_id": reference.get("reference_id"),
        "reference_sha256": sha256_json(reference),
        "entry": None,
        "calls": [],
    }
    references = reference.get("references", [])
    if reference.get("status") != "ok" or len(references) < 2:
        rec.update(status="reference_unresolved",
                   verdicts={"with_note": "unresolved", "bare": "unresolved"},
                   usage=usage_totals([]))
        return rec
    attempts = {}
    paired_seed = stable_seed(master_seed, reference["reference_id"], "paired-attempt")
    for arm, arm_note in (("with_note", note), ("bare", "")):
        answer, calls, error = _attempt(item["question"], arm_note, url, seed=paired_seed,
                                        owner_id=owner_id, phase=f"dev-attempt-{arm}",
                                        )
        rec["calls"] += calls
        attempts[arm] = {"answer": answer, "error": error, "seed": paired_seed}
    rec["attempts"] = attempts
    rec["attempt_protocol"] = {
        "signature": "note, question -> answer",
        "paired_seed": paired_seed,
        "only_manipulated_field": "note",
    }
    if any(attempt["error"] for attempt in attempts.values()):
        rec.update(status="attempt_error", usage=usage_totals(rec["calls"]))
        return rec

    rec["reference_ids"] = [reference["derivation_id"] for reference in references]
    rec["verdicts"], rec["deltas"], rec["adjudications"] = {}, {}, {}
    for arm, attempt in attempts.items():
        verdict, delta, checks, calls = _judge_attempt(
            item, attempt["answer"], references, url, master_seed=master_seed,
            owner_id=owner_id, phase=f"dev-adjudication-{arm}",
            seed_namespace=reference["reference_id"],
            seed_phase="dev-paired-adjudication")
        rec["calls"] += calls
        rec["verdicts"][arm] = verdict
        rec["deltas"][arm] = delta
        rec["adjudications"][arm] = checks
    rec["adjudication_protocol"] = {
        "signature": "AdjudicateSig",
        "seed_namespace": reference["reference_id"],
        "seed_phase": "dev-paired-adjudication",
        "paired_seeds": [
            stable_seed(
                master_seed, reference["reference_id"], "dev-paired-adjudication", index)
            for index in range(len(references))
        ],
        "only_manipulated_field": "attempt",
        "audit_phases": {
            arm: f"dev-adjudication-{arm}" for arm in ("with_note", "bare")
        },
    }
    rec["status"] = "ok"
    rec["usage"] = usage_totals(rec["calls"])
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

def _jsonl(records: list[dict]) -> str:
    return "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                   for record in records)


def _read_jsonl(path: Path) -> list[dict]:
    text = read_artifact_bytes(path).decode("utf-8")
    return [
        strict_json_loads(line, label=f"{path}:{line_number}")
        for line_number, line in enumerate(text.splitlines(), 1)
        if line.strip()
    ]


def _study_dir(args) -> Path:
    try:
        validate_id(args.study_id, "study ID")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.task not in CORPORA:
        raise SystemExit(f"unknown task: {args.task!r}")
    base = ROOT / "study-selfquiz"
    sdir = base / "studies" / args.study_id / args.task
    cursor = base
    for part in ("studies", args.study_id, args.task):
        if cursor.is_symlink():
            raise SystemExit(f"study artifact path must not traverse a symlink: {cursor}")
        cursor /= part
    if cursor.is_symlink():
        raise SystemExit(f"study artifact path must not traverse a symlink: {cursor}")
    return sdir


def _study_round_lock_path(args) -> Path:
    """Keep coordination state outside every study artifact inventory."""
    sdir = _study_dir(args)
    if not isinstance(args.round, int) or isinstance(args.round, bool) or args.round < 1:
        raise SystemExit("round must be a positive integer")
    return ROOT / ".studybench-locks" / "selfquiz" / sdir.parts[-2] / sdir.name \
        / f"r{args.round}.lock"


@contextmanager
def _study_round_lock(args):
    """Hold one nonblocking process-wide lock across all writes and model calls."""
    path = _study_round_lock_path(args)
    try:
        with exclusive_process_lock(path):
            yield path
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(
            "study round is already active or has an unsafe owner-only lock: "
            f"{args.study_id}/{args.task}/r{args.round} ({error})"
        ) from error


def _environment_complete(environment: dict[str, object]) -> bool:
    packages = environment.get("packages")
    package_ready = isinstance(packages, dict) and all(
        packages.get(name)
        for name in ("dspy", "openai", "pydantic")
    )
    return bool(
        environment_is_claim_ready(environment)
        and package_ready
        and all(environment.get(field) for field in (
            "python", "implementation", "machine", "platform",
        ))
    )


def validate_audit_protocol(path: Path) -> tuple[str, dict]:
    """Read and validate one blinded human-audit pre-registration without writing."""
    try:
        protocol_bytes = read_artifact_bytes(path)
        text = protocol_bytes.decode("utf-8")
        protocol = strict_json_loads(protocol_bytes, label="human-audit protocol")
        validate_human_audit_protocol(protocol)
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"invalid blinded-audit protocol: {error}") from error
    return text, protocol


def _snapshot_audit_protocol(path: Path | None, sdir: Path) -> dict | None:
    """Snapshot a human-audit protocol before any study artifact is generated."""
    if path is None:
        return None
    text, protocol = validate_audit_protocol(path)
    digest = sha256_text(text)
    relative = Path("audit-protocols") / f"{digest}.json"
    write_immutable_text(sdir / relative, text)
    return {
        "sha256": digest,
        "path": str(relative),
        "protocol_id": protocol["protocol_id"],
        "protocol": protocol,
    }


def _write_task_manifest(
    args, corpus, sdir: Path, urls: list[str]
) -> tuple[dict, dict[str, object]]:
    """Create/validate the stable study contract and snapshot this launch."""

    try:
        corpus_info = corpus_record(corpus)
        source = source_record()
        environment = environment_record()
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"cannot record study provenance: {error}") from error
    if corpus_info.get("commit") != corpus.commit:
        raise SystemExit(
            f"{corpus.name} is at {corpus_info.get('commit')}, expected {corpus.commit}"
        )
    if corpus_info.get("dirty"):
        raise SystemExit(f"refusing to study a dirty corpus checkout: {corpus.repo}")

    task_manifest_path = sdir / "manifest.json"
    existing_manifest = None
    if task_manifest_path.exists():
        try:
            existing_manifest = load_json_artifact(task_manifest_path)
        except (OSError, UnicodeError, ValueError) as error:
            raise SystemExit(f"invalid existing task manifest: {error}") from error
        baseline_environment = (
            existing_manifest.get("environment")
            if isinstance(existing_manifest, dict)
            else None
        )
        if not isinstance(baseline_environment, dict):
            raise SystemExit("existing task manifest has no environment baseline")
        if not environments_compatible(baseline_environment, environment):
            raise SystemExit(
                "study environment has substantive drift; choose a new --study-id"
            )
    else:
        baseline_environment = environment

    try:
        declared_server_count = int(environment["server_count"])
    except (KeyError, TypeError, ValueError):
        declared_server_count = None
    current_provenance_readiness = {
        "corpus_pinned_clean": True,
        "source_pinned_clean": bool(source.get("git_commit")
                                    and source.get("tree_sha256")
                                    and source.get("files")
                                    and not source.get("dirty")),
        "environment_complete": _environment_complete(environment),
        "model_revision_pinned": bool(MODEL_REVISION),
        "server_count_matches_environment": declared_server_count == len(urls),
    }
    if not args.smoke and not all(current_provenance_readiness.values()):
        failed = [
            name for name, ready in current_provenance_readiness.items() if not ready
        ]
        raise SystemExit(
            "research self-study requires complete clean provenance; failed: "
            + ", ".join(failed)
        )

    try:
        baseline_server_count = int(baseline_environment["server_count"])
    except (KeyError, TypeError, ValueError):
        baseline_server_count = None
    provenance_readiness = {
        **current_provenance_readiness,
        "environment_complete": _environment_complete(baseline_environment),
        "server_count_matches_environment": baseline_server_count == len(urls),
    }
    audit_protocol_path = getattr(args, "audit_protocol", None)
    if audit_protocol_path is not None and args.round != 1:
        raise SystemExit("audit protocols must be pre-registered in round 1")
    if audit_protocol_path is None and existing_manifest is not None:
        audit_protocol = existing_manifest.get("human_audit_protocol")
        if isinstance(audit_protocol, dict):
            relative = PurePosixPath(str(audit_protocol.get("path", "")))
            snapshot = sdir.joinpath(*relative.parts)
            if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                    or not snapshot.is_file()
                    or sha256_file(snapshot) != audit_protocol.get("sha256")):
                raise SystemExit("pre-registered audit-protocol snapshot is missing or changed")
    else:
        audit_protocol = _snapshot_audit_protocol(audit_protocol_path, sdir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": args.study_id,
        "task": args.task,
        "master_seed": args.seed,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "sampling": SAMPLING,
        "corpus_commit": corpus.commit,
        "corpus": corpus_info,
        "source": source,
        # This exact first-launch record remains the immutable substantive
        # contract.  Per-invocation records live below each round and bind all
        # newly generated model artifacts.
        "environment": baseline_environment,
        "environment_contract": environment_contract_record(
            baseline_environment
        ),
        "server_transport": {
            "scope": "loopback",
            "protocol": "openai-compatible-http",
            "server_count": len(urls),
            "assignment": "stable_seed(master_seed, owner_id, server) modulo server_count",
        },
        "provenance_readiness": provenance_readiness,
        "automated_provenance_ready": all(provenance_readiness.values()),
        "human_audit_protocol": audit_protocol,
        "config": {
            "chapters_per_round": args.chapters,
            "questions_per_chapter": 3 if args.smoke else args.questions,
            "smoke": args.smoke,
            "quiz_max_iters": QUIZ_MAX_ITERS,
            "derive_max_iters": DERIVE_MAX_ITERS,
            "train_ensemble": TRAIN_ENSEMBLE,
            "dev_ensemble": DEV_ENSEMBLE,
            "retest_fraction": RETEST_FRAC,
            "freshness_near_jaccard": FRESHNESS_NEAR_JACCARD,
            "max_freshness_near_rate": MAX_FRESHNESS_NEAR_RATE,
            "concurrency": args.concurrency,
            "provider_retries": 0,
        },
    }
    if existing_manifest is not None:
        if canonical_json_bytes(existing_manifest) != canonical_json_bytes(manifest):
            raise SystemExit(
                "study task manifest drifted; choose a new --study-id"
            )
        manifest = existing_manifest
    else:
        write_immutable_json(task_manifest_path, manifest)
    try:
        launch_environment = write_environment_snapshot(
            sdir,
            PurePosixPath(f"r{args.round}/environments"),
            environment,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot snapshot study launch environment: {error}") from error
    return manifest, launch_environment


def _validate_artifact_environment(
    sdir: Path,
    task_manifest: dict,
    artifact: dict,
    *,
    label: str,
) -> None:
    """Fail closed unless a model artifact binds one compatible exact launch."""

    baseline = task_manifest.get("environment")
    readiness = task_manifest.get("provenance_readiness")
    if not isinstance(baseline, dict) or not isinstance(readiness, dict):
        raise SystemExit("task manifest has no valid environment contract")
    if not environment_contract_is_valid(
        task_manifest.get("environment_contract"), baseline
    ):
        raise SystemExit("task manifest environment contract fingerprint is invalid")
    try:
        validate_environment_snapshot(
            sdir,
            artifact.get("environment_snapshot"),
            baseline=baseline,
            require_claim_ready=readiness.get("environment_complete") is True,
        )
    except ValueError as error:
        raise SystemExit(f"{label} has invalid launch-environment provenance") from error


def _bind_launch_environment(
    artifact: dict, launch_environment: dict[str, object]
) -> dict:
    """Bind a newly generated model artifact to this exact invocation."""

    if not isinstance(artifact, dict) or "environment_snapshot" in artifact:
        raise SystemExit("new model artifact has an invalid environment binding")
    return {**artifact, "environment_snapshot": launch_environment}


def _launch_environment_inventory(
    sdir: Path, round_number: int, task_manifest: dict
) -> list[dict[str, object]]:
    """List and revalidate every content-addressed launch through a round."""

    records: list[dict[str, object]] = []
    for value in range(1, round_number + 1):
        environment_dir = sdir / f"r{value}" / "environments"
        for path in sorted(environment_dir.glob("environment-*.json")):
            try:
                data = read_artifact_bytes(path)
            except (OSError, ValueError) as error:
                raise SystemExit(f"unsafe launch-environment artifact: {path}") from error
            record: dict[str, object] = {
                "schema_version": 1,
                "sha256": sha256_bytes(data),
                "bytes": len(data),
                "snapshot": path.relative_to(sdir).as_posix(),
            }
            if path.name != f"environment-{record['sha256']}.json":
                raise SystemExit(f"launch-environment filename/hash mismatch: {path}")
            _validate_artifact_environment(
                sdir,
                task_manifest,
                {"environment_snapshot": record},
                label=f"launch environment {path.name}",
            )
            records.append(record)
    if not records:
        raise SystemExit("study has no exact launch-environment snapshots")
    return records


def _error_rate(verdicts: list[str]) -> dict:
    resolved = [verdict for verdict in verdicts if verdict in ("correct", "partial", "wrong")]
    errors = sum(verdict in ("partial", "wrong") for verdict in resolved)
    return {"resolved": len(resolved), "errors": errors,
            "error_rate": errors / len(resolved) if resolved else None}


def _json_identity_equal(observed: object, expected: object) -> bool:
    """Compare manifest identities without Python's bool/int equivalence."""

    if type(expected) is int:
        return type(observed) is int and observed == expected
    if type(expected) is bool:
        return observed is expected
    return observed == expected


def _artifact_call_phases(record: dict, *, owner_id: str, label: str) \
        -> dict[str, list[dict]]:
    """Index one artifact's exact, provider-auditable model-call graph."""

    calls = record.get("calls")
    if (not isinstance(calls, list)
            or any(not isinstance(call, dict) for call in calls)):
        raise SystemExit(f"{label} has incomplete model-call usage")
    try:
        expected_usage = usage_totals(calls)
    except (TypeError, ValueError):
        raise SystemExit(f"{label} has incomplete model-call usage") from None
    if record.get("usage") != expected_usage:
        raise SystemExit(f"{label} has incomplete model-call usage")
    audit = usage_ledger_audit(calls, calls)
    if not audit["complete"]:
        raise SystemExit(f"{label} has invalid model calls: {audit['errors']}")
    phases: dict[str, list[dict]] = defaultdict(list)
    for call in calls:
        if call.get("owner_id") != owner_id:
            raise SystemExit(f"{label} has a call bound to another artifact")
        phases[call["phase"]].append(call)
    return phases


def _consume_call_phase(phases: dict[str, list[dict]], *, owner_id: str,
                        phase: str, seed: int, label: str,
                        required: bool) -> None:
    """Consume one LM invocation and verify its deterministic call identities."""

    calls = phases.pop(phase, [])
    if required and not calls:
        raise SystemExit(f"{label} is missing calls for phase {phase}")
    for index, call in enumerate(calls):
        if (not _json_identity_equal(call.get("seed"), seed)
                or call.get("call_id")
                != _record_id("call", owner_id, phase, seed, index)):
            raise SystemExit(f"{label} has drifted calls for phase {phase}")


def _finish_call_graph(phases: dict[str, list[dict]], *, label: str) -> None:
    if phases:
        raise SystemExit(f"{label} has unexpected model-call phases: {sorted(phases)}")


def _validate_adjudication_check(
    check: object,
    *,
    reference_id: str,
    seed: int,
    owner_id: str,
    call_phase: str,
    phases: dict[str, list[dict]],
    label: str,
    metadata: dict[str, object] | None = None,
) -> dict:
    if not isinstance(check, dict):
        raise SystemExit(f"{label} has a malformed adjudication")
    status = check.get("status")
    if (status not in {"ok", "error"}
            or not _json_identity_equal(check.get("seed"), seed)
            or check.get("reference_id") != reference_id
            or check.get("verdict") not in {"correct", "partial", "wrong", "unresolved"}
            or not isinstance(check.get("delta"), str)
            or (metadata is not None
                and any(not _json_identity_equal(check.get(key), value)
                        for key, value in metadata.items()))
            or (status == "error"
                and (not isinstance(check.get("error"), str) or not check["error"]))):
        raise SystemExit(f"{label} has a drifted adjudication")
    _consume_call_phase(
        phases,
        owner_id=owner_id,
        phase=call_phase,
        seed=seed,
        label=label,
        required=status == "ok",
    )
    return check


def _adjudication_result(checks: list[dict], reference_count: int) -> tuple[str, str]:
    verdicts = [check["verdict"] for check in checks if check["status"] == "ok"]
    if len(verdicts) != reference_count or reference_count == 0:
        return "unresolved", "adjudication failed"
    if all(verdict == "correct" for verdict in verdicts):
        verdict = "correct"
    elif all(verdict in {"wrong", "partial"} for verdict in verdicts):
        verdict = "wrong" if "wrong" in verdicts else "partial"
    else:
        verdict = "unresolved"
    delta = " | ".join(check["delta"] for check in checks if check.get("delta"))
    return verdict, delta


def _validate_derivations(
    record: dict,
    *,
    owner_id: str,
    master_seed: int,
    ensemble: int,
    rt: RepoTools,
    phases: dict[str, list[dict]],
    label: str,
) -> list[dict]:
    derivations = record.get("derivations")
    if not isinstance(derivations, list) or len(derivations) != ensemble:
        raise SystemExit(f"{label} does not contain the exact derivation ensemble")
    for index, derivation in enumerate(derivations):
        seed = stable_seed(master_seed, owner_id, "derive", index)
        derivation_id = _record_id("derivation", owner_id, index, seed)
        if (not isinstance(derivation, dict)
                or derivation.get("derivation_id") != derivation_id
                or not _json_identity_equal(derivation.get("seed"), seed)
                or derivation.get("status") not in {"ok", "invalid", "error"}
                or derivation.get("evidence_class") != "quote-only"
                or not isinstance(derivation.get("answer"), str)
                or not isinstance(derivation.get("raw_evidence"), list)
                or not isinstance(derivation.get("evidence"), list)
                or not isinstance(derivation.get("rejected_evidence"), list)
                or not _trajectory_hash_valid(derivation)
                or (derivation.get("status") == "error"
                    and (not isinstance(derivation.get("error"), str)
                         or not derivation["error"]))
                or any(validate_evidence(rt, evidence) != evidence
                       for evidence in derivation.get("evidence", []))):
            raise SystemExit(f"{label} has a drifted derivation at index {index}")
        expected_evidence, expected_rejected = [], []
        for raw in derivation["raw_evidence"]:
            evidence = validate_evidence(rt, raw)
            if evidence is None:
                expected_rejected.append(raw)
            elif evidence not in expected_evidence:
                expected_evidence.append(evidence)
        if (derivation["evidence"] != expected_evidence
                or derivation["rejected_evidence"] != expected_rejected):
            raise SystemExit(f"{label} has drifted raw-evidence lineage at index {index}")
        _consume_call_phase(
            phases,
            owner_id=owner_id,
            phase=f"derive-{index}",
            seed=seed,
            label=label,
            required=derivation["status"] != "error",
        )
        support = derivation.get("reference_support")
        if support is None:
            if (derivation["status"] == "ok"
                    or (derivation["status"] == "invalid"
                        and derivation["answer"].strip()
                        and derivation["evidence"])):
                raise SystemExit(f"{label} has an unverified derivation at index {index}")
            continue
        if (derivation["status"] == "error"
                or not derivation["answer"].strip()
                or not derivation["evidence"]):
            raise SystemExit(f"{label} has impossible reference support at index {index}")
        support_seed = stable_seed(seed, "reference-support")
        if (not isinstance(support, dict)
                or support.get("status") not in {"ok", "error"}
                or not _json_identity_equal(support.get("seed"), support_seed)
                or type(support.get("supported")) is not bool
                or not isinstance(support.get("rationale"), str)
                or (support.get("status") == "error"
                    and (not isinstance(support.get("error"), str)
                         or not support["error"]))):
            raise SystemExit(f"{label} has drifted reference support at index {index}")
        _consume_call_phase(
            phases,
            owner_id=owner_id,
            phase=f"reference-support-{index}",
            seed=support_seed,
            label=label,
            required=support["status"] == "ok",
        )
        supported = (bool(derivation["answer"].strip())
                     and bool(derivation["evidence"])
                     and support["status"] == "ok"
                     and support["supported"] is True)
        if (derivation["status"] == "ok") != supported:
            raise SystemExit(f"{label} has inconsistent derivation status at index {index}")
    return derivations


def _validate_reference_consensus(
    record: dict,
    derivations: list[dict],
    *,
    owner_id: str,
    master_seed: int,
    phases: dict[str, list[dict]],
    label: str,
    stored_references: object | None = None,
) -> list[dict]:
    valid = [derivation for derivation in derivations if derivation["status"] == "ok"]
    checks = record.get("reference_consensus")
    if len(valid) < 2:
        if checks != [] or record.get("reference_ids") != []:
            raise SystemExit(f"{label} has consensus without two valid derivations")
        expected_references: list[dict] = []
    else:
        if not isinstance(checks, list) or len(checks) != 2:
            raise SystemExit(f"{label} lacks reciprocal reference consensus")
        pairs = ((valid[0], valid[1], "a-to-b"),
                 (valid[1], valid[0], "b-to-a"))
        validated = []
        for check, (left, right, direction) in zip(checks, pairs):
            seed = stable_seed(master_seed, owner_id, "reference-consensus", direction)
            validated.append(_validate_adjudication_check(
                check,
                reference_id=left["derivation_id"],
                seed=seed,
                owner_id=owner_id,
                call_phase=f"reference-consensus-{direction}",
                phases=phases,
                label=label,
                metadata={"candidate_id": right["derivation_id"]},
            ))
        expected_references = valid[:2] if all(
            check["status"] == "ok" and check["verdict"] == "correct"
            for check in validated
        ) else []
        expected_ids = [reference["derivation_id"] for reference in expected_references]
        if record.get("reference_ids") != expected_ids:
            raise SystemExit(f"{label} has drifted consensus reference IDs")
    if stored_references is not None and stored_references != expected_references:
        raise SystemExit(f"{label} has drifted or duplicated consensus references")
    return expected_references


def _validate_attempt_adjudications(
    checks: object,
    references: list[dict],
    *,
    owner_id: str,
    master_seed: int,
    phase: str,
    phases: dict[str, list[dict]],
    label: str,
    seed_namespace: str | None = None,
    seed_phase: str | None = None,
) -> tuple[str, str]:
    if not isinstance(checks, list) or len(checks) != len(references):
        raise SystemExit(f"{label} has incomplete attempt adjudications")
    effective_seed_phase = seed_phase or phase
    validated = []
    for index, (check, reference) in enumerate(zip(checks, references)):
        seed = stable_seed(
            master_seed, seed_namespace or owner_id, effective_seed_phase, index)
        audit_phase = f"{phase}-{index}"
        validated.append(_validate_adjudication_check(
            check,
            reference_id=reference["derivation_id"],
            seed=seed,
            owner_id=owner_id,
            call_phase=audit_phase,
            phases=phases,
            label=label,
            metadata={"audit_phase": audit_phase, "seed_phase": effective_seed_phase},
        ))
    return _adjudication_result(validated, len(references))


def _validate_distillation(
    expected: dict,
    record: dict,
    references: list[dict],
    *,
    master_seed: int,
    rt: RepoTools,
    phases: dict[str, list[dict]],
    label: str,
) -> None:
    owner_id = expected["item_id"]
    distill_seed = stable_seed(master_seed, owner_id, "distill")
    entry = record.get("entry")
    bounced = record.get("entry_bounced")
    if (entry is None) == (bounced is None):
        raise SystemExit(f"{label} must contain exactly one distillation outcome")
    if entry is not None:
        if not isinstance(entry, dict):
            raise SystemExit(f"{label} has a malformed admitted entry")
        raw = entry
        support_checks = entry.get("correction_support")
        _consume_call_phase(
            phases, owner_id=owner_id, phase="distill", seed=distill_seed,
            label=label, required=True)
    else:
        if (not isinstance(bounced, dict)
                or not isinstance(bounced.get("raw_entry"), dict)
                or not isinstance(bounced.get("reasons"), list)
                or not bounced["reasons"]
                or any(not isinstance(reason, str) or not reason
                       for reason in bounced["reasons"])):
            raise SystemExit(f"{label} has a malformed bounced distillation")
        raw = bounced["raw_entry"]
        support_checks = bounced.get("support_checks")
        failed_before_output = any(
            reason.startswith("distill_error:") for reason in bounced["reasons"])
        _consume_call_phase(
            phases, owner_id=owner_id, phase="distill", seed=distill_seed,
            label=label, required=not failed_before_output)

    if entry is None and failed_before_output and support_checks is not None:
        raise SystemExit(f"{label} has correction support after a failed distillation")
    if entry is None and not failed_before_output and support_checks is None:
        raise SystemExit(f"{label} omitted correction support after distillation")

    if support_checks is not None:
        if not isinstance(support_checks, list) or len(support_checks) != len(references):
            raise SystemExit(f"{label} has incomplete correction-support checks")
        for index, (check, reference) in enumerate(zip(support_checks, references)):
            seed = stable_seed(master_seed, owner_id, "correction-support", index)
            _validate_adjudication_check(
                check,
                reference_id=reference["derivation_id"],
                seed=seed,
                owner_id=owner_id,
                call_phase=f"correction-support-{index}",
                phases=phases,
                label=label,
            )
    elif entry is not None:
        raise SystemExit(f"{label} admitted an entry without correction support")

    if entry is None:
        return
    selected = sorted(references, key=lambda reference: reference["derivation_id"])[0]
    citation = validate_evidence(rt, entry)
    stored_citation = {key: entry.get(key) for key in ("file", "line", "quote")}
    expected_entry_id = _record_id(
        "entry", owner_id, entry.get("belief"), entry.get("correction"), citation)
    if (not isinstance(entry.get("belief"), str) or not entry["belief"].strip()
            or not isinstance(entry.get("correction"), str)
            or not entry["correction"].strip()
            or citation is None
            or stored_citation != citation
            or citation not in selected["evidence"]
            or entry.get("entry_id") != expected_entry_id
            or entry.get("origin_item_id") != expected["origin_item_id"]
            or not _json_identity_equal(entry.get("origin_round"), expected["origin_round"])
            or entry.get("chapter") != expected["chapter"]
            or entry.get("verdict") != record.get("verdict")
            or entry.get("reference_ids")
            != [reference["derivation_id"] for reference in references]
            or entry.get("evidence_class") != selected["evidence_class"]
            or any(check.get("status") != "ok" or check.get("verdict") != "correct"
                   for check in support_checks)):
        raise SystemExit(f"{label} has an invalid admitted entry or citation")


def _validate_completed_item(expected: dict, record: dict, note_sha256: str, *,
                             master_seed: int, rt: RepoTools) -> None:
    identity = ("schema_version", "item_id", "origin_item_id", "origin_round",
                "round", "kind", "split", "question", "qtype", "anchors", "chapter",
                "writer_sketch", "quiz_episode_id", "quiz_ordinal", "retest_of")
    if any(
        not _json_identity_equal(record.get(key), expected.get(key))
        for key in identity
    ):
        raise SystemExit(f"completed item drifted from question record: {expected['item_id']}")
    if record.get("input_note_sha256") != note_sha256:
        raise SystemExit(f"completed item used a different note: {expected['item_id']}")
    if expected["kind"] == "retest" and record.get("entry") is not None:
        raise SystemExit(f"retest illegally produced a note entry: {expected['item_id']}")
    label = f"completed item {expected['item_id']}"
    owner_id = expected["item_id"]
    phases = _artifact_call_phases(record, owner_id=owner_id, label=label)
    attempt_seed = stable_seed(master_seed, owner_id, "attempt")
    if (not isinstance(record.get("attempt"), str)
            or not _json_identity_equal(record.get("attempt_seed"), attempt_seed)):
        raise SystemExit(f"{label} has drifted attempt lineage")
    status = record.get("status")
    if status == "attempt_error":
        if (not isinstance(record.get("error"), str) or not record["error"]
                or record.get("entry") is not None
                or any(key in record for key in
                       ("derivations", "reference_consensus", "reference_ids",
                        "adjudications", "verdict", "delta", "entry_bounced"))):
            raise SystemExit(f"{label} has an invalid attempt-error state")
        _consume_call_phase(
            phases, owner_id=owner_id, phase="attempt", seed=attempt_seed,
            label=label, required=False)
        _finish_call_graph(phases, label=label)
        return
    if status != "ok":
        raise SystemExit(f"{label} has an invalid terminal status")
    if not record["attempt"].strip() or "error" in record:
        raise SystemExit(f"{label} has an invalid successful attempt")
    _consume_call_phase(
        phases, owner_id=owner_id, phase="attempt", seed=attempt_seed,
        label=label, required=True)
    derivations = _validate_derivations(
        record, owner_id=owner_id, master_seed=master_seed,
        ensemble=TRAIN_ENSEMBLE, rt=rt, phases=phases, label=label)
    references = _validate_reference_consensus(
        record, derivations, owner_id=owner_id, master_seed=master_seed,
        phases=phases, label=label)
    if not references:
        if (record.get("verdict") != "unresolved"
                or record.get("delta")
                != "independent derivations did not substantively agree"
                or record.get("entry") is not None
                or any(key in record for key in
                       ("adjudications", "entry_bounced"))):
            raise SystemExit(f"{label} has an invalid unresolved-consensus state")
        _finish_call_graph(phases, label=label)
        return
    verdict, delta = _validate_attempt_adjudications(
        record.get("adjudications"), references,
        owner_id=owner_id, master_seed=master_seed,
        phase="attempt-adjudication", phases=phases, label=label)
    if record.get("verdict") != verdict or record.get("delta") != delta:
        raise SystemExit(f"{label} reports a verdict inconsistent with its adjudications")
    if verdict in {"wrong", "partial"} and is_distillable_item(expected):
        _validate_distillation(
            expected, record, references, master_seed=master_seed,
            rt=rt, phases=phases, label=label)
    elif record.get("entry") is not None or "entry_bounced" in record:
        raise SystemExit(f"{label} contains an ineligible distillation")
    _finish_call_graph(phases, label=label)


def _validate_dev_reference(item: dict, record: dict, *, study_id: str, task: str,
                            master_seed: int, rt: RepoTools) -> None:
    expected_id = _record_id("dev-reference", study_id, task, item["item_id"])
    if any((
        not _json_identity_equal(record.get("schema_version"), SCHEMA_VERSION),
        record.get("reference_id") != expected_id,
        record.get("origin_item_id") != item["item_id"],
        not _json_identity_equal(record.get("origin_round"), item["origin_round"]),
        not _json_identity_equal(record.get("created_round"), item["origin_round"]),
        record.get("question") != item["question"],
        record.get("qtype") != item["qtype"],
        record.get("anchors") != item["anchors"],
        record.get("chapter") != item["chapter"],
        any(key in record for key in ("attempt", "attempts", "entry", "input_note_sha256")),
    )):
        raise SystemExit(f"dev reference has invalid blind lineage: {expected_id}")
    label = f"dev reference {expected_id}"
    stored_references = record.get("references")
    if not isinstance(stored_references, list):
        raise SystemExit(f"dev reference is incomplete: {expected_id}")
    phases = _artifact_call_phases(record, owner_id=expected_id, label=label)
    derivations = _validate_derivations(
        record, owner_id=expected_id, master_seed=master_seed,
        ensemble=DEV_ENSEMBLE, rt=rt, phases=phases, label=label)
    references = _validate_reference_consensus(
        record, derivations, owner_id=expected_id, master_seed=master_seed,
        phases=phases, label=label, stored_references=stored_references)
    _finish_call_graph(phases, label=label)
    status = record.get("status")
    if ((status == "ok") != (len(references) == DEV_ENSEMBLE)
            or status not in {"ok", "unresolved"}):
        raise SystemExit(f"dev reference consensus drifted: {expected_id}")


def _validate_dev_exam(item: dict, record: dict, reference: dict, *,
                       note_sha256: str, master_seed: int, exam_round: int) -> None:
    if (not isinstance(reference, dict)
            or not isinstance(reference.get("reference_id"), str)
            or not isinstance(reference.get("references"), list)):
        raise SystemExit("dev exam has a malformed reference artifact")
    exam_id = _record_id("dev-exam", item["item_id"], exam_round)
    paired_seed = stable_seed(master_seed, reference["reference_id"], "paired-attempt")
    attempts = record.get("attempts", {})
    references = reference.get("references", [])
    adjudication_seed_phase = "dev-paired-adjudication"
    paired_adjudication_seeds = [
        stable_seed(
            master_seed, reference["reference_id"], adjudication_seed_phase, index)
        for index in range(len(references))
    ]
    if any((
        not _json_identity_equal(record.get("schema_version"), SCHEMA_VERSION),
        record.get("origin_item_id") != item["item_id"],
        record.get("item_id") != exam_id,
        not _json_identity_equal(record.get("round"), exam_round),
        not _json_identity_equal(record.get("exam_round"), exam_round),
        not _json_identity_equal(record.get("origin_round"), item["origin_round"]),
        record.get("kind") != "dev_exam",
        record.get("split") != "dev",
        record.get("input_note_sha256") != note_sha256,
        record.get("reference_id") != reference["reference_id"],
        record.get("reference_sha256") != sha256_json(reference),
        record.get("entry") is not None,
        record.get("question") != item["question"],
        record.get("qtype") != item["qtype"],
        record.get("anchors") != item["anchors"],
        record.get("chapter") != item["chapter"],
        record.get("writer_sketch") != item.get("writer_sketch"),
        record.get("quiz_episode_id") != item.get("quiz_episode_id"),
        not _json_identity_equal(
            record.get("quiz_ordinal"), item.get("quiz_ordinal")),
        record.get("retest_of") != item.get("retest_of"),
    )):
        raise SystemExit(f"dev exam record has invalid lineage: {exam_id}")
    label = f"dev exam {exam_id}"
    phases = _artifact_call_phases(record, owner_id=exam_id, label=label)
    status = record.get("status")
    if status == "reference_unresolved":
        if (reference.get("status") != "unresolved" or attempts or phases
                or record.get("verdicts") != {
                    "with_note": "unresolved", "bare": "unresolved"}
                or any(key in record for key in
                       ("attempt_protocol", "reference_ids", "deltas",
                        "adjudications", "adjudication_protocol"))):
            raise SystemExit(f"dev exam unresolved-reference record drifted: {exam_id}")
    elif status in {"ok", "attempt_error"} and reference.get("status") == "ok":
        reference_ids = [candidate.get("derivation_id") for candidate in references
                         if isinstance(candidate, dict)]
        if (len(references) != DEV_ENSEMBLE
                or len(reference_ids) != DEV_ENSEMBLE
                or len(set(reference_ids)) != DEV_ENSEMBLE):
            raise SystemExit(f"dev exam reference ensemble drifted: {exam_id}")
        if not isinstance(attempts, dict) or set(attempts) != {"with_note", "bare"}:
            raise SystemExit(f"dev exam record lacks paired attempts: {exam_id}")
        protocol = record.get("attempt_protocol")
        if (not isinstance(protocol, dict)
                or set(protocol) != {
                    "signature", "paired_seed", "only_manipulated_field"}
                or protocol.get("signature") != "note, question -> answer"
                or not _json_identity_equal(protocol.get("paired_seed"), paired_seed)
                or protocol.get("only_manipulated_field") != "note"):
            raise SystemExit(f"dev exam arms used different protocols: {exam_id}")
        for arm in ("with_note", "bare"):
            attempt = attempts[arm]
            if (not isinstance(attempt, dict)
                    or not isinstance(attempt.get("answer"), str)
                    or not _json_identity_equal(attempt.get("seed"), paired_seed)
                    or (attempt.get("error") is not None
                        and (not isinstance(attempt.get("error"), str)
                             or not attempt["error"]))):
                raise SystemExit(f"dev exam arm has drifted attempt lineage: {exam_id}")
            _consume_call_phase(
                phases,
                owner_id=exam_id,
                phase=f"dev-attempt-{arm}",
                seed=paired_seed,
                label=label,
                required=attempt["error"] is None,
            )
        has_attempt_error = any(attempt["error"] is not None
                                for attempt in attempts.values())
        if (status == "attempt_error") != has_attempt_error:
            raise SystemExit(f"dev exam attempt status is inconsistent: {exam_id}")
        if status == "attempt_error":
            if any(key in record for key in
                   ("reference_ids", "verdicts", "deltas", "adjudications",
                    "adjudication_protocol")):
                raise SystemExit(f"dev exam adjudicated failed attempts: {exam_id}")
            _finish_call_graph(phases, label=label)
            return
        if status == "ok":
            protocol = record.get("adjudication_protocol")
            expected_audit_phases = {
                arm: f"dev-adjudication-{arm}" for arm in ("with_note", "bare")
            }
            adjudications = record.get("adjudications")
            if (not isinstance(protocol, dict)
                    or set(protocol) != {
                        "signature", "seed_namespace", "seed_phase", "paired_seeds",
                        "only_manipulated_field", "audit_phases"}
                    or protocol.get("signature") != "AdjudicateSig"
                    or protocol.get("seed_namespace") != reference["reference_id"]
                    or protocol.get("seed_phase") != adjudication_seed_phase
                    or not isinstance(protocol.get("paired_seeds"), list)
                    or len(protocol["paired_seeds"]) != len(paired_adjudication_seeds)
                    or any(not _json_identity_equal(observed, expected)
                           for observed, expected in zip(
                               protocol["paired_seeds"], paired_adjudication_seeds))
                    or protocol.get("only_manipulated_field") != "attempt"
                    or protocol.get("audit_phases") != expected_audit_phases
                    or not isinstance(adjudications, dict)
                    or set(adjudications) != {"with_note", "bare"}):
                raise SystemExit(f"dev exam adjudication protocol drifted: {exam_id}")
            if record.get("reference_ids") != reference_ids:
                raise SystemExit(f"dev exam reference IDs drifted: {exam_id}")
            expected_verdicts, expected_deltas = {}, {}
            for arm in ("with_note", "bare"):
                verdict, delta = _validate_attempt_adjudications(
                    adjudications[arm],
                    references,
                    owner_id=exam_id,
                    master_seed=master_seed,
                    phase=f"dev-adjudication-{arm}",
                    phases=phases,
                    label=label,
                    seed_namespace=reference["reference_id"],
                    seed_phase=adjudication_seed_phase,
                )
                expected_verdicts[arm] = verdict
                expected_deltas[arm] = delta
            if (record.get("verdicts") != expected_verdicts
                    or record.get("deltas") != expected_deltas):
                raise SystemExit(
                    f"dev exam reported verdicts drifted from adjudications: {exam_id}"
                )
            _finish_call_graph(phases, label=label)
    else:
        raise SystemExit(f"dev exam has invalid status/reference pairing: {exam_id}")


def _load_prior_rounds(sdir: Path, round_number: int, *, study_id: str, task: str,
                       rt: RepoTools,
                       ) -> tuple[list[list[dict]], list[dict], list[dict], list[dict]]:
    question_sets, items, calls, manifests = [], [], [], []
    previous_note_sha256 = sha256_text("")
    try:
        task_manifest = load_json_artifact(sdir / "manifest.json")
        master_seed = task_manifest["master_seed"]
        task_config = task_manifest["config"]
        protocol_questions = task_config["questions_per_chapter"]
        protocol_smoke = task_config["smoke"]
        protocol_chapters = task_config["chapters_per_round"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"cannot validate prior-round task protocol: {error}") from error
    syllabus = chapters(rt)
    if (type(master_seed) is not int
            or type(protocol_questions) is not int or protocol_questions < 2
            or type(protocol_smoke) is not bool
            or type(protocol_chapters) is not int or protocol_chapters < 1
            or not syllabus):
        raise SystemExit("cannot validate prior-round task protocol: invalid config")
    chapters_per_round = min(protocol_chapters, len(syllabus))
    for prior_round in range(1, round_number):
        rdir = sdir / f"r{prior_round}"
        start = (prior_round - 1) * chapters_per_round
        planned_chapters = [
            syllabus[(start + index) % len(syllabus)]
            for index in range(chapters_per_round)
        ]
        expected_quiz_chapters = planned_chapters[:1] if protocol_smoke \
            else planned_chapters
        required = {
            "round_manifest": rdir / "manifest.json",
            "questions": rdir / "questions.jsonl",
            "items": rdir / "items.jsonl",
            "freshness": rdir / "freshness.json",
            "dev": rdir / "dev-exam.jsonl",
            "usage": rdir / "usage.jsonl",
            "cumulative_usage": rdir / "cumulative-usage.jsonl",
            "summary": rdir / "summary.json",
            "note_manifest": sdir / "notes" / f"note-r{prior_round}.manifest.json",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise SystemExit("cannot skip or partially inherit a prior round; missing: "
                             + ", ".join(missing))
        questions = _read_jsonl(required["questions"])
        round_items = _read_jsonl(required["items"])
        dev_records = _read_jsonl(required["dev"])
        for subdir, records in (("items", round_items), ("dev-exam", dev_records)):
            for record in records:
                _validate_artifact_environment(
                    sdir,
                    task_manifest,
                    record,
                    label=f"prior {subdir} artifact",
                )
                record_id = record.get("item_id")
                if not _safe_artifact_id(record_id):
                    raise SystemExit(f"prior aggregate has an unsafe item ID: {record_id!r}")
                path = rdir / subdir / f"{record_id}.json"
                if not path.exists() or load_json_artifact(path) != record:
                    raise SystemExit(f"prior aggregate drifted from item artifact: {path}")
        quiz_episodes = []
        episode_paths = sorted((rdir / "quiz-episodes").glob("*.json"))
        for path in episode_paths:
            episode_id = path.stem
            if not _safe_artifact_id(episode_id):
                raise SystemExit(f"prior round has an unsafe episode ID: {episode_id!r}")
            episode = load_json_artifact(path)
            if episode.get("owner_id") != episode_id:
                raise SystemExit(f"prior quiz episode filename drifted: {path}")
            _validate_artifact_environment(
                sdir, task_manifest, episode, label="prior quiz episode"
            )
            quiz_episodes.append(episode)
        protocol_args = argparse.Namespace(
            study_id=study_id,
            task=task,
            round=prior_round,
            seed=master_seed,
            smoke=protocol_smoke,
            questions=protocol_questions,
        )
        question_errors = [
            (item.get("item_id"), _question_record_error(item, rt)) for item in questions
        ]
        if any(error for _, error in question_errors):
            raise SystemExit(
                f"prior round has invalid question records: {question_errors}"
            )
        _validate_question_provenance(
            protocol_args,
            questions,
            quiz_episodes,
            rt,
            expected_chapters=expected_quiz_chapters,
            expected_question_count=protocol_questions,
        )
        expected_train = {item["item_id"]: item for item in questions
                          if item.get("split") == "train"}
        observed_train = {record.get("item_id"): record for record in round_items}
        if (len(expected_train) != len([item for item in questions
                                       if item.get("split") == "train"])
                or len(observed_train) != len(round_items)
                or set(observed_train) != set(expected_train)):
            raise SystemExit(f"prior round {prior_round} train aggregate is incomplete")
        for item_id, expected in expected_train.items():
            _validate_completed_item(
                expected,
                observed_train[item_id],
                previous_note_sha256,
                master_seed=master_seed,
                rt=rt,
            )

        all_dev_references = []
        for path in sorted((sdir / "dev-references").glob("*.json")):
            reference = load_json_artifact(path)
            _validate_artifact_environment(
                sdir, task_manifest, reference, label="prior dev reference"
            )
            all_dev_references.append(reference)
        dev_references = [
            reference for reference in all_dev_references
            if _json_identity_equal(reference.get("created_round"), prior_round)
        ]
        round_calls = _read_jsonl(required["usage"])
        expected_round_calls = [
            call
            for artifact in quiz_episodes + round_items + dev_records + dev_references
            for call in artifact.get("calls", [])
        ]
        if not artifact_usage_consistent(
                quiz_episodes + round_items + dev_records + dev_references):
            raise SystemExit(f"prior round artifact usage drifted in round {prior_round}")
        round_audit = usage_ledger_audit(expected_round_calls, round_calls)
        if not round_audit["complete"]:
            raise SystemExit(f"prior round usage is invalid in round {prior_round}: "
                             f"{round_audit['errors']}")
        question_sets.append(questions)
        items += round_items
        calls += round_calls
        cumulative = _read_jsonl(required["cumulative_usage"])
        audit = usage_ledger_audit(calls, cumulative)
        if not audit["complete"]:
            raise SystemExit(f"prior cumulative usage is invalid in round {prior_round}: "
                             f"{audit['errors']}")
        manifest = load_json_artifact(required["note_manifest"])
        summary = load_json_artifact(required["summary"])
        round_manifest = load_json_artifact(required["round_manifest"])
        _validate_artifact_environment(
            sdir,
            task_manifest,
            {"environment_snapshot": round_manifest.get(
                "initial_environment_snapshot")}
            if isinstance(round_manifest, dict)
            else {},
            label="prior round manifest",
        )
        freshness = load_json_artifact(required["freshness"])
        _validate_construction_artifacts(sdir, manifest)
        prior_readiness = manifest.get("automated_readiness")
        if (manifest.get("study_id") != study_id or manifest.get("task") != task
                or not _json_identity_equal(manifest.get("round"), prior_round)
                or manifest.get("input_note_sha256") != previous_note_sha256
                or manifest.get("round_usage") != usage_totals(round_calls)
                or manifest.get("cumulative_usage") != usage_totals(calls)
                or manifest.get("round_usage_by_phase") != usage_by_phase(round_calls)
                or manifest.get("cumulative_usage_by_phase") != usage_by_phase(calls)
                or summary.get("round_usage") != usage_totals(round_calls)
                or summary.get("cumulative_usage") != usage_totals(calls)
                or summary.get("round_usage_by_phase") != usage_by_phase(round_calls)
                or summary.get("cumulative_usage_by_phase") != usage_by_phase(calls)
                or summary.get("round_usage_audit") != round_audit
                or summary.get("cumulative_usage_audit") != audit
                or not isinstance(prior_readiness, dict) or not prior_readiness
                or not all(value is True for value in prior_readiness.values())
                or manifest.get("automated_claim_ready") is not True
                or summary.get("automated_readiness") != prior_readiness
                or summary.get("automated_claim_ready")
                != manifest.get("automated_claim_ready")
                or summary.get("note_sha256") != manifest.get("note_sha256")
                or summary.get("freshness") != freshness
                or freshness.get("audit_complete") is not True
                or freshness.get("fresh") is not True
                or not freshness_sources_complete(rdir, freshness)
                or round_manifest.get("study_id") != study_id
                or round_manifest.get("task") != task
                or round_manifest.get("round") != prior_round
                or round_manifest.get("chapters") != planned_chapters
                or round_manifest.get("master_seed") != master_seed
                or round_manifest.get("input_note_sha256") != previous_note_sha256
                or round_manifest.get("task_manifest_sha256")
                != sha256_json(task_manifest)):
            raise SystemExit(
                f"prior note manifest has invalid lineage: {required['note_manifest']}")
        relative = PurePosixPath(str(manifest.get("note_path", "")))
        note_path = required["note_manifest"].parent.joinpath(*relative.parts)
        if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                or not note_path.is_file()
                or sha256_file(note_path) != manifest.get("note_sha256")):
            raise SystemExit(f"prior note artifact is missing or changed: {note_path}")

        cumulative_dev = collect_dev_questions(question_sets)
        references_by_origin = {
            reference.get("origin_item_id"): reference
            for reference in all_dev_references
        }
        exams_by_origin = {record.get("origin_item_id"): record for record in dev_records}
        if (len(references_by_origin) != len(all_dev_references)
                or len(exams_by_origin) != len(dev_records)
                or set(exams_by_origin) != {item["item_id"] for item in cumulative_dev}):
            raise SystemExit(f"prior round {prior_round} dev aggregate is incomplete")
        for item in cumulative_dev:
            reference = references_by_origin.get(item["item_id"])
            if reference is None:
                raise SystemExit(f"prior dev reference is missing: {item['item_id']}")
            _validate_dev_reference(
                item,
                reference,
                study_id=study_id,
                task=task,
                master_seed=master_seed,
                rt=rt,
            )
            _validate_dev_exam(
                item,
                exams_by_origin[item["item_id"]],
                reference,
                note_sha256=manifest["note_sha256"],
                master_seed=master_seed,
                exam_round=prior_round,
            )
        previous_note_sha256 = manifest["note_sha256"]
        manifests.append(manifest)
    return question_sets, items, calls, manifests


def _load_input_note(sdir: Path, round_number: int, entries: list[dict]) -> str:
    if round_number == 1:
        return ""
    manifest_path = sdir / "notes" / f"note-r{round_number - 1}.manifest.json"
    manifest = load_json_artifact(manifest_path)
    relative = PurePosixPath(manifest.get("note_path", ""))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SystemExit(f"unsafe note path in {manifest_path}")
    note_path = manifest_path.parent.joinpath(*relative.parts)
    try:
        note_text = read_artifact_bytes(note_path).decode("utf-8")
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"prior note artifact is unsafe or unreadable: {note_path}") from error
    if sha256_text(note_text) != manifest.get("note_sha256"):
        raise SystemExit(f"prior note bytes no longer match {manifest_path}")
    expected_entries = [entry["entry_id"] for entry in entries]
    if manifest.get("entry_ids") != expected_entries:
        raise SystemExit(f"prior note entry lineage no longer matches {manifest_path}")
    return note_text


def _prepare_questions(args, rt: RepoTools, tools_fns, urls: list[str], rdir: Path,
                       todo: list[str], prior_questions: list[list[dict]],
                       prior_items: list[dict], *, task_manifest: dict,
                       launch_environment: dict[str, object]) \
        -> tuple[list[dict], list[dict]]:
    qfile = rdir / "questions.jsonl"
    quiz_dir = rdir / "quiz-episodes"
    count = 3 if args.smoke else args.questions
    chapters_to_run = todo[:1] if args.smoke else todo
    if qfile.is_symlink() or quiz_dir.is_symlink():
        raise SystemExit("question artifacts must not traverse symlinks")
    expected_episode_ids = {
        _record_id("quiz", args.study_id, args.task, args.round, chapter)
        for chapter in chapters_to_run
    }
    episode_paths = sorted(quiz_dir.glob("*.json"))
    if any(path.stem not in expected_episode_ids for path in episode_paths):
        raise SystemExit("quiz-episode directory does not match the exact chapter plan")
    if qfile.exists():
        records = _read_jsonl(qfile)
        episodes = [load_json_artifact(path) for path in episode_paths]
        if not episodes:
            raise SystemExit("questions exist without immutable quiz episodes")
        errors = [(item.get("item_id"), _question_record_error(item, rt)) for item in records]
        errors = [(item_id, error) for item_id, error in errors if error]
        if errors:
            raise SystemExit(f"invalid immutable question records: {errors}")
        for path, episode in zip(episode_paths, episodes):
            if episode.get("owner_id") != path.stem:
                raise SystemExit(f"quiz episode filename drifted: {path}")
            _validate_artifact_environment(
                rdir.parent, task_manifest, episode, label="quiz episode"
            )
        _validate_question_provenance(
            args,
            records,
            episodes,
            rt,
            expected_chapters=chapters_to_run,
            expected_question_count=count,
        )
        return records, episodes

    specs = []
    episodes_by_owner = {}
    for chapter in chapters_to_run:
        owner_id = _record_id("quiz", args.study_id, args.task, args.round, chapter)
        path = quiz_dir / f"{owner_id}.json"
        if path.exists():
            episode = load_json_artifact(path)
            _validate_artifact_environment(
                rdir.parent, task_manifest, episode, label="quiz episode"
            )
            episodes_by_owner[owner_id] = episode
        else:
            seed = stable_seed(args.seed, args.study_id, args.task, args.round, "quiz", chapter)
            specs.append((chapter, owner_id, path, seed))

    def one(spec):
        chapter, owner_id, path, seed = spec
        episode = run_quiz(chapter, tools_fns, _server_url(urls, args.seed, owner_id), count,
                           seed=seed, owner_id=owner_id)
        episode = _bind_launch_environment(episode, launch_environment)
        _validate_artifact_environment(
            rdir.parent, task_manifest, episode, label="quiz episode"
        )
        write_immutable_json(path, episode)
        return owner_id, episode

    with ThreadPoolExecutor(max_workers=max(1, len(specs))) as pool:
        for owner_id, episode in pool.map(one, specs):
            episodes_by_owner[owner_id] = episode

    seen = [item["question"] for questions in prior_questions for item in questions
            if item.get("kind") == "quiz"]
    records, rejected, protocol_errors = [], [], []
    accepted_by_chapter: dict[str, list[dict]] = {}
    for chapter in chapters_to_run:
        owner_id = _record_id("quiz", args.study_id, args.task, args.round, chapter)
        episode = episodes_by_owner[owner_id]
        if episode["status"] != "ok":
            raise SystemExit(f"quiz episode failed for {chapter}; inspect {owner_id}.json")
        raw_questions = episode.get("questions")
        if not isinstance(raw_questions, list) or len(raw_questions) != count:
            raw_count = len(raw_questions) if isinstance(raw_questions, list) else 0
            protocol_errors.append(
                f"{chapter} returned {raw_count} "
                f"raw questions; exactly {count} were requested"
            )
            raw_questions = raw_questions if isinstance(raw_questions, list) else []
        accepted = []
        for ordinal, raw in enumerate(raw_questions):
            if not isinstance(raw, dict):
                rejected.append({"chapter": chapter, "ordinal": ordinal,
                                 "raw_question": raw,
                                 "reasons": ["question payload is not an object"]})
                continue
            anchors = validate_anchors(rt, raw.get("anchors"))
            question = raw.get("question")
            reasons = []
            if not isinstance(question, str) or not question.strip():
                reasons.append("empty question")
            if anchors is None:
                reasons.append("one or more anchors are not exact corpus files")
            if not reasons and dedup(question, seen):
                reasons.append("near-duplicate question")
            if reasons:
                rejected.append({"chapter": chapter, "ordinal": ordinal,
                                 "raw_question": raw, "reasons": reasons})
                continue
            seen.append(question)
            item_id = _record_id("question", args.study_id, args.task, args.round,
                                 chapter, ordinal, question)
            accepted.append({
                "schema_version": SCHEMA_VERSION,
                "item_id": item_id,
                "origin_item_id": item_id,
                "origin_round": args.round,
                "round": args.round,
                "kind": "quiz",
                "split": "train",
                "question": question,
                "qtype": raw["qtype"],
                "anchors": anchors,
                "chapter": chapter,
                "writer_sketch": raw.get("writer_sketch", ""),
                "quiz_episode_id": owner_id,
                "quiz_ordinal": ordinal,
            })
        if len(accepted) != count:
            protocol_errors.append(
                f"{chapter} produced {len(accepted)} valid questions; exactly {count} "
                "are required for a fixed train/dev protocol"
            )
        accepted_by_chapter[chapter] = accepted

    if rejected:
        write_immutable_text(rdir / "rejected-questions.jsonl", _jsonl(rejected))
    if protocol_errors:
        raise SystemExit("quiz protocol failed: " + "; ".join(protocol_errors))

    for chapter in chapters_to_run:
        accepted = accepted_by_chapter[chapter]
        dev_index = stable_seed(args.seed, args.study_id, args.task, args.round,
                                "dev-split", chapter) % len(accepted)
        accepted[dev_index]["split"] = "dev"
        records += accepted

    candidates = sorted((item for item in prior_items if eligible_retest(item)),
                        key=lambda item: item["item_id"])
    if args.round > 1 and candidates:
        n_retest = max(1, int(len(records) * RETEST_FRAC))
        rng = random.Random(stable_seed(args.seed, args.study_id, args.task,
                                        args.round, "retest-sample"))
        selected = rng.sample(candidates, min(n_retest, len(candidates)))
        records += [make_retest_item(item, task=args.task, study_id=args.study_id,
                                     round_number=args.round) for item in selected]
    records.sort(key=lambda item: item["item_id"])
    errors = [(item["item_id"], _question_record_error(item, rt)) for item in records]
    if any(error for _, error in errors):
        raise SystemExit(f"generated invalid question records: {errors}")
    episodes = [episodes_by_owner[key] for key in sorted(episodes_by_owner)]
    _validate_question_provenance(
        args,
        records,
        episodes,
        rt,
        expected_chapters=chapters_to_run,
        expected_question_count=count,
    )
    write_immutable_text(qfile, _jsonl(records))
    return records, episodes


def _write_note_bytes(args, sdir: Path, note_text: str) -> tuple[str, Path]:
    note_sha256 = sha256_text(note_text)
    notes_dir = sdir / "notes"
    relative = Path("by-sha256") / f"{note_sha256}.md"
    write_immutable_text(notes_dir / relative, note_text)
    write_immutable_text(notes_dir / f"note-r{args.round}.md", note_text)
    return note_sha256, relative


def _has_symlink_component(root: Path, relative: PurePosixPath) -> bool:
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            return True
    return False


def _construction_artifacts(sdir: Path, round_number: int) -> dict[str, dict[str, object]]:
    """Hash every artifact on which the cumulative note/readiness depends."""
    paths = {sdir / "manifest.json"}
    for prior_round in range(1, round_number + 1):
        paths.update(path for path in (sdir / f"r{prior_round}").rglob("*") if path.is_file())
        note_alias = sdir / "notes" / f"note-r{prior_round}.md"
        if note_alias.is_file():
            paths.add(note_alias)
        prior_manifest = sdir / "notes" / f"note-r{prior_round}.manifest.json"
        if prior_manifest.is_file():
            paths.add(prior_manifest)
            value = load_json_artifact(prior_manifest)
            relative = PurePosixPath(str(value.get("note_path", "")))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise SystemExit(f"unsafe prior note artifact path: {prior_manifest}")
            paths.add(prior_manifest.parent.joinpath(*relative.parts))
    current_alias = sdir / "notes" / f"note-r{round_number}.md"
    if current_alias.is_file():
        paths.add(current_alias)
        paths.add(sdir / "notes" / "by-sha256" / f"{sha256_file(current_alias)}.md")
    for path in (sdir / "dev-references").glob("*.json"):
        reference = load_json_artifact(path)
        if reference.get("created_round", round_number + 1) <= round_number:
            paths.add(path)
    task_manifest = load_json_artifact(sdir / "manifest.json")
    protocol = task_manifest.get("human_audit_protocol")
    if isinstance(protocol, dict):
        relative = PurePosixPath(str(protocol.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise SystemExit("task manifest contains an unsafe audit-protocol path")
        paths.add(sdir.joinpath(*relative.parts))

    inventory = {}
    study_root = sdir.resolve(strict=True)
    for path in sorted(paths):
        relative_path = path.relative_to(sdir)
        relative = PurePosixPath(relative_path.as_posix())
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise SystemExit(f"construction artifact is missing: {path}") from error
        if (_has_symlink_component(sdir, relative)
                or not resolved.is_relative_to(study_root) or not resolved.is_file()):
            raise SystemExit(f"construction artifact is missing or unsafe: {path}")
        inventory[str(relative)] = {
            "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size,
        }
    return inventory


def _validate_construction_artifacts(sdir: Path, manifest: dict) -> Path:
    """Re-hash the complete recorded dependency inventory and exact note bytes."""
    inventory = manifest.get("construction_artifacts")
    if (not isinstance(inventory, dict) or not inventory
            or manifest.get("construction_artifacts_sha256") != sha256_json(inventory)):
        raise SystemExit("construction artifact inventory is missing or has drifted")
    round_number = manifest.get("round")
    if not isinstance(round_number, int) or isinstance(round_number, bool) \
            or round_number < 1:
        raise SystemExit("construction manifest has no valid round")
    expected_inventory = _construction_artifacts(sdir, round_number)
    expected_inventory.pop(f"notes/note-r{round_number}.manifest.json", None)
    if inventory != expected_inventory:
        raise SystemExit("construction artifact inventory is not complete and exact")
    try:
        study_root = sdir.resolve(strict=True)
    except OSError as error:
        raise SystemExit(f"construction study directory is unavailable: {sdir}") from error
    for raw_relative, recorded in inventory.items():
        relative = _canonical_file(raw_relative)
        if relative is None or not isinstance(recorded, dict) \
                or set(recorded) != {"sha256", "bytes"}:
            raise SystemExit(f"invalid construction artifact record: {raw_relative!r}")
        digest = recorded.get("sha256")
        size = recorded.get("bytes")
        if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise SystemExit(f"invalid construction artifact metadata: {raw_relative}")
        path = sdir.joinpath(*PurePosixPath(relative).parts)
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise SystemExit(f"construction artifact is missing: {path}") from error
        if (_has_symlink_component(sdir, PurePosixPath(relative))
                or not resolved.is_relative_to(study_root)
                or not resolved.is_file() or resolved.stat().st_size != size
                or sha256_file(resolved) != digest):
            raise SystemExit(f"construction artifact is missing or changed: {path}")

    note_relative = _canonical_file(manifest.get("note_path"))
    note_digest = manifest.get("note_sha256")
    if note_relative is None or not isinstance(note_digest, str) \
            or not re.fullmatch(r"[0-9a-f]{64}", note_digest):
        raise SystemExit("construction manifest has invalid note identity")
    note_path = sdir / "notes"
    note_path = note_path.joinpath(*PurePosixPath(note_relative).parts)
    try:
        resolved_note = note_path.resolve(strict=True)
    except OSError as error:
        raise SystemExit(f"construction note is missing: {note_path}") from error
    note_from_study = PurePosixPath("notes") / PurePosixPath(note_relative)
    if (_has_symlink_component(sdir, note_from_study)
            or not resolved_note.is_relative_to(study_root)
            or not resolved_note.is_file() or sha256_file(resolved_note) != note_digest):
        raise SystemExit(f"construction note is missing or changed: {note_path}")
    return note_path


def _write_note(args, sdir: Path, note_text: str, entries: list[dict], *,
                input_note_sha256: str, round_calls: list[dict],
                cumulative_calls: list[dict], round_construction_calls: list[dict],
                cumulative_construction_calls: list[dict], corpus_commit: str,
                automated_claim_ready: bool,
                automated_readiness: dict[str, bool]) -> dict:
    notes_dir = sdir / "notes"
    note_sha256, relative = _write_note_bytes(args, sdir, note_text)
    artifacts = _construction_artifacts(sdir, args.round)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": args.study_id,
        "task": args.task,
        "round": args.round,
        "corpus_commit": corpus_commit,
        # Automated self-judgment is not a publication/confirmatory audit. This
        # immutable construction manifest therefore cannot claim readiness by
        # itself; a separate pre-registered, blinded human audit is required.
        "claim_ready": False,
        "publication_claim_ready": False,
        "confirmatory_claim_ready": False,
        "automated_claim_ready": automated_claim_ready,
        "automated_readiness": automated_readiness,
        "human_audit": {
            "required": True,
            "status": "not_performed",
            "protocol": "pre-registered blinded verdict and evidence audit",
        },
        "note_sha256": note_sha256,
        "note_path": str(relative),
        "input_note_sha256": input_note_sha256,
        "entry_ids": [entry["entry_id"] for entry in entries],
        "entries": entries,
        "construction_artifacts": artifacts,
        "construction_artifacts_sha256": sha256_json(artifacts),
        "usage": usage_totals(cumulative_calls),
        "round_usage": usage_totals(round_calls),
        "cumulative_usage": usage_totals(cumulative_calls),
        "round_usage_by_phase": usage_by_phase(round_calls),
        "cumulative_usage_by_phase": usage_by_phase(cumulative_calls),
        "round_construction_usage": usage_totals(round_construction_calls),
        "cumulative_construction_usage": usage_totals(cumulative_construction_calls),
        "round_construction_usage_by_phase": usage_by_phase(round_construction_calls),
        "cumulative_construction_usage_by_phase": usage_by_phase(
            cumulative_construction_calls),
        "note_chars": len(note_text),
    }
    write_immutable_json(notes_dir / f"note-r{args.round}.manifest.json", manifest)
    return manifest


def _construction_artifact_bytes(sdir: Path, construction: dict) -> dict[str, bytes]:
    """Load construction bytes after ``_validate_construction_artifacts`` passes."""

    loaded = {}
    for raw_relative in construction["construction_artifacts"]:
        relative = PurePosixPath(raw_relative)
        try:
            loaded[raw_relative] = read_artifact_bytes(
                sdir.joinpath(*relative.parts))
        except (OSError, ValueError) as error:
            raise SystemExit(
                f"cannot load construction artifact for human audit: {raw_relative}"
            ) from error
    return loaded


def _human_audit_matches_binding(
    audit: object,
    *,
    required: dict[str, object],
    round_number: int,
) -> bool:
    return (
        isinstance(audit, dict)
        and type(audit.get("schema_version")) is int
        and audit["schema_version"] == HUMAN_AUDIT_SCHEMA_VERSION
        and type(audit.get("round")) is int
        and audit["round"] == round_number
        and all(audit.get(key) == value for key, value in required.items())
        and type(audit.get("blinding_preserved")) is bool
        and type(audit.get("reviewer_independent")) is bool
        and audit.get("decision") in {"pass", "fail"}
    )


def _auditor_id_is_valid(audit: dict) -> bool:
    try:
        validate_id(audit.get("auditor_id"), "auditor ID")
    except (TypeError, ValueError):
        return False
    return True


def _validate_promoted_auditor_identity(promoted: dict, audit: dict) -> None:
    """Require the produced manifest to preserve the exact validated auditor ID."""

    human = promoted.get("human_audit") if isinstance(promoted, dict) else None
    if (
        not _auditor_id_is_valid(audit)
        or not isinstance(human, dict)
        or not _auditor_id_is_valid(human)
        or human.get("auditor_id") != audit.get("auditor_id")
    ):
        raise SystemExit("promoted human audit has invalid auditor identity")


def _reject_prior_failed_human_audit(
    sdir: Path,
    construction: dict,
    artifacts: dict[str, bytes],
    *,
    required: dict[str, object],
    round_number: int,
) -> None:
    """Permanently bind a valid failed audit to its exact construction.

    Invalid and unrelated archive files have no scientific meaning and are
    ignored.  Filesystem indirection is different: any symlink in the archive
    path fails closed so a relevant failure cannot be hidden during promotion.
    """

    failed_relative = PurePosixPath("notes/audits/failed/by-sha256")
    if _has_symlink_component(sdir, failed_relative):
        raise SystemExit("failed human-audit archive must not traverse a symlink")
    failed_root = sdir.joinpath(*failed_relative.parts)
    if not failed_root.exists():
        return
    try:
        study_root = sdir.resolve(strict=True)
        resolved_root = failed_root.resolve(strict=True)
    except OSError as error:
        raise SystemExit("failed human-audit archive is unavailable") from error
    if not resolved_root.is_relative_to(study_root) or not resolved_root.is_dir():
        raise SystemExit("failed human-audit archive is unsafe")

    try:
        archived_paths = sorted(failed_root.iterdir())
    except OSError as error:
        raise SystemExit("cannot inspect the failed human-audit archive") from error
    for path in archived_paths:
        if path.is_symlink():
            raise SystemExit("failed human-audit archive must not contain symlinks")
        if not path.is_file() or path.suffix != ".json" \
                or re.fullmatch(r"[0-9a-f]{64}", path.stem) is None:
            continue
        try:
            archived_bytes = read_artifact_bytes(path)
        except (OSError, ValueError) as error:
            raise SystemExit(f"cannot inspect failed human audit: {path.name}") from error
        if sha256_bytes(archived_bytes) != path.stem:
            continue
        try:
            archived = strict_json_loads(
                archived_bytes, label=f"archived failed human audit {path.name}")
        except ValueError:
            continue
        if (not _human_audit_matches_binding(
                archived, required=required, round_number=round_number)
                or archived.get("decision") != "fail"
                or not _auditor_id_is_valid(archived)):
            continue
        try:
            validation = validate_human_audit_result(
                archived, construction, artifacts)
        except HumanAuditError:
            continue
        if not validation.passed:
            raise SystemExit(
                "this exact construction has a previously archived valid failing "
                "human audit; start a new study instead of promoting a later pass"
            )


def _promote_human_audit_locked(args, audit_path: Path) -> Path:
    """Validate one exact audit, archiving valid failures or promoting a pass.

    The automated construction manifest is never modified. Promotion requires a
    protocol that was snapshotted into the immutable task manifest before study,
    plus a passing review of every cumulative train/dev record and note entry.
    """
    sdir = _study_dir(args)
    task_manifest_path = sdir / "manifest.json"
    construction_path = sdir / "notes" / f"note-r{args.round}.manifest.json"
    try:
        task_manifest = load_json_artifact(task_manifest_path)
        construction = load_json_artifact(construction_path)
        audit_bytes = read_artifact_bytes(audit_path)
        audit_text = audit_bytes.decode("utf-8")
        audit = strict_json_loads(audit_bytes, label="completed human-audit result")
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"cannot load human-audit promotion inputs: {error}") from error
    if (construction.get("study_id") != args.study_id
            or construction.get("task") != args.task
            or construction.get("round") != args.round
            or task_manifest.get("study_id") != args.study_id
            or task_manifest.get("task") != args.task
            or task_manifest.get("master_seed") != args.seed
            or construction.get("automated_claim_ready") is not True
            or construction.get("claim_ready") is not False):
        raise SystemExit("construction manifest is not eligible for human-audit promotion")
    automated_readiness = construction.get("automated_readiness")
    if (not isinstance(automated_readiness, dict) or not automated_readiness
            or not all(value is True for value in automated_readiness.values())
            or construction.get("publication_claim_ready") is not False
            or construction.get("confirmatory_claim_ready") is not False):
        raise SystemExit("construction manifest did not pass every automated gate")
    _validate_construction_artifacts(sdir, construction)
    construction_artifacts = _construction_artifact_bytes(sdir, construction)

    protocol_record = task_manifest.get("human_audit_protocol")
    if not isinstance(protocol_record, dict):
        raise SystemExit("study has no pre-registered blinded-audit protocol")
    protocol_relative = PurePosixPath(str(protocol_record.get("path", "")))
    if protocol_relative.is_absolute() or ".." in protocol_relative.parts \
            or not protocol_relative.parts:
        raise SystemExit("task manifest has an unsafe audit-protocol path")
    protocol_path = sdir.joinpath(*protocol_relative.parts)
    try:
        protocol = load_json_artifact(protocol_path)
        if sha256_file(protocol_path) != protocol_record.get("sha256"):
            raise ValueError("protocol hash mismatch")
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"pre-registered audit protocol is invalid: {error}") from error
    try:
        validate_human_audit_protocol(
            protocol,
            expected_protocol_id=protocol_record.get("protocol_id"),
        )
    except HumanAuditError as error:
        raise SystemExit(
            f"pre-registered audit protocol no longer satisfies its contract: {error}"
        ) from error

    construction_hash = sha256_file(construction_path)
    audit_required = {
        "study_id": args.study_id,
        "task": args.task,
        "protocol_sha256": protocol_record["sha256"],
        "construction_manifest_sha256": construction_hash,
        "note_sha256": construction["note_sha256"],
    }
    if not _human_audit_matches_binding(
            audit, required=audit_required, round_number=args.round):
        raise SystemExit("human audit does not match the pre-registration or study artifacts")
    if not _auditor_id_is_valid(audit):
        raise SystemExit("invalid auditor ID")
    try:
        validation = validate_human_audit_result(
            audit, construction, construction_artifacts)
    except HumanAuditError as error:
        raise SystemExit(str(error)) from error
    passed = validation.passed

    audit_hash = sha256_text(audit_text)
    if not passed:
        failed_relative = (
            Path("audits") / "failed" / "by-sha256" / f"{audit_hash}.json"
        )
        write_immutable_text(sdir / "notes" / failed_relative, audit_text)
        raise SystemExit(
            "human audit decision is 'fail'; archived exact result at "
            f"{failed_relative}; no audited note manifest was created"
        )

    _reject_prior_failed_human_audit(
        sdir,
        construction,
        construction_artifacts,
        required=audit_required,
        round_number=args.round,
    )
    audit_relative = Path("audits") / "by-sha256" / f"{audit_hash}.json"
    write_immutable_text(sdir / "notes" / audit_relative, audit_text)
    protocol_note_relative = Path("audits") / "protocols" / f"{protocol_record['sha256']}.json"
    write_immutable_text(
        sdir / "notes" / protocol_note_relative,
        read_artifact_bytes(protocol_path).decode("utf-8"),
    )
    promoted = {
        **construction,
        "manifest_type": "human-audited-note",
        "claim_ready": True,
        "publication_claim_ready": True,
        "confirmatory_claim_ready": True,
        "construction_manifest": {
            "path": construction_path.name,
            "sha256": construction_hash,
        },
        "human_audit": {
            "required": True,
            "status": "passed",
            "auditor_id": audit["auditor_id"],
            "protocol_sha256": protocol_record["sha256"],
            "protocol_path": str(protocol_note_relative),
            "result_sha256": audit_hash,
            "result_path": str(audit_relative),
        },
    }
    _validate_promoted_auditor_identity(promoted, audit)
    promoted_path = sdir / "notes" / f"note-r{args.round}.audited.manifest.json"
    write_immutable_json(promoted_path, promoted)
    return promoted_path


def promote_human_audit(args, audit_path: Path) -> Path:
    """Serialize promotion with study construction and recheck inside the lock."""
    with _study_round_lock(args):
        promoted_path = _study_dir(args) / "notes" \
            / f"note-r{args.round}.audited.manifest.json"
        if promoted_path.exists():
            try:
                existing = load_json_artifact(promoted_path)
                supplied_hash = sha256_bytes(read_artifact_bytes(audit_path))
            except (OSError, UnicodeError, ValueError) as error:
                raise SystemExit(f"cannot recheck existing human-audit promotion: {error}") \
                    from error
            if existing.get("human_audit", {}).get("result_sha256") != supplied_hash:
                raise SystemExit(
                    "round already has a different immutable human-audit promotion"
                )
        return _promote_human_audit_locked(args, audit_path)


def _validate_round_environment_provenance(
    sdir: Path, round_number: int, task_manifest: dict
) -> None:
    """Revalidate every exact model-launch binding used through one round."""

    rdir = sdir / f"r{round_number}"
    round_manifest = load_json_artifact(rdir / "manifest.json")
    _validate_artifact_environment(
        sdir,
        task_manifest,
        {"environment_snapshot": round_manifest.get(
            "initial_environment_snapshot")}
        if isinstance(round_manifest, dict)
        else {},
        label="round manifest",
    )
    paths = [
        *sorted((rdir / "quiz-episodes").glob("*.json")),
        *sorted((rdir / "items").glob("*.json")),
        *sorted((rdir / "dev-exam").glob("*.json")),
    ]
    for path in paths:
        artifact = load_json_artifact(path)
        if not isinstance(artifact, dict):
            raise SystemExit(f"model artifact is not an object: {path}")
        _validate_artifact_environment(
            sdir, task_manifest, artifact, label=f"model artifact {path.name}"
        )
    for path in sorted((sdir / "dev-references").glob("*.json")):
        reference = load_json_artifact(path)
        if (
            not isinstance(reference, dict)
            or type(reference.get("created_round")) is not int
        ):
            raise SystemExit(f"dev reference has invalid round provenance: {path}")
        if reference["created_round"] <= round_number:
            _validate_artifact_environment(
                sdir,
                task_manifest,
                reference,
                label=f"dev reference {path.name}",
            )


def _completed_round_is_exact(args) -> bool:
    """Validate and stop at an already-finalized immutable round."""
    sdir = _study_dir(args)
    construction_path = sdir / "notes" / f"note-r{args.round}.manifest.json"
    if not construction_path.exists():
        return False
    try:
        construction = load_json_artifact(construction_path)
        task_manifest = load_json_artifact(sdir / "manifest.json")
    except (OSError, UnicodeError, ValueError) as error:
        raise SystemExit(f"finalized study round is unreadable: {error}") from error
    expected_config = {
        "chapters_per_round": args.chapters,
        "questions_per_chapter": 3 if args.smoke else args.questions,
        "smoke": args.smoke,
        "concurrency": args.concurrency,
    }
    config = task_manifest.get("config")
    try:
        urls = validate_local_server_urls(args.base_urls)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if (construction.get("study_id") != args.study_id
            or construction.get("task") != args.task
            or construction.get("round") != args.round
            or task_manifest.get("study_id") != args.study_id
            or task_manifest.get("task") != args.task
            or task_manifest.get("master_seed") != args.seed
            or not isinstance(config, dict)
            or any(config.get(key) != value for key, value in expected_config.items())
            or task_manifest.get("server_transport", {}).get("server_count") != len(urls)):
        raise SystemExit("completed study round does not match the requested protocol")
    if args.audit_protocol is not None:
        try:
            protocol_sha256 = sha256_bytes(read_artifact_bytes(args.audit_protocol))
        except (OSError, UnicodeError, ValueError) as error:
            raise SystemExit(f"cannot read requested audit protocol: {error}") from error
        if task_manifest.get("human_audit_protocol", {}).get("sha256") != protocol_sha256:
            raise SystemExit("completed study round used a different audit protocol")
    _validate_construction_artifacts(sdir, construction)
    if isinstance(task_manifest.get("environment"), dict):
        _validate_round_environment_provenance(
            sdir, args.round, task_manifest
        )
    return True


def run_round(args):
    """Serialize one round and never repeat calls after finalization."""
    with _study_round_lock(args):
        if _completed_round_is_exact(args):
            log.info(
                "ROUND %d %s already finalized for study %s",
                args.round, args.task, args.study_id,
            )
            return
        _run_round_locked(args)


def _run_round_locked(args):
    corpus = CORPORA[args.task]
    sdir = _study_dir(args)
    rdir = sdir / f"r{args.round}"
    protected_directories = (rdir, sdir / "notes", sdir / "dev-references")
    if any(path.is_symlink() for path in protected_directories):
        raise SystemExit("study round artifacts must not traverse symlinks")
    try:
        urls = validate_local_server_urls(args.base_urls)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    # Build and byte-validate the complete pinned corpus snapshot before any
    # immutable task manifest can be created.
    rt = RepoTools(corpus, read_max_lines=READ_MAX_LINES)
    manifest, launch_environment = _write_task_manifest(args, corpus, sdir, urls)
    if not args.smoke:
        try:
            validate_local_server_urls(
                args.base_urls,
                expected_count=int(manifest["environment"]["server_count"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SystemExit(
                "claim-ready self-study requires loopback endpoints matching SB_NSERVE"
            ) from error
    tools_fns = make_tools(rt)

    chaps = chapters(rt)
    if not chaps:
        raise SystemExit(f"{args.task} has no study chapters")
    k = min(args.chapters, len(chaps))
    start = (args.round - 1) * k
    todo = [chaps[(start + index) % len(chaps)] for index in range(k)]
    prior_questions, prior_items, prior_calls, prior_manifests = _load_prior_rounds(
        sdir, args.round, study_id=args.study_id, task=args.task, rt=rt)
    prior_entries = collect_note_entries(prior_items)
    input_note = _load_input_note(sdir, args.round, prior_entries)
    round_contract = {
        "schema_version": SCHEMA_VERSION,
        "study_id": args.study_id,
        "task": args.task,
        "round": args.round,
        "chapters": todo,
        "master_seed": args.seed,
        "input_note_sha256": sha256_text(input_note),
        "task_manifest_sha256": sha256_json(manifest),
    }
    round_manifest_path = rdir / "manifest.json"
    if round_manifest_path.exists():
        try:
            round_manifest = load_json_artifact(round_manifest_path)
        except (OSError, UnicodeError, ValueError) as error:
            raise SystemExit(f"invalid existing round manifest: {error}") from error
        if (
            not isinstance(round_manifest, dict)
            or set(round_manifest) != {
                *round_contract,
                "initial_environment_snapshot",
            }
            or canonical_json_bytes({
                key: round_manifest[key] for key in round_contract
            }) != canonical_json_bytes(round_contract)
        ):
            raise SystemExit("study round manifest drifted")
        _validate_artifact_environment(
            sdir,
            manifest,
            {"environment_snapshot": round_manifest.get(
                "initial_environment_snapshot")},
            label="round manifest",
        )
    else:
        round_manifest = {
            **round_contract,
            "initial_environment_snapshot": launch_environment,
        }
        write_immutable_json(round_manifest_path, round_manifest)
    log.info("ROUND %d %s: chapters=%s (syllabus has %d)",
             args.round, args.task, todo, len(chaps))

    questions, quiz_episodes = _prepare_questions(
        args,
        rt,
        tools_fns,
        urls,
        rdir,
        todo,
        prior_questions,
        prior_items,
        task_manifest=manifest,
        launch_environment=launch_environment,
    )
    freshness = freshness_audit(
        questions,
        task=args.task,
        study_dir=sdir,
        snapshot_dir=rdir / "freshness-sources",
    )
    write_immutable_json(rdir / "freshness.json", freshness)
    freshness_snapshots_complete = freshness_sources_complete(rdir, freshness)
    if not args.smoke and (not freshness["fresh"] or not freshness_snapshots_complete):
        quiz_calls = sorted(
            (call for episode in quiz_episodes for call in episode.get("calls", [])),
            key=lambda call: call["call_id"],
        )
        write_immutable_text(rdir / "usage.jsonl", _jsonl(quiz_calls))
        write_immutable_text(
            rdir / "cumulative-usage.jsonl",
            _jsonl(sorted(prior_calls + quiz_calls, key=lambda call: call["call_id"])),
        )
        raise SystemExit(
            "question freshness gate failed before training; inspect "
            f"{rdir / 'freshness.json'}"
        )
    train_items = [item for item in questions if item["split"] == "train"]
    item_dir = rdir / "items"
    records_by_id = {}
    pending = []
    for item in train_items:
        path = item_dir / f"{item['item_id']}.json"
        if path.exists():
            record = load_json_artifact(path)
            _validate_artifact_environment(
                sdir, manifest, record, label="training item"
            )
            _validate_completed_item(
                item,
                record,
                sha256_text(input_note),
                master_seed=args.seed,
                rt=rt,
            )
            records_by_id[item["item_id"]] = record
        else:
            pending.append((item, path))
    log.info("%d train/retest items pending; %d cumulative dev questions held out",
             len(pending), len(collect_dev_questions(prior_questions + [questions])))

    def one_train(spec):
        item, path = spec
        record = run_item(item, input_note, tools_fns,
                          _server_url(urls, args.seed, item["item_id"]),
                          TRAIN_ENSEMBLE, rt, master_seed=args.seed)
        record = _bind_launch_environment(record, launch_environment)
        _validate_artifact_environment(
            sdir, manifest, record, label="training item"
        )
        write_immutable_json(path, record)
        return item["item_id"], record

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for item_id, record in pool.map(one_train, pending):
            records_by_id[item_id] = record
    records = [records_by_id[item["item_id"]] for item in train_items]
    records.sort(key=lambda item: item["item_id"])
    for item, record in ((item, records_by_id[item["item_id"]]) for item in train_items):
        _validate_artifact_environment(
            sdir, manifest, record, label="training item"
        )
        _validate_completed_item(
            item,
            record,
            sha256_text(input_note),
            master_seed=args.seed,
            rt=rt,
        )
    write_immutable_text(rdir / "items.jsonl", _jsonl(records))

    quiz_calls = [call for episode in quiz_episodes for call in episode.get("calls", [])]
    train_calls = [call for record in records for call in record.get("calls", [])]
    construction_round_calls = sorted(quiz_calls + train_calls, key=lambda call: call["call_id"])
    all_entries = collect_note_entries(prior_items + records)
    note_text = render_note(rt, chaps, all_entries, corpus.display)
    prior_construction_calls = [call for call in prior_calls if _is_construction_call(call)]
    # Dev evaluation needs the exact final note bytes, but the authoritative
    # manifest is intentionally withheld until every round artifact is complete.
    _write_note_bytes(args, sdir, note_text)

    dev_pool = collect_dev_questions(prior_questions + [questions])
    reference_dir = sdir / "dev-references"
    references_by_origin = {}
    pending_references = []
    for item in dev_pool:
        reference_id = _record_id("dev-reference", args.study_id, args.task, item["item_id"])
        path = reference_dir / f"{reference_id}.json"
        if path.exists():
            reference = load_json_artifact(path)
            _validate_artifact_environment(
                sdir, manifest, reference, label="dev reference"
            )
            _validate_dev_reference(
                item,
                reference,
                study_id=args.study_id,
                task=args.task,
                master_seed=args.seed,
                rt=rt,
            )
            references_by_origin[item["item_id"]] = reference
        else:
            if item["origin_round"] != args.round:
                raise SystemExit(f"immutable prior dev reference is missing: {path}")
            pending_references.append((item, path))

    def one_reference(spec):
        item, path = spec
        reference_id = _record_id("dev-reference", args.study_id, args.task, item["item_id"])
        reference = build_dev_reference(
            item, tools_fns, _server_url(urls, args.seed, reference_id), master_seed=args.seed,
            created_round=args.round, rt=rt, study_id=args.study_id, task=args.task)
        reference = _bind_launch_environment(reference, launch_environment)
        _validate_artifact_environment(
            sdir, manifest, reference, label="dev reference"
        )
        _validate_dev_reference(
            item,
            reference,
            study_id=args.study_id,
            task=args.task,
            master_seed=args.seed,
            rt=rt,
        )
        write_immutable_json(path, reference)
        return item["item_id"], reference

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for origin_id, reference in pool.map(one_reference, pending_references):
            references_by_origin[origin_id] = reference
    dev_references = [references_by_origin[item["item_id"]] for item in dev_pool]
    current_reference_calls = [
        call
        for reference in dev_references
        if reference["created_round"] == args.round
        for call in reference.get("calls", [])
    ]

    dev_dir = rdir / "dev-exam"
    dev_by_origin = {}
    pending_dev = []
    for item in dev_pool:
        exam_id = _record_id("dev-exam", item["item_id"], args.round)
        path = dev_dir / f"{exam_id}.json"
        reference = references_by_origin[item["item_id"]]
        if path.exists():
            record = load_json_artifact(path)
            _validate_artifact_environment(
                sdir, manifest, record, label="dev exam"
            )
            _validate_dev_exam(
                item, record, reference, note_sha256=sha256_text(note_text),
                master_seed=args.seed, exam_round=args.round)
            dev_by_origin[item["item_id"]] = record
        else:
            pending_dev.append((item, reference, path))

    def one_dev(spec):
        item, reference, path = spec
        exam_id = _record_id("dev-exam", item["item_id"], args.round)
        record = run_dev_item(
            item, note_text, reference, _server_url(urls, args.seed, exam_id),
            master_seed=args.seed, exam_round=args.round)
        record = _bind_launch_environment(record, launch_environment)
        _validate_artifact_environment(
            sdir, manifest, record, label="dev exam"
        )
        _validate_dev_exam(
            item, record, reference, note_sha256=sha256_text(note_text),
            master_seed=args.seed, exam_round=args.round)
        write_immutable_json(path, record)
        return item["item_id"], record

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for origin_id, record in pool.map(one_dev, pending_dev):
            dev_by_origin[origin_id] = record
    dev_records = [dev_by_origin[item["item_id"]] for item in dev_pool]
    dev_records.sort(key=lambda item: item["origin_item_id"])
    write_immutable_text(rdir / "dev-exam.jsonl", _jsonl(dev_records))

    dev_calls = [call for record in dev_records for call in record.get("calls", [])]
    round_calls = sorted(
        construction_round_calls + current_reference_calls + dev_calls,
        key=lambda call: call["call_id"],
    )
    expected_round_calls = [
        call
        for artifact in quiz_episodes + records
        + [reference for reference in dev_references
           if reference["created_round"] == args.round]
        + dev_records
        for call in artifact.get("calls", [])
    ]
    round_usage_audit = usage_ledger_audit(expected_round_calls, round_calls)
    write_immutable_text(rdir / "usage.jsonl", _jsonl(round_calls))
    cumulative_calls = sorted(prior_calls + round_calls, key=lambda call: call["call_id"])
    write_immutable_text(rdir / "cumulative-usage.jsonl", _jsonl(cumulative_calls))
    cumulative_usage_audit = usage_ledger_audit(
        prior_calls + round_calls, _read_jsonl(rdir / "cumulative-usage.jsonl"))
    train_originals = [record for record in records if is_distillable_item(record)]
    retests = [record for record in records if record.get("kind") == "retest"]
    verdict_counts = defaultdict(int)
    for record in records:
        verdict_counts[record.get("verdict", record["status"])] += 1
    all_originals = [record for record in prior_items + records if is_distillable_item(record)]
    original_ids = {record["item_id"] for record in all_originals}
    successful_derivations = [
        derivation
        for artifact in records + dev_references
        for derivation in artifact.get("derivations", [])
        if derivation.get("status") == "ok"
    ]
    evidence_safe = bool(successful_derivations) and all(
        _trajectory_hash_valid(derivation)
        and derivation.get("evidence_class") == "quote-only"
        and derivation.get("reference_support", {}).get("status") == "ok"
        and derivation.get("reference_support", {}).get("supported") is True
        and bool(derivation.get("evidence"))
        and all(validate_evidence(rt, evidence) == evidence
                for evidence in derivation.get("evidence", []))
        for derivation in successful_derivations
    )
    entry_lineage_safe = all(
        entry.get("origin_item_id") in original_ids
        and entry.get("evidence_class") == "quote-only"
        and validate_evidence(rt, entry) == {
            "file": entry.get("file"), "line": entry.get("line"), "quote": entry.get("quote")}
        and bool(entry.get("correction_support"))
        and all(check.get("status") == "ok" and check.get("verdict") == "correct"
                for check in entry.get("correction_support", []))
        for entry in all_entries
    )
    response_models = sorted({call.get("response_model") for call in cumulative_calls
                              if isinstance(call.get("response_model"), str)
                              and call.get("response_model")})
    launch_environments = _launch_environment_inventory(
        sdir, args.round, manifest
    )
    launch_environment_hashes = {
        sha256_json(record) for record in launch_environments
    }
    model_artifacts = quiz_episodes + records + dev_references + dev_records
    launch_environments_bound = bool(model_artifacts) and all(
        isinstance(artifact.get("environment_snapshot"), dict)
        and sha256_json(artifact["environment_snapshot"])
        in launch_environment_hashes
        for artifact in model_artifacts
    )
    automated_readiness = {
        "non_smoke": not args.smoke,
        "provenance_complete": manifest["automated_provenance_ready"] is True,
        "launch_environments_bound": launch_environments_bound,
        "prior_rounds_automated_ready": all(
            prior.get("automated_claim_ready") is True for prior in prior_manifests),
        "question_freshness": (freshness["fresh"] is True
                               and freshness_snapshots_complete),
        "quiz_episodes_complete": bool(quiz_episodes)
        and all(episode.get("status") == "ok" and _trajectory_hash_valid(episode)
                for episode in quiz_episodes),
        "training_complete": bool(train_originals)
        and all(record.get("status") == "ok"
                and record.get("verdict") in {"correct", "partial", "wrong"}
                for record in records),
        "dev_references_complete": bool(dev_references)
        and all(reference.get("status") == "ok" for reference in dev_references),
        "dev_exam_complete": bool(dev_records) and all(
            record.get("status") == "ok"
            and set(record.get("verdicts", {})) == {"with_note", "bare"}
            and set(record["verdicts"].values()) <= {"correct", "partial", "wrong"}
            for record in dev_records),
        "lineage_clean": entry_lineage_safe,
        "evidence_safe": evidence_safe,
        "usage_complete": (round_usage_audit["complete"]
                           and cumulative_usage_audit["complete"]
                           and artifact_usage_consistent(
                               quiz_episodes + records + dev_references + dev_records)),
        "response_model_homogeneous": len(response_models) == 1,
        "response_model_expected": response_models == [MODEL_ID.removeprefix("openai/")],
    }
    automated_claim_ready = all(automated_readiness.values())
    note_sha256 = sha256_text(note_text)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "study_id": args.study_id,
        "round": args.round,
        "task": args.task,
        "chapters": todo,
        "train_items": len(train_originals),
        "retest_items": len(retests),
        "cumulative_dev_items": len(dev_records),
        "verdicts": dict(verdict_counts),
        "train": _error_rate([record.get("verdict", "unresolved") for record in train_originals]),
        "retest": _error_rate([record.get("verdict", "unresolved") for record in retests]),
        "dev_with_note": _error_rate([
            record.get("verdicts", {}).get("with_note", "unresolved") for record in dev_records]),
        "dev_bare": _error_rate([
            record.get("verdicts", {}).get("bare", "unresolved") for record in dev_records]),
        "entries_admitted": sum(record.get("entry") is not None for record in train_originals),
        "entries_bounced": sum("entry_bounced" in record for record in train_originals),
        "note_entries_total": len(all_entries),
        "note_sha256": note_sha256,
        "claim_ready": False,
        "publication_claim_ready": False,
        "confirmatory_claim_ready": False,
        "automated_claim_ready": automated_claim_ready,
        "automated_readiness": automated_readiness,
        "human_audit": {"required": True, "status": "not_performed"},
        "freshness": freshness,
        "response_models": response_models,
        "environment_contract": manifest["environment_contract"],
        "launch_environments": launch_environments,
        "round_usage": usage_totals(round_calls),
        "cumulative_usage": usage_totals(cumulative_calls),
        "round_usage_by_phase": usage_by_phase(round_calls),
        "cumulative_usage_by_phase": usage_by_phase(cumulative_calls),
        "round_usage_audit": round_usage_audit,
        "cumulative_usage_audit": cumulative_usage_audit,
    }
    write_immutable_json(rdir / "summary.json", summary)
    # This is deliberately the final authoritative write. A crash anywhere
    # earlier leaves no manifest that evaluation could mistake for a complete,
    # claim-ready round.
    _write_note(
        args, sdir, note_text, all_entries,
        input_note_sha256=sha256_text(input_note),
        round_calls=round_calls,
        cumulative_calls=cumulative_calls,
        round_construction_calls=construction_round_calls,
        cumulative_construction_calls=prior_construction_calls + construction_round_calls,
        corpus_commit=manifest["corpus_commit"],
        automated_claim_ready=automated_claim_ready,
        automated_readiness=automated_readiness,
    )
    log.info("ROUND %d SUMMARY %s", args.round, json.dumps(summary, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=list(CORPORA))
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--study-id", required=True,
                   help="immutable namespace for one study replication")
    p.add_argument("--seed", type=int, required=True,
                   help="master seed; every model phase derives and records its own seed")
    p.add_argument("--chapters", type=int, default=K_CHAPTERS)
    p.add_argument("--questions", type=int, default=M_QUESTIONS)
    p.add_argument("--base-urls", default="http://localhost:8100/v1")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument(
        "--audit-protocol", type=Path,
        help="pre-registered blinded human-audit protocol to snapshot before round 1",
    )
    p.add_argument(
        "--promote-human-audit", type=Path,
        help="offline: validate a completed blinded audit and write a separate audited manifest",
    )
    p.add_argument("--smoke", action="store_true", help="1 chapter, 3 questions")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    # Validate before using the study ID in any filesystem path.
    _study_dir(args)
    if args.round < 1 or args.chapters < 1 or args.questions < 2 or args.concurrency < 1:
        p.error("round/chapters/concurrency must be positive and questions must be at least 2")
    if args.audit_protocol is not None and args.round != 1:
        p.error("--audit-protocol may only be pre-registered in round 1")
    if args.promote_human_audit is not None:
        if args.smoke or args.audit_protocol is not None:
            p.error("human-audit promotion cannot be combined with --smoke or --audit-protocol")
        promoted = promote_human_audit(args, args.promote_human_audit)
        print(promoted)
        return

    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(
                      ROOT / "logs" / f"selfquiz-{args.study_id}-{args.task}.log")])
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run_round(args)


if __name__ == "__main__":
    main()
