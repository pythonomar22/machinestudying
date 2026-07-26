"""The 100-question rev-3 practice set and its deterministic 70/30 split.

The set is our StudyBench-pipeline replication at fulldspy scope
(data/dspy_validation.jsonl, provenance in data_collection/ rev-3,
experiments/009). The split is stratified by topic and derived from the
master seed; the study slice feeds fold-back mining, the dev slice is the
offline iteration signal and never enters the study object.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import PurePosixPath

from studybench.dataset import CORPORA, ROOT

PRACTICE_PATH = ROOT / "data" / "dspy_validation.jsonl"
DEV_TOTAL = 30  # stratified: 30 // topics per topic, remainder to the first topics


def practice_dataset_sha256() -> str:
    return hashlib.sha256(PRACTICE_PATH.read_bytes()).hexdigest()


def load_practice_questions() -> tuple[dict, ...]:
    corpus = CORPORA["dspy"]
    rows = tuple(
        json.loads(line)
        for line in PRACTICE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    ids: set[str] = set()
    for row in rows:
        if set(row) != {"id", "topic", "question", "gold_answer", "rubric", "evidence"}:
            raise ValueError(f"unexpected fields in practice row: {row.get('id')}")
        if row["id"] in ids:
            raise ValueError(f"duplicate practice id: {row['id']}")
        ids.add(row["id"])
        if sum(claim["weight"] for claim in row["rubric"]) != 100:
            raise ValueError(f"practice rubric weights do not sum to 100: {row['id']}")
        span_ids = {span["span_id"] for span in row["evidence"]}
        for claim in row["rubric"]:
            if not set(claim["span_ids"]).issubset(span_ids):
                raise ValueError(f"unknown evidence span: {row['id']}/{claim['claim_id']}")
        for span in row["evidence"]:
            logical = PurePosixPath(span["path"])
            if not logical.parts or logical.parts[0] not in corpus.roots or ".." in logical.parts:
                raise ValueError(f"practice evidence escapes corpus roots: {span['path']}")
    if len(rows) != 100:
        raise ValueError(f"expected 100 practice questions, found {len(rows)}")
    return rows


def split_practice(rows: tuple[dict, ...], master_seed: int) -> dict:
    """Stratified 70/30 split: per topic, a seeded sample joins the dev slice."""

    by_topic: dict[str, list[dict]] = {}
    for row in rows:
        by_topic.setdefault(row["topic"], []).append(row)
    topics = sorted(by_topic)
    rng = random.Random(f"foldback-split-{master_seed}")
    per_topic = DEV_TOTAL // len(topics)
    extra = DEV_TOTAL - per_topic * len(topics)
    dev_ids: set[str] = set()
    for position, topic in enumerate(topics):
        want = per_topic + (1 if position < extra else 0)
        members = sorted(row["id"] for row in by_topic[topic])
        dev_ids.update(rng.sample(members, want))
    return {
        "master_seed": master_seed,
        "dataset_sha256": practice_dataset_sha256(),
        "study_ids": sorted(row["id"] for row in rows if row["id"] not in dev_ids),
        "dev_ids": sorted(dev_ids),
    }
