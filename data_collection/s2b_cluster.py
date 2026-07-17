# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy<2.3",
#     "scikit-learn<1.6",
#     "umap-learn==0.5.7",
#     "hdbscan==0.8.40",
#     "matplotlib==3.10.3",
#     "openai>=1.90",
# ]
# ///
"""s2b: paper Stage 2 — UMAP + HDBSCAN clustering, GPT-5.4 labels, selection.

Paper: embeddings are projected to ten dimensions with UMAP (n_neighbors=15)
and clustered with HDBSCAN (min_cluster_size=30, min_samples=5); GPT-5.4
assigns a behavioral label to each cluster after reviewing 30 representative
sessions, and clusters are then selected for generation.

Documented adaptations (see README fidelity table):
- The paper's HDBSCAN parameters assume a session pool in the thousands; our
  issue-derived pool has ~300 seeds. We run BOTH the paper-literal config and
  a pool-scaled config (min_cluster_size=10), record both, and label the
  scaled clusters.
- Selection prefers the paper's path: a cluster whose GPT label matches the
  held-out test topic (react_agents_and_tools). Measured fact: DSPy GitHub
  issues do not contain enough ReAct/tools/agents Q&A mass to form such a
  cluster (the paper's community-session source differs from public issues),
  so when no cluster matches, we fall back to LABEL-CONDITIONED selection:
  the Stage-1 seeds matching TOPIC_PATTERN become the seed pool for the
  target label. Both paths are recorded in clusters.json.
- Representative sessions are the members closest to the cluster centroid by
  cosine similarity in the original embedding space; UMAP is seeded (the
  paper specifies neither).

All artifacts are persisted BEFORE selection so a failed selection still
leaves the full clustering on disk for review.

Usage: uv run data_collection/s2b_cluster.py
"""

from __future__ import annotations

import json
import re

import hdbscan
import matplotlib
import numpy as np
import umap
from openai import OpenAI

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ARTIFACTS, MASTER_SEED, OPENAI_MODEL, REASONING_EFFORT, chat, load_env, read_json, read_jsonl, write_json
from prompts import DSPY_VALUES, LABEL_TEMPLATE, TARGET_DESCRIPTION, TARGET_LABEL

UMAP_DIMS, UMAP_NEIGHBORS = 10, 15                           # paper
PAPER_HDBSCAN = {"min_cluster_size": 30, "min_samples": 5}   # paper-literal
SCALED_HDBSCAN = {"min_cluster_size": 10, "min_samples": 5}  # pool-scaled (ours)
REPRESENTATIVES = 30                                         # paper
TARGET_TOKENS = {"react", "agent", "agents", "tool", "tools"}
TOPIC_PATTERN = re.compile(
    r"\breact\b|\btools?\b|\btoolkits?\b|\bagents?\b|\bagentic\b"
    r"|\bfunction[- ]calling\b|\btool[- ]calls?\b",
    re.IGNORECASE,
)
MIN_SELECTED = 20  # must at least cover the paper's 20-seed sample

# dataviz reference categorical palette, fixed order; muted gray = noise.
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
           "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
NOISE_COLOR = "#b8bcc2"

LABEL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "cluster_label",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "description": {"type": "string"},
                "coherent": {"type": "boolean"},
            },
            "required": ["label", "description", "coherent"],
            "additionalProperties": False,
        },
    },
}


def cluster_summary(labels: np.ndarray) -> dict:
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    return {
        "clusters": int(len(unique)),
        "noise": int((labels < 0).sum()),
        "sizes": {int(c): int(n) for c, n in zip(unique, counts)},
    }


def representatives(members: np.ndarray, embeddings: np.ndarray) -> list[int]:
    centroid = embeddings[members].mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    order = np.argsort(-(embeddings[members] @ centroid))
    return [int(members[i]) for i in order[:REPRESENTATIVES]]


def main() -> None:
    load_env()
    seeds = {str(row["number"]): row for row in read_jsonl(ARTIFACTS / "seed_pool.jsonl")}
    index = read_json(ARTIFACTS / "embeddings_index.json")
    matrix = np.load(ARTIFACTS / "embeddings.npy")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    seed_rows = [i for i, row in enumerate(index["rows"]) if row["kind"] == "seed"]
    test_rows = [i for i, row in enumerate(index["rows"]) if row["kind"] == "test"]
    seed_keys = [index["rows"][i]["key"] for i in seed_rows]
    if set(seed_keys) != set(seeds):
        raise RuntimeError("embeddings and seed pool disagree; rerun s2a")
    seed_matrix, test_matrix = matrix[seed_rows], matrix[test_rows]

    projected = umap.UMAP(
        n_components=UMAP_DIMS, n_neighbors=UMAP_NEIGHBORS,
        metric="cosine", random_state=MASTER_SEED,
    ).fit_transform(seed_matrix)
    paper_labels = hdbscan.HDBSCAN(**PAPER_HDBSCAN).fit_predict(projected)
    scaled_labels = hdbscan.HDBSCAN(**SCALED_HDBSCAN).fit_predict(projected)
    print(f"paper-literal HDBSCAN: {cluster_summary(paper_labels)}")
    print(f"pool-scaled HDBSCAN:   {cluster_summary(scaled_labels)}")

    client = OpenAI(timeout=600, max_retries=2)
    clusters = []
    for cluster_id in sorted(set(scaled_labels[scaled_labels >= 0])):
        members = np.flatnonzero(scaled_labels == cluster_id)
        reps = representatives(members, seed_matrix)
        sessions = [
            {"number": int(seed_keys[i]), "question": seeds[seed_keys[i]]["question_text"][:1200]}
            for i in reps
        ]
        prompt = LABEL_TEMPLATE.format(
            library_name=DSPY_VALUES["library_name"],
            num_sessions=len(sessions),
            sessions_json=json.dumps(sessions, ensure_ascii=False, indent=2),
        )
        response = chat(client, "s2b_label", {
            "model": OPENAI_MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": LABEL_SCHEMA,
        })
        verdict = json.loads(response.choices[0].message.content)
        clusters.append({
            "cluster_id": int(cluster_id),
            "size": int(len(members)),
            **verdict,
            "representative_numbers": [int(seed_keys[i]) for i in reps],
            "member_numbers": [int(seed_keys[i]) for i in members],
        })
        print(f"cluster {cluster_id} (n={len(members)}): {verdict['label']} "
              f"(coherent={verdict['coherent']})")

    # Diagnostics: where do the held-out test questions land, and where do
    # target-topic (keyword) seeds sit?
    centroids = np.stack([
        seed_matrix[scaled_labels == cluster["cluster_id"]].mean(axis=0)
        for cluster in clusters
    ])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    test_assignment = {
        index["rows"][row]["key"]:
            int(clusters[int(np.argmax(centroids @ matrix[row]))]["cluster_id"])
        for row in test_rows
    }
    keyword_mask = np.array([
        bool(TOPIC_PATTERN.search(seeds[key]["question_text"])) for key in seed_keys
    ])
    keyword_numbers = sorted(int(seed_keys[i]) for i in np.flatnonzero(keyword_mask))
    keyword_spread = {
        int(c): int((keyword_mask & (scaled_labels == c)).sum())
        for c in sorted(set(scaled_labels))
    }

    payload = {
        "umap": {"n_components": UMAP_DIMS, "n_neighbors": UMAP_NEIGHBORS,
                 "metric": "cosine", "random_state": MASTER_SEED},
        "hdbscan_paper_literal": {**PAPER_HDBSCAN, "summary": cluster_summary(paper_labels)},
        "hdbscan_pool_scaled": {**SCALED_HDBSCAN, "summary": cluster_summary(scaled_labels)},
        "labeling_model": OPENAI_MODEL,
        "clusters": clusters,
        "diagnostics": {
            "test_question_assignment": test_assignment,
            "topic_pattern": TOPIC_PATTERN.pattern,
            "topic_keyword_numbers": keyword_numbers,
            "topic_keyword_cluster_spread": keyword_spread,
        },
    }
    write_json(ARTIFACTS / "clusters.json", payload)  # persist before selection

    def target_score(cluster: dict) -> int:
        tokens = set(cluster["label"].lower().replace("-", "_").split("_"))
        return len(tokens & TARGET_TOKENS)

    best = max(clusters, key=target_score)
    if target_score(best) > 0:
        selection = {
            "mode": "cluster_label",
            "cluster_id": best["cluster_id"],
            "label": best["label"],
            "description": best["description"],
            "member_numbers": best["member_numbers"],
        }
        if best["cluster_id"] != max(
            set(test_assignment.values()), key=list(test_assignment.values()).count
        ):
            raise RuntimeError(
                "label-selected cluster disagrees with the modal test-question "
                "cluster; inspect clusters.json before proceeding"
            )
    else:
        print(f"no cluster label matches {sorted(TARGET_TOKENS)}; "
              "falling back to label-conditioned keyword selection")
        selection = {
            "mode": "keyword_fallback",
            "cluster_id": None,
            "label": TARGET_LABEL,
            "description": TARGET_DESCRIPTION,
            "member_numbers": keyword_numbers,
        }
    if len(selection["member_numbers"]) < MIN_SELECTED:
        raise RuntimeError(
            f"selected seed pool has {len(selection['member_numbers'])} members; "
            f"need at least {MIN_SELECTED}"
        )
    payload["selection"] = selection
    write_json(ARTIFACTS / "clusters.json", payload)

    # Diagnostic 2-D map (display only), dataviz-style.
    plane = umap.UMAP(
        n_components=2, n_neighbors=UMAP_NEIGHBORS,
        metric="cosine", random_state=MASTER_SEED,
    ).fit_transform(np.vstack([seed_matrix, test_matrix]))
    seed_plane, test_plane = plane[: len(seed_rows)], plane[len(seed_rows):]
    selected_mask = np.array([
        int(key) in set(selection["member_numbers"]) for key in seed_keys
    ])
    figure, axis = plt.subplots(figsize=(11, 8))
    noise = scaled_labels < 0
    axis.scatter(seed_plane[noise, 0], seed_plane[noise, 1], s=14,
                 color=NOISE_COLOR, label=f"noise (n={int(noise.sum())})")
    for position, cluster in enumerate(clusters):
        mask = scaled_labels == cluster["cluster_id"]
        axis.scatter(seed_plane[mask, 0], seed_plane[mask, 1], s=26,
                     color=PALETTE[position % len(PALETTE)],
                     marker="o" if position < len(PALETTE) else "^",
                     label=f"{cluster['label']} (n={cluster['size']})")
        center = seed_plane[mask].mean(axis=0)
        axis.annotate(cluster["label"], center, fontsize=8,
                      color="#333333", ha="center")
    axis.scatter(seed_plane[selected_mask, 0], seed_plane[selected_mask, 1],
                 s=90, facecolors="none", edgecolors="#1a1a1a", linewidths=1.2,
                 label=f"selected: {selection['label']} "
                       f"({selection['mode']}, n={int(selected_mask.sum())})")
    axis.scatter(test_plane[:, 0], test_plane[:, 1], s=110, color="#1a1a1a",
                 marker="*", label="held-out test questions (5)")
    axis.set_title("SmallDSPy seed pool — UMAP(2) of Qwen3-Embedding-8B, "
                   "pool-scaled HDBSCAN clusters")
    axis.set_xticks([]), axis.set_yticks([])
    for side in axis.spines.values():
        side.set_color("#d0d0d0")
    axis.legend(loc="best", fontsize=8, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(ARTIFACTS / "clusters.png", dpi=150)
    print(f"selection: {selection['mode']} → {selection['label']} "
          f"(n={len(selection['member_numbers'])}); "
          f"test questions land in clusters {test_assignment}")


if __name__ == "__main__":
    main()
