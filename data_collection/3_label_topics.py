# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy<2.3",
#     "pandas==2.2.3",
#     "scikit-learn<1.6",
#     "umap-learn==0.5.7",
#     "hdbscan==0.8.40",
#     "matplotlib==3.10.3",
#     "openai>=1.90",
# ]
# ///
"""Stage 2 (paper): embed seed sessions, cluster them, GPT-label the topics.

Paper, verbatim: "Each session is represented by its first substantive user
question, embedded using Qwen3-Embedding-8B with a domain-aware prefix
prompt. Embeddings are projected to ten dimensions with UMAP
(n_neighbors=15) and clustered using HDBSCAN (min_cluster_size=30,
min_samples=5). GPT-5.4 assigns a behavioral label to each cluster after
reviewing 30 representative sessions."

Their HDBSCAN/UMAP numbers were tuned to their session pool; ours is a
different source and size (302 issues), so the clustering parameters are
re-tuned here from first principles via a reproducible sweep (see
FIDELITY.md, Stage 2). The paper's values stay in the sweep grid so the
comparison is on record.

Phases (embedding needs the vLLM env + one GPU; the rest run under uv):
  srun --overlap --jobid=<JOBID> .venv-vllm/bin/python data_collection/3_label_topics.py embed
  uv run data_collection/3_label_topics.py sweep
  uv run data_collection/3_label_topics.py label [--min-cluster-size N] [--min-samples M] [--n-neighbors K]

Artifacts: 3_embeddings.npy, 3_embeddings_index.json, 3_cluster_sweep.json,
3_umap10.npy, 3_clusters.png, and the labeled clone 3_seed_sessions_topic.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DC = Path(__file__).resolve().parent
ROOT = DC.parent
ARTIFACTS = DC / "artifacts" / "3_label_topics"
INPUT = DC / "artifacts" / "2_filter_sessions" / "2_seed_sessions.json"

EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"          # paper
INSTRUCT = (                                      # inference: wording is ours
    "Instruct: Identify the topic and user intent of a question about the "
    "DSPy language-model programming library.\nQuery: "
)
EMBED_MAX_CHARS = 4_000                           # inference
UMAP_DIMS = 10                                    # paper
UMAP_SEED = 20260716                              # inference (paper: unseeded)
DEFAULT_N_NEIGHBORS = 15                          # paper

# Sweep grid; the paper's (30, 5) is included so its collapse is on record.
SWEEP_N_NEIGHBORS = (10, 15, 25)
SWEEP_MIN_CLUSTER_SIZE = (8, 10, 12, 15, 20, 25, 30)
SWEEP_MIN_SAMPLES = (3, 5, 8, 10)
# First-principles constraints (rationale in FIDELITY.md):
MIN_KEPT_CLUSTER = 15      # a topic must (nearly) support the 20-seed sample
MAX_NOISE_FRACTION = 0.45  # noise must not eat the pool
MIN_CLUSTERS = 4           # enough topical diversity to be worth labeling

REPRESENTATIVES = 30                              # paper (all members if fewer)
OPENAI_MODEL, REASONING_EFFORT = "gpt-5.4", "xhigh"
REPRESENTATIVE_QUESTION_CHARS = 1_200

LABEL_TEMPLATE = """You are naming one cluster of real user questions about DSPy, an open-source library for programming language models. The {num_sessions} questions below are representative members of a single behavioral cluster discovered by embedding and clustering a large pool of DSPy GitHub issues.

Assign the cluster a behavioral label describing what its users are trying to accomplish, in the style of these examples from a related benchmark: `gepa_optimizer_usage`, `rag_and_retrieval_pipelines`, `react_agents_and_tools`, `signature_schema_and_pydantic_types`, `evaluation_metrics_and_custom_eval`.

Return JSON with:
- `label`: a short snake_case behavioral label
- `description`: 1-2 sentences describing what users in this cluster are trying to do
- `coherent`: false if the cluster mixes several unrelated behaviors, true otherwise

## Representative questions
{sessions_json}

Return JSON that matches the schema exactly."""

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

# dataviz reference categorical palette, fixed order; muted gray = noise.
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
           "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
NOISE_COLOR = "#b8bcc2"


def load_sessions() -> list[dict]:
    return json.loads(INPUT.read_text(encoding="utf-8"))["sessions"]


def load_embeddings():
    import numpy as np

    index = json.loads((ARTIFACTS / "3_embeddings_index.json").read_text(encoding="utf-8"))
    matrix = np.load(ARTIFACTS / "3_embeddings.npy")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix, index


def openai_key() -> str:
    import os

    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        key, _, value = line.strip().partition("=")
        if key == "OPENAI_API_KEY" and value:
            return value.strip().strip("'\"")
    return os.environ["OPENAI_API_KEY"]


def project(matrix, n_neighbors: int, dims: int):
    import umap

    return umap.UMAP(
        n_components=dims, n_neighbors=n_neighbors,
        metric="cosine", random_state=UMAP_SEED,
    ).fit_transform(matrix)


def cluster(projected, min_cluster_size: int, min_samples: int):
    import hdbscan

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples,
        gen_min_span_tree=True,
    )
    labels = model.fit_predict(projected)
    return labels, float(model.relative_validity_)


def summarize(labels) -> dict:
    import numpy as np

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    return {
        "clusters": int(len(unique)),
        "noise_fraction": round(float((labels < 0).mean()), 3),
        "sizes": [int(count) for count in sorted(counts, reverse=True)],
    }


def embed() -> None:
    import numpy as np
    from vllm import LLM

    sessions = load_sessions()
    llm = LLM(model=EMBED_MODEL, runner="pooling", max_model_len=8192,
              gpu_memory_utilization=0.85, enforce_eager=True)
    outputs = llm.embed([
        INSTRUCT + row["question_text"][:EMBED_MAX_CHARS] for row in sessions
    ])
    matrix = np.array([out.outputs.embedding for out in outputs], dtype=np.float32)
    if matrix.shape[0] != len(sessions):
        raise RuntimeError(f"embedded {matrix.shape[0]} of {len(sessions)} sessions")
    norms = np.linalg.norm(matrix, axis=1)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS / "3_embeddings.npy", matrix)
    (ARTIFACTS / "3_embeddings_index.json").write_text(
        json.dumps({
            "model": EMBED_MODEL,
            "instruct": INSTRUCT,
            "max_chars": EMBED_MAX_CHARS,
            "dimensions": int(matrix.shape[1]),
            "normalized": bool(np.allclose(norms, 1.0, atol=1e-3)),
            "numbers": [row["number"] for row in sessions],
        }, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"embedded {matrix.shape[0]} sessions, dim={matrix.shape[1]}, "
          f"norm range [{norms.min():.4f}, {norms.max():.4f}]")


def sweep() -> None:
    matrix, index = load_embeddings()
    if index["numbers"] != [row["number"] for row in load_sessions()]:
        raise RuntimeError("embeddings and seed sessions disagree; rerun embed")
    rows = []
    for n_neighbors in SWEEP_N_NEIGHBORS:
        projected = project(matrix, n_neighbors, UMAP_DIMS)
        for min_cluster_size in SWEEP_MIN_CLUSTER_SIZE:
            for min_samples in SWEEP_MIN_SAMPLES:
                if min_samples > min_cluster_size:
                    continue
                labels, dbcv = cluster(projected, min_cluster_size, min_samples)
                summary = summarize(labels)
                rows.append({
                    "n_neighbors": n_neighbors,
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "dbcv": round(dbcv, 4),
                    **summary,
                })

    def admissible(row: dict) -> bool:
        return (
            row["clusters"] >= MIN_CLUSTERS
            and row["noise_fraction"] <= MAX_NOISE_FRACTION
            and (row["sizes"] and min(row["sizes"]) >= MIN_KEPT_CLUSTER)
        )

    candidates = [row for row in rows if admissible(row)]
    recommendation = max(candidates, key=lambda row: row["dbcv"]) if candidates else None
    (ARTIFACTS / "3_cluster_sweep.json").write_text(
        json.dumps({
            "constraints": {
                "min_kept_cluster": MIN_KEPT_CLUSTER,
                "max_noise_fraction": MAX_NOISE_FRACTION,
                "min_clusters": MIN_CLUSTERS,
            },
            "grid": rows,
            "recommendation": recommendation,
        }, indent=1) + "\n",
        encoding="utf-8",
    )
    header = f"{'nn':>3} {'mcs':>4} {'ms':>3} {'dbcv':>7} {'k':>3} {'noise':>6}  sizes"
    print(header)
    for row in sorted(rows, key=lambda r: -r["dbcv"]):
        flag = " *" if admissible(row) else ""
        print(f"{row['n_neighbors']:>3} {row['min_cluster_size']:>4} "
              f"{row['min_samples']:>3} {row['dbcv']:>7.4f} {row['clusters']:>3} "
              f"{row['noise_fraction']:>6.3f}  {row['sizes']}{flag}")
    print(f"\nadmissible configs: {len(candidates)} (marked *)")
    print(f"recommendation: {recommendation}")


def label(min_cluster_size: int | None, min_samples: int | None,
          n_neighbors: int | None) -> None:
    import numpy as np
    from openai import OpenAI

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep_data = json.loads((ARTIFACTS / "3_cluster_sweep.json").read_text(encoding="utf-8"))
    chosen = dict(sweep_data["recommendation"] or {})
    if min_cluster_size:
        chosen["min_cluster_size"] = min_cluster_size
    if min_samples:
        chosen["min_samples"] = min_samples
    if n_neighbors:
        chosen["n_neighbors"] = n_neighbors
    if not all(chosen.get(k) for k in ("min_cluster_size", "min_samples", "n_neighbors")):
        raise SystemExit("no sweep recommendation; pass --min-cluster-size/--min-samples/--n-neighbors")

    sessions = load_sessions()
    matrix, index = load_embeddings()
    if index["numbers"] != [row["number"] for row in sessions]:
        raise RuntimeError("embeddings and seed sessions disagree; rerun embed")
    projected = project(matrix, chosen["n_neighbors"], UMAP_DIMS)
    np.save(ARTIFACTS / "3_umap10.npy", projected)
    labels, dbcv = cluster(projected, chosen["min_cluster_size"], chosen["min_samples"])
    print(f"final clustering {chosen['n_neighbors']}/{chosen['min_cluster_size']}/"
          f"{chosen['min_samples']}: {summarize(labels)} dbcv={dbcv:.4f}")

    client = OpenAI(api_key=openai_key(), timeout=600, max_retries=2)
    log_path = ARTIFACTS / "3_labeling_log.jsonl"
    clusters = []
    for cluster_id in sorted(set(int(value) for value in labels if value >= 0)):
        members = np.flatnonzero(labels == cluster_id)
        centroid = matrix[members].mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        order = np.argsort(-(matrix[members] @ centroid))
        representatives = [int(members[i]) for i in order[:REPRESENTATIVES]]
        payload = [
            {
                "number": sessions[i]["number"],
                "question": sessions[i]["question_text"][:REPRESENTATIVE_QUESTION_CHARS],
            }
            for i in representatives
        ]
        prompt = LABEL_TEMPLATE.format(
            num_sessions=len(payload),
            sessions_json=json.dumps(payload, ensure_ascii=False, indent=2),
        )
        request = {
            "model": OPENAI_MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": LABEL_SCHEMA,
        }
        response = client.chat.completions.create(**request)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "timestamp": time.time(),
                "request": request,
                "response": response.model_dump(exclude_none=True),
            }, ensure_ascii=False) + "\n")
        verdict = json.loads(response.choices[0].message.content)
        clusters.append({
            "cluster_id": cluster_id,
            "size": int(len(members)),
            **verdict,
            "representative_numbers": [sessions[i]["number"] for i in representatives],
        })
        print(f"cluster {cluster_id} (n={len(members)}): {verdict['label']} "
              f"(coherent={verdict['coherent']})")

    by_id = {cluster["cluster_id"]: cluster["label"] for cluster in clusters}
    labeled_sessions = [
        {
            **row,
            "cluster_id": int(value) if value >= 0 else None,
            "topic": by_id.get(int(value)) if value >= 0 else None,
        }
        for row, value in zip(sessions, labels)
    ]
    (ARTIFACTS / "3_seed_sessions_topic.json").write_text(
        json.dumps({
            "source": INPUT.name,
            "embedding": {k: index[k] for k in ("model", "instruct", "max_chars", "dimensions")},
            "umap": {"n_components": UMAP_DIMS, "n_neighbors": chosen["n_neighbors"],
                     "metric": "cosine", "random_state": UMAP_SEED},
            "hdbscan": {"min_cluster_size": chosen["min_cluster_size"],
                        "min_samples": chosen["min_samples"], "dbcv": round(dbcv, 4)},
            "paper_reference": {"n_neighbors": 15, "min_cluster_size": 30, "min_samples": 5},
            "representatives_per_cluster": REPRESENTATIVES,
            "labeling_model": OPENAI_MODEL,
            "clusters": clusters,
            "noise_count": int((labels < 0).sum()),
            "sessions": labeled_sessions,
        }, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    plane = project(matrix, chosen["n_neighbors"], 2)
    figure, axis = plt.subplots(figsize=(11, 8))
    noise = labels < 0
    axis.scatter(plane[noise, 0], plane[noise, 1], s=14, color=NOISE_COLOR,
                 label=f"noise (n={int(noise.sum())})")
    for position, entry in enumerate(clusters):
        mask = labels == entry["cluster_id"]
        axis.scatter(plane[mask, 0], plane[mask, 1], s=26,
                     color=PALETTE[position % len(PALETTE)],
                     marker="o" if position < len(PALETTE) else "^",
                     label=f"{entry['label']} (n={entry['size']})")
        center = plane[mask].mean(axis=0)
        axis.annotate(entry["label"], center, fontsize=8, color="#333333", ha="center")
    axis.set_title("DSPy seed sessions - UMAP(2) of Qwen3-Embedding-8B, "
                   f"HDBSCAN({chosen['min_cluster_size']},{chosen['min_samples']}), "
                   f"n_neighbors={chosen['n_neighbors']}")
    axis.set_xticks([]), axis.set_yticks([])
    for side in axis.spines.values():
        side.set_color("#d0d0d0")
    axis.legend(loc="best", fontsize=8, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(ARTIFACTS / "3_clusters.png", dpi=150)
    print(f"wrote 3_seed_sessions_topic.json ({len(clusters)} topics, "
          f"{int(noise.sum())} noise) and 3_clusters.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["embed", "sweep", "label"])
    parser.add_argument("--min-cluster-size", type=int)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--n-neighbors", type=int)
    args = parser.parse_args()
    if args.phase == "embed":
        embed()
    elif args.phase == "sweep":
        sweep()
    else:
        label(args.min_cluster_size, args.min_samples, args.n_neighbors)


if __name__ == "__main__":
    main()
