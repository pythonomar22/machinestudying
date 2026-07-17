# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy<2.3",
#     "scikit-learn<1.6",
#     "umap-learn==0.5.7",
#     "matplotlib==3.10.3",
# ]
# ///
"""Audit: place the full 30-question Study-DSPy benchmark in our embedding space.

SmallDSPy is our own 5-question subset (exactly the react_agents_and_tools
five of the released 30), so as a broader sanity check this embeds every
question in data/dspy.jsonl (all 6 released topics) with the same
model/instruct as s2a, lays them over the issue seed pool with the same
UMAP settings as s2b, and reports which issue cluster each topic's questions
sit nearest in the original embedding space. Existing artifacts are not
touched; outputs are embeddings_fulldspy.npy, clusters_fulldspy.png, and
audit_fulldspy.json.

Two phases (embedding needs the vLLM env, plotting needs matplotlib):
  srun --overlap --jobid=<JOBID> .venv-vllm/bin/python data_collection/audit_fulldspy.py embed
  uv run data_collection/audit_fulldspy.py plot
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DC = Path(__file__).resolve().parent
ROOT = DC.parent
UMAP_NEIGHBORS, UMAP_SEED = 15, 20260716  # match s2b

# dataviz reference categorical palette, fixed order (one hue per topic).
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
           "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
SEED_CLUSTERED, SEED_NOISE = "#9aa0a6", "#d9dbde"


def load_benchmark() -> list[dict]:
    return [json.loads(line) for line in open(ROOT / "data/dspy.jsonl", encoding="utf-8")]


def embed() -> None:
    import numpy as np
    from vllm import LLM

    from s2a_embed import INSTRUCT, MAX_CHARS, MODEL

    benchmark = load_benchmark()
    ours = [json.loads(line) for line in
            open(ROOT / "data/smalldspy_ourvalidationset.jsonl", encoding="utf-8")]
    llm = LLM(model=MODEL, runner="pooling", max_model_len=8192,
              gpu_memory_utilization=0.85, enforce_eager=True)
    texts = [INSTRUCT + row["question"][:MAX_CHARS] for row in benchmark + ours]
    matrix = np.array([out.outputs.embedding for out in llm.embed(texts)], dtype=np.float32)
    np.save(DC / "artifacts/embeddings_fulldspy.npy", matrix[: len(benchmark)])
    np.save(DC / "artifacts/embeddings_ourval.npy", matrix[len(benchmark):])
    print(f"embedded {len(benchmark)} benchmark + {len(ours)} generated questions")


def plot() -> None:
    import matplotlib
    import numpy as np
    import umap

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = json.loads((DC / "artifacts/embeddings_index.json").read_text(encoding="utf-8"))
    matrix = np.load(DC / "artifacts/embeddings.npy")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    seed_rows = [i for i, row in enumerate(index["rows"]) if row["kind"] == "seed"]
    seed_keys = [index["rows"][i]["key"] for i in seed_rows]
    seed_matrix = matrix[seed_rows]

    benchmark = load_benchmark()
    smalldspy_ids = {
        json.loads(line)["id"]
        for line in open(ROOT / "data/smalldspy.jsonl", encoding="utf-8")
    }
    topics = sorted({row["topic"] for row in benchmark})
    bench_matrix = np.load(DC / "artifacts/embeddings_fulldspy.npy")
    bench_matrix /= np.linalg.norm(bench_matrix, axis=1, keepdims=True)

    clusters = json.loads((DC / "artifacts/clusters.json").read_text(encoding="utf-8"))["clusters"]
    assignment = {}  # seed key -> cluster position (index into clusters)
    for position, cluster in enumerate(clusters):
        for number in cluster["member_numbers"]:
            assignment[str(number)] = position
    centroids = np.stack([
        seed_matrix[[i for i, key in enumerate(seed_keys) if assignment.get(key) == position]].mean(axis=0)
        for position in range(len(clusters))
    ])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    # Per-question nearest issue cluster (original space), rolled up by topic.
    per_topic: dict[str, dict] = {}
    per_question = []
    for row, vector in zip(benchmark, bench_matrix):
        sims = centroids @ vector
        nearest = int(np.argmax(sims))
        per_question.append({
            "id": row["id"],
            "topic": row["topic"],
            "in_smalldspy": row["id"] in smalldspy_ids,
            "nearest_cluster": clusters[nearest]["label"],
            "nearest_cluster_cosine": round(float(sims[nearest]), 4),
        })
        bucket = per_topic.setdefault(row["topic"], {"counts": {}, "cosines": []})
        bucket["counts"][clusters[nearest]["label"]] = bucket["counts"].get(clusters[nearest]["label"], 0) + 1
        bucket["cosines"].append(float(sims[nearest]))
    for topic, bucket in per_topic.items():
        members = bench_matrix[[i for i, row in enumerate(benchmark) if row["topic"] == topic]]
        intra = members @ members.T
        bucket["mean_nearest_cosine"] = round(float(np.mean(bucket["cosines"])), 4)
        bucket["mean_intra_topic_cosine"] = round(
            float((intra.sum() - np.trace(intra)) / (len(members) * (len(members) - 1))), 4
        )
        del bucket["cosines"]

    # Register check: do OUR generated questions sit inside the benchmark's
    # own mode? Calibrate with the released set's leave-one-out neighbors.
    ours = [json.loads(line) for line in
            open(ROOT / "data/smalldspy_ourvalidationset.jsonl", encoding="utf-8")]
    ours_matrix = np.load(DC / "artifacts/embeddings_ourval.npy")
    ours_matrix /= np.linalg.norm(ours_matrix, axis=1, keepdims=True)
    bench_sims = bench_matrix @ bench_matrix.T
    np.fill_diagonal(bench_sims, -1.0)
    bench_nn = bench_sims.max(axis=1)
    react_rows = [i for i, row in enumerate(benchmark) if row["topic"] == "react_agents_and_tools"]
    register = {
        "benchmark_nn_cosine": {
            "mean": round(float(bench_nn.mean()), 4),
            "min": round(float(bench_nn.min()), 4),
            "max": round(float(bench_nn.max()), 4),
        },
        "react5_mutual_mean": round(float(
            (bench_matrix[react_rows] @ bench_matrix[react_rows].T)
            [~np.eye(len(react_rows), dtype=bool)].mean()), 4),
        "ours": [],
    }
    for row, vector in zip(ours, ours_matrix):
        to_bench = bench_matrix @ vector
        nearest = int(np.argmax(to_bench))
        register["ours"].append({
            "id": row["id"],
            "nearest_benchmark_cosine": round(float(to_bench[nearest]), 4),
            "nearest_benchmark_id": benchmark[nearest]["id"],
            "nearest_benchmark_topic": benchmark[nearest]["topic"],
            "max_react5_cosine": round(float(to_bench[react_rows].max()), 4),
            "max_seed_cosine": round(float((seed_matrix @ vector).max()), 4),
        })
    (DC / "artifacts/audit_fulldspy.json").write_text(
        json.dumps({"umap_seed": UMAP_SEED, "topics": per_topic,
                    "questions": per_question, "register_check": register},
                   indent=2) + "\n",
        encoding="utf-8",
    )

    plane = umap.UMAP(n_components=2, n_neighbors=UMAP_NEIGHBORS, metric="cosine",
                      random_state=UMAP_SEED).fit_transform(
        np.vstack([seed_matrix, bench_matrix, ours_matrix])
    )
    seed_plane = plane[: len(seed_keys)]
    bench_plane = plane[len(seed_keys): len(seed_keys) + len(benchmark)]
    ours_plane = plane[len(seed_keys) + len(benchmark):]

    figure, axis = plt.subplots(figsize=(12, 8.5))
    clustered_mask = np.array([key in assignment for key in seed_keys])
    axis.scatter(seed_plane[~clustered_mask, 0], seed_plane[~clustered_mask, 1],
                 s=13, color=SEED_NOISE, label="issue seeds: noise")
    axis.scatter(seed_plane[clustered_mask, 0], seed_plane[clustered_mask, 1],
                 s=13, color=SEED_CLUSTERED, label="issue seeds: clustered")
    for position, cluster in enumerate(clusters):
        mask = np.array([assignment.get(key) == position for key in seed_keys])
        center = seed_plane[mask].mean(axis=0)
        axis.annotate(cluster["label"], center, fontsize=8, color="#5f6368", ha="center")
    for position, topic in enumerate(topics):
        mask = np.array([row["topic"] == topic for row in benchmark])
        axis.scatter(bench_plane[mask, 0], bench_plane[mask, 1], s=150, marker="*",
                     color=PALETTE[position % len(PALETTE)],
                     label=f"{topic} (n={int(mask.sum())})")
    subset_mask = np.array([row["id"] in smalldspy_ids for row in benchmark])
    axis.scatter(bench_plane[subset_mask, 0], bench_plane[subset_mask, 1],
                 s=290, facecolors="none", edgecolors="#1a1a1a", linewidths=1.1,
                 label="in SmallDSPy subset (held-out test)")
    axis.scatter(ours_plane[:, 0], ours_plane[:, 1], s=150, marker="D",
                 color="#1a1a1a", label="our generated validation questions (5)")
    axis.set_title("Full Study-DSPy benchmark (stars by topic) and our generated "
                   "questions (diamonds) over the DSPy issue seed pool")
    axis.set_xticks([]), axis.set_yticks([])
    for side in axis.spines.values():
        side.set_color("#d0d0d0")
    axis.legend(loc="best", fontsize=8, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(DC / "artifacts/clusters_fulldspy_ours.png", dpi=150)

    for topic in topics:
        bucket = per_topic[topic]
        print(f"{topic}: nearest clusters {bucket['counts']} | "
              f"mean nearest cosine {bucket['mean_nearest_cosine']} | "
              f"intra-topic {bucket['mean_intra_topic_cosine']}")
    print(f"\nbenchmark leave-one-out NN cosine: {register['benchmark_nn_cosine']}")
    print(f"react-five mutual mean: {register['react5_mutual_mean']}")
    for entry in register["ours"]:
        print(f"{entry['id']}: nearest benchmark {entry['nearest_benchmark_cosine']} "
              f"({entry['nearest_benchmark_topic']}) | max react5 "
              f"{entry['max_react5_cosine']} | max seed {entry['max_seed_cosine']}")
    print("wrote clusters_fulldspy_ours.png and audit_fulldspy.json")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if phase == "embed":
        embed()
    elif phase == "plot":
        plot()
    else:
        raise SystemExit("usage: audit_fulldspy.py [embed|plot]")
