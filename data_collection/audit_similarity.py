"""Audit: embedding-space similarity of the generated validation questions
to the held-out test questions and to their seed anchors.

Complements s5's shingle-based decontamination with a semantic check using
the same Qwen3-Embedding-8B setup as s2a. Reports, for every generated
question, its cosine similarity to each test question and to its nearest
seed; high test similarity (>= 0.95) is flagged for human review.

Run on a GPU node inside the interactive Slurm allocation:
  srun --overlap --jobid=<JOBID> .venv-vllm/bin/python data_collection/audit_similarity.py
"""

from __future__ import annotations

import json
from pathlib import Path

from s2a_embed import INSTRUCT, MAX_CHARS, MODEL

DC = Path(__file__).resolve().parent
ROOT = DC.parent
FLAG_THRESHOLD = 0.95


def main() -> None:
    import numpy as np
    from vllm import LLM

    generated = [
        json.loads(line)
        for line in open(ROOT / "data/smalldspy_ourvalidationset.jsonl", encoding="utf-8")
    ]
    tests = [
        json.loads(line)
        for line in open(ROOT / "data/smalldspy.jsonl", encoding="utf-8")
    ]
    seeds = [
        json.loads(line)
        for line in open(DC / "artifacts/seed_pool.jsonl", encoding="utf-8")
    ]
    texts = (
        [INSTRUCT + row["question"][:MAX_CHARS] for row in generated]
        + [INSTRUCT + row["question"][:MAX_CHARS] for row in tests]
        + [INSTRUCT + row["question_text"][:MAX_CHARS] for row in seeds]
    )
    llm = LLM(model=MODEL, runner="pooling", max_model_len=8192,
              gpu_memory_utilization=0.85, enforce_eager=True)
    matrix = np.array([out.outputs.embedding for out in llm.embed(texts)], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    n_generated, n_tests = len(generated), len(tests)
    G = matrix[:n_generated]
    T = matrix[n_generated : n_generated + n_tests]
    S = matrix[n_generated + n_tests :]

    result = {"model": MODEL, "flag_threshold": FLAG_THRESHOLD, "questions": []}
    for position, row in enumerate(generated):
        test_sims = {tests[j]["id"]: round(float(G[position] @ T[j]), 4)
                     for j in range(n_tests)}
        seed_sims = S @ G[position]
        nearest = int(np.argmax(seed_sims))
        worst = max(test_sims, key=test_sims.get)
        result["questions"].append({
            "id": row["id"],
            "max_test_similarity": test_sims[worst],
            "nearest_test": worst,
            "test_similarities": test_sims,
            "nearest_seed_issue": seeds[nearest]["number"],
            "nearest_seed_similarity": round(float(seed_sims[nearest]), 4),
            "flagged": test_sims[worst] >= FLAG_THRESHOLD,
        })
        print(f"{row['id']}: max test sim {test_sims[worst]:.3f} ({worst}), "
              f"nearest seed #{seeds[nearest]['number']} "
              f"({seed_sims[nearest]:.3f})"
              + ("  <-- FLAGGED" if test_sims[worst] >= FLAG_THRESHOLD else ""))
    (DC / "artifacts" / "audit_similarity.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    flagged = [q["id"] for q in result["questions"] if q["flagged"]]
    print(f"flagged: {flagged or 'none'}")


if __name__ == "__main__":
    main()
