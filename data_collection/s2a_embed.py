"""s2a: embed seed questions with Qwen3-Embedding-8B (paper Stage 2, part 1).

The paper embeds each session's first substantive user question with
Qwen3-Embedding-8B using a domain-aware prefix prompt; the exact prefix is
unpublished, so INSTRUCT below is our reconstruction and is recorded in the
output index. The 5 held-out SmallDSPy test questions are embedded in the
same batch — they take no part in clustering (s2b excludes them), but s2b
uses them to sanity-check cluster selection and s5 reuses the similarities
for decontamination.

Run on a GPU node inside the interactive Slurm allocation:
  srun --overlap --jobid=<JOBID> .venv-vllm/bin/python data_collection/s2a_embed.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DC = Path(__file__).resolve().parent
ROOT = DC.parent
MODEL = "Qwen/Qwen3-Embedding-8B"
INSTRUCT = (
    "Instruct: Identify the topic and user intent of a question about the "
    "DSPy language-model programming library.\nQuery: "
)
MAX_CHARS = 4_000


def main() -> None:
    from vllm import LLM

    seeds = [
        json.loads(line)
        for line in open(DC / "artifacts/seed_pool.jsonl", encoding="utf-8")
    ]
    tests = [
        json.loads(line)
        for line in open(ROOT / "data/smalldspy.jsonl", encoding="utf-8")
    ]
    index = [
        {"kind": "seed", "key": str(row["number"])} for row in seeds
    ] + [
        {"kind": "test", "key": row["id"]} for row in tests
    ]
    texts = [
        INSTRUCT + row["question_text"][:MAX_CHARS] for row in seeds
    ] + [
        INSTRUCT + row["question"][:MAX_CHARS] for row in tests
    ]

    llm = LLM(
        model=MODEL,
        runner="pooling",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    outputs = llm.embed(texts)
    matrix = np.array([out.outputs.embedding for out in outputs], dtype=np.float32)
    if matrix.shape[0] != len(index):
        raise RuntimeError(f"embedded {matrix.shape[0]} of {len(index)} texts")
    norms = np.linalg.norm(matrix, axis=1)
    np.save(DC / "artifacts/embeddings.npy", matrix)
    (DC / "artifacts/embeddings_index.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "instruct": INSTRUCT,
                "max_chars": MAX_CHARS,
                "dimensions": int(matrix.shape[1]),
                "normalized": bool(np.allclose(norms, 1.0, atol=1e-3)),
                "rows": index,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"embedded {matrix.shape[0]} texts, dim={matrix.shape[1]}, "
          f"norm range [{norms.min():.4f}, {norms.max():.4f}]")


if __name__ == "__main__":
    main()
