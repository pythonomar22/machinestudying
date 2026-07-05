#!/bin/bash
# Launch one vLLM server per allocated GPU for Qwen/Qwen3.5-9B.
# Run inside a slurm job (CUDA_VISIBLE_DEVICES is cgroup-set there); this script
# re-pins each server to a single GPU. Ports are SB_PORT_BASE+i (derived from the
# slurm job id by the caller — the a3 nodes are shared, fixed ports collide).
# Logs to logs/vllm-<i>.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_LINK_MODE=copy
unset ROCR_VISIBLE_DEVICES

if [ ! -x .venv-vllm/bin/vllm ]; then
    uv venv .venv-vllm -p 3.12
    uv pip install -p .venv-vllm vllm==0.24.0
fi

NGPU=${SB_NGPU:-$(nvidia-smi -L | wc -l)}
PORT_BASE=${SB_PORT_BASE:-8100}
LOG_PREFIX=${SB_VLLM_LOG_PREFIX:-logs/vllm}  # job-unique prefix: stale logs from a
mkdir -p logs                                # previous run must not be re-read
for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i .venv-vllm/bin/vllm serve Qwen/Qwen3.5-9B \
        --port $((PORT_BASE + i)) \
        --max-model-len 262144 \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder \
        --language-model-only \
        > "$LOG_PREFIX-$i.log" 2>&1 &
done
wait
