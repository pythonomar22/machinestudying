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
# flashinfer JIT-compiles kernels at startup; it needs nvcc and ninja on PATH
# (ninja is pip-installed into the venv, whose bin/ is not on PATH otherwise)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$PWD/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"

if [ ! -x .venv-vllm/bin/vllm ]; then
    uv venv .venv-vllm -p 3.12
    uv pip install -p .venv-vllm vllm==0.24.0 ninja
fi

NGPU=${SB_NGPU:-$(nvidia-smi -L | wc -l)}
TP=${SB_TP:-1}  # tensor-parallel size per server (e.g. 2 on 48GB L40S if KV is tight)
PORT_BASE=${SB_PORT_BASE:-8100}
LOG_PREFIX=${SB_VLLM_LOG_PREFIX:-logs/vllm}  # job-unique prefix: stale logs from a
mkdir -p logs                                # previous run must not be re-read
for i in $(seq 0 $((NGPU / TP - 1))); do
    CUDA_VISIBLE_DEVICES=$(seq -s, $((i * TP)) $((i * TP + TP - 1))) \
    .venv-vllm/bin/vllm serve Qwen/Qwen3.5-9B \
        --port $((PORT_BASE + i)) \
        --tensor-parallel-size "$TP" \
        --max-model-len 262144 \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder \
        --language-model-only \
        > "$LOG_PREFIX-$i.log" 2>&1 &
done
wait
