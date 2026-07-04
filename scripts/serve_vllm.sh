#!/bin/bash
# Launch one vLLM server per allocated GPU for Qwen/Qwen3.5-9B (ports 8100+i).
# Run inside a slurm job (CUDA_VISIBLE_DEVICES is cgroup-set there); serve_vllm.sh
# re-pins each server to a single GPU. Logs to logs/vllm-<i>.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_LINK_MODE=copy
unset ROCR_VISIBLE_DEVICES

if [ ! -d .venv-vllm ]; then
    uv venv .venv-vllm -p 3.12
    uv pip install -p .venv-vllm vllm
fi

NGPU=${SB_NGPU:-$(nvidia-smi -L | wc -l)}
mkdir -p logs
for i in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$i .venv-vllm/bin/vllm serve Qwen/Qwen3.5-9B \
        --port $((8100 + i)) \
        --max-model-len 262144 \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder \
        --language-model-only \
        > "logs/vllm-$i.log" 2>&1 &
done
wait
