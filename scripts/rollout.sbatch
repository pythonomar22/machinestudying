#!/bin/bash
#SBATCH --job-name=sb-rollout
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
# Generation only (grading is offline): serve Qwen3.5-9B, then drive the ReAct
# rollouts. Resumable — finished episodes in runs/ are skipped on resubmit.
# Scope for smoke tests via: sbatch --gpus-per-node=1 --time=2:00:00 \
#   --export=ALL,SB_TASKS=dspy,SB_BUDGETS=k5,SB_ROLLOUTS=1,SB_LIMIT=2 scripts/rollout.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_LINK_MODE=copy

TASKS=${SB_TASKS:-dspy,openclaw}
BUDGETS=${SB_BUDGETS:-direct,k5,k20,k20f}
ROLLOUTS=${SB_ROLLOUTS:-3}
LIMIT=${SB_LIMIT:-0}
export SB_NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
# nodes are shared and jobs can run concurrently: job-unique ports (stride 8 so
# consecutive job ids cannot overlap) and job-unique vLLM logs
export SB_PORT_BASE=$((20000 + (SLURM_JOB_ID % 2000) * 8))
export SB_VLLM_LOG_PREFIX="logs/vllm-$SLURM_JOB_ID"

uv sync
setsid bash scripts/serve_vllm.sh &
SERVE_PID=$!
trap 'kill -TERM -- -$SERVE_PID 2>/dev/null || true' EXIT

BASE_URLS=""
for i in $(seq 0 $((SB_NGPU - 1))); do
    BASE_URLS+="${BASE_URLS:+,}http://localhost:$((SB_PORT_BASE + i))/v1"
    for _ in $(seq 1 240); do  # up to 40 min (first startup compiles kernels)
        curl -sf "http://localhost:$((SB_PORT_BASE + i))/health" > /dev/null && break
        if grep -q "Traceback (most recent call last)" "$SB_VLLM_LOG_PREFIX-$i.log" 2>/dev/null; then
            echo "FATAL: vLLM server $i crashed during startup:"
            tail -50 "$SB_VLLM_LOG_PREFIX-$i.log"
            exit 1
        fi
        sleep 10
    done
    curl -sf "http://localhost:$((SB_PORT_BASE + i))/health" > /dev/null \
        || { echo "FATAL: vLLM server $i never became healthy"; tail -50 "$SB_VLLM_LOG_PREFIX-$i.log"; exit 1; }
done
echo "all $SB_NGPU vLLM servers healthy: $BASE_URLS"

for task in ${TASKS//,/ }; do
    .venv/bin/python -m studybench.rollout --task "$task" \
        --budgets "$BUDGETS" --rollouts "$ROLLOUTS" --limit "$LIMIT" \
        --base-urls "$BASE_URLS" --concurrency $((8 * SB_NGPU))
done
echo "rollouts complete"
