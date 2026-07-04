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
# Scope for smoke tests via: sbatch --export=ALL,SB_TASKS=dspy,SB_BUDGETS=k5,SB_ROLLOUTS=1,SB_LIMIT=2 ...
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_LINK_MODE=copy

TASKS=${SB_TASKS:-dspy,openclaw}
BUDGETS=${SB_BUDGETS:-direct,k5,k20,k20f}
ROLLOUTS=${SB_ROLLOUTS:-3}
LIMIT=${SB_LIMIT:-0}
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

[ -d .venv ] || uv sync
bash scripts/serve_vllm.sh &
SERVE_PID=$!

BASE_URLS=""
for i in $(seq 0 $((NGPU - 1))); do
    BASE_URLS+="${BASE_URLS:+,}http://localhost:$((8100 + i))/v1"
    for _ in $(seq 1 240); do  # up to 40 min for first-time startup
        curl -sf "http://localhost:$((8100 + i))/health" > /dev/null && break
        sleep 10
    done
    curl -sf "http://localhost:$((8100 + i))/health" > /dev/null \
        || { echo "FATAL: vLLM server $i never became healthy"; kill $SERVE_PID; exit 1; }
done
echo "all $NGPU vLLM servers healthy: $BASE_URLS"

for task in ${TASKS//,/ }; do
    .venv/bin/python -m studybench.rollout --task "$task" \
        --budgets "$BUDGETS" --rollouts "$ROLLOUTS" --limit "$LIMIT" \
        --base-urls "$BASE_URLS" --concurrency $((8 * NGPU))
done

kill $SERVE_PID 2>/dev/null
wait
echo "rollouts complete"
