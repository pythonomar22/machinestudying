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
# Partition/GPU count are submit-time knobs on other clusters, e.g.
#   sbatch -p <partition> --gpus-per-node=7 scripts/rollout.sbatch
# Scope for smoke tests via (2 GPUs: one TP=2 server on 48GB L40S nodes):
#   sbatch --gpus-per-node=2 --time=2:00:00 \
#     --export=ALL,SB_TASKS=dspy,SB_BUDGETS=k5,SB_ROLLOUTS=1,SB_LIMIT=2 scripts/rollout.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_LINK_MODE=copy

TASKS=${SB_TASKS:-dspy,openclaw}
BUDGETS=${SB_BUDGETS:-direct,k5,k20,k20f}
ROLLOUTS=${SB_ROLLOUTS:-3}
LIMIT=${SB_LIMIT:-0}
VARIANT=${SB_VARIANT:-}

uv sync
source scripts/serve_and_wait.sh

for task in ${TASKS//,/ }; do
    .venv/bin/python -m studybench.rollout --task "$task" \
        --budgets "$BUDGETS" --rollouts "$ROLLOUTS" --limit "$LIMIT" \
        --variant "$VARIANT" \
        --base-urls "$BASE_URLS" --concurrency $((8 * SB_NGPU))
done
echo "rollouts complete"
