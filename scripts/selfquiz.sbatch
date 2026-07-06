#!/bin/bash
#SBATCH --job-name=sb-selfquiz
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
# One self-quizzing study round (studybench/selfquiz.py). Resumable at item
# granularity. SB_ROUND selects the round; SB_SMOKE=1 runs 1 chapter x 3
# questions; SB_TASKS selects corpora.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_LINK_MODE=copy

TASKS=${SB_TASKS:-dspy,openclaw}
ROUND=${SB_ROUND:-1}
SMOKE=${SB_SMOKE:-0}
export SB_NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
export SB_PORT_BASE=$((20000 + (SLURM_JOB_ID % 2000) * 8))
export SB_VLLM_LOG_PREFIX="logs/vllm-$SLURM_JOB_ID"

setsid bash scripts/serve_vllm.sh &
SERVE_PID=$!
trap 'kill -TERM -- -$SERVE_PID 2>/dev/null || true' EXIT

BASE_URLS=""
for i in $(seq 0 $((SB_NGPU - 1))); do
    BASE_URLS+="${BASE_URLS:+,}http://localhost:$((SB_PORT_BASE + i))/v1"
    for _ in $(seq 1 240); do
        curl -sf "http://localhost:$((SB_PORT_BASE + i))/health" > /dev/null && break
        if grep -q "EngineCore failed to start" "$SB_VLLM_LOG_PREFIX-$i.log" 2>/dev/null; then
            echo "FATAL: vLLM server $i crashed"; tail -50 "$SB_VLLM_LOG_PREFIX-$i.log"; exit 1
        fi
        sleep 10
    done
    curl -sf "http://localhost:$((SB_PORT_BASE + i))/health" > /dev/null \
        || { echo "FATAL: vLLM server $i never became healthy"; tail -50 "$SB_VLLM_LOG_PREFIX-$i.log"; exit 1; }
done
echo "all $SB_NGPU vLLM servers healthy: $BASE_URLS"

FLAGS=""
[ "$SMOKE" = 1 ] && FLAGS="--smoke --debug"
for task in ${TASKS//,/ }; do
    .venv-dspy/bin/python -m studybench.selfquiz --task "$task" --round "$ROUND" \
        --base-urls "$BASE_URLS" --concurrency $((4 * SB_NGPU)) $FLAGS
done
echo "selfquiz round $ROUND complete"
