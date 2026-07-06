#!/bin/bash
#SBATCH --job-name=sb-study
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
# Cheatsheet study runs (paper §B Table 4): one forced 50-tool-call ReAct episode
# per corpus; each writes cheatsheets/{task}.md. Cheap: two episodes total.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_LINK_MODE=copy

TASKS=${SB_TASKS:-dspy,openclaw}
export SB_NGPU=1
export SB_PORT_BASE=$((20000 + (SLURM_JOB_ID % 2000) * 8))
export SB_VLLM_LOG_PREFIX="logs/vllm-$SLURM_JOB_ID"

uv sync
setsid bash scripts/serve_vllm.sh &
SERVE_PID=$!
trap 'kill -TERM -- -$SERVE_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
    curl -sf "http://localhost:$SB_PORT_BASE/health" > /dev/null && break
    if grep -q "EngineCore failed to start" "$SB_VLLM_LOG_PREFIX-0.log" 2>/dev/null; then
        echo "FATAL: vLLM crashed during startup"; tail -50 "$SB_VLLM_LOG_PREFIX-0.log"; exit 1
    fi
    sleep 10
done
curl -sf "http://localhost:$SB_PORT_BASE/health" > /dev/null \
    || { echo "FATAL: vLLM never became healthy"; tail -50 "$SB_VLLM_LOG_PREFIX-0.log"; exit 1; }
echo "vLLM healthy"

for task in ${TASKS//,/ }; do
    .venv/bin/python -m studybench.study --task "$task" \
        --base-url "http://localhost:$SB_PORT_BASE/v1"
done
echo "study complete"
