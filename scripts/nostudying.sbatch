#!/bin/bash
# Run the no-studying Qwen3.5-9B baseline on the held-out SmallDSPy questions.
# Example:
#   SB_RUN_ID=smalldspy-base SB_SEED=0 sbatch scripts/nostudying.sbatch
#SBATCH --job-name=studybench-eval
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=6
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job with sbatch}"

RUN_ID=${SB_RUN_ID:?set SB_RUN_ID}
SEED=${SB_SEED:?set SB_SEED}
BUDGETS=${SB_BUDGETS:-direct,k5,k20,k20f}
ROLLOUTS=${SB_ROLLOUTS:-3}
LIMIT=${SB_LIMIT:-0}
SMOKE=${SB_SMOKE:-0}
DEBUG=${SB_DEBUG:-0}

die() {
    echo "FATAL: $*" >&2
    exit 1
}

[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] \
    || die "SB_RUN_ID must contain only letters, digits, '-' and '_'"
[[ "$SEED" =~ ^-?[0-9]+$ ]] || die "SB_SEED must be an integer"
[[ "$ROLLOUTS" =~ ^[1-9][0-9]*$ ]] || die "ROLLOUTS must be positive"
[[ "$LIMIT" =~ ^[0-9]+$ ]] || die "LIMIT must be non-negative"
[[ "$SMOKE" = 0 || "$SMOKE" = 1 ]] || die "SMOKE must be 0 or 1"
[[ "$DEBUG" = 0 || "$DEBUG" = 1 ]] || die "DEBUG must be 0 or 1"
if [[ "$SMOKE" = 1 && "$LIMIT" = 0 ]] || [[ "$SMOKE" = 0 && "$LIMIT" != 0 ]]; then
    die "SB_SMOKE=1 requires a positive SB_LIMIT; full runs use neither"
fi

IFS=',' read -r -a BUDGET_LIST <<< "$BUDGETS"
[ "${#BUDGET_LIST[@]}" -gt 0 ] || die "BUDGETS must not be empty"
for budget in "${BUDGET_LIST[@]}"; do
    [[ "$budget" = direct || "$budget" = k5 || "$budget" = k20 \
        || "$budget" = k20f ]] || die "unsupported budget: $budget"
done

command -v curl >/dev/null || die "curl is required"
command -v setsid >/dev/null || die "setsid is required"
[ -x .venv-dspy/bin/python ] || die "run scripts/setup.sh first"
[ -x .venv-vllm/bin/vllm ] || die "run scripts/setup.sh first"
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPUs}"
: "${SLURM_JOB_ID:?nostudying.sbatch must run inside Slurm}"
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
[ -x "$CUDA_HOME/bin/nvcc" ] || die "nvcc not found under CUDA_HOME=$CUDA_HOME"
export CUDA_HOME
export PATH="$PWD/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NGPU=${#GPUS[@]}
[ "$NGPU" -eq 6 ] || die "expected six visible GPUs"
export SB_EXPECTED_NGPU=$NGPU

GPU_INFO=$(.venv-vllm/bin/python - <<'PY'
import os
import torch

count = torch.cuda.device_count()
expected = int(os.environ["SB_EXPECTED_NGPU"])
if count != expected:
    raise SystemExit(f"expected {expected} CUDA devices, found {count}")
print(min(torch.cuda.get_device_properties(i).total_memory for i in range(count)) // 2**20)
print(", ".join(torch.cuda.get_device_name(i) for i in range(count)))
PY
)
MIN_MEMORY=${GPU_INFO%%$'\n'*}
GPU_NAMES=${GPU_INFO#*$'\n'}
[[ "$MIN_MEMORY" =~ ^[1-9][0-9]*$ ]] || die "could not inspect GPU memory"
TP=2
NSERVE=$((NGPU / TP))
echo "GPUs: $GPU_NAMES; minimum ${MIN_MEMORY} MiB; TP=$TP; servers=$NSERVE"

MODEL=Qwen/Qwen3.5-9B
REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
PORT_BASE=$((20000 + (SLURM_JOB_ID % 2000) * 8))
export SB_VLLM_API_KEY
SB_VLLM_API_KEY=$(.venv-vllm/bin/python -c \
    'import secrets; print(secrets.token_urlsafe(48))')
export VLLM_API_KEY=$SB_VLLM_API_KEY
export SB_NSERVE=$NSERVE SB_NGPU=$NGPU SB_TP_EFFECTIVE=$TP
export SB_MODEL_ID=$MODEL SB_MODEL_REVISION=$REVISION SB_VLLM_VERSION=0.24.0
export SB_GPU_NAMES="$GPU_NAMES" SB_GPU_MEMORY_MIB=$MIN_MEMORY
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY

PIDS=()
URLS=()
LOGS=()
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    for pid in "${PIDS[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p logs
for ((i = 0; i < NSERVE; i++)); do
    devices=$(IFS=,; printf '%s' "${GPUS[*]:$((i * TP)):$TP}")
    port=$((PORT_BASE + i))
    log="logs/vllm-${SLURM_JOB_ID}-${i}.log"
    setsid env CUDA_VISIBLE_DEVICES="$devices" \
        VLLM_API_KEY="$SB_VLLM_API_KEY" \
        .venv-vllm/bin/vllm serve "$MODEL" \
        --revision "$REVISION" \
        --host 127.0.0.1 --port "$port" \
        --tensor-parallel-size "$TP" \
        --max-model-len 262144 \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder \
        --language-model-only >"$log" 2>&1 &
    PIDS+=("$!")
    URLS+=("http://127.0.0.1:$port/v1")
    LOGS+=("$log")
done

servers_alive() {
    local pid
    for pid in "${PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null || return 1
    done
}

for ((i = 0; i < NSERVE; i++)); do
    ready=0
    for _ in $(seq 1 360); do
        if ! servers_alive; then
            tail -n 80 "${LOGS[@]}" >&2 || true
            die "a vLLM server crashed during startup"
        fi
        if curl --noproxy '*' --fail --silent \
            --connect-timeout 2 --max-time 5 \
            -H "Authorization: Bearer $SB_VLLM_API_KEY" \
            -o /dev/null "${URLS[$i]%/v1}/health"; then
            ready=1
            break
        fi
        sleep 5
    done
    if [ "$ready" = 0 ]; then
        tail -n 80 "${LOGS[$i]}" >&2 || true
        die "vLLM server $i did not become healthy within 30 minutes"
    fi
done
servers_alive || die "a vLLM server crashed after startup"
BASE_URLS=$(IFS=,; printf '%s' "${URLS[*]}")
echo "vLLM ready: $BASE_URLS"

args=(
    --task smalldspy
    --condition baseline
    --run-id "$RUN_ID"
    --seed "$SEED"
    --budgets "$BUDGETS"
    --rollouts "$ROLLOUTS"
    --limit "$LIMIT"
    --base-urls "$BASE_URLS"
    --concurrency "$((8 * NSERVE))"
)
[ "$SMOKE" = 0 ] || args+=(--smoke)
[ "$DEBUG" = 0 ] || args+=(--debug)
servers_alive || die "a vLLM server crashed before evaluation"
.venv-dspy/bin/python -m studybench.react "${args[@]}"
servers_alive || die "a vLLM server crashed during evaluation"

echo "no-studying evaluation complete: runs/$RUN_ID/smalldspy"
