#!/bin/bash
# Grade one complete StudyBench run with a local Qwen3.5-9B judge.
# Example: sbatch scripts/grading.sbatch runs/smalldspy-nostudy-20260715
#SBATCH --job-name=studybench-grade
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job from the repository with sbatch}"

RUN_PATH=${1:?usage: sbatch scripts/grading.sbatch runs/RUN_ID}
JUDGE=${SB_JUDGE:-local10k}
DEBUG=${SB_DEBUG:-0}

die() {
    echo "FATAL: $*" >&2
    exit 1
}

[[ "$JUDGE" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] \
    || die "SB_JUDGE must contain only letters, digits, '-' and '_'"
[[ "$DEBUG" = 0 || "$DEBUG" = 1 ]] || die "SB_DEBUG must be 0 or 1"
command -v curl >/dev/null || die "curl is required"
command -v setsid >/dev/null || die "setsid is required"
[ -x .venv/bin/python ] || die "run scripts/setup.sh first"
[ -x .venv-vllm/bin/vllm ] || die "run scripts/setup.sh first"
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPUs}"
: "${SLURM_JOB_ID:?grading.sbatch must run inside Slurm}"

mapfile -t TARGETS < <(.venv/bin/python - "$RUN_PATH" <<'PY'
import sys

from studybench.grade import resolve_run_path

for run_id, task in resolve_run_path(sys.argv[1]):
    print(f"{run_id}/{task}")
PY
) || die "invalid run path: $RUN_PATH"
[ "${#TARGETS[@]}" -gt 0 ] || die "run path contains no task runs"
echo "grading: ${TARGETS[*]}"

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
[ -x "$CUDA_HOME/bin/nvcc" ] || die "nvcc not found under CUDA_HOME=$CUDA_HOME"
export CUDA_HOME
export PATH="$PWD/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
NGPU=${#GPUS[@]}
[ "$NGPU" -eq 4 ] || die "expected four visible GPUs"
export SB_EXPECTED_NGPU=$NGPU

GPU_NAMES=$(.venv-vllm/bin/python - <<'PY'
import os
import torch

count = torch.cuda.device_count()
expected = int(os.environ["SB_EXPECTED_NGPU"])
if count != expected:
    raise SystemExit(f"expected {expected} CUDA devices, found {count}")
print(", ".join(torch.cuda.get_device_name(i) for i in range(count)))
PY
)

mapfile -t JUDGE_INFO < <(.venv/bin/python - "$JUDGE" <<'PY'
import json
import sys

from studybench.grade import JUDGES

try:
    judge = JUDGES[sys.argv[1]]
except KeyError:
    raise SystemExit(f"unknown local judge profile: {sys.argv[1]}")
if judge["contract"] != "local":
    raise SystemExit(f"judge profile is not local: {sys.argv[1]}")
runtime = judge["runtime"]
for value in (
    judge["grade_id"],
    judge["model"],
    judge["revision"],
    runtime["vllm"],
    runtime["model_runner"],
    runtime["tensor_parallel"],
    runtime["max_model_len"],
    runtime["max_num_seqs"],
    runtime["reasoning_parser"],
    json.dumps(runtime["structured_outputs"], separators=(",", ":")),
):
    print(value)
PY
)
[ "${#JUDGE_INFO[@]}" -eq 10 ] || die "could not read local judge configuration"
GRADE_ID=${SB_GRADE_ID:-${JUDGE_INFO[0]}}
[[ "$GRADE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] \
    || die "SB_GRADE_ID must contain only letters, digits, '-' and '_'"
MODEL=${JUDGE_INFO[1]}
REVISION=${JUDGE_INFO[2]}
VLLM_VERSION=${JUDGE_INFO[3]}
MODEL_RUNNER=${JUDGE_INFO[4]}
TP=${JUDGE_INFO[5]}
MAX_MODEL_LEN=${JUDGE_INFO[6]}
MAX_NUM_SEQS=${JUDGE_INFO[7]}
REASONING_PARSER=${JUDGE_INFO[8]}
STRUCTURED_OUTPUTS=${JUDGE_INFO[9]}
[ "$MODEL_RUNNER" = v1 ] || die "unsupported vLLM model runner: $MODEL_RUNNER"
[ "$((NGPU % TP))" -eq 0 ] || die "$NGPU GPUs cannot be split into TP=$TP replicas"
NSERVE=$((NGPU / TP))
CONCURRENCY=$NSERVE
echo "GPUs: $GPU_NAMES; replicas=$NSERVE; tensor_parallel=$TP"
INSTALLED_VLLM=$(.venv-vllm/bin/python -c \
    'from importlib.metadata import version; print(version("vllm"))')
[ "$INSTALLED_VLLM" = "$VLLM_VERSION" ] \
    || die "expected vLLM $VLLM_VERSION, found $INSTALLED_VLLM"
: "${HF_HOME:?HF_HOME must point to the Hugging Face cache}"
MODEL_PATH="$HF_HOME/hub/models--${MODEL//\//--}/snapshots/$REVISION"
[ -f "$MODEL_PATH/config.json" ] || die "cached Qwen snapshot is missing: $MODEL_PATH"
PORT_BASE=$((24000 + (SLURM_JOB_ID % 1000) * 8))
export OPENAI_API_KEY
OPENAI_API_KEY=$(.venv-vllm/bin/python -c \
    'import secrets; print(secrets.token_urlsafe(48))')
export VLLM_API_KEY=$OPENAI_API_KEY
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY

PIDS=()
URLS=()
LOGS=()
PREEXISTING_GRADE_ROOTS=()
for target in "${TARGETS[@]}"; do
    grade_root="grades/${target%%/*}/$GRADE_ID/${target#*/}"
    [ ! -d "$grade_root" ] || PREEXISTING_GRADE_ROOTS+=("$grade_root")
done
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for _ in $(seq 1 30); do
        alive=0
        for pid in "${PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive=1
        done
        [ "$alive" = 1 ] || break
        sleep 1
    done
    for pid in "${PIDS[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    if [ "$status" -ne 0 ]; then
        for target in "${TARGETS[@]}"; do
            grade_root="grades/${target%%/*}/$GRADE_ID/${target#*/}"
            preexisting=0
            for existing in "${PREEXISTING_GRADE_ROOTS[@]}"; do
                [ "$existing" != "$grade_root" ] || preexisting=1
            done
            if [ "$preexisting" = 0 ] && [ -d "$grade_root" ]; then
                find "$grade_root" -type f -delete
                find "$grade_root" -depth -type d -empty -delete
            fi
        done
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

servers_alive() {
    local pid
    for pid in "${PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null || return 1
    done
}

mkdir -p logs
LOG_TAG="${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
for ((i = 0; i < NSERVE; i++)); do
    devices=$(IFS=,; printf '%s' "${GPUS[*]:$((i * TP)):$TP}")
    port=$((PORT_BASE + i))
    log="logs/grading-vllm-${LOG_TAG}-${i}.log"
    setsid env CUDA_VISIBLE_DEVICES="$devices" \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        VLLM_USE_V2_MODEL_RUNNER=0 \
        VLLM_API_KEY="$OPENAI_API_KEY" \
        .venv-vllm/bin/vllm serve "$MODEL_PATH" \
        --served-model-name "$MODEL" \
        --host 127.0.0.1 --port "$port" \
        --tensor-parallel-size "$TP" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --structured-outputs-config "$STRUCTURED_OUTPUTS" \
        --reasoning-parser "$REASONING_PARSER" \
        --language-model-only >"$log" 2>&1 &
    PIDS+=("$!")
    URLS+=("http://127.0.0.1:$port/v1")
    LOGS+=("$log")

    ready=0
    for _ in $(seq 1 360); do
        if ! servers_alive; then
            tail -n 80 "${LOGS[@]}" >&2 || true
            die "a Qwen judge server crashed during startup"
        fi
        if curl --noproxy '*' --fail --silent \
            --connect-timeout 2 --max-time 5 \
            -H "Authorization: Bearer $OPENAI_API_KEY" \
            -o /dev/null "http://127.0.0.1:$port/health"; then
            ready=1
            break
        fi
        sleep 5
    done
    if [ "$ready" = 0 ]; then
        tail -n 80 "$log" >&2 || true
        die "Qwen judge server $i did not become healthy within 30 minutes"
    fi
done
BASE_URLS=$(IFS=,; printf '%s' "${URLS[*]}")
echo "local Qwen judge ready: $BASE_URLS"

args=(
    "$RUN_PATH"
    --judge "$JUDGE"
    --grade-id "$GRADE_ID"
    --base-urls "$BASE_URLS"
    --concurrency "$CONCURRENCY"
)
[ "$DEBUG" = 0 ] || args+=(--debug)

servers_alive || die "a Qwen judge server crashed before grading"
.venv/bin/python -m studybench.grade "${args[@]}" --limit 5
servers_alive || die "a Qwen judge server crashed during the smoke grade"
.venv/bin/python -m studybench.grade "${args[@]}"
servers_alive || die "a Qwen judge server crashed during grading"

for target in "${TARGETS[@]}"; do
    run_id=${target%%/*}
    task=${target#*/}
    echo "local grading complete: grades/$run_id/$GRADE_ID/$task/report.json"
done
