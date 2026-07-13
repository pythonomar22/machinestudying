#!/bin/bash
# Launch vLLM servers for Qwen/Qwen3.5-9B across the allocated GPUs.
# GPU topology is decided here and only here: the GPU list comes from
# CUDA_VISIBLE_DEVICES (slurm cgroup or manual pinning; nvidia-smi otherwise),
# and the tensor-parallel size defaults to what fits the 262k context — TP=1 on
# 80GB (H100), TP=2 on 48GB (L40S), where one full-length sequence needs ~36GB
# of KV cache. One server per TP group, ports SB_PORT_BASE+i (job-unique from
# the caller — nodes may be shared, fixed ports collide). Overrides: SB_NGPU,
# SB_TP, SB_PORT_BASE, SB_VLLM_LOG_PREFIX. After launching, writes
# $SB_VLLM_LOG_PREFIX.env (BASE_URLS/SB_NGPU/SB_NSERVE) for serve_and_wait.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_LINK_MODE=copy
unset ROCR_VISIBLE_DEVICES
# flashinfer JIT-compiles kernels at startup; it needs nvcc and ninja on PATH
# (ninja is pip-installed into the venv, whose bin/ is not on PATH otherwise)
export CUDA_HOME=${CUDA_HOME:-$(ls -d /usr/local/cuda /usr/local/cuda-12* 2>/dev/null | head -1)}
[ -n "$CUDA_HOME" ] || echo "WARNING: no CUDA toolkit found; set CUDA_HOME if vLLM startup needs nvcc" >&2
export PATH="$PWD/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"

PORT_BASE=${SB_PORT_BASE:-8100}
LOG_PREFIX=${SB_VLLM_LOG_PREFIX:-logs/vllm}  # job-unique prefix: stale logs/env
mkdir -p logs                                # from a previous run must not be re-read
rm -f "$LOG_PREFIX.env"

if [ ! -x .venv-vllm/bin/vllm ]; then
    uv venv .venv-vllm -p 3.12
    uv pip install -p .venv-vllm vllm==0.24.0 ninja
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
NGPU=${SB_NGPU:-${#GPUS[@]}}
MIN_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
TP=${SB_TP:-$(( MIN_MIB >= 70000 ? 1 : 2 ))}
NSERVE=$((NGPU / TP))
if [ "$NSERVE" -lt 1 ]; then
    echo "FATAL: $NGPU GPU(s) < TP=$TP (a ${MIN_MIB}MiB GPU cannot hold the 262k context alone)" >&2
    exit 1
fi
[ $((NGPU % TP)) -eq 0 ] || echo "WARNING: $((NGPU % TP)) of $NGPU GPUs idle (TP=$TP)" >&2

URLS=""
for i in $(seq 0 $((NSERVE - 1))); do
    DEVS=$(IFS=,; echo "${GPUS[*]:$((i * TP)):$TP}")
    echo "server $i: GPUs [$DEVS] TP=$TP port $((PORT_BASE + i))"
    CUDA_VISIBLE_DEVICES=$DEVS \
    .venv-vllm/bin/vllm serve Qwen/Qwen3.5-9B \
        --port $((PORT_BASE + i)) \
        --tensor-parallel-size "$TP" \
        --max-model-len 262144 \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder \
        --language-model-only \
        > "$LOG_PREFIX-$i.log" 2>&1 &
    URLS+="${URLS:+,}http://localhost:$((PORT_BASE + i))/v1"
done
printf 'BASE_URLS=%s\nSB_NGPU=%s\nSB_NSERVE=%s\n' "$URLS" "$NGPU" "$NSERVE" > "$LOG_PREFIX.env"
wait
