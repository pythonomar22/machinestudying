#!/bin/bash
# Launch authenticated vLLM servers using exactly the GPUs exposed by Slurm.
# Every identity needed for a claim-ready run is snapshotted before launch.
set -euo pipefail
umask 077
cd "$(dirname "$0")/.."
export UV_LINK_MODE=copy
unset ROCR_VISIBLE_DEVICES
: "${SLURM_JOB_ID:?serve_vllm.sh must run inside a Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs for this job}"
source scripts/setup_common.sh
verify_env_file  # validates path/type/owner/mode only; never reads secret contents
require_command hostname
require_command env
require_command nvidia-smi
require_command sha256sum
NVIDIA_SMI=$(command -v nvidia-smi)
[[ "$NVIDIA_SMI" = /* && -x "$NVIDIA_SMI" ]] \
    || { echo "FATAL: nvidia-smi must resolve to an absolute executable" >&2; exit 1; }
[ -x .venv-vllm/bin/vllm ] \
    || { echo "FATAL: missing pinned .venv-vllm; run scripts/setup.sh first" >&2; exit 1; }
verify_vllm_environment \
    || { echo "FATAL: vLLM environment drift; rerun setup" >&2; exit 1; }

# flashinfer JIT-compiles kernels at startup and requires both nvcc and ninja.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
[ -x "$CUDA_HOME/bin/nvcc" ] \
    || { echo "FATAL: CUDA toolkit not found at $CUDA_HOME" >&2; exit 1; }
CUDA_HOME=$(cd "$CUDA_HOME" && pwd -P)
export CUDA_HOME
export PATH="$PWD/.venv-vllm/bin:$CUDA_HOME/bin:$PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Keep inherited vLLM/PyTorch/NCCL/Python/proxy settings from silently changing
# model behavior.  Cluster CUDA installs can require LD_LIBRARY_PATH, so its
# exact value is the sole inherited tuning path and is recorded below.
: "${HOME:?HOME is required for the pinned model cache}"
SERVER_HOME=$(cd "$HOME" && pwd -P)
SERVER_PATH="$PWD/.venv-vllm/bin:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin"
SERVER_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
SANITIZED_ENV=(
    env -i
    "HOME=$SERVER_HOME"
    "PATH=$SERVER_PATH"
    "LD_LIBRARY_PATH=$SERVER_LD_LIBRARY_PATH"
    "CUDA_HOME=$CUDA_HOME"
    "LANG=C.UTF-8"
    "LC_ALL=C.UTF-8"
    "PYTHONHASHSEED=0"
    "PYTHONNOUSERSITE=1"
)

MODEL_ID=Qwen/Qwen3.5-9B
MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
PORT_BASE=${SB_PORT_BASE:-8100}
LOG_PREFIX=${SB_VLLM_LOG_PREFIX:-logs/vllm}
[[ "$PORT_BASE" =~ ^[1-9][0-9]*$ ]] && [ "$PORT_BASE" -ge 1024 ] \
    && [ "$PORT_BASE" -le 65535 ] \
    || { echo "FATAL: invalid SB_PORT_BASE: $PORT_BASE" >&2; exit 1; }
[[ "$LOG_PREFIX" =~ ^logs/[A-Za-z0-9._-]+$ ]] \
    || { echo "FATAL: SB_VLLM_LOG_PREFIX must be one name under logs/" >&2; exit 1; }
mkdir -p logs
[ -d logs ] && [ ! -L logs ] \
    || { echo "FATAL: logs/ must be a real directory" >&2; exit 1; }

ARTIFACTS=(
    "$LOG_PREFIX.topology"
    "$LOG_PREFIX.env"
    "$LOG_PREFIX.packages.txt"
    "$LOG_PREFIX.vllm-runtime.json"
    "$LOG_PREFIX.model-cache.json"
    "$LOG_PREFIX.gpus.json"
)
for artifact in "${ARTIFACTS[@]}"; do
    if [ -e "$artifact" ] || [ -L "$artifact" ]; then
        echo "FATAL: log prefix already has artifact: $artifact" >&2
        exit 1
    fi
done
if compgen -G "$LOG_PREFIX-*.log" > /dev/null; then
    echo "FATAL: log prefix already has server logs" >&2
    exit 1
fi

PIDS=()
TEMP_FILES=()
cleanup() {
    if [ "${#PIDS[@]}" -gt 0 ]; then
        kill "${PIDS[@]}" 2>/dev/null || true
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
    if [ "${#TEMP_FILES[@]}" -gt 0 ]; then
        rm -f "${TEMP_FILES[@]}"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$CUDA_VISIBLE_DEVICES" != ,* && "$CUDA_VISIBLE_DEVICES" != *, \
    && "$CUDA_VISIBLE_DEVICES" != *,,* ]] \
    || { echo "FATAL: CUDA_VISIBLE_DEVICES contains an empty identifier" >&2; exit 1; }
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NGPU=${#GPUS[@]}
[ "$NGPU" -gt 0 ] || { echo "FATAL: empty CUDA_VISIBLE_DEVICES" >&2; exit 1; }
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
    [[ "$gpu" =~ ^([0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9-]+)$ ]] \
        || { echo "FATAL: invalid visible GPU identifier: $gpu" >&2; exit 1; }
    [ -z "${SEEN_GPUS[$gpu]:-}" ] \
        || { echo "FATAL: duplicate CUDA_VISIBLE_DEVICES identifier: $gpu" >&2; exit 1; }
    SEEN_GPUS[$gpu]=1
done
if [ -n "${SB_NGPU:-}" ]; then
    [[ "$SB_NGPU" =~ ^[1-9][0-9]*$ ]] \
        || { echo "FATAL: SB_NGPU must be a positive integer" >&2; exit 1; }
    [ "$SB_NGPU" -eq "$NGPU" ] \
        || { echo "FATAL: SB_NGPU disagrees with CUDA_VISIBLE_DEVICES" >&2; exit 1; }
fi

GPU_ROWS="$LOG_PREFIX.gpus.tsv.tmp-$$"
CUDA_GPU_ROWS="$LOG_PREFIX.cuda-gpus.tsv.tmp-$$"
TEMP_FILES+=("$GPU_ROWS" "$CUDA_GPU_ROWS")
: > "$GPU_ROWS"
MEMORIES=()
declare -A SEEN_GPU_UUIDS=()
query_allocated_gpu() {
    local selector=$1 field=$2 value
    value=$("${SANITIZED_ENV[@]}" "$NVIDIA_SMI" -i "$selector" \
        --query-gpu="$field" --format=csv,noheader,nounits) \
        || return 1
    [ -n "$value" ] && [[ "$value" != *$'\n'* ]] && [[ "$value" != *$'\t'* ]] \
        || return 1
    printf '%s' "$value"
}
# A Slurm CUDA ordinal can be remapped and need not be the same physical index
# understood by nvidia-smi. Resolve logical CUDA devices to UUIDs first, then
# use only those UUIDs as nvidia-smi selectors.
"${SANITIZED_ENV[@]}" \
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
    .venv-vllm/bin/python - "$CUDA_VISIBLE_DEVICES" "$CUDA_GPU_ROWS" <<'PY'
from pathlib import Path
import sys

import torch

tokens = sys.argv[1].split(",")
if torch.cuda.device_count() != len(tokens):
    raise SystemExit(
        "CUDA runtime visible-device count disagrees with CUDA_VISIBLE_DEVICES"
    )
rows = []
for logical_index, token in enumerate(tokens):
    uuid = torch.cuda.get_device_properties(logical_index).uuid
    if not isinstance(uuid, str) or not uuid or "\t" in uuid or "\n" in uuid:
        raise SystemExit(f"CUDA logical device {logical_index} has no safe UUID")
    rows.append(f"{token}\t{uuid}\n")
Path(sys.argv[2]).write_text("".join(rows), encoding="utf-8")
PY

while IFS=$'\t' read -r gpu uuid; do
    [ -n "$gpu" ] && [ -n "$uuid" ] \
        || { echo "FATAL: CUDA runtime returned an incomplete GPU identity" >&2; exit 1; }
    [ -z "${SEEN_GPU_UUIDS[$uuid]:-}" ] \
        || { echo "FATAL: allocated GPU identifiers alias the same UUID: $uuid" >&2; exit 1; }
    SEEN_GPU_UUIDS[$uuid]=1
    name=$(query_allocated_gpu "$uuid" name) \
        || { echo "FATAL: cannot query name for allocated GPU $gpu" >&2; exit 1; }
    memory=$(query_allocated_gpu "$uuid" memory.total) \
        || { echo "FATAL: cannot query memory for allocated GPU $gpu" >&2; exit 1; }
    driver=$(query_allocated_gpu "$uuid" driver_version) \
        || { echo "FATAL: cannot query driver for allocated GPU $gpu" >&2; exit 1; }
    [[ "$memory" =~ ^[1-9][0-9]*$ ]] \
        || { echo "FATAL: invalid memory size for allocated GPU $gpu" >&2; exit 1; }
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$gpu" "$uuid" "$name" "$memory" "$driver" >> "$GPU_ROWS"
    MEMORIES+=("$memory")
done < "$CUDA_GPU_ROWS"
[ "${#MEMORIES[@]}" -eq "$NGPU" ] \
    || { echo "FATAL: CUDA runtime omitted an allocated GPU" >&2; exit 1; }

GPU_INVENTORY="$LOG_PREFIX.gpus.json"
"${SANITIZED_ENV[@]}" .venv-vllm/bin/python - "$GPU_ROWS" "$GPU_INVENTORY" \
    "$CUDA_VISIBLE_DEVICES" "$SLURM_JOB_ID" "${SLURM_JOB_GPUS:-}" \
    "${SLURM_STEP_GPUS:-}" "${SLURM_JOB_NODELIST:-}" "${SLURM_NODEID:-}" \
    "$(hostname)" <<'PY'
import json
import os
from pathlib import Path
import sys

rows_path, destination = map(Path, sys.argv[1:3])
gpus = []
for line in rows_path.read_text(encoding="utf-8").splitlines():
    fields = line.split("\t")
    if len(fields) != 5:
        raise SystemExit("invalid allocated-GPU inventory row")
    identifier, uuid, name, memory, driver = fields
    gpus.append({
        "cuda_identifier": identifier,
        "uuid": uuid,
        "name": name,
        "memory_mib": int(memory),
        "driver_version": driver,
    })
record = {
    "schema_version": 1,
    "hostname": sys.argv[9],
    "cuda_visible_devices": sys.argv[3],
    "gpu_count": len(gpus),
    "gpus": gpus,
    "slurm": {
        "job_id": sys.argv[4],
        "job_gpus": sys.argv[5] or None,
        "step_gpus": sys.argv[6] or None,
        "job_nodelist": sys.argv[7] or None,
        "node_id": sys.argv[8] or None,
    },
}
payload = (json.dumps(record, allow_nan=False, ensure_ascii=False,
                      separators=(",", ":"), sort_keys=True) + "\n")
destination.write_text(payload, encoding="utf-8")
os.chmod(destination, 0o600)
PY
GPU_INVENTORY_SHA256=$(sha256sum "$GPU_INVENTORY" | awk '{print $1}')

MIN_MIB=$(printf '%s\n' "${MEMORIES[@]}" | sort -n | head -1)
TP=${SB_TP:-$(( MIN_MIB >= 70000 ? 1 : 2 ))}
[[ "$TP" =~ ^[1-9][0-9]*$ ]] \
    || { echo "FATAL: invalid tensor parallel size: $TP" >&2; exit 1; }
NSERVE=$((NGPU / TP))
[ "$NSERVE" -ge 1 ] \
    || { echo "FATAL: $NGPU allocated GPU(s) cannot satisfy TP=$TP" >&2; exit 1; }
[ $((NGPU % TP)) -eq 0 ] \
    || { echo "FATAL: $NGPU allocated GPUs is not divisible by TP=$TP" >&2; exit 1; }
[ $((PORT_BASE + NSERVE - 1)) -le 65535 ] \
    || { echo "FATAL: server port range exceeds 65535" >&2; exit 1; }

# Refuse a known collision up front. Authentication and process identity checks
# below also prevent a racing or stale process from being mistaken for ours.
"${SANITIZED_ENV[@]}" .venv-vllm/bin/python - "$PORT_BASE" "$NSERVE" <<'PY'
import socket
import sys

sockets = []
try:
    for port in range(int(sys.argv[1]), int(sys.argv[1]) + int(sys.argv[2])):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", port))
        sockets.append(server)
except OSError as error:
    raise SystemExit(f"vLLM port range is not free: {error}")
finally:
    for server in sockets:
        server.close()
PY

VLLM_ENV_INVENTORY="$LOG_PREFIX.packages.txt"
(
    unset PYTHONHOME PYTHONPATH
    export PYTHONNOUSERSITE=1
    write_vllm_inventory "$VLLM_ENV_INVENTORY"
)
chmod 600 "$VLLM_ENV_INVENTORY"
VLLM_ENV_SHA256=$(sha256sum "$VLLM_ENV_INVENTORY" | awk '{print $1}')

# Resolve exactly one model-cache root before either inventorying or launching
# the server. Alternative Hugging Face cache variables are not inherited by the
# server process.
MODEL_CACHE_ROOT=$("${SANITIZED_ENV[@]}" .venv-vllm/bin/python - \
    "${HF_HUB_CACHE:-}" "${HUGGINGFACE_HUB_CACHE:-}" \
    "${HF_HOME:-}" "${XDG_CACHE_HOME:-}" <<'PY'
from pathlib import Path
import sys

hf_hub_cache, legacy_cache, hf_home, xdg_cache = sys.argv[1:5]
if hf_hub_cache:
    path = Path(hf_hub_cache)
elif legacy_cache:
    path = Path(legacy_cache)
elif hf_home:
    path = Path(hf_home) / "hub"
elif xdg_cache:
    path = Path(xdg_cache) / "huggingface" / "hub"
else:
    path = Path.home() / ".cache" / "huggingface" / "hub"
try:
    print(path.resolve(strict=True))
except OSError as error:
    raise SystemExit(f"pinned Hugging Face cache is absent: {error}") from error
PY
)

VLLM_RUNTIME_INVENTORY="$LOG_PREFIX.vllm-runtime.json"
"${SANITIZED_ENV[@]}" .venv-vllm/bin/python - \
    "$VLLM_ENV_INVENTORY" "$VLLM_RUNTIME_INVENTORY" \
    scripts/vllm-requirements.lock .venv-vllm/bin/vllm \
    "$SERVER_HOME" "$SERVER_PATH" "$SERVER_LD_LIBRARY_PATH" \
    "$MODEL_CACHE_ROOT" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import torch

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

packages, destination, lock, entrypoint = map(Path, sys.argv[1:5])
executable = Path(sys.executable)
resolved_executable = executable.resolve(strict=True)
cuda_home = Path(os.environ["CUDA_HOME"]).resolve(strict=True)
nvcc = cuda_home / "bin" / "nvcc"
resolved_nvcc = nvcc.resolve(strict=True)
nvcc_process = subprocess.run(
    [str(resolved_nvcc), "--version"], capture_output=True, check=True
)
nvcc_version = nvcc_process.stdout + nvcc_process.stderr
if not nvcc_version:
    raise SystemExit("nvcc returned no version identity")
record = {
    "schema_version": 1,
    "python": {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": str(executable),
        "resolved_executable": str(resolved_executable),
        "executable_sha256": digest(resolved_executable),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
    },
    "vllm_entrypoint": {
        "path": str(entrypoint),
        "sha256": digest(entrypoint),
    },
    "cuda_toolkit": {
        "cuda_home": str(cuda_home),
        "nvcc": {
            "path": str(nvcc),
            "resolved_path": str(resolved_nvcc),
            "sha256": digest(resolved_nvcc),
            "version_text": nvcc_version.decode("utf-8"),
            "version_sha256": hashlib.sha256(nvcc_version).hexdigest(),
        },
    },
    "torch": {
        "version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    },
    "server_environment": {
        "policy": "clear-and-allowlist-v1",
        "home": sys.argv[5],
        "path": sys.argv[6],
        "ld_library_path": sys.argv[7] or None,
        "model_cache_root": sys.argv[8],
        "lang": "C.UTF-8",
        "lc_all": "C.UTF-8",
        "python_hash_seed": "0",
        "python_no_user_site": "1",
        "hf_hub_offline": "1",
        "transformers_offline": "1",
        "proxy_policy": "cleared",
    },
    "package_inventory_sha256": digest(packages),
    "lock_sha256": digest(lock),
}
payload = (json.dumps(record, allow_nan=False, ensure_ascii=False,
                      separators=(",", ":"), sort_keys=True) + "\n")
destination.write_text(payload, encoding="utf-8")
os.chmod(destination, 0o600)
PY
VLLM_RUNTIME_SHA256=$(sha256sum "$VLLM_RUNTIME_INVENTORY" | awk '{print $1}')

# Hash every logical snapshot entry and its resolved regular cache blob through
# stable, no-follow descriptors. A missing or concurrently changing snapshot is
# fatal: claim-ready jobs never download mutable model state at run time.
MODEL_CACHE_INVENTORY="$LOG_PREFIX.model-cache.json"
"${SANITIZED_ENV[@]}" .venv-vllm/bin/python -I studybench/model_cache.py \
    create "$MODEL_ID" "$MODEL_REVISION" "$MODEL_CACHE_ROOT" \
    "$MODEL_CACHE_INVENTORY"
MODEL_CACHE_SHA256=$(sha256sum "$MODEL_CACHE_INVENTORY" | awk '{print $1}')

for fingerprint in \
    "$GPU_INVENTORY_SHA256" "$VLLM_ENV_SHA256" \
    "$VLLM_RUNTIME_SHA256" "$MODEL_CACHE_SHA256"; do
    [[ "$fingerprint" =~ ^[0-9a-f]{64}$ ]] \
        || { echo "FATAL: invalid launcher inventory hash" >&2; exit 1; }
done

SB_VLLM_API_KEY=$("${SANITIZED_ENV[@]}" .venv-vllm/bin/python -c \
    'import secrets; print(secrets.token_urlsafe(48))')
SB_VLLM_API_KEY_SHA256=$(printf '%s' "$SB_VLLM_API_KEY" | sha256sum | awk '{print $1}')
[[ "$SB_VLLM_API_KEY_SHA256" =~ ^[0-9a-f]{64}$ ]] \
    || { echo "FATAL: could not create ephemeral server identity" >&2; exit 1; }
SB_SERVER_LAUNCH_ID=$SB_VLLM_API_KEY_SHA256

# Bracket vLLM's model load with the canonical inventory. This second complete
# pass is intentionally immediately before starting the server processes; the
# runner performs the matching post-readiness pass before any episode begins.
"${SANITIZED_ENV[@]}" .venv-vllm/bin/python -I studybench/model_cache.py \
    verify "$MODEL_ID" "$MODEL_REVISION" "$MODEL_CACHE_INVENTORY" \
    "$MODEL_CACHE_SHA256"

URLS=""
for ((i = 0; i < NSERVE; i++)); do
    DEVS=$(IFS=,; echo "${GPUS[*]:$((i * TP)):$TP}")
    echo "server $i: allocated GPUs [$DEVS], TP=$TP, port $((PORT_BASE + i))"
    # Feed the ephemeral credential over a pipe to a tiny exec launcher.  It
    # enters vLLM's environment only at execve; it is never an argv element of
    # env, Python, or vLLM (including during the launcher transition).
    printf '%s' "$SB_VLLM_API_KEY" | \
    "${SANITIZED_ENV[@]}" \
        "CUDA_VISIBLE_DEVICES=$DEVS" \
        "HF_HUB_CACHE=$MODEL_CACHE_ROOT" \
        "HF_HUB_OFFLINE=1" \
        "TRANSFORMERS_OFFLINE=1" \
        .venv-vllm/bin/python -c '
import os
import sys

secret = sys.stdin.buffer.read()
try:
    api_key = secret.decode("ascii")
except UnicodeDecodeError as error:
    raise SystemExit("invalid server credential encoding") from error
if not api_key or any(character.isspace() for character in api_key):
    raise SystemExit("invalid server credential")
os.environ["VLLM_API_KEY"] = api_key
os.execv(sys.argv[1], sys.argv[1:])
' .venv-vllm/bin/vllm serve "$MODEL_ID" \
        --revision "$MODEL_REVISION" \
        --host 127.0.0.1 \
        --port $((PORT_BASE + i)) \
        --tensor-parallel-size "$TP" \
        --max-model-len 262144 \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder \
        --language-model-only \
        > "$LOG_PREFIX-$i.log" 2>&1 &
    PIDS+=("$!")
    URLS+="${URLS:+,}http://localhost:$((PORT_BASE + i))/v1"
done

SERVER_PIDS=$(IFS=,; echo "${PIDS[*]}")
TOPOLOGY="$LOG_PREFIX.topology"
TOPOLOGY_TMP="$TOPOLOGY.tmp-$$"
TEMP_FILES+=("$TOPOLOGY_TMP")
TOPOLOGY_FORMAT='BASE_URLS=%q\nSB_NGPU=%q\nSB_NSERVE=%q\nSB_VLLM_VERSION=%q\n'
TOPOLOGY_FORMAT+='SB_TP_EFFECTIVE=%q\nSB_VLLM_ENV_INVENTORY=%q\nSB_VLLM_ENV_SHA256=%q\n'
TOPOLOGY_FORMAT+='SB_VLLM_RUNTIME_INVENTORY=%q\nSB_VLLM_RUNTIME_SHA256=%q\n'
TOPOLOGY_FORMAT+='SB_MODEL_CACHE_INVENTORY=%q\nSB_MODEL_CACHE_SHA256=%q\n'
TOPOLOGY_FORMAT+='SB_GPU_INVENTORY=%q\nSB_GPU_INVENTORY_SHA256=%q\n'
TOPOLOGY_FORMAT+='SB_MODEL_ID=%q\nSB_MODEL_REVISION=%q\nSB_CUDA_VISIBLE_DEVICES=%q\n'
TOPOLOGY_FORMAT+='SB_SLURM_JOB_ID=%q\nSB_SLURM_JOB_GPUS=%q\nSB_SLURM_STEP_GPUS=%q\n'
TOPOLOGY_FORMAT+='SB_SERVER_LAUNCH_ID=%q\nSB_VLLM_API_KEY=%q\nSB_VLLM_API_KEY_SHA256=%q\n'
TOPOLOGY_FORMAT+='SB_LAUNCHER_PID=%q\nSB_SERVER_PIDS=%q\nSB_SERVER_HOSTNAME=%q\n'
printf "$TOPOLOGY_FORMAT" \
    "$URLS" "$NGPU" "$NSERVE" "$VLLM_VERSION" "$TP" \
    "$VLLM_ENV_INVENTORY" "$VLLM_ENV_SHA256" \
    "$VLLM_RUNTIME_INVENTORY" "$VLLM_RUNTIME_SHA256" \
    "$MODEL_CACHE_INVENTORY" "$MODEL_CACHE_SHA256" \
    "$GPU_INVENTORY" "$GPU_INVENTORY_SHA256" \
    "$MODEL_ID" "$MODEL_REVISION" "$CUDA_VISIBLE_DEVICES" \
    "$SLURM_JOB_ID" "${SLURM_JOB_GPUS:-}" "${SLURM_STEP_GPUS:-}" \
    "$SB_SERVER_LAUNCH_ID" "$SB_VLLM_API_KEY" "$SB_VLLM_API_KEY_SHA256" \
    "$$" "$SERVER_PIDS" "$(hostname)" > "$TOPOLOGY_TMP"
chmod 600 "$TOPOLOGY_TMP"
ln "$TOPOLOGY_TMP" "$TOPOLOGY"
rm -f "$TOPOLOGY_TMP"

# A server exiting for any reason invalidates the topology. Keep this launcher
# alive through readiness and for the entire lifetime of all server processes.
set +e
wait -n
status=$?
set -e
echo "FATAL: a vLLM server exited unexpectedly (status $status)" >&2
exit "$(( status == 0 ? 1 : status ))"
