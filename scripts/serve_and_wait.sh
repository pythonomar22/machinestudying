# Sourced by sbatch runners. Launch a fresh authenticated vLLM topology, bind
# every exported identity to this Slurm job, and wait for authenticated model
# discovery while both the launcher and every server process remain alive.
set -euo pipefail
umask 077

# This file is sourced, so the policy also applies to every subsequent Python
# model client in the Slurm runner. Loopback requests must never traverse an
# inherited proxy or curl configuration.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_AUDIT
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=$NO_PROXY
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1

source scripts/setup_common.sh
verify_env_file  # validates path/type/owner/mode only; never reads secret contents
require_command curl
require_command grep
require_command hostname
require_command setsid
require_command sha256sum
require_command stat
: "${SLURM_JOB_ID:?serve_and_wait.sh must run inside a Slurm allocation}"
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose allocated GPUs for this job}"
export SB_PORT_BASE=${SB_PORT_BASE:-$((20000 + (SLURM_JOB_ID % 2000) * 8))}
export SB_VLLM_LOG_PREFIX=${SB_VLLM_LOG_PREFIX:-logs/vllm-${SLURM_JOB_ID}}
[[ "$SB_VLLM_LOG_PREFIX" =~ ^logs/[A-Za-z0-9._-]+$ ]] \
    || { echo "FATAL: SB_VLLM_LOG_PREFIX must be one name under logs/" >&2; exit 1; }
mkdir -p logs
[ -d logs ] && [ ! -L logs ] \
    || { echo "FATAL: logs/ must be a real directory" >&2; exit 1; }

setsid bash scripts/serve_vllm.sh &
SERVE_PID=$!
cleanup_server() {
    kill -TERM -- -"$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
}
trap cleanup_server EXIT

assert_launcher_alive() {
    kill -0 "$SERVE_PID" 2>/dev/null \
        || { echo "FATAL: vLLM launcher is no longer alive" >&2; return 1; }
    if [ -n "${SB_SERVER_PIDS:-}" ]; then
        local pid
        IFS=',' read -ra server_processes <<< "$SB_SERVER_PIDS"
        [ "${#server_processes[@]}" -eq "$SB_NSERVE" ] || return 1
        for pid in "${server_processes[@]}"; do
            [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null \
                || { echo "FATAL: vLLM server process $pid is no longer alive" >&2; return 1; }
        done
    fi
}

TOPOLOGY="$SB_VLLM_LOG_PREFIX.topology"
for _ in $(seq 1 180); do
    [ -f "$TOPOLOGY" ] && break
    assert_launcher_alive
    sleep 5
done
[ -f "$TOPOLOGY" ] \
    || { echo "FATAL: no server topology after 15 min" >&2; exit 1; }

verify_owner_only_file() {
    local path=$1
    [ -f "$path" ] && [ ! -L "$path" ] \
        && [ "$(stat -c '%u' "$path")" = "$(id -u)" ] \
        && [ "$(stat -c '%a' "$path")" = 600 ] \
        || { echo "FATAL: expected owner-only regular file: $path" >&2; return 1; }
}
verify_owner_only_file "$TOPOLOGY"
source "$TOPOLOGY"

for required in \
    BASE_URLS SB_NGPU SB_NSERVE SB_VLLM_VERSION SB_TP_EFFECTIVE \
    SB_VLLM_ENV_INVENTORY SB_VLLM_ENV_SHA256 \
    SB_VLLM_RUNTIME_INVENTORY SB_VLLM_RUNTIME_SHA256 \
    SB_MODEL_CACHE_INVENTORY SB_MODEL_CACHE_SHA256 \
    SB_GPU_INVENTORY SB_GPU_INVENTORY_SHA256 \
    SB_MODEL_ID SB_MODEL_REVISION SB_CUDA_VISIBLE_DEVICES SB_SLURM_JOB_ID \
    SB_SERVER_LAUNCH_ID SB_VLLM_API_KEY SB_VLLM_API_KEY_SHA256 \
    SB_LAUNCHER_PID SB_SERVER_PIDS SB_SERVER_HOSTNAME; do
    [ -n "${!required:-}" ] \
        || { echo "FATAL: topology omits $required" >&2; exit 1; }
done
[[ "$SB_NGPU" =~ ^[1-9][0-9]*$ && "$SB_NSERVE" =~ ^[1-9][0-9]*$ ]] \
    || { echo "FATAL: invalid server topology counts" >&2; exit 1; }
[[ "$SB_TP_EFFECTIVE" =~ ^[1-9][0-9]*$ ]] \
    || { echo "FATAL: invalid effective tensor parallel size" >&2; exit 1; }
[ "$SB_VLLM_VERSION" = "$VLLM_VERSION" ] \
    || { echo "FATAL: unexpected vLLM server version" >&2; exit 1; }
[ "$SB_MODEL_ID" = "Qwen/Qwen3.5-9B" ] \
    && [ "$SB_MODEL_REVISION" = c202236235762e1c871ad0ccb60c8ee5ba337b9a ] \
    || { echo "FATAL: unexpected model identity" >&2; exit 1; }
[ "$SB_LAUNCHER_PID" = "$SERVE_PID" ] \
    || { echo "FATAL: topology came from a different launcher" >&2; exit 1; }
[ "$SB_SLURM_JOB_ID" = "$SLURM_JOB_ID" ] \
    && [ "$SB_CUDA_VISIBLE_DEVICES" = "$CUDA_VISIBLE_DEVICES" ] \
    && [ "$SB_SERVER_HOSTNAME" = "$(hostname)" ] \
    || { echo "FATAL: topology came from a different Slurm allocation" >&2; exit 1; }

computed_key_hash=$(printf '%s' "$SB_VLLM_API_KEY" | sha256sum | awk '{print $1}')
[ "$computed_key_hash" = "$SB_VLLM_API_KEY_SHA256" ] \
    && [ "$SB_SERVER_LAUNCH_ID" = "$SB_VLLM_API_KEY_SHA256" ] \
    || { echo "FATAL: invalid ephemeral vLLM server identity" >&2; exit 1; }

verify_snapshot() {
    local path=$1 expected_path=$2 fingerprint=$3
    [ "$path" = "$expected_path" ] \
        || { echo "FATAL: unexpected launcher inventory path" >&2; return 1; }
    [[ "$fingerprint" =~ ^[0-9a-f]{64}$ ]] \
        || { echo "FATAL: invalid launcher inventory hash" >&2; return 1; }
    verify_owner_only_file "$path"
    [ "$(sha256sum "$path" | awk '{print $1}')" = "$fingerprint" ] \
        || { echo "FATAL: launcher inventory changed after launch" >&2; return 1; }
}
verify_snapshot "$SB_VLLM_ENV_INVENTORY" \
    "$SB_VLLM_LOG_PREFIX.packages.txt" "$SB_VLLM_ENV_SHA256"
verify_snapshot "$SB_VLLM_RUNTIME_INVENTORY" \
    "$SB_VLLM_LOG_PREFIX.vllm-runtime.json" "$SB_VLLM_RUNTIME_SHA256"
verify_snapshot "$SB_MODEL_CACHE_INVENTORY" \
    "$SB_VLLM_LOG_PREFIX.model-cache.json" "$SB_MODEL_CACHE_SHA256"
verify_snapshot "$SB_GPU_INVENTORY" \
    "$SB_VLLM_LOG_PREFIX.gpus.json" "$SB_GPU_INVENTORY_SHA256"

.venv-vllm/bin/python -I - \
    "$SB_VLLM_RUNTIME_INVENTORY" "$SB_MODEL_CACHE_INVENTORY" \
    "$SB_GPU_INVENTORY" "$SB_VLLM_ENV_SHA256" "$SB_MODEL_ID" \
    "$SB_MODEL_REVISION" "$SB_CUDA_VISIBLE_DEVICES" "$SB_SLURM_JOB_ID" \
    "$SB_NGPU" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def load(path: str) -> dict:
    data = Path(path).read_bytes()
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SystemExit(f"duplicate key in launcher inventory: {key}")
            result[key] = value
        return result
    value = json.loads(
        data.decode("utf-8"), object_pairs_hook=no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite value {value}")),
    )
    canonical = (json.dumps(value, allow_nan=False, ensure_ascii=False,
                            separators=(",", ":"), sort_keys=True) + "\n").encode()
    if canonical != data or not isinstance(value, dict):
        raise SystemExit("launcher inventory is not a canonical JSON object")
    return value

runtime = load(sys.argv[1])
model = load(sys.argv[2])
allocation = load(sys.argv[3])
cuda_toolkit = runtime.get("cuda_toolkit")
nvcc = cuda_toolkit.get("nvcc") if isinstance(cuda_toolkit, dict) else None
torch_identity = runtime.get("torch")
server_environment = runtime.get("server_environment")
server_path = (
    server_environment.get("path")
    if isinstance(server_environment, dict)
    else None
)
if (runtime.get("schema_version") != 1
        or runtime.get("package_inventory_sha256") != sys.argv[4]
        or runtime.get("python", {}).get("version") != "3.12.11"
        or not isinstance(cuda_toolkit, dict)
        or not Path(str(cuda_toolkit.get("cuda_home", ""))).is_absolute()
        or not isinstance(nvcc, dict)
        or not Path(str(nvcc.get("resolved_path", ""))).is_absolute()
        or hashlib.sha256(str(nvcc.get("version_text", "")).encode()).hexdigest()
        != nvcc.get("version_sha256")
        or not isinstance(torch_identity, dict)
        or not torch_identity.get("version")
        or not torch_identity.get("cuda_version")
        or not isinstance(server_environment, dict)
        or set(server_environment) != {
            "policy", "home", "path", "ld_library_path", "model_cache_root",
            "lang", "lc_all", "python_hash_seed", "python_no_user_site",
            "hf_hub_offline", "transformers_offline", "proxy_policy",
        }
        or server_environment.get("policy") != "clear-and-allowlist-v1"
        or not Path(str(server_environment.get("home", ""))).is_absolute()
        or not isinstance(server_path, str)
        or not server_path
        or not all(Path(part).is_absolute() for part in server_path.split(":"))
        or (server_environment.get("ld_library_path") is not None
            and not isinstance(server_environment.get("ld_library_path"), str))
        or not Path(str(server_environment.get("model_cache_root", ""))).is_absolute()
        or server_environment.get("lang") != "C.UTF-8"
        or server_environment.get("lc_all") != "C.UTF-8"
        or server_environment.get("python_hash_seed") != "0"
        or server_environment.get("python_no_user_site") != "1"
        or server_environment.get("hf_hub_offline") != "1"
        or server_environment.get("transformers_offline") != "1"
        or server_environment.get("proxy_policy") != "cleared"):
    raise SystemExit("vLLM runtime inventory does not bind the pinned environment")
if (model.get("attestation_policy") != "stable-openat-sha256-v1"
        or model.get("model") != sys.argv[5]
        or model.get("revision") != sys.argv[6]):
    raise SystemExit("model-cache inventory does not bind the launched model")
if server_environment["model_cache_root"] != model.get("cache_root"):
    raise SystemExit("server environment uses a different model-cache root")
files = model.get("files")
if (not isinstance(files, list) or not files
        or model.get("file_count") != len(files)
        or model.get("total_bytes") != sum(row.get("bytes", -1) for row in files)):
    raise SystemExit("model-cache inventory is incomplete")
tree = (json.dumps(files, allow_nan=False, ensure_ascii=False,
                   separators=(",", ":"), sort_keys=True) + "\n").encode()
if model.get("tree_sha256") != hashlib.sha256(tree).hexdigest():
    raise SystemExit("model-cache tree hash is invalid")
gpus = allocation.get("gpus")
expected_ids = sys.argv[7].split(",")
if (allocation.get("cuda_visible_devices") != sys.argv[7]
        or allocation.get("slurm", {}).get("job_id") != sys.argv[8]
        or allocation.get("gpu_count") != int(sys.argv[9])
        or not isinstance(gpus, list)
        or [row.get("cuda_identifier") for row in gpus] != expected_ids):
    raise SystemExit("allocated-GPU inventory does not bind this Slurm job")
PY

[ $((SB_NGPU % SB_TP_EFFECTIVE)) -eq 0 ] \
    && [ $((SB_NGPU / SB_TP_EFFECTIVE)) -eq "$SB_NSERVE" ] \
    || { echo "FATAL: inconsistent GPU/TP/server topology" >&2; exit 1; }
IFS=',' read -ra SERVER_URLS <<< "$BASE_URLS"
[ "${#SERVER_URLS[@]}" -eq "$SB_NSERVE" ] \
    || { echo "FATAL: URL count disagrees with server count" >&2; exit 1; }
declare -A seen_urls=()
for url in "${SERVER_URLS[@]}"; do
    [[ "$url" =~ ^http://localhost:[1-9][0-9]*/v1$ ]] \
        || { echo "FATAL: invalid local server URL: $url" >&2; exit 1; }
    [ -z "${seen_urls[$url]:-}" ] \
        || { echo "FATAL: duplicate local server URL: $url" >&2; exit 1; }
    seen_urls[$url]=1
done

export BASE_URLS SB_NGPU SB_NSERVE SB_VLLM_VERSION SB_TP_EFFECTIVE
export SB_VLLM_ENV_INVENTORY SB_VLLM_ENV_SHA256
export SB_VLLM_RUNTIME_INVENTORY SB_VLLM_RUNTIME_SHA256
export SB_MODEL_CACHE_INVENTORY SB_MODEL_CACHE_SHA256
export SB_GPU_INVENTORY SB_GPU_INVENTORY_SHA256
export SB_MODEL_ID SB_MODEL_REVISION SB_CUDA_VISIBLE_DEVICES
export SB_SLURM_JOB_ID SB_SLURM_JOB_GPUS SB_SLURM_STEP_GPUS
export SB_SERVER_LAUNCH_ID SB_VLLM_API_KEY SB_VLLM_API_KEY_SHA256
export SB_SERVER_HOSTNAME

probe_server_identity() {
    local url=$1 index=$2 response status
    response="$SB_VLLM_LOG_PREFIX-$index.models.tmp"
    rm -f "$response"
    if ! curl --disable --noproxy '*' \
        --fail --silent --show-error --connect-timeout 3 --max-time 20 \
        --config - --output "$response" "$url/models" <<EOF
header = "Authorization: Bearer $SB_VLLM_API_KEY"
EOF
    then
        rm -f "$response"
        return 1
    fi
    set +e
    .venv-vllm/bin/python -I - "$response" "$SB_MODEL_ID" <<'PY'
import json
from pathlib import Path
import sys

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid authenticated model response: {error}") from error
models = payload.get("data") if isinstance(payload, dict) else None
if (not isinstance(models, list)
        or [row.get("id") for row in models if isinstance(row, dict)] != [sys.argv[2]]):
    raise SystemExit("authenticated endpoint serves a different model identity")
PY
    status=$?
    set -e
    rm -f "$response"
    return "$status"
}

for ((i = 0; i < SB_NSERVE; i++)); do
    url=${SERVER_URLS[$i]}
    ready=false
    for _ in $(seq 1 240); do
        assert_launcher_alive
        if probe_server_identity "$url" "$i"; then
            ready=true
            break
        fi
        if grep -q "EngineCore failed to start" "$SB_VLLM_LOG_PREFIX-$i.log" 2>/dev/null; then
            echo "FATAL: vLLM server $i crashed during startup:" >&2
            tail -50 "$SB_VLLM_LOG_PREFIX-$i.log" >&2
            exit 1
        fi
        sleep 10
    done
    [ "$ready" = true ] && assert_launcher_alive \
        && probe_server_identity "$url" "$i" \
        || {
            echo "FATAL: authenticated vLLM server $i never became ready" >&2
            tail -50 "$SB_VLLM_LOG_PREFIX-$i.log" >&2
            exit 1
        }
done
assert_launcher_alive
# Readiness means vLLM has completed its model load. Rebuild the complete cache
# inventory now and require byte-for-byte equality with the prelaunch record.
# Failure exits this sourced runner, whose EXIT trap terminates the topology.
.venv-vllm/bin/python -I studybench/model_cache.py verify \
    "$SB_MODEL_ID" "$SB_MODEL_REVISION" "$SB_MODEL_CACHE_INVENTORY" \
    "$SB_MODEL_CACHE_SHA256"
assert_launcher_alive
echo "all $SB_NSERVE authenticated vLLM servers ready on $SB_NGPU allocated GPUs: $BASE_URLS"
