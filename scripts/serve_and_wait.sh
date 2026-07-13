# Sourced by the sbatch runners: launches scripts/serve_vllm.sh (which decides
# GPU topology and writes $SB_VLLM_LOG_PREFIX.env), waits for every server to
# become healthy, and leaves BASE_URLS + SB_NGPU set in the caller. Ports and
# log names derive from the job id — nodes may be shared, fixed ports collide.
export SB_PORT_BASE=${SB_PORT_BASE:-$((20000 + (${SLURM_JOB_ID:-$$} % 2000) * 8))}
export SB_VLLM_LOG_PREFIX=${SB_VLLM_LOG_PREFIX:-logs/vllm-${SLURM_JOB_ID:-$$}}
mkdir -p logs

setsid bash scripts/serve_vllm.sh &
SERVE_PID=$!
trap 'kill -TERM -- -$SERVE_PID 2>/dev/null || true' EXIT

# the topology file appears once the servers are launched (the first run also
# bootstraps the vLLM venv, which takes minutes)
for _ in $(seq 1 180); do
    [ -f "$SB_VLLM_LOG_PREFIX.env" ] && break
    kill -0 "$SERVE_PID" 2>/dev/null \
        || { echo "FATAL: serve_vllm.sh died before launching servers"; exit 1; }
    sleep 5
done
[ -f "$SB_VLLM_LOG_PREFIX.env" ] \
    || { echo "FATAL: no server topology after 15 min"; exit 1; }
source "$SB_VLLM_LOG_PREFIX.env"

i=0
for url in ${BASE_URLS//,/ }; do
    for _ in $(seq 1 240); do  # up to 40 min (first startup downloads weights + compiles kernels)
        curl -sf "$url/health" > /dev/null && break
        # fatal marker only — warmup WARNINGs also print tracebacks
        if grep -q "EngineCore failed to start" "$SB_VLLM_LOG_PREFIX-$i.log" 2>/dev/null; then
            echo "FATAL: vLLM server $i crashed during startup:"
            tail -50 "$SB_VLLM_LOG_PREFIX-$i.log"
            exit 1
        fi
        sleep 10
    done
    curl -sf "$url/health" > /dev/null \
        || { echo "FATAL: vLLM server $i never became healthy"; tail -50 "$SB_VLLM_LOG_PREFIX-$i.log"; exit 1; }
    i=$((i + 1))
done
echo "all $SB_NSERVE vLLM servers healthy on $SB_NGPU GPUs: $BASE_URLS"
