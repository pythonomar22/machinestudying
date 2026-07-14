#!/bin/bash
#SBATCH --job-name=sb-selfquiz
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=6
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
# One namespaced, deterministically seeded DSPy selfquiz construction.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source scripts/run_args.sh
SMOKE=${SB_SMOKE:-0}
DEBUG=${SB_DEBUG:-$SMOKE}
STUDY_ID=${SB_STUDY_ID:?set a unique SB_STUDY_ID}
STUDY_SEED=${SB_STUDY_SEED:?set SB_STUDY_SEED explicitly}
AUDIT_PROTOCOL=${SB_AUDIT_PROTOCOL:-}
ATTEMPT_ACCESS=${SB_ATTEMPT_ACCESS:-react-corpus}

validate_bool SB_SMOKE "$SMOKE"
validate_bool SB_DEBUG "$DEBUG"
validate_id SB_STUDY_ID "$STUDY_ID"
validate_seed SB_STUDY_SEED "$STUDY_SEED"
[[ "$ATTEMPT_ACCESS" = closed-book || "$ATTEMPT_ACCESS" = react-corpus ]] \
    || fail_arg "SB_ATTEMPT_ACCESS must be closed-book or react-corpus"
if [ -n "$AUDIT_PROTOCOL" ]; then
    [ -f "$AUDIT_PROTOCOL" ] || fail_arg "SB_AUDIT_PROTOCOL must be a regular file"
    [ ! -L "$AUDIT_PROTOCOL" ] || fail_arg "SB_AUDIT_PROTOCOL must not be a symlink"
    [ -r "$AUDIT_PROTOCOL" ] || fail_arg "SB_AUDIT_PROTOCOL must be readable"
    [ -s "$AUDIT_PROTOCOL" ] || fail_arg "SB_AUDIT_PROTOCOL must not be empty"
fi

source scripts/setup_common.sh
verify_env_file
sync_dspy_environment
if [ -n "$AUDIT_PROTOCOL" ]; then
    .venv-dspy/bin/python - "$AUDIT_PROTOCOL" <<'PY'
import sys
from pathlib import Path

from studybench.selfquiz import validate_audit_protocol

validate_audit_protocol(Path(sys.argv[1]))
PY
fi
source scripts/serve_and_wait.sh

rounds=(1 2 3 4)
[ "$SMOKE" = 0 ] || rounds=(1)
for round in "${rounds[@]}"; do
    flags=(--task dspy --round "$round" --study-id "$STUDY_ID" \
        --seed "$STUDY_SEED" --chapters 4 --questions 5 \
        --attempt-access "$ATTEMPT_ACCESS")
    [ "$SMOKE" = 0 ] || flags+=(--smoke)
    [ "$DEBUG" = 0 ] || flags+=(--debug)
    if [ "$round" = 1 ] && [ -n "$AUDIT_PROTOCOL" ]; then
        flags+=(--audit-protocol "$AUDIT_PROTOCOL")
    fi
    .venv-dspy/bin/python -m studybench.selfquiz \
        --base-urls "$BASE_URLS" --concurrency "$((4 * SB_NGPU))" "${flags[@]}"
done
echo "semantic selfquiz construction complete for study $STUDY_ID"
