#!/bin/bash
#SBATCH --job-name=sb-selfquiz
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=6
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
# One namespaced, deterministically seeded self-quizzing study round.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source scripts/run_args.sh
TASKS=${SB_TASKS:-dspy,openclaw}
ROUND=${SB_ROUND:-1}
SMOKE=${SB_SMOKE:-0}
DEBUG=${SB_DEBUG:-$SMOKE}
STUDY_ID=${SB_STUDY_ID:?set a unique SB_STUDY_ID}
STUDY_SEED=${SB_STUDY_SEED:?set SB_STUDY_SEED explicitly}
AUDIT_PROTOCOL=${SB_AUDIT_PROTOCOL:-}

validate_csv_members SB_TASKS "$TASKS" dspy,openclaw
validate_positive_int SB_ROUND "$ROUND"
validate_bool SB_SMOKE "$SMOKE"
validate_bool SB_DEBUG "$DEBUG"
validate_id SB_STUDY_ID "$STUDY_ID"
validate_seed SB_STUDY_SEED "$STUDY_SEED"
if [ -n "$AUDIT_PROTOCOL" ]; then
    [ "$ROUND" = 1 ] || fail_arg "SB_AUDIT_PROTOCOL may only be registered in round 1"
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

flags=(--study-id "$STUDY_ID" --seed "$STUDY_SEED")
[ -z "$AUDIT_PROTOCOL" ] || flags+=(--audit-protocol "$AUDIT_PROTOCOL")
[ "$SMOKE" = 0 ] || flags+=(--smoke)
[ "$DEBUG" = 0 ] || flags+=(--debug)
for task in ${TASKS//,/ }; do
    .venv-dspy/bin/python -m studybench.selfquiz --task "$task" --round "$ROUND" \
        --base-urls "$BASE_URLS" --concurrency "$((4 * SB_NGPU))" "${flags[@]}"
done
echo "selfquiz round $ROUND complete for study $STUDY_ID"
