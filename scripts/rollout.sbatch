#!/bin/bash
#SBATCH --job-name=sb-rollout
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=6
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
# Native tool-calling generation. Grading remains a separate offline step.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_LINK_MODE=copy

source scripts/run_args.sh
TASKS=${SB_TASKS:-dspy,openclaw}
BUDGETS=${SB_BUDGETS:-direct,k5,k20,k20f}
ROLLOUTS=${SB_ROLLOUTS:-3}
LIMIT=${SB_LIMIT:-0}
SMOKE=${SB_SMOKE:-0}
EXPLORATORY=${SB_EXPLORATORY:-0}
DEBUG=${SB_DEBUG:-0}
NO_THINK=${SB_NO_THINK_HISTORY:-0}
ALLOW_DIRTY=${SB_ALLOW_DIRTY:-0}
RUN_ID=${SB_RUN_ID:?set a unique SB_RUN_ID}
RUN_SEED=${SB_RUN_SEED:?set SB_RUN_SEED explicitly}
SEED_GROUP=${SB_SEED_GROUP:?set the paired evaluation SB_SEED_GROUP}
PREREGISTRATION=${SB_PREREGISTRATION:-}
PREREGISTRATION_ROLE=${SB_PREREGISTRATION_ROLE:-}

validate_csv_members SB_TASKS "$TASKS" dspy,openclaw
validate_csv_members SB_BUDGETS "$BUDGETS" direct,k5,k20,k20f
validate_positive_int SB_ROLLOUTS "$ROLLOUTS"
validate_nonnegative_int SB_LIMIT "$LIMIT"
validate_bool SB_SMOKE "$SMOKE"
validate_bool SB_EXPLORATORY "$EXPLORATORY"
validate_bool SB_DEBUG "$DEBUG"
validate_bool SB_NO_THINK_HISTORY "$NO_THINK"
validate_bool SB_ALLOW_DIRTY "$ALLOW_DIRTY"
validate_id SB_RUN_ID "$RUN_ID"
validate_seed SB_RUN_SEED "$RUN_SEED"
validate_id SB_SEED_GROUP "$SEED_GROUP"
[ "$ALLOW_DIRTY" = 0 ] || [ "$SMOKE" = 1 ] \
    || fail_arg "SB_ALLOW_DIRTY is permitted only for isolated smoke runs"
[ "$SMOKE" = 0 ] || [ "$EXPLORATORY" = 0 ] \
    || fail_arg "SB_SMOKE and SB_EXPLORATORY are mutually exclusive"
if [ "$LIMIT" -gt 0 ] && [ "$SMOKE" != 1 ]; then
    fail_arg "SB_LIMIT is diagnostic and requires SB_SMOKE=1"
fi
if [ "$SMOKE" = 1 ] && [ "$LIMIT" -eq 0 ]; then
    fail_arg "SB_SMOKE=1 requires a positive SB_LIMIT"
fi
if [ -n "$PREREGISTRATION" ] || [ -n "$PREREGISTRATION_ROLE" ]; then
    [ -n "$PREREGISTRATION" ] && [ -n "$PREREGISTRATION_ROLE" ] \
        || fail_arg "SB_PREREGISTRATION and SB_PREREGISTRATION_ROLE are required together"
    [[ "$PREREGISTRATION_ROLE" = control || "$PREREGISTRATION_ROLE" = treatment ]] \
        || fail_arg "SB_PREREGISTRATION_ROLE must be control or treatment"
    require_single_csv_value SB_TASKS "$TASKS"
    [ -f "$PREREGISTRATION" ] || fail_arg "preregistration not found: $PREREGISTRATION"
    [ ! -L "$PREREGISTRATION" ] \
        || fail_arg "preregistration must not be a symlink: $PREREGISTRATION"
elif [ "$SMOKE" = 0 ] && [ "$EXPLORATORY" = 0 ]; then
    fail_arg "confirmatory runs require SB_PREREGISTRATION and SB_PREREGISTRATION_ROLE"
fi
if [ -n "${SB_NOTE_MANIFEST:-}" ] && [ -z "${SB_NOTE_PATH:-}" ]; then
    fail_arg "SB_NOTE_MANIFEST requires SB_NOTE_PATH"
fi
if [ -n "${SB_NOTE_PATH:-}" ]; then
    require_single_csv_value SB_TASKS "$TASKS"
    [ -f "$SB_NOTE_PATH" ] || fail_arg "note not found: $SB_NOTE_PATH"
    if [ "$SMOKE" != 1 ]; then
        : "${SB_NOTE_MANIFEST:?non-smoke notes require SB_NOTE_MANIFEST}"
    fi
    if [ -n "${SB_NOTE_MANIFEST:-}" ]; then
        [ -f "$SB_NOTE_MANIFEST" ] \
            || fail_arg "note manifest not found: $SB_NOTE_MANIFEST"
    fi
fi

source scripts/setup_common.sh
verify_env_file
sync_main_environment
source scripts/serve_and_wait.sh

for task in ${TASKS//,/ }; do
    args=(--task "$task" --run-id "$RUN_ID" --seed "$RUN_SEED"
          --seed-group "$SEED_GROUP"
          --budgets "$BUDGETS" --rollouts "$ROLLOUTS" --limit "$LIMIT"
          --base-urls "$BASE_URLS" --concurrency "$((8 * SB_NGPU))")
    [ "$SMOKE" = 0 ] || args+=(--smoke)
    [ "$EXPLORATORY" = 0 ] || args+=(--exploratory)
    [ "$DEBUG" = 0 ] || args+=(--debug)
    [ "$NO_THINK" = 0 ] || args+=(--no-think-history)
    [ "$ALLOW_DIRTY" = 0 ] || args+=(--allow-dirty)
    if [ -n "$PREREGISTRATION" ]; then
        args+=(--preregistration "$PREREGISTRATION"
              --preregistration-role "$PREREGISTRATION_ROLE")
    fi
    if [ -n "${SB_NOTE_PATH:-}" ]; then
        args+=(--note "$SB_NOTE_PATH")
        [ -z "${SB_NOTE_MANIFEST:-}" ] \
            || args+=(--note-manifest "$SB_NOTE_MANIFEST")
    fi
    .venv/bin/python -m studybench.rollout "${args[@]}"
done
echo "native rollouts complete"
