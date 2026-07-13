#!/bin/bash
#SBATCH --job-name=sb-react
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=6
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
# Faithful dspy.ReAct generation. Every invocation requires a deterministic
# seed and either an immutable evaluation run ID or an immutable study ID.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source scripts/run_args.sh
TASKS=${SB_TASKS:-dspy,openclaw}
BUDGETS=${SB_BUDGETS:-direct,k5,k20,k20f}
ROLLOUTS=${SB_ROLLOUTS:-3}
LIMIT=${SB_LIMIT:-0}
SMOKE=${SB_SMOKE:-0}
EXPLORATORY=${SB_EXPLORATORY:-0}
STUDY=${SB_STUDY:-0}
ALLOW_DIRTY=${SB_ALLOW_DIRTY:-0}
RUN_SEED=${SB_RUN_SEED:?set SB_RUN_SEED explicitly}
PREREGISTRATION=${SB_PREREGISTRATION:-}
PREREGISTRATION_ROLE=${SB_PREREGISTRATION_ROLE:-}

validate_csv_members SB_TASKS "$TASKS" dspy,openclaw
validate_csv_members SB_BUDGETS "$BUDGETS" direct,k5,k20,k20f
validate_positive_int SB_ROLLOUTS "$ROLLOUTS"
validate_nonnegative_int SB_LIMIT "$LIMIT"
validate_bool SB_SMOKE "$SMOKE"
validate_bool SB_EXPLORATORY "$EXPLORATORY"
validate_bool SB_STUDY "$STUDY"
validate_bool SB_ALLOW_DIRTY "$ALLOW_DIRTY"
validate_seed SB_RUN_SEED "$RUN_SEED"
[ "$ALLOW_DIRTY" = 0 ] || [ "$SMOKE" = 1 ] \
    || fail_arg "SB_ALLOW_DIRTY is permitted only for isolated smoke runs"
[ "$SMOKE" = 0 ] || [ "$EXPLORATORY" = 0 ] \
    || fail_arg "SB_SMOKE and SB_EXPLORATORY are mutually exclusive"

if [ "$STUDY" = 1 ]; then
    STUDY_ID=${SB_STUDY_ID:?set SB_STUDY_ID for a cheatsheet study}
    validate_id SB_STUDY_ID "$STUDY_ID"
    [ "$LIMIT" = 0 ] || fail_arg "SB_LIMIT does not apply to study mode"
    [ -z "${SB_NOTE_PATH:-}${SB_NOTE_MANIFEST:-}" ] \
        || fail_arg "study mode cannot consume an evaluation note"
    [ -z "$PREREGISTRATION$PREREGISTRATION_ROLE" ] && [ "$EXPLORATORY" = 0 ] \
        || fail_arg "study mode cannot consume evaluation preregistration options"
else
    RUN_ID=${SB_RUN_ID:?set a unique SB_RUN_ID}
    SEED_GROUP=${SB_SEED_GROUP:?set the paired evaluation SB_SEED_GROUP}
    validate_id SB_RUN_ID "$RUN_ID"
    validate_id SB_SEED_GROUP "$SEED_GROUP"
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
        [ -f "$PREREGISTRATION" ] \
            || fail_arg "preregistration not found: $PREREGISTRATION"
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
fi

source scripts/setup_common.sh
verify_env_file
sync_dspy_environment
source scripts/serve_and_wait.sh

for task in ${TASKS//,/ }; do
    if [ "$STUDY" = 1 ]; then
        args=(--task "$task" --study --study-id "$STUDY_ID" --seed "$RUN_SEED"
              --base-urls "$BASE_URLS")
        [ "$SMOKE" = 0 ] || args+=(--smoke)
        [ "$ALLOW_DIRTY" = 0 ] || args+=(--allow-dirty)
    else
        args=(--task "$task" --run-id "$RUN_ID" --seed "$RUN_SEED"
              --seed-group "$SEED_GROUP"
              --budgets "$BUDGETS" --rollouts "$ROLLOUTS" --limit "$LIMIT"
              --base-urls "$BASE_URLS" --concurrency "$((8 * SB_NGPU))")
        [ "$SMOKE" = 0 ] || args+=(--smoke)
        [ "$EXPLORATORY" = 0 ] || args+=(--exploratory)
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
    fi
    .venv-dspy/bin/python -m studybench.react "${args[@]}"
done
echo "react generation complete"
