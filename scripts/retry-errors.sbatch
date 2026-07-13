#!/bin/bash
#SBATCH --job-name=sb-retry-errors
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=6
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
# Retry one immutable run namespace. The runner's manifest check prevents this
# job from changing an arm's inputs, and residual failures propagate nonzero.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source scripts/run_args.sh
RUNNER=${SB_RUNNER:?set SB_RUNNER to react or rollout}
TASKS=${SB_TASKS:-dspy,openclaw}
BUDGETS=${SB_BUDGETS:-direct,k5,k20,k20f}
ROLLOUTS=${SB_ROLLOUTS:-3}
RUN_ID=${SB_RUN_ID:?set the existing SB_RUN_ID}
RUN_SEED=${SB_RUN_SEED:?set the run's original SB_RUN_SEED}
SEED_GROUP=${SB_SEED_GROUP:?set the run's original SB_SEED_GROUP}
NO_THINK=${SB_NO_THINK_HISTORY:-0}
EXPLORATORY=${SB_EXPLORATORY:-0}
PREREGISTRATION=${SB_PREREGISTRATION:-}
PREREGISTRATION_ROLE=${SB_PREREGISTRATION_ROLE:-}

[[ "$RUNNER" = react || "$RUNNER" = rollout ]] \
    || fail_arg "SB_RUNNER must be react or rollout"
validate_csv_members SB_TASKS "$TASKS" dspy,openclaw
validate_csv_members SB_BUDGETS "$BUDGETS" direct,k5,k20,k20f
validate_positive_int SB_ROLLOUTS "$ROLLOUTS"
validate_id SB_RUN_ID "$RUN_ID"
validate_seed SB_RUN_SEED "$RUN_SEED"
validate_id SB_SEED_GROUP "$SEED_GROUP"
validate_bool SB_NO_THINK_HISTORY "$NO_THINK"
validate_bool SB_EXPLORATORY "$EXPLORATORY"
if [ -n "$PREREGISTRATION" ] || [ -n "$PREREGISTRATION_ROLE" ]; then
    [ -n "$PREREGISTRATION" ] && [ -n "$PREREGISTRATION_ROLE" ] \
        || fail_arg "SB_PREREGISTRATION and SB_PREREGISTRATION_ROLE are required together"
    [[ "$PREREGISTRATION_ROLE" = control || "$PREREGISTRATION_ROLE" = treatment ]] \
        || fail_arg "SB_PREREGISTRATION_ROLE must be control or treatment"
    require_single_csv_value SB_TASKS "$TASKS"
    [ -f "$PREREGISTRATION" ] || fail_arg "preregistration not found: $PREREGISTRATION"
    [ ! -L "$PREREGISTRATION" ] \
        || fail_arg "preregistration must not be a symlink: $PREREGISTRATION"
elif [ "$EXPLORATORY" = 0 ]; then
    fail_arg "confirmatory retries require the run's original preregistration and role"
fi
if [ -n "${SB_NOTE_MANIFEST:-}" ] && [ -z "${SB_NOTE_PATH:-}" ]; then
    fail_arg "SB_NOTE_MANIFEST requires SB_NOTE_PATH"
fi
if [ -n "${SB_NOTE_PATH:-}" ]; then
    require_single_csv_value SB_TASKS "$TASKS"
    : "${SB_NOTE_MANIFEST:?research note retries require SB_NOTE_MANIFEST}"
    [ -f "$SB_NOTE_PATH" ] || fail_arg "note not found: $SB_NOTE_PATH"
    [ -f "$SB_NOTE_MANIFEST" ] || fail_arg "note manifest not found: $SB_NOTE_MANIFEST"
fi

source scripts/setup_common.sh
verify_env_file
if [ "$RUNNER" = react ]; then
    sync_dspy_environment
    .venv-dspy/bin/python -c \
        'import dspy, studybench.react'
else
    sync_main_environment
    .venv/bin/python -c 'import openai, regex, studybench.rollout'
fi
source scripts/serve_and_wait.sh
for task in ${TASKS//,/ }; do
    args=(--task "$task" --run-id "$RUN_ID" --seed "$RUN_SEED"
          --seed-group "$SEED_GROUP"
          --budgets "$BUDGETS" --rollouts "$ROLLOUTS"
          --base-urls "$BASE_URLS" --concurrency "$((8 * SB_NGPU))")
    [ "$EXPLORATORY" = 0 ] || args+=(--exploratory)
    if [ -n "$PREREGISTRATION" ]; then
        args+=(--preregistration "$PREREGISTRATION"
              --preregistration-role "$PREREGISTRATION_ROLE")
    fi
    if [ -n "${SB_NOTE_PATH:-}" ]; then
        args+=(--note "$SB_NOTE_PATH" --note-manifest "$SB_NOTE_MANIFEST")
    fi
    if [ "$RUNNER" = react ]; then
        [ "$NO_THINK" = 0 ] || fail_arg "no-think-history is native-runner only"
        .venv-dspy/bin/python -m studybench.react "${args[@]}"
    else
        [ "$NO_THINK" = 0 ] || args+=(--no-think-history)
        .venv/bin/python -m studybench.rollout "${args[@]}"
    fi
done
echo "retry complete with no residual runner failures"
