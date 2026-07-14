#!/bin/bash
#SBATCH --job-name=sb-grade-local
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
# Exploratory Qwen grading and reporting. This path is diagnostic only and
# cannot create claim-ready or paper-comparable artifacts.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_LINK_MODE=copy

source scripts/run_args.sh
TASK=${SB_TASK:-dspy}
RUN_ID=${SB_RUN_ID:?set the exploratory SB_RUN_ID}
GRADE_ID=${SB_GRADE_ID:?set a unique SB_GRADE_ID}
EVIDENCE_MODE=${SB_EVIDENCE_MODE:-excerpt_evidence}
CONCURRENCY=${SB_GRADE_CONCURRENCY:-8}
CI_REPLICATES=${SB_CI_REPLICATES:-1000}
CI_SEED=${SB_CI_SEED:-0}
DEBUG=${SB_DEBUG:-0}
LOCAL_SMOKE=${SB_LOCAL_SMOKE:-0}

validate_csv_members SB_TASK "$TASK" dspy,openclaw
require_single_csv_value SB_TASK "$TASK"
validate_id SB_RUN_ID "$RUN_ID"
validate_id SB_GRADE_ID "$GRADE_ID"
validate_positive_int SB_GRADE_CONCURRENCY "$CONCURRENCY"
validate_nonnegative_int SB_CI_REPLICATES "$CI_REPLICATES"
validate_seed SB_CI_SEED "$CI_SEED"
validate_bool SB_DEBUG "$DEBUG"
validate_bool SB_LOCAL_SMOKE "$LOCAL_SMOKE"
case "$EVIDENCE_MODE" in
    whole_files)
        EVIDENCE_FLAG=--whole-files
        ;;
    excerpt_evidence)
        EVIDENCE_FLAG=--excerpt-evidence
        ;;
    *)
        fail_arg "SB_EVIDENCE_MODE must be whole_files or excerpt_evidence"
        ;;
esac

# Exactly one TP=2 server makes the endpoint identity unambiguous on both H100
# and L40S allocations. The authenticated key is generated inside the launcher.
export SB_TP=2
export GRADER_MODEL=local
# Local mode never needs an external judge credential. Keep .env values from
# entering these processes; load_private_env preserves existing empty values.
export OPENAI_API_KEY=
export SAKANA_API_KEY=

source scripts/setup_common.sh
verify_env_file
sync_main_environment
source scripts/serve_and_wait.sh

[ "$SB_NGPU" -eq 2 ] && [ "$SB_NSERVE" -eq 1 ] \
    && [ "$SB_TP_EFFECTIVE" -eq 2 ] \
    || fail_arg "local grading requires exactly one TP=2 server on two GPUs"
require_single_csv_value BASE_URLS "$BASE_URLS"
JUDGE_BASE_URL=$BASE_URLS

grade_args=(
    --task "$TASK"
    --run-id "$RUN_ID"
    --grade-id "$GRADE_ID"
    --judge-base-url "$JUDGE_BASE_URL"
    "$EVIDENCE_FLAG"
    --concurrency "$CONCURRENCY"
)
[ "$DEBUG" = 0 ] || grade_args+=(--debug)
[ "$LOCAL_SMOKE" = 0 ] || grade_args+=(--local-smoke)
.venv/bin/python -m studybench.grade "${grade_args[@]}"

if [ "$LOCAL_SMOKE" = 1 ]; then
    echo "local grading smoke complete; partial namespace is intentionally unreportable"
    exit 0
fi

.venv/bin/python -m studybench.report \
    --tasks "$TASK" \
    --run-id "$RUN_ID" \
    --grader local \
    --grade-id "$GRADE_ID" \
    --judge-base-url "$JUDGE_BASE_URL" \
    "$EVIDENCE_FLAG" \
    --ci "$CI_REPLICATES" \
    --ci-seed "$CI_SEED"

echo "exploratory local grading and diagnostic report complete"
