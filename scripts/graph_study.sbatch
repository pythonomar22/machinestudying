#!/bin/bash
#SBATCH --job-name=sb-graph-study
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=6
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
# One namespaced, deterministically seeded full-DSPy call-neighborhood study.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source scripts/run_args.sh
SMOKE=${SB_SMOKE:-0}
DEBUG=${SB_DEBUG:-$SMOKE}
STUDY_ID=${SB_STUDY_ID:?set a unique SB_STUDY_ID}
STUDY_SEED=${SB_STUDY_SEED:?set SB_STUDY_SEED explicitly}

validate_bool SB_SMOKE "$SMOKE"
validate_bool SB_DEBUG "$DEBUG"
validate_id SB_STUDY_ID "$STUDY_ID"
validate_seed SB_STUDY_SEED "$STUDY_SEED"

source scripts/setup_common.sh
verify_env_file
sync_dspy_environment
source scripts/serve_and_wait.sh

flags=(--study-id "$STUDY_ID" --seed "$STUDY_SEED")
[ "$SMOKE" = 0 ] || flags+=(--smoke)
[ "$DEBUG" = 0 ] || flags+=(--debug)
.venv-dspy/bin/python -m studybench.graph_study \
    --base-urls "$BASE_URLS" --concurrency "$((4 * SB_NGPU))" "${flags[@]}"
echo "call-neighborhood study construction complete for study $STUDY_ID"
