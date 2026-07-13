#!/bin/bash
#SBATCH --job-name=sb-selfquiz
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
# One self-quizzing study round (studybench/selfquiz.py). Resumable at item
# granularity. SB_ROUND selects the round; SB_SMOKE=1 runs 1 chapter x 3
# questions; SB_TASKS selects corpora.
# On other clusters: sbatch -p <partition> --gpus-per-node=N scripts/selfquiz.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TASKS=${SB_TASKS:-dspy,openclaw}
ROUND=${SB_ROUND:-1}
SMOKE=${SB_SMOKE:-0}

source scripts/serve_and_wait.sh

FLAGS=""
[ "$SMOKE" = 1 ] && FLAGS="--smoke --debug"
[ "${SB_COMPACT:-0}" = 1 ] && FLAGS="--compact"
[ "${SB_USAGE:-0}" = 1 ] && FLAGS="--usage"
[ "${SB_STUDIED:-0}" = 1 ] && FLAGS="--studied"
for task in ${TASKS//,/ }; do
    .venv-dspy/bin/python -m studybench.selfquiz --task "$task" --round "$ROUND" \
        --base-urls "$BASE_URLS" --concurrency $((4 * SB_NGPU)) $FLAGS
done
echo "selfquiz round $ROUND complete"
