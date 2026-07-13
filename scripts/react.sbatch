#!/bin/bash
#SBATCH --job-name=sb-react
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
# Faithful dspy.ReAct rollouts (studybench/react.py) -> runs/react/.
# Resumable; smoke via SB_TASKS/SB_BUDGETS/SB_ROLLOUTS/SB_LIMIT like rollout.sbatch.
# On other clusters: sbatch -p <partition> --gpus-per-node=N scripts/react.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TASKS=${SB_TASKS:-dspy,openclaw}
BUDGETS=${SB_BUDGETS:-direct,k5,k20,k20f}
ROLLOUTS=${SB_ROLLOUTS:-3}
LIMIT=${SB_LIMIT:-0}
VARIANT=${SB_VARIANT:-}
STUDY=${SB_STUDY:-0}   # 1 = run the forced-50 study loop instead of the eval grid

source scripts/serve_and_wait.sh

for task in ${TASKS//,/ }; do
    if [ "$STUDY" = 1 ]; then
        .venv-dspy/bin/python -m studybench.react --task "$task" --study \
            ${SB_STUDY_SEED:+--study-seed "$SB_STUDY_SEED"} \
            --base-urls "$BASE_URLS"
    else
        .venv-dspy/bin/python -m studybench.react --task "$task" \
            --budgets "$BUDGETS" --rollouts "$ROLLOUTS" --limit "$LIMIT" \
            --variant "$VARIANT" \
            --base-urls "$BASE_URLS" --concurrency $((8 * SB_NGPU))
    fi
done
echo "react rollouts complete"
