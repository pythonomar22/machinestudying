#!/bin/bash
# Study the pinned SmallDSPy corpus for 50 forced ReAct iterations, save the
# resulting cheatsheet, then run the same held-out evaluation as the baseline.
# Example:
#   SB_RUN_ID=smalldspy-cheatsheet SB_SEED=0 sbatch scripts/cheatsheet.sbatch
#SBATCH --job-name=studybench-cheat
#SBATCH --account=matx
#SBATCH --partition=matx
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job with sbatch}"
export SB_CONDITION=cheatsheet
export SB_DEBUG=${SB_DEBUG:-1}
exec bash scripts/nostudying.sbatch
