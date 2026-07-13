#!/bin/bash
#SBATCH --job-name=sb-retry-errors
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
# Retry infra-error episodes across every react-harness arm (resume logic
# retries status=error/forced_short only). One server startup, all arms.
set -uo pipefail   # no -e: per-arm exit 1 (residual errors) must not stop the sweep
cd "$SLURM_SUBMIT_DIR"

source scripts/serve_and_wait.sh

# variant:rollouts pairs (must match each arm's original rollout count)
for spec in ":3" "cheatsheet:6" "selfquiz-r1:3" "selfquiz-r2:3" "selfquiz-r4:3" \
            "selfquiz-select:3" "selfquiz-usage:3" "selfquiz-hybrid:6" \
            "selfquiz-hybrid2:6" "selfquiz-studied:6"; do
    v="${spec%%:*}"; r="${spec##*:}"
    for task in dspy openclaw; do
        .venv-dspy/bin/python -m studybench.react --task "$task" --rollouts "$r" \
            ${v:+--variant "$v"} --base-urls "$BASE_URLS" --concurrency $((8 * SB_NGPU)) \
            || echo "residual errors in variant='$v' task=$task (will show in validation)"
    done
done
echo "error-retry sweep complete"
