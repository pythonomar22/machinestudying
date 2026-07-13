#!/bin/bash
# Idempotent setup from a fresh clone. Existing wrong/dirty corpus snapshots and
# drifted environments fail closed; the script never resets research inputs.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_LINK_MODE=copy

source scripts/setup_common.sh
require_command git
require_command uv
require_command stat
verify_env_file

mkdir -p corpora logs/slurm
ensure_corpus "$PWD/corpora/dspy" "$DSPY_URL" "$DSPY_SHA" DSPy
ensure_corpus "$PWD/corpora/openclaw" "$OPENCLAW_URL" "$OPENCLAW_SHA" OpenClaw
sync_main_environment
sync_dspy_environment
sync_vllm_environment

echo "setup complete: pinned corpora and environments verified"
