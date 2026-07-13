#!/bin/bash
# Offline grading setup. This intentionally omits the GPU/vLLM environment.
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

echo "grading environments ready: pinned corpora and parser imports verified"
