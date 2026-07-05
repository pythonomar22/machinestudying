#!/bin/bash
# One-time setup for offline grading: a sandbox venv with the pinned DSPy checkout
# installed (used by sandbox.py to actually run DSPy answers), plus the project venv.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_LINK_MODE=copy

[ -d .venv ] || uv sync
if [ ! -d .venv-dspy ]; then
    uv venv .venv-dspy -p 3.12
    uv pip install -p .venv-dspy ./corpora/dspy
fi
# optional deps exercised by gold answers (MIPROv2 needs optuna)
uv pip install -p .venv-dspy optuna
echo "grading environments ready"
