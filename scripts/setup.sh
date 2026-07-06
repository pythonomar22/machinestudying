#!/bin/bash
# One-time environment setup from a fresh clone (no secrets needed for rollouts).
# Creates: corpora checkouts at the paper's pinned commits (Table 2), the runner
# venvs. Requires uv (https://docs.astral.sh/uv/) and git.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_LINK_MODE=copy

DSPY_SHA=9cdb0aac28b2a04b064e40697ccd301872cf6a43
OPENCLAW_SHA=da228660306b55a9cce3b973946f3aacfc515848

mkdir -p corpora logs/slurm
if [ ! -d corpora/dspy ]; then
    git clone https://github.com/stanfordnlp/dspy corpora/dspy
    git -C corpora/dspy checkout $DSPY_SHA
fi
if [ ! -d corpora/openclaw ]; then
    git clone https://github.com/openclaw/openclaw corpora/openclaw
    git -C corpora/openclaw checkout $OPENCLAW_SHA
fi

[ -d .venv ] || uv sync
if [ ! -d .venv-dspy ]; then
    uv venv .venv-dspy -p 3.12
    uv pip install -p .venv-dspy ./corpora/dspy optuna regex
fi
if [ ! -x .venv-vllm/bin/vllm ]; then
    uv venv .venv-vllm -p 3.12
    uv pip install -p .venv-vllm vllm==0.24.0 ninja
fi
echo "setup complete"
