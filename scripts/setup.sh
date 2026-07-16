#!/bin/bash
# Reproduce the paper's source snapshots and three Python environments.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export UV_LINK_MODE=copy

DSPY_URL=https://github.com/stanfordnlp/dspy
DSPY_COMMIT=9cdb0aac28b2a04b064e40697ccd301872cf6a43
PYTHON=3.12.11

die() {
    echo "FATAL: $*" >&2
    exit 1
}

for command in git uv; do
    command -v "$command" >/dev/null || die "missing required command: $command"
done

ensure_corpus() {
    local path=$1 url=$2 commit=$3 name=$4 head dirty
    if [ ! -e "$path" ]; then
        git clone "$url" "$path"
        git -C "$path" checkout --detach "$commit"
    fi
    [ -d "$path/.git" ] || die "$name is not a Git checkout: $path"
    head=$(git -C "$path" rev-parse HEAD)
    [ "$head" = "$commit" ] \
        || die "$name is at $head, expected $commit; refusing to change it"
    dirty=$(git -C "$path" status --porcelain=v1 --untracked-files=all)
    [ -z "$dirty" ] || die "$name is dirty; refusing to change it"
}

ensure_venv() {
    local path=$1 python=$2 label=$3 actual
    if [ -e "$path" ] && [ ! -x "$path/bin/python" ]; then
        die "$label environment is incomplete: $path"
    fi
    if [ ! -e "$path" ]; then
        uv venv "$path" --python "$python"
    fi
    actual=$("$path/bin/python" -I -c \
        'import sys; print(".".join(map(str, sys.version_info[:3])))')
    [ "$actual" = "$python" ] \
        || die "$label environment uses Python $actual, expected $python"
}

mkdir -p corpora logs/slurm
ensure_corpus corpora/smalldspy "$DSPY_URL" "$DSPY_COMMIT" SmallDSPy

ensure_venv .venv "$PYTHON" root
UV_PROJECT_ENVIRONMENT="$ROOT/.venv" \
    uv sync --project "$ROOT" --frozen --python "$PYTHON"

ensure_venv .venv-dspy "$PYTHON" DSPy
UV_PROJECT_ENVIRONMENT="$ROOT/.venv-dspy" \
    uv sync --project "$ROOT/corpora/smalldspy" --frozen --no-dev \
    --python "$PYTHON"
uv pip install --python .venv-dspy/bin/python --no-deps 'regex==2026.6.28'

ensure_venv .venv-vllm "$PYTHON" vLLM
uv pip sync --python .venv-vllm/bin/python scripts/vllm-requirements.lock

echo "setup complete"
