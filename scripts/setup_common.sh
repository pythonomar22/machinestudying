#!/bin/bash
# Shared, fail-closed setup helpers. This file is sourced; it performs no setup
# by itself. Existing corpora and environments are never reset automatically.

if [ "${STUDYBENCH_SETUP_COMMON_LOADED:-0}" = 1 ]; then
    return 0
fi
readonly STUDYBENCH_SETUP_COMMON_LOADED=1

readonly REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly DSPY_SHA=9cdb0aac28b2a04b064e40697ccd301872cf6a43
readonly OPENCLAW_SHA=da228660306b55a9cce3b973946f3aacfc515848
readonly DSPY_URL=https://github.com/stanfordnlp/dspy
readonly OPENCLAW_URL=https://github.com/openclaw/openclaw
readonly MAIN_PYTHON_VERSION=3.14.6
readonly AUX_PYTHON_VERSION=3.12.11
readonly TREE_SITTER_VERSION=0.26.0
readonly TREE_SITTER_TYPESCRIPT_VERSION=0.23.2
readonly VLLM_VERSION=0.24.0
readonly VLLM_LOCK="$REPO_ROOT/scripts/vllm-requirements.lock"

die() {
    echo "FATAL: $*" >&2
    return 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

verify_env_file() {
    local path="$REPO_ROOT/.env"
    if [ ! -e "$path" ] && [ ! -L "$path" ]; then
        return 0
    fi
    if [ ! -f "$path" ] || [ -L "$path" ]; then
        die ".env must be a regular, non-symlink file"
        return 1
    fi
    if [ "$(stat -c '%a' "$path")" != 600 ]; then
        die ".env must have mode 0600 (contents were not read)"
        return 1
    fi
    if [ "$(stat -c '%u' "$path")" != "$(id -u)" ]; then
        die ".env must be owned by the current user (contents were not read)"
        return 1
    fi
}

verify_python_version() {
    local executable=$1 expected=$2 label=$3 observed
    if [ ! -x "$executable" ]; then
        die "$label Python is unavailable: $executable"
        return 1
    fi
    if ! observed=$("$executable" -I -c \
        'import sys; print(".".join(map(str, sys.version_info[:3])))'); then
        die "cannot inspect $label Python"
        return 1
    fi
    if [ "$observed" != "$expected" ]; then
        die "$label uses Python $observed, expected exactly $expected"
        return 1
    fi
}

ensure_venv() {
    local path=$1 expected=$2 label=$3
    if [ -e "$path" ] && [ ! -x "$path/bin/python" ]; then
        die "$label environment exists but is incomplete: $path"
        return 1
    fi
    if [ ! -e "$path" ]; then
        uv venv "$path" --python "$expected" || return 1
    fi
    verify_python_version "$path/bin/python" "$expected" "$label" || return 1
}

verify_corpus() {
    local path=$1 expected=$2 name=$3
    if [ ! -d "$path/.git" ]; then
        die "$name is not a git checkout: $path"
        return 1
    fi
    local observed
    if ! observed=$(git -C "$path" rev-parse HEAD); then
        die "cannot read $name HEAD"
        return 1
    fi
    if [ "$observed" != "$expected" ]; then
        die "$name is at $observed, expected $expected; refusing to change it"
        return 1
    fi
    local dirty
    if ! dirty=$(git -C "$path" status --porcelain=v1 --untracked-files=all); then
        die "cannot inspect $name status"
        return 1
    fi
    if [ -n "$dirty" ]; then
        die "$name checkout is dirty; preserve or remove those changes before setup"
        return 1
    fi
}

ensure_corpus() {
    local path=$1 url=$2 expected=$3 name=$4
    if [ ! -e "$path" ]; then
        git clone "$url" "$path" || return 1
        git -C "$path" checkout --detach "$expected" || return 1
    fi
    verify_corpus "$path" "$expected" "$name" || return 1
}

sync_main_environment() {
    require_command uv || return 1
    ensure_venv "$REPO_ROOT/.venv" "$MAIN_PYTHON_VERSION" "main" || return 1
    (cd "$REPO_ROOT" && UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv" uv sync --frozen) \
        || return 1
    verify_python_version \
        "$REPO_ROOT/.venv/bin/python" "$MAIN_PYTHON_VERSION" "main" || return 1
    "$REPO_ROOT/.venv/bin/python" - <<PY
from importlib.metadata import version

expected = {
    "tree-sitter": "$TREE_SITTER_VERSION",
    "tree-sitter-typescript": "$TREE_SITTER_TYPESCRIPT_VERSION",
}
observed = {name: version(name) for name in expected}
if observed != expected:
    raise SystemExit(f"TypeScript parser version drift: {observed}, expected {expected}")
PY
}

sync_dspy_environment() {
    # DSPy's own lock at the pinned commit freezes the selfquiz dependency graph.
    # Selfquiz validates quoted repository evidence; it does not execute or
    # syntax-check generated code, so grading-only parser packages do not belong
    # in this environment.
    require_command git || return 1
    require_command uv || return 1
    verify_corpus "$REPO_ROOT/corpora/dspy" "$DSPY_SHA" "DSPy harness" \
        || return 1
    ensure_venv "$REPO_ROOT/.venv-dspy" "$AUX_PYTHON_VERSION" "DSPy" \
        || return 1
    UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv-dspy" \
        uv sync --project "$REPO_ROOT/corpora/dspy" --frozen --no-dev \
        || return 1
    verify_python_version \
        "$REPO_ROOT/.venv-dspy/bin/python" "$AUX_PYTHON_VERSION" "DSPy" \
        || return 1
    "$REPO_ROOT/.venv-dspy/bin/python" - <<PY
import dspy, pydantic
PY
}

verify_vllm_environment() {
    verify_python_version \
        "$REPO_ROOT/.venv-vllm/bin/python" "$AUX_PYTHON_VERSION" "vLLM" \
        || return 1
    if [ ! -f "$VLLM_LOCK" ]; then
        die "missing vLLM requirements lock: $VLLM_LOCK"
        return 1
    fi
    "$REPO_ROOT/.venv-vllm/bin/python" - "$VLLM_LOCK" <<'PY'
from importlib.metadata import distributions
from pathlib import Path
import re
import sys

expected = sorted(
    line.strip() for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
)
observed = sorted(
    f"{re.sub(r'[-_.]+', '-', dist.metadata['Name']).lower()}=={dist.version}"
    for dist in distributions()
)
if observed != expected:
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    raise SystemExit(f"vLLM environment differs from lock; missing={missing}, extra={extra}")
PY
}

write_vllm_inventory() {
    local destination=$1
    "$REPO_ROOT/.venv-vllm/bin/python" -I - "$destination" "$REPO_ROOT" <<'PY'
from pathlib import Path
import os
import sys

sys.path.insert(0, sys.argv[2])
from studybench.integrity import canonical_json_bytes
from studybench.provenance import installed_distribution_inventory

destination = Path(sys.argv[1])
payload = canonical_json_bytes(installed_distribution_inventory())
temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
temporary.write_bytes(payload)
if destination.exists():
    if not destination.is_file() or destination.read_bytes() != payload:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"refusing to replace a different environment inventory: {destination}"
        )
    temporary.unlink()
else:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if not destination.is_file() or destination.read_bytes() != payload:
            temporary.unlink(missing_ok=True)
            raise SystemExit(
                f"refusing to replace a different environment inventory: {destination}"
            )
    temporary.unlink()
PY
}

sync_vllm_environment() {
    ensure_venv "$REPO_ROOT/.venv-vllm" "$AUX_PYTHON_VERSION" "vLLM" \
        || return 1
    if [ ! -f "$VLLM_LOCK" ]; then
        die "missing vLLM requirements lock: $VLLM_LOCK"
        return 1
    fi
    uv pip sync --python "$REPO_ROOT/.venv-vllm/bin/python" "$VLLM_LOCK" \
        || return 1
    if ! verify_vllm_environment; then
        die "vLLM environment differs from its complete lock"
        return 1
    fi
}
