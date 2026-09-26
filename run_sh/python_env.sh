#!/usr/bin/env bash
# Shared interpreter selection; callers may override PYTHON_BIN explicitly.
_python_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${UV_PROJECT_ENVIRONMENT:-$_python_repo_root/.venv}/bin/python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python interpreter: $PYTHON_BIN. Run 'uv sync --locked' in $_python_repo_root." >&2
  return 2
fi
export PYTHON_BIN
unset _python_repo_root
