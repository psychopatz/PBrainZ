#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${PROJECT_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install "${PROJECT_ROOT}[dev]"

echo "Installed P BrainZ into ${VENV_DIR}"
echo "The system Python was not modified."
echo "Next: ./scripts/run.sh, then configure providers in the native control panel"
