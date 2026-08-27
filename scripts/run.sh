#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${PROJECT_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "Python 3.11 or newer is required. Install Python, then run this script again." >&2
        exit 1
    fi

    echo "First run: creating the private HoomansLLM environment..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if ! "${VENV_PYTHON}" -c 'import fastapi, google.genai, openai, pydantic_settings, uvicorn, hoomans_llm' >/dev/null 2>&1; then
    echo "First run: installing HoomansLLM dependencies..."
    "${VENV_PYTHON}" -m ensurepip --upgrade >/dev/null 2>&1 || true
    "${VENV_PYTHON}" -m pip install --upgrade pip
    "${VENV_PYTHON}" -m pip install "${PROJECT_ROOT}"
fi

cd "${PROJECT_ROOT}"
exec "${VENV_PYTHON}" -m hoomans_llm "$@"
