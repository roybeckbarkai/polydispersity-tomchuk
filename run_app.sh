#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

cd "$SCRIPT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found on PATH."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in .venv"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing requirements"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Starting Streamlit"
exec streamlit run streamlit_app.py
