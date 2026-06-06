#!/usr/bin/env bash
# Force the project venv so you don't accidentally use Homebrew Python 3.13
# (which has no plotly unless you bypass PEP 668).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/src"
exec "${ROOT}/venv/bin/python" -m streamlit run "${ROOT}/streamlit_apps/psf_3d_viewer.py" "$@"
