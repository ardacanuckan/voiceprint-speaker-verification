#!/bin/bash
set -e; cd "$(dirname "$0")"; ROOT="$(cd ../.. && pwd)"
source "$ROOT/venv/bin/activate"; export PYTHONPATH="$ROOT"
python benchmark.py "${1:-benchmark}" "${@:2}"
