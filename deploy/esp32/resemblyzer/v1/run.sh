#!/bin/bash
set -e; cd "$(dirname "$0")"; ROOT="$(cd ../../../.. && pwd)"
source "$ROOT/venv/bin/activate"; export PYTHONPATH="$ROOT"

MODE=${1:-all}

echo "========================================"
echo "  DEPLOY // ESP32 // RESEMBLYZER v1"
echo "========================================"

case "$MODE" in
    quantize)   python quantize.py ;;
    export)     python export_weights.py ;;
    simulator)  python simulator_gui.py ;;
    benchmark)  python benchmark_test.py ;;
    eval)       python scientific_eval.py ;;
    all)
        echo "--- TFLite quantization ---"
        python quantize.py
        echo ""
        echo "--- Pure C weight export ---"
        python export_weights.py
        echo ""
        echo "--- Launching simulator ---"
        python simulator_gui.py
        ;;
    *)
        echo "Usage: ./run.sh [quantize|export|simulator|benchmark|all]"
        ;;
esac
