#!/bin/bash
# ============================================
# VOICEPRINT — One-click setup
# ============================================
set -e
cd "$(dirname "$0")"

echo "========================================"
echo "  VOICEPRINT // SETUP"
echo "========================================"
echo ""

# Virtual environment
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/4] Virtual environment exists"
fi

source venv/bin/activate

# Dependencies
echo "[2/4] Installing dependencies..."
pip install -q -r requirements.txt

# Models
echo "[3/4] Downloading pretrained models..."
mkdir -p cache

if [ ! -f "cache/campplus_LM.onnx" ]; then
    echo "  WeSpeaker CAM++ (28MB)..."
    curl -sL "https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus-LM/resolve/main/voxceleb_CAM%2B%2B_LM.onnx" -o cache/campplus_LM.onnx
fi

if [ ! -f "cache/resnet34_LM.onnx" ]; then
    echo "  WeSpeaker ResNet34 (25MB)..."
    curl -sL "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx" -o cache/resnet34_LM.onnx
fi

echo "  SpeechBrain + Resemblyzer auto-download on first use"

# Directories
echo "[4/4] Creating directories..."
mkdir -p data

echo ""
echo "========================================"
echo "  SETUP COMPLETE"
echo "========================================"
echo ""
echo "  Compare all models:          python compare.py"
echo "  Single model:                cd models/resemblyzer && ./run.sh"
echo "  ESP32 deployment:            cd deploy/esp32/resemblyzer/v1 && ./run.sh"
echo ""
