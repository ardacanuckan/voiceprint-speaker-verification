#!/bin/bash
# ============================================
#  VOICEPRINT — Speaker Verification
#  One-click setup & launch
# ============================================
set -e
cd "$(dirname "$0")"

echo ""
echo "  ██╗   ██╗ ██████╗ ██╗ ██████╗███████╗"
echo "  ██║   ██║██╔═══██╗██║██╔════╝██╔════╝"
echo "  ██║   ██║██║   ██║██║██║     █████╗  "
echo "  ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝  "
echo "   ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗"
echo "    ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝"
echo "  SPEAKER VERIFICATION SYSTEM"
echo ""

# --- Python check ---
if ! command -v python3 &>/dev/null; then
    echo "  ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

# --- Venv ---
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/3] Virtual environment ready"
fi
source venv/bin/activate

# --- Dependencies ---
echo "[2/3] Installing dependencies (first run may take a few minutes)..."
pip install -q -r requirements.txt 2>/dev/null

# --- Models ---
echo "[3/3] Checking models..."
mkdir -p cache

if [ ! -f "cache/campplus_LM.onnx" ]; then
    echo "  Downloading WeSpeaker CAM++ (28MB)..."
    curl -sL "https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus-LM/resolve/main/voxceleb_CAM%2B%2B_LM.onnx" -o cache/campplus_LM.onnx
fi

echo "  Resemblyzer + SpeechBrain will auto-download on first use"
echo ""
echo "  All ready! Launching..."
echo ""

python3 app.py
