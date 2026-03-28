# VOICEPRINT — Speaker Verification

Guided speaker verification app with 3 models side-by-side.

## Quick Start

```bash
./run.sh
```

That's it. Everything downloads and installs automatically.

## What It Does

1. **Load Models** — loads 3 models in parallel:
   - WeSpeaker CAM++ (0.654% EER, 28MB) — best accuracy
   - Resemblyzer float32 (~5-7% EER, 17MB) — full model
   - Resemblyzer int8 (~5-7% EER, 1.4MB) — ESP32 version

2. **Enroll** — read 5 sentences aloud (8s each), or load a WAV file.
   The app builds your voiceprint from multiple recordings.

3. **Verify** — speak into mic or load a WAV. See match/reject for each model,
   with inference time, CPU usage, and similarity score.

## Models Compared

| Model | EER | Size | What |
|---|---|---|---|
| WeSpeaker CAM++ | 0.654% | 28 MB | Best accuracy, runs on Mac |
| Resemblyzer float32 | ~5-7% | 17 MB | Full model, runs on Mac |
| Resemblyzer int8 | ~5-7% | 1.4 MB | Quantized for ESP32-S3 |

## Files

```
speaker_verification_production/
├── run.sh              # One-click setup + launch
├── requirements.txt    # Python deps
├── app.py              # GUI application
├── engine.py           # Model backends (full + quantized + CAM++)
├── README.md           # This file
├── cache/              # Downloaded models (auto-created)
└── data/               # Recordings & embeddings (auto-created)
```
