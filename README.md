# VOICEPRINT

Lightweight speaker verification for edge devices. Compares pretrained models, quantizes to int8 for ESP32-S3, and adds noise hardening for real-world use.

## Results

| Model | EER | Size | Inference | Edge |
|---|---|---|---|---|
| WeSpeaker CAM++ | 0.654% | 28 MB | 413 ms | -- |
| WeSpeaker ResNet34 | 0.723% | 25 MB | 83 ms | -- |
| SpeechBrain ECAPA | 0.800% | 83 MB | 37 ms | -- |
| **Resemblyzer (int8)** | ~5-7% | **1.4 MB** | ~1s | **ESP32-S3** |

| Quantization | Accuracy | Noise Hardening |
|---|---|---|
| Per-row int8: 0.993 cos sim | 17.3% flash | +0.20 boost at 5dB SNR |

## Quick Start

```bash
./setup.sh                                  # Install everything
python compare.py                           # Compare all 4 models (GUI)
cd models/resemblyzer && ./run.sh benchmark # Single model CLI
cd deploy/esp32/resemblyzer/v1 && ./run.sh  # Quantize + simulate
```

## Structure

```
.
├── compare.py                  # 4-model comparison GUI
│
├── models/                     # One folder per model
│   ├── resemblyzer/            #   ~5-7% EER, 17MB — edge target
│   ├── wespeaker_campp/        #   0.654% EER, 28MB — best accuracy
│   ├── wespeaker_resnet34/     #   0.723% EER, 25MB — compact
│   └── speechbrain_ecapa/      #   0.800% EER, 83MB — production API
│
├── deploy/                     # Edge deployment
│   └── esp32/resemblyzer/v1/   #   Int8 quantized + Arduino firmware
│       ├── quantize.py         #     TFLite pipeline
│       ├── export_weights.py   #     Pure C int8 export
│       ├── simulator_gui.py    #     Run on Mac before flashing
│       └── firmware/           #     Ready-to-flash .ino
│
├── lib/                        # Shared utilities
│   ├── audio.py                #   Recording, save/load WAV
│   ├── noise.py                #   Noise augmentation + spectral gating
│   └── metrics.py              #   Cosine similarity, timing
│
├── paper/paper.md              # Academic paper
└── training_data/arda_2023/    # Voice samples
```

## Per-Model Usage

Each model folder has the same interface:

```bash
cd models/<model_name>
./run.sh benchmark              # Measure speed, RAM, embedding size
./run.sh enroll                 # Record 10s from mic, save embedding
./run.sh verify                 # Record 5s, compare against enrollment
./run.sh enroll --file voice.wav
```

## ESP32-S3 Deployment

```bash
cd deploy/esp32/resemblyzer/v1
./run.sh quantize               # Export ONNX → TFLite (float32, int8, dynamic)
./run.sh export                 # Generate int8 C headers for Arduino
./run.sh simulator              # Test quantized model on Mac (GUI)
```

Then flash `firmware/speaker_verify.ino` to XIAO ESP32-S3 Sense.
