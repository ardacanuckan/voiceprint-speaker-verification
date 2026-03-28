# VOICEPRINT — Lightweight Speaker Verification for Edge Devices

A research project exploring speaker verification on resource-constrained hardware. The goal: take a person's voice recording, build a voiceprint, and verify their identity — all running on a microcontroller with no cloud dependency.

## The Problem

Speaker verification models like ECAPA-TDNN and CAM++ achieve excellent accuracy (<1% EER) but require 25-83 MB of memory — far too large for microcontrollers like the ESP32-S3 (8 MB flash, 8 MB PSRAM). We need a model that fits in ~1.5 MB while still reliably distinguishing one person's voice from others, even in noisy environments.

## Approach

### Phase 1: Model Selection

We benchmarked 4 open-source pretrained speaker verification models to find the best candidate for edge deployment:

| Model | Architecture | EER | Model Size | CPU Inference | Embedding |
|---|---|---|---|---|---|
| WeSpeaker CAM++ | CAM++ + Large-Margin | **0.654%** | 28 MB | 413 ms | 512d |
| WeSpeaker ResNet34-LM | ResNet34 + Large-Margin | 0.723% | 25 MB | 83 ms | 256d |
| SpeechBrain ECAPA-TDNN | ECAPA-TDNN + SE blocks | 0.800% | 83 MB | 37 ms | 192d |
| **Resemblyzer GE2E** | **3-layer LSTM** | **~5-7%** | **17 MB (5.4 MB weights)** | 403 ms | **256d** |

CAM++ has the best accuracy, but at 28 MB it's too large for ESP32. Resemblyzer's 3-layer LSTM with 1.42M parameters is the only model that can realistically fit after quantization.

### Phase 2: Quantization for ESP32-S3

We explored two quantization approaches to compress Resemblyzer from 5.4 MB to fit on ESP32-S3:

**Attempt 1 — TFLite quantization:**

| Variant | Size | Accuracy (cos sim vs float32) |
|---|---|---|
| TFLite float32 | 6,854 KB | 1.0000 |
| TFLite int8 | 5,785 KB | 0.6311 — unusable |
| TFLite dynamic | 2,781 KB | 0.9910 |

Standard int8 quantization failed catastrophically (0.63 cosine similarity) because the LSTM had to be unrolled into 160 timesteps, and per-tensor quantization couldn't preserve accuracy across all those operations.

**Attempt 2 — Per-row int8 quantization (our solution):**

Instead of TFLite, we quantize the weights ourselves and implement the LSTM forward pass in pure C.

The key insight: **per-tensor** quantization uses one scale factor for the entire weight matrix. The first LSTM layer (`weight_ih_l0`) had outlier rows with scale 0.48, while most rows needed only ~0.01. This mismatch destroyed 90% of the precision in those rows.

**Per-row** quantization gives each row its own scale factor:

| Layer | Per-Tensor Error | Per-Row Error | Improvement |
|---|---|---|---|
| lstm.weight_ih_l0 | 0.0902 | 0.0076 | **11.9x** |
| lstm.weight_hh_l0 | 0.0323 | 0.0099 | 3.3x |
| lstm.weight_ih_l1 | 0.0238 | 0.0084 | 2.8x |
| linear.weight | 0.0291 | 0.0116 | 2.5x |

**Final quantized model:**

| Metric | Value |
|---|---|
| Weight size | **1,390 KB** (int8) |
| Scale factors | 25 KB (float32, one per row) |
| Total | **1,415 KB** |
| ESP32-S3 flash usage | **17.3%** of 8 MB |
| Accuracy vs float32 | **0.993** cosine similarity |
| Compression ratio | 74% smaller than float32 |

### Phase 3: Noise Hardening

In real-world use, the device operates in environments where multiple people talk. A clean enrollment recording tested against noisy verification audio scores poorly.

We developed three techniques applied without retraining the model:

1. **Augmented enrollment** — The clean enrollment audio is augmented with 5 synthetic noise types (white noise, babble, reverb, etc.). Each variant produces an embedding. All embeddings are averaged into a noise-tolerant centroid that covers a broader region of embedding space.

2. **Spectral gating** — Verification audio passes through adaptive spectral subtraction to remove background noise before feature extraction.

3. **Multi-segment averaging** — Verification audio is split into overlapping segments, each producing its own embedding. The average reduces the impact of transient noise.

### Phase 4: Impostor Testing

We tested with real voice data from 2 speakers: target (Arda) and impostor (Zeliha).

**Score distributions (197 target segments, 44 impostor segments):**

| Speaker | Mean Score | Min | Max |
|---|---|---|---|
| Arda (target) | 0.734 | 0.620 | 0.811 |
| Zeliha (impostor) | 0.605 | 0.529 | 0.691 |

**Performance at optimal threshold (0.645):**

| Metric | Value |
|---|---|
| True Accept Rate | 98.0% (193/197) |
| False Accept Rate | 9.1% (4/44) |
| Precision | 98.0% |
| F1 Score | 0.980 |
| Estimated EER | **5.1%** |

**Noise robustness (target speaker, all conditions PASS):**

| Noise Type | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|
| White | 100% | 100% | 100% | 100% | 100% |
| Babble | 100% | 100% | 100% | 100% | 100% |
| Street | 100% | 100% | 100% | 100% | 100% |
| Fan | 100% | 100% | 100% | 100% | 100% |
| Music | 93% | 100% | 100% | 100% | 100% |

## ESP32-S3 Implementation

The `deploy/esp32/resemblyzer/v1/firmware/` directory contains a complete Arduino sketch for XIAO ESP32-S3 Sense. The implementation is pure C — no TFLite dependency:

- `lstm_engine.h` — LSTM forward pass with on-the-fly per-row dequantization
- `mel_features.h` — 512-point FFT + 40-channel mel filterbank
- `model_weights.h` — 1,390 KB int8 weight arrays with per-row scale factors
- `speaker_verify.ino` — Main sketch: I2S mic capture → mel → LSTM → cosine similarity → LED

**Workflow on device:**
1. Hold BOOT button → record 3 seconds → enrollment saved to SPIFFS flash
2. Speak near mic → voice activity detection → inference → LED feedback
3. Enrollment persists across power cycles

## Production App

`speaker_verification_production/` contains a standalone GUI app for testing on Mac:

```bash
cd speaker_verification_production
./run.sh    # One command: installs deps, downloads models, launches GUI
```

**Features:**
- Guided 5-minute enrollment with on-screen reading text and NEXT button
- Toggle between clean and noise-augmented enrollment
- Side-by-side comparison: Resemblyzer float32 vs int8 quantized
- Real-time inference time, CPU usage, cosine similarity score
- Verify from microphone (5s) or WAV file

## Repository Structure

```
.
├── models/                         # Per-model benchmarks
│   ├── resemblyzer/                #   GE2E LSTM — edge target
│   ├── wespeaker_campp/            #   CAM++ — best accuracy
│   ├── wespeaker_resnet34/         #   ResNet34-LM — compact
│   └── speechbrain_ecapa/          #   ECAPA-TDNN — production API
│
├── deploy/esp32/resemblyzer/v1/    # Edge deployment pipeline
│   ├── quantize.py                 #   ONNX + TFLite quantization
│   ├── export_weights.py           #   Per-row int8 C header export
│   ├── simulator_gui.py            #   Test quantized model on Mac
│   ├── benchmark_test.py           #   Automated accuracy test
│   ├── impostor_test.py            #   Multi-speaker rejection test
│   ├── scientific_eval.py          #   Full evaluation with EER
│   └── firmware/                   #   Arduino sketch for ESP32-S3
│
├── speaker_verification_production/ # Standalone GUI app
│   ├── run.sh                      #   One-click setup + launch
│   ├── app.py                      #   GUI with guided enrollment
│   └── engine.py                   #   Model backends (float32 + int8)
│
├── lib/                            # Shared code
│   ├── audio.py                    #   Recording, WAV I/O
│   ├── noise.py                    #   Augmentation + spectral gating
│   └── metrics.py                  #   Cosine similarity, timing
│
├── paper/paper.md                  # Research paper
├── compare.py                      # 4-model comparison GUI
├── setup.sh                        # Project-wide setup
└── training_data/                  # Voice samples (not in git)
```

## Limitations

- **EER of 5.1%** — Resemblyzer is significantly less accurate than CAM++ (0.654%). For high-security use, this is insufficient.
- **Score gap of 0.13** between target and impostor means some impostor segments score close to the threshold, especially with certain noise types.
- **ESP32-S3 inference time** not yet measured on hardware — the pure C LSTM with dequantization may take several seconds at 240 MHz.
- **Single-speaker design** — verifies against one enrolled person only.
- **No liveness detection** — vulnerable to replay attacks.

## Requirements

- Python 3.10+
- macOS (tested on Apple Silicon) or Linux
- Microphone for recording, or WAV files
- ~500 MB disk for models and dependencies
- XIAO ESP32-S3 Sense for hardware deployment (optional)
