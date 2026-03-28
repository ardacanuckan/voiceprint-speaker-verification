# ESP32-S3 / Resemblyzer v1

Int8 per-row quantized Resemblyzer for XIAO ESP32-S3 Sense.

## Results

| Metric | Value |
|---|---|
| Original size | 5,430 KB (float32) |
| Quantized size | **1,390 KB** (int8) |
| Scale factors | 25 KB |
| ESP32-S3 flash usage | 17.3% of 8 MB |
| Accuracy (cos sim) | 0.993 vs float32 |
| Noise hardening boost | +0.20 at 5dB babble |

## Run

```bash
./run.sh                # Run everything (quantize + export + simulator)
./run.sh quantize       # TFLite quantization pipeline only
./run.sh export         # Pure C int8 weight export only
./run.sh simulator      # Launch ESP32 simulator GUI
```

## Files

- `quantize.py` — ONNX export + TFLite conversion + accuracy comparison
- `export_weights.py` — Per-row int8 quantization + C header generation
- `simulator_gui.py` — Run quantized model on Mac, see memory/speed/accuracy
- `firmware/` — Complete Arduino sketch for XIAO ESP32-S3 Sense

## Hardware Deployment

1. Open `firmware/speaker_verify.ino` in Arduino IDE
2. Board: **XIAO ESP32S3 Sense**
3. Upload, Serial Monitor 115200
4. BOOT button = enroll, speak = verify
