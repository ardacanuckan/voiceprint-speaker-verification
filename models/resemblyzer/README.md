# Resemblyzer GE2E

| Metric | Value |
|---|---|
| Architecture | 3-layer LSTM (40→256) + Dense(256) |
| Parameters | 1.42M |
| EER | ~5-7% (VoxCeleb1) |
| Model size | 17 MB |
| Embedding | 256d |
| Best for | Edge deployment (smallest model) |

## Run

```bash
./run.sh benchmark --file ../../training_data/arda_2023/sound_train1.wav
./run.sh enroll --file ../../training_data/arda_2023/sound_train1.wav
./run.sh verify --file ../../training_data/arda_2023/sound_train2.wav
./run.sh enroll     # Record from mic
```
