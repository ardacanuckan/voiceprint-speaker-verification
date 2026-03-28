# SpeechBrain ECAPA-TDNN

| Metric | Value |
|---|---|
| Architecture | ECAPA-TDNN (SE blocks + multi-scale) |
| Parameters | 22.15M |
| EER | 0.800% (VoxCeleb1) |
| Model size | 83 MB |
| Embedding | 192d |
| Best for | Production-ready API |

## Run

```bash
./run.sh benchmark --file ../../training_data/arda_2023/sound_train1.wav
./run.sh enroll
./run.sh verify
```
