# WeSpeaker ResNet34-LM

| Metric | Value |
|---|---|
| Architecture | ResNet34 with Large-Margin fine-tuning |
| Parameters | 6.70M |
| EER | 0.723% (VoxCeleb1) |
| Model size | 25 MB |
| Embedding | 256d |
| Best for | Compact + accurate |

## Run

```bash
./run.sh benchmark --file ../../training_data/arda_2023/sound_train1.wav
./run.sh enroll
./run.sh verify
```
