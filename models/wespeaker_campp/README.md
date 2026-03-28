# WeSpeaker CAM++

| Metric | Value |
|---|---|
| Architecture | CAM++ with Large-Margin fine-tuning |
| Parameters | 7.18M |
| EER | 0.654% (VoxCeleb1) |
| Model size | 28 MB |
| Embedding | 512d |
| Best for | Best accuracy/speed ratio |

## Run

```bash
./run.sh benchmark --file ../../training_data/arda_2023/sound_train1.wav
./run.sh enroll
./run.sh verify
```
