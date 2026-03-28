"""Audio recording, saving, loading utilities."""

import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 16000


def record_audio(duration, label="Recording"):
    """Record audio from microphone with countdown."""
    print(f"\n{label} — {duration}s, speak after countdown...")
    print("3..."); sd.sleep(1000)
    print("2..."); sd.sleep(1000)
    print("1..."); sd.sleep(1000)
    print(">> RECORDING <<")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    print(">> DONE <<")
    return audio.squeeze()


def save_wav(audio, path, sr=SAMPLE_RATE):
    """Save audio array to WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def load_wav(path):
    """Load WAV file, return (audio_float32, sample_rate)."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr
