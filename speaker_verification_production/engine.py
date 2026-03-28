"""
Speaker Verification Engine — Full + ESP32 quantized models.
Shared backend for the GUI app.
"""

import time
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
import psutil
from pathlib import Path

SAMPLE_RATE = 16000
CACHE_DIR = Path(__file__).parent / "cache"
DATA_DIR = Path(__file__).parent / "data"


# ============================================================
# FULL MODEL — Resemblyzer (float32, ~17MB)
# ============================================================

class FullModel:
    name = "RESEMBLYZER FLOAT32"
    tag = "FULL MODEL"

    def __init__(self):
        self.encoder = None

    def load(self):
        from resemblyzer import VoiceEncoder
        self.encoder = VoiceEncoder()

    @property
    def loaded(self):
        return self.encoder is not None

    def embed(self, audio_np):
        return self.encoder.embed_utterance(audio_np)

    def info(self):
        return {
            "name": "Resemblyzer GE2E",
            "precision": "float32",
            "params": "1.42M",
            "size": "17 MB",
            "embedding": "256d",
            "eer": "~5-7%",
        }


# ============================================================
# ESP32 MODEL — Quantized int8 per-row (~1.4MB)
# ============================================================

class ESP32Model:
    name = "RESEMBLYZER INT8"
    tag = "ESP32 MODEL"

    def __init__(self):
        self.weights = {}
        self.scales = {}
        self._loaded = False

    def load(self):
        from resemblyzer import VoiceEncoder
        for key, tensor in VoiceEncoder().state_dict().items():
            t = tensor.numpy()
            if t.ndim == 2:
                s = np.array([max(np.abs(t[r]).max(), 1e-10) / 127.0
                              for r in range(t.shape[0])], dtype=np.float32)
                self.weights[key] = np.round(t / s[:, None]).clip(-127, 127).astype(np.int8)
                self.scales[key] = s
            else:
                s = max(np.abs(t).max(), 1e-10) / 127.0
                self.weights[key] = np.round(t / s).clip(-127, 127).astype(np.int8)
                self.scales[key] = np.array([s], dtype=np.float32)
        self._loaded = True

    @property
    def loaded(self):
        return self._loaded

    def _dq(self, name):
        s, w = self.scales[name], self.weights[name].astype(np.float32)
        return w * s[:, None] if (s.ndim == 1 and w.ndim == 2 and s.shape[0] == w.shape[0]) else w * s[0]

    def embed(self, audio_np):
        from resemblyzer.audio import wav_to_mel_spectrogram
        mel = wav_to_mel_spectrogram(audio_np)
        x = mel
        for l in range(3):
            T, H = x.shape[0], 256
            wih, whh = self._dq(f"lstm.weight_ih_l{l}"), self._dq(f"lstm.weight_hh_l{l}")
            bih, bhh = self._dq(f"lstm.bias_ih_l{l}"), self._dq(f"lstm.bias_hh_l{l}")
            h, c, out = np.zeros(H, np.float32), np.zeros(H, np.float32), np.zeros((T, H), np.float32)
            for t in range(T):
                g = wih @ x[t] + bih + whh @ h + bhh
                ig = 1/(1+np.exp(-np.clip(g[0:H],-20,20)))
                fg = 1/(1+np.exp(-np.clip(g[H:2*H],-20,20)))
                gg = np.tanh(g[2*H:3*H])
                og = 1/(1+np.exp(-np.clip(g[3*H:],-20,20)))
                c = fg*c + ig*gg; h = og*np.tanh(c); out[t] = h
            x = out
        p = np.maximum(self._dq("linear.weight") @ x[-1] + self._dq("linear.bias"), 0)
        return p / (np.linalg.norm(p) + 1e-8)

    def info(self):
        wb = sum(w.nbytes for w in self.weights.values())
        sb = sum(s.nbytes for s in self.scales.values())
        return {
            "name": "Resemblyzer GE2E",
            "precision": "int8 per-row",
            "params": "1.42M",
            "size": f"{(wb+sb)/1024:.0f} KB",
            "embedding": "256d",
            "eer": "~5-7%",
            "weight_bytes": wb,
            "scale_bytes": sb,
            "esp32_flash_pct": f"{(wb+sb)/(8*1024*1024)*100:.1f}%",
        }


# ============================================================
# FULL MODEL — WeSpeaker CAM++ (ONNX, ~28MB)
# ============================================================

class CAMPlusPlusModel:
    name = "WESPEAKER CAM++"
    tag = "BEST ACCURACY"

    def __init__(self):
        self.session = None

    def load(self):
        import onnxruntime as ort
        path = CACHE_DIR / "campplus_LM.onnx"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}\nRun ./run.sh first")
        self.session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])

    @property
    def loaded(self):
        return self.session is not None

    def embed(self, audio_np):
        import torch
        import torchaudio
        if not hasattr(torchaudio, 'list_audio_backends'):
            torchaudio.list_audio_backends = lambda: ['soundfile']
        waveform = torch.from_numpy(audio_np).unsqueeze(0).float()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, num_mel_bins=80, sample_frequency=SAMPLE_RATE, dither=0.0)
        fbank = (fbank - fbank.mean(dim=0, keepdim=True)).unsqueeze(0).numpy()
        inp = self.session.get_inputs()[0].name
        out = self.session.get_outputs()[0].name
        return self.session.run([out], {inp: fbank})[0].squeeze()

    def info(self):
        return {
            "name": "WeSpeaker CAM++",
            "precision": "float32",
            "params": "7.18M",
            "size": "28 MB",
            "embedding": "512d",
            "eer": "0.654%",
        }


# ============================================================
# Utilities
# ============================================================

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def record_audio(duration, sr=SAMPLE_RATE):
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()


def noise_reduce(audio, sr=SAMPLE_RATE):
    import noisereduce as nr
    return nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.65).astype(np.float32)


# --- Noise augmentation for enrollment ---
NOISE_AUGMENTATIONS = [
    ("white_5dB",  lambda a: a + np.random.randn(len(a)).astype(np.float32) * (np.sqrt(np.mean(a**2)) / 10**(5/20))),
    ("white_15dB", lambda a: a + np.random.randn(len(a)).astype(np.float32) * (np.sqrt(np.mean(a**2)) / 10**(15/20))),
    ("babble_5dB", lambda a: _add_babble(a, 5)),
    ("babble_10dB", lambda a: _add_babble(a, 10)),
    ("reverb",     lambda a: _add_reverb(a)),
]

def _add_babble(audio, snr_db, n_voices=5):
    babble = np.zeros_like(audio)
    t = np.arange(len(audio)) / SAMPLE_RATE
    for _ in range(n_voices):
        freq = np.random.uniform(2, 6)
        env = 0.5 * (1 + np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi)))
        babble += np.random.randn(len(audio)).astype(np.float32) * env
    babble = babble / (np.std(babble) + 1e-8)
    rms = np.sqrt(np.mean(audio**2))
    return audio + babble * (rms / 10**(snr_db/20))

def _add_reverb(audio, decay=0.4, delay_ms=40):
    d = int(delay_ms * SAMPLE_RATE / 1000)
    out = audio.copy()
    if d < len(audio):
        out[d:] += decay * audio[:-d]
    return out


def augmented_enroll_segments(segments, embed_fn, progress_cb=None):
    """
    Build a robust voiceprint from audio segments.
    For each segment: 1 clean embedding (3x weight) + 5 noise-augmented embeddings.
    Returns centroid embedding.
    """
    all_embs = []
    for i, seg in enumerate(segments):
        # Clean — triple weight
        try:
            emb = embed_fn(seg)
            all_embs.extend([emb, emb, emb])
        except Exception:
            continue

        # Augmented
        for _, aug_fn in NOISE_AUGMENTATIONS:
            try:
                noisy = np.clip(aug_fn(seg.copy()), -1.0, 1.0)
                all_embs.append(embed_fn(noisy))
            except Exception:
                continue

        if progress_cb and (i + 1) % 3 == 0:
            progress_cb(i + 1, len(segments), len(all_embs))

    if not all_embs:
        raise ValueError("No embeddings extracted")

    centroid = np.mean(all_embs, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    return centroid, len(all_embs)


def measure(fn, *args):
    """Run fn, return (result, ms, ram_mb, cpu_pct)."""
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)
    mem0 = proc.memory_info().rss
    t0 = time.perf_counter()
    result = fn(*args)
    ms = (time.perf_counter() - t0) * 1000
    mem1 = proc.memory_info().rss
    cpu = proc.cpu_percent(interval=0.1)
    ram = max(0, (mem1 - mem0)) / 1024 / 1024
    return result, ms, ram, cpu
