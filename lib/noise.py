"""
Noise Hardening Module for Speaker Verification
================================================
Makes enrollment robust to noisy environments by:

1. AUGMENTED ENROLLMENT
   Clean voice → add 8 types of synthetic noise → extract embedding from each
   → average all embeddings into one "hardened" centroid.
   Result: enrollment embedding that tolerates noise.

2. NOISE REDUCTION (verification side)
   Incoming noisy audio → spectral gating → cleaned audio → embedding.
   Uses noisereduce for real-time spectral subtraction.

3. MULTI-SEGMENT AVERAGING (verification side)
   Split audio into overlapping segments → embedding per segment → average.
   Removes transient noise spikes that affect single-pass embedding.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 16000


# ============================================================
# 1. NOISE AUGMENTATION — synthetic noise generation
# ============================================================

def add_white_noise(audio, snr_db=10):
    """Add white Gaussian noise at given SNR."""
    rms_signal = np.sqrt(np.mean(audio ** 2))
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.randn(len(audio)).astype(np.float32) * rms_noise
    return audio + noise


def add_pink_noise(audio, snr_db=10):
    """Add pink (1/f) noise."""
    n = len(audio)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1  # avoid div by zero
    pink_spectrum = 1.0 / np.sqrt(freqs)
    pink_spectrum[0] = 0
    phases = np.random.uniform(0, 2 * np.pi, len(freqs))
    pink_fft = pink_spectrum * np.exp(1j * phases)
    pink = np.fft.irfft(pink_fft, n=n).astype(np.float32)
    pink = pink / (np.std(pink) + 1e-8)
    rms_signal = np.sqrt(np.mean(audio ** 2))
    rms_target = rms_signal / (10 ** (snr_db / 20))
    return audio + pink * rms_target


def add_babble_noise(audio, snr_db=10, n_voices=5):
    """Simulate babble noise (multiple random speech-like signals)."""
    babble = np.zeros_like(audio)
    for _ in range(n_voices):
        # Random modulated noise simulates speech-like babble
        t = np.arange(len(audio)) / SAMPLE_RATE
        mod_freq = np.random.uniform(2, 6)  # speech rate modulation
        envelope = 0.5 * (1 + np.sin(2 * np.pi * mod_freq * t + np.random.uniform(0, 2 * np.pi)))
        voice = np.random.randn(len(audio)).astype(np.float32) * envelope
        babble += voice
    babble = babble / (np.std(babble) + 1e-8)
    rms_signal = np.sqrt(np.mean(audio ** 2))
    rms_target = rms_signal / (10 ** (snr_db / 20))
    return audio + babble * rms_target


def add_reverb(audio, decay=0.3, delay_ms=30):
    """Simple reverb simulation with single reflection."""
    delay_samples = int(delay_ms * SAMPLE_RATE / 1000)
    reverbed = audio.copy()
    if delay_samples < len(audio):
        reverbed[delay_samples:] += decay * audio[:-delay_samples]
    return reverbed


def add_street_noise(audio, snr_db=10):
    """Low-frequency rumble + random impulses (cars, steps)."""
    n = len(audio)
    t = np.arange(n) / SAMPLE_RATE
    # Low freq rumble
    rumble = np.sin(2 * np.pi * 50 * t) * 0.5 + np.sin(2 * np.pi * 120 * t) * 0.3
    rumble = rumble.astype(np.float32)
    # Random impulses
    impulses = np.zeros(n, dtype=np.float32)
    n_impulses = np.random.randint(5, 20)
    for _ in range(n_impulses):
        pos = np.random.randint(0, n)
        width = np.random.randint(100, 1000)
        end = min(pos + width, n)
        impulses[pos:end] = np.random.randn(end - pos) * np.random.uniform(0.5, 2.0)
    noise = rumble + impulses
    noise = noise / (np.std(noise) + 1e-8)
    rms_signal = np.sqrt(np.mean(audio ** 2))
    rms_target = rms_signal / (10 ** (snr_db / 20))
    return audio + noise * rms_target


def add_fan_noise(audio, snr_db=15):
    """Constant fan/AC hum — narrow-band low frequency."""
    n = len(audio)
    t = np.arange(n) / SAMPLE_RATE
    # Fan harmonics at 60Hz, 120Hz, 180Hz
    fan = (np.sin(2 * np.pi * 60 * t) * 1.0 +
           np.sin(2 * np.pi * 120 * t) * 0.6 +
           np.sin(2 * np.pi * 180 * t) * 0.3 +
           np.random.randn(n) * 0.1).astype(np.float32)
    fan = fan / (np.std(fan) + 1e-8)
    rms_signal = np.sqrt(np.mean(audio ** 2))
    rms_target = rms_signal / (10 ** (snr_db / 20))
    return audio + fan * rms_target


def add_music_noise(audio, snr_db=10):
    """Random tonal noise simulating background music."""
    n = len(audio)
    t = np.arange(n) / SAMPLE_RATE
    music = np.zeros(n, dtype=np.float32)
    n_tones = np.random.randint(3, 8)
    for _ in range(n_tones):
        freq = np.random.choice([220, 330, 440, 550, 660, 880, 1100])
        amp = np.random.uniform(0.2, 1.0)
        phase = np.random.uniform(0, 2 * np.pi)
        music += amp * np.sin(2 * np.pi * freq * t + phase)
    music = music / (np.std(music) + 1e-8)
    rms_signal = np.sqrt(np.mean(audio ** 2))
    rms_target = rms_signal / (10 ** (snr_db / 20))
    return audio + music.astype(np.float32) * rms_target


# All augmentation functions with their SNR ranges
AUGMENTATIONS = [
    ("white_5dB",   lambda a: add_white_noise(a, snr_db=5)),
    ("white_15dB",  lambda a: add_white_noise(a, snr_db=15)),
    ("pink_10dB",   lambda a: add_pink_noise(a, snr_db=10)),
    ("babble_10dB", lambda a: add_babble_noise(a, snr_db=10)),
    ("babble_5dB",  lambda a: add_babble_noise(a, snr_db=5)),
    ("reverb",      lambda a: add_reverb(a, decay=0.4, delay_ms=40)),
    ("street_10dB", lambda a: add_street_noise(a, snr_db=10)),
    ("fan_15dB",    lambda a: add_fan_noise(a, snr_db=15)),
    ("music_10dB",  lambda a: add_music_noise(a, snr_db=10)),
]


def augmented_enroll(audio, embed_fn, n_augmentations=None):
    """
    Create noise-hardened enrollment embedding.

    Args:
        audio: float32 numpy array, clean enrollment audio
        embed_fn: function(audio_array) -> embedding_array
        n_augmentations: how many augmentations to use (None = all)

    Returns:
        hardened_embedding: averaged embedding from clean + augmented versions
        individual_embeddings: list of (name, embedding) pairs
    """
    augs = AUGMENTATIONS if n_augmentations is None else AUGMENTATIONS[:n_augmentations]

    embeddings = []
    details = []

    # Clean embedding (weighted higher)
    emb_clean = embed_fn(audio)
    embeddings.append(emb_clean)
    embeddings.append(emb_clean)  # double weight for clean
    details.append(("clean", emb_clean))

    # Augmented embeddings
    for name, aug_fn in augs:
        try:
            noisy_audio = aug_fn(audio.copy())
            # Clip to prevent overflow
            noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
            emb = embed_fn(noisy_audio)
            embeddings.append(emb)
            details.append((name, emb))
        except Exception:
            pass

    # Average all embeddings → hardened centroid
    centroid = np.mean(embeddings, axis=0)
    # Re-normalize
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

    return centroid, details


# ============================================================
# 2. NOISE REDUCTION — spectral gating
# ============================================================

def reduce_noise(audio, sr=SAMPLE_RATE):
    """Apply spectral gating noise reduction."""
    import noisereduce as nr
    # Stationary noise reduction — works well for constant background noise
    cleaned = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=True,
        prop_decrease=0.75,  # how aggressively to remove noise
        n_fft=512,
        hop_length=128,
    )
    return cleaned.astype(np.float32)


def reduce_noise_adaptive(audio, sr=SAMPLE_RATE):
    """Adaptive noise reduction — better for non-stationary noise."""
    import noisereduce as nr
    cleaned = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=False,
        prop_decrease=0.65,
        n_fft=512,
        hop_length=128,
    )
    return cleaned.astype(np.float32)


# ============================================================
# 3. MULTI-SEGMENT AVERAGING
# ============================================================

def multi_segment_embed(audio, embed_fn, segment_sec=2.0, hop_sec=1.0, sr=SAMPLE_RATE):
    """
    Split audio into overlapping segments, embed each, average.
    Removes effect of transient noise spikes.

    Args:
        audio: float32 audio array
        embed_fn: function(audio_array) -> embedding
        segment_sec: segment duration in seconds
        hop_sec: hop between segments in seconds

    Returns:
        averaged_embedding, n_segments
    """
    seg_len = int(segment_sec * sr)
    hop_len = int(hop_sec * sr)

    if len(audio) < sr:  # less than 1 second — too short
        return embed_fn(audio), 1

    if len(audio) < seg_len:
        return embed_fn(audio), 1

    embeddings = []
    pos = 0
    while pos + seg_len <= len(audio):
        segment = audio[pos:pos + seg_len]
        emb = embed_fn(segment)
        embeddings.append(emb)
        pos += hop_len

    if not embeddings:
        return embed_fn(audio), 1

    avg = np.mean(embeddings, axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-8)
    return avg, len(embeddings)


# ============================================================
# FULL HARDENED PIPELINE
# ============================================================

def hardened_enroll(audio, embed_fn):
    """
    Full noise-hardened enrollment pipeline.
    1. Noise reduce the audio (in case enrollment itself has some noise)
    2. Create augmented enrollment with 9 noise types
    3. Return hardened centroid embedding

    Returns: (centroid_embedding, details_list, stats_dict)
    """
    # Light noise reduction on enrollment audio
    cleaned = reduce_noise(audio)

    # Augmented enrollment
    centroid, details = augmented_enroll(cleaned, embed_fn)

    stats = {
        "n_augmentations": len(details) - 1,  # minus clean
        "clean_vs_centroid": float(np.dot(details[0][1], centroid)),
    }

    return centroid, details, stats


def training_data_enroll(audio_files, embed_fn, segment_sec=3.0, hop_sec=1.5,
                         sr=SAMPLE_RATE):
    """
    Build a super-robust voiceprint from multiple training audio files.

    Strategy:
      1. Load all audio files, resample to 16kHz
      2. Split each into overlapping segments (3s window, 1.5s hop)
      3. For each segment: extract clean embedding + 9 noise-augmented embeddings
      4. Average ALL embeddings into one centroid
      5. L2 normalize

    More data = better centroid = more noise-tolerant voiceprint.

    Args:
        audio_files: list of Path objects to WAV files
        embed_fn: function(audio_array) -> embedding
        segment_sec: segment length in seconds
        hop_sec: hop between segments

    Returns: (centroid, stats_dict)
    """
    import soundfile as sf
    import torchaudio
    import torch

    if not audio_files:
        raise ValueError("No audio files provided")

    seg_len = int(segment_sec * sr)
    hop_len = int(hop_sec * sr)

    all_embeddings = []
    total_segments = 0
    total_duration = 0.0

    for fpath in audio_files:
        audio, file_sr = sf.read(str(fpath), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) < sr:  # less than 1 second — skip
            print(f"  SKIP {fpath.name}: too short ({len(audio)/file_sr:.1f}s)")
            continue

        if file_sr != sr:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), file_sr, sr).numpy()

        total_duration += len(audio) / sr
        cleaned = reduce_noise(audio, sr)

        # Segment and embed
        pos = 0
        file_segs = 0
        while pos + seg_len <= len(cleaned):
            segment = cleaned[pos:pos + seg_len]
            pos += hop_len
            total_segments += 1
            file_segs += 1

            # Clean embedding (triple weight)
            try:
                emb_clean = embed_fn(segment)
                all_embeddings.extend([emb_clean, emb_clean, emb_clean])
            except (RuntimeError, ValueError):
                continue

            # Noise-augmented embeddings
            for _, aug_fn in AUGMENTATIONS:
                try:
                    noisy = np.clip(aug_fn(segment.copy()), -1.0, 1.0)
                    all_embeddings.append(embed_fn(noisy))
                except (RuntimeError, ValueError):
                    continue

        print(f"  {fpath.name}: {file_segs} segments")

    if not all_embeddings:
        raise ValueError("No embeddings extracted — check audio files")

    # Average into centroid
    centroid = np.mean(all_embeddings, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

    stats = {
        "n_files": len(audio_files),
        "total_duration_sec": total_duration,
        "n_segments": total_segments,
        "n_embeddings": len(all_embeddings),
        "n_augmentations": len(AUGMENTATIONS),
    }

    return centroid, stats


def hardened_verify(audio, embed_fn):
    """
    Full noise-hardened verification pipeline.
    1. Noise reduce the audio
    2. Multi-segment embedding averaging
    3. Return robust embedding

    Returns: (embedding, stats_dict)
    """
    # Aggressive noise reduction
    cleaned = reduce_noise_adaptive(audio)

    # Multi-segment averaging
    embedding, n_segments = multi_segment_embed(cleaned, embed_fn)

    stats = {
        "n_segments": n_segments,
        "noise_reduction": "adaptive",
    }

    return embedding, stats
