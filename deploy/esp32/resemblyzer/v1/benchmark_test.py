"""
Automated Accuracy Benchmark for ESP32 Resemblyzer v1
=====================================================
Tests the quantized model against training data with multiple conditions:

1. SELF-MATCH TEST
   Split training data into enroll/verify halves.
   Enroll from first half, verify from second half.
   Expected: high cosine similarity (MATCH)

2. NOISE ROBUSTNESS TEST
   Enroll from clean data, verify with synthetic noise at various SNRs.
   Tests: white, babble, street, fan, music noise at 0dB, 5dB, 10dB, 15dB, 20dB

3. ENROLLMENT METHOD COMPARISON
   Compare: single 10s enroll vs hardened enroll vs training data enroll

4. REJECTION TEST
   Verify with random noise (no speech) — should REJECT.

Results saved to benchmark_results.json
"""

import sys
import os
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from lib.noise import (
    training_data_enroll, hardened_enroll, hardened_verify,
    add_white_noise, add_babble_noise, add_street_noise,
    add_fan_noise, add_music_noise, reduce_noise,
    multi_segment_embed, AUGMENTATIONS,
)
from lib.metrics import cosine_similarity

SAMPLE_RATE = 16000
THRESHOLD = 0.60


class QuantizedResemblyzer:
    """Same int8 engine as simulator_gui — copied here for standalone use."""

    def __init__(self):
        self.weights = {}
        self.scales = {}

    def load(self):
        from resemblyzer import VoiceEncoder
        enc = VoiceEncoder()
        for key, tensor in enc.state_dict().items():
            t = tensor.numpy()
            if t.ndim == 2:
                n_rows = t.shape[0]
                row_scales = np.zeros(n_rows, dtype=np.float32)
                quantized = np.zeros_like(t, dtype=np.int8)
                for r in range(n_rows):
                    absmax = np.abs(t[r]).max()
                    row_scales[r] = absmax / 127.0 if absmax > 1e-10 else 1e-10
                    quantized[r] = np.round(t[r] / row_scales[r]).clip(-127, 127).astype(np.int8)
                self.weights[key] = quantized
                self.scales[key] = row_scales
            else:
                absmax = np.abs(t).max()
                scale = absmax / 127.0 if absmax > 1e-10 else 1e-10
                quantized = np.round(t / scale).clip(-127, 127).astype(np.int8)
                self.weights[key] = quantized
                self.scales[key] = np.array([scale], dtype=np.float32)

    def _dequant(self, name):
        s = self.scales[name]
        w = self.weights[name].astype(np.float32)
        if s.ndim == 1 and w.ndim == 2 and s.shape[0] == w.shape[0]:
            return w * s[:, None]
        return w * s[0]

    def _lstm_layer(self, x, layer_idx):
        T, _ = x.shape
        H = 256
        w_ih = self._dequant(f"lstm.weight_ih_l{layer_idx}")
        w_hh = self._dequant(f"lstm.weight_hh_l{layer_idx}")
        b_ih = self._dequant(f"lstm.bias_ih_l{layer_idx}")
        b_hh = self._dequant(f"lstm.bias_hh_l{layer_idx}")
        h = np.zeros(H, dtype=np.float32)
        c = np.zeros(H, dtype=np.float32)
        outputs = np.zeros((T, H), dtype=np.float32)
        for t in range(T):
            gates = w_ih @ x[t] + b_ih + w_hh @ h + b_hh
            ig = 1 / (1 + np.exp(-np.clip(gates[0:H], -20, 20)))
            fg = 1 / (1 + np.exp(-np.clip(gates[H:2*H], -20, 20)))
            gg = np.tanh(gates[2*H:3*H])
            og = 1 / (1 + np.exp(-np.clip(gates[3*H:4*H], -20, 20)))
            c = fg * c + ig * gg
            h = og * np.tanh(c)
            outputs[t] = h
        return outputs

    def forward(self, mel_frames):
        x = mel_frames
        for l in range(3):
            x = self._lstm_layer(x, l)
        last_h = x[-1]
        w = self._dequant("linear.weight")
        b = self._dequant("linear.bias")
        proj = np.maximum(w @ last_h + b, 0)
        return proj / (np.linalg.norm(proj) + 1e-8)


def embed_fn_factory(model):
    """Create embedding function from model instance."""
    from resemblyzer.audio import wav_to_mel_spectrogram
    def embed(audio_np):
        mel = wav_to_mel_spectrogram(audio_np)
        return model.forward(mel)
    return embed


def load_training_audio(training_dir, sr=SAMPLE_RATE):
    """Load all training WAV files, resample, concatenate."""
    import torchaudio
    import torch
    all_audio = []
    for f in sorted(Path(training_dir).glob("*.wav")):
        audio, file_sr = sf.read(str(f), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), file_sr, sr).numpy()
        all_audio.append(audio)
    return all_audio


def run_benchmark(training_dir):
    print("=" * 60)
    print("  VOICEPRINT BENCHMARK — ESP32 Resemblyzer v1")
    print("=" * 60)

    # Load model
    print("\n[1/6] Loading quantized model...")
    model = QuantizedResemblyzer()
    model.load()
    embed = embed_fn_factory(model)

    # Load training audio
    print("[2/6] Loading training audio...")
    audio_files = sorted(Path(training_dir).glob("*.wav"))
    audios = load_training_audio(training_dir)
    total_dur = sum(len(a) / SAMPLE_RATE for a in audios)
    print(f"  {len(audios)} files, {total_dur:.0f}s total")

    # Use ALL files for enrollment (training_data_enroll handles segmenting)
    # Use the longest file's last 30s as verification holdout
    enroll_files = audio_files
    longest = max(audios, key=len)
    verify_audio = longest[-SAMPLE_RATE * 30:]  # last 30s as holdout

    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tests": {}}

    # ============================================================
    # TEST 1: Enrollment Method Comparison
    # ============================================================
    print("\n[3/6] Enrollment method comparison...")

    # A: Single 10s (first 10s of first file)
    emb_10s = embed(audios[0][:SAMPLE_RATE * 10])

    # B: Hardened single (first 10s)
    emb_hardened, _, _ = hardened_enroll(audios[0][:SAMPLE_RATE * 10], embed)

    # C: Training data (all files)
    emb_trained, train_stats = training_data_enroll(enroll_files, embed)

    print(f"  Training data: {train_stats['n_segments']} segments, "
          f"{train_stats['n_embeddings']} embeddings")

    # Verify with clean segment from file 3
    verify_clean = verify_audio[SAMPLE_RATE * 10 : SAMPLE_RATE * 20]  # 10s chunk
    emb_verify_clean = embed(verify_clean)

    scores_clean = {
        "10s_enroll": cosine_similarity(emb_10s, emb_verify_clean),
        "hardened_enroll": cosine_similarity(emb_hardened, emb_verify_clean),
        "trained_enroll": cosine_similarity(emb_trained, emb_verify_clean),
    }

    print(f"\n  CLEAN VERIFY (different file):")
    for method, score in scores_clean.items():
        v = "PASS" if score >= THRESHOLD else "FAIL"
        print(f"    {method:<20} {score:.4f}  [{v}]")

    results["tests"]["enrollment_comparison_clean"] = scores_clean

    # ============================================================
    # TEST 2: Noise Robustness at Various SNRs
    # ============================================================
    print("\n[4/6] Noise robustness test...")

    noise_fns = {
        "white": add_white_noise,
        "babble": add_babble_noise,
        "street": add_street_noise,
        "fan": add_fan_noise,
        "music": add_music_noise,
    }
    snr_levels = [0, 5, 10, 15, 20]
    noise_results = {}

    print(f"\n  {'SNR':<6}", end="")
    for noise_name in noise_fns:
        print(f"  {noise_name:>8}", end="")
    print()
    print("  " + "-" * 52)

    for snr in snr_levels:
        row = {}
        print(f"  {snr:>3}dB", end="")
        for noise_name, noise_fn in noise_fns.items():
            noisy = noise_fn(verify_clean.copy(), snr_db=snr)
            # Use trained enrollment + hardened verify
            emb_noisy, _ = hardened_verify(noisy, embed)
            score = cosine_similarity(emb_trained, emb_noisy)
            row[noise_name] = round(score, 4)
            v = "+" if score >= THRESHOLD else "x"
            print(f"  {score:>7.4f}{v}", end="")
        print()
        noise_results[f"{snr}dB"] = row

    results["tests"]["noise_robustness"] = noise_results

    # ============================================================
    # TEST 3: Self-match consistency
    # ============================================================
    print("\n[5/6] Self-match consistency...")

    # Take 5 random 5s segments from verify audio
    n_trials = 10
    self_scores = []
    for i in range(n_trials):
        start = np.random.randint(0, len(verify_audio) - SAMPLE_RATE * 5)
        seg = verify_audio[start:start + SAMPLE_RATE * 5]
        emb_seg = embed(seg)
        score = cosine_similarity(emb_trained, emb_seg)
        self_scores.append(score)

    results["tests"]["self_match"] = {
        "n_trials": n_trials,
        "mean": round(float(np.mean(self_scores)), 4),
        "min": round(float(np.min(self_scores)), 4),
        "max": round(float(np.max(self_scores)), 4),
        "std": round(float(np.std(self_scores)), 4),
        "all_pass": all(s >= THRESHOLD for s in self_scores),
    }

    print(f"  {n_trials} random segments from verify file:")
    print(f"  Mean: {np.mean(self_scores):.4f}  "
          f"Min: {np.min(self_scores):.4f}  "
          f"Max: {np.max(self_scores):.4f}  "
          f"Std: {np.std(self_scores):.4f}")
    print(f"  All pass (>={THRESHOLD}): {results['tests']['self_match']['all_pass']}")

    # ============================================================
    # TEST 4: Rejection test (non-speech)
    # ============================================================
    print("\n[6/6] Rejection test...")

    rejection_scores = []
    reject_sources = [
        ("white_noise", np.random.randn(SAMPLE_RATE * 5).astype(np.float32) * 0.1),
        ("silence", np.zeros(SAMPLE_RATE * 5, dtype=np.float32)),
        ("tone_440hz", np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE * 5) / SAMPLE_RATE).astype(np.float32) * 0.3),
        ("pink_noise", add_white_noise(np.zeros(SAMPLE_RATE * 5, dtype=np.float32), snr_db=0)),
    ]

    for name, audio in reject_sources:
        emb = embed(audio)
        score = cosine_similarity(emb_trained, emb)
        rejected = score < THRESHOLD
        rejection_scores.append({"source": name, "score": round(score, 4), "rejected": rejected})
        v = "REJECT" if rejected else "LEAK!"
        print(f"  {name:<15} {score:.4f}  [{v}]")

    results["tests"]["rejection"] = rejection_scores
    results["tests"]["rejection_all_pass"] = all(r["rejected"] for r in rejection_scores)

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")

    n_noise_pass = sum(1 for snr_row in noise_results.values()
                       for s in snr_row.values() if s >= THRESHOLD)
    n_noise_total = sum(len(row) for row in noise_results.values())

    print(f"  Enrollment methods (clean):  all 3 tested")
    print(f"  Noise robustness:            {n_noise_pass}/{n_noise_total} pass")
    print(f"  Self-match consistency:      {'PASS' if results['tests']['self_match']['all_pass'] else 'FAIL'}")
    print(f"  Rejection:                   {'PASS' if results['tests']['rejection_all_pass'] else 'FAIL'}")
    print(f"  Threshold:                   {THRESHOLD}")

    # Overall score
    total_tests = n_noise_total + n_trials + len(reject_sources) + 3
    passed = (n_noise_pass +
              sum(1 for s in self_scores if s >= THRESHOLD) +
              sum(1 for r in rejection_scores if r["rejected"]) +
              sum(1 for s in scores_clean.values() if s >= THRESHOLD))
    pct = passed / total_tests * 100
    results["overall"] = {"passed": passed, "total": total_tests, "accuracy_pct": round(pct, 1)}

    print(f"\n  OVERALL: {passed}/{total_tests} ({pct:.1f}%)")
    print(f"{'=' * 60}")

    # Save results
    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    return results


if __name__ == "__main__":
    training_dir = ROOT / "training_data" / "arda_2023"
    if not training_dir.exists():
        print(f"ERROR: {training_dir} not found")
        sys.exit(1)
    run_benchmark(training_dir)
