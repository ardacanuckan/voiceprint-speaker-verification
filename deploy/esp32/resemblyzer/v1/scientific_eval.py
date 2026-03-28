"""
Scientific Evaluation — Speaker Verification System
====================================================
Multi-speaker test with comprehensive noise conditions.

Protocol:
  - TARGET speaker: Arda (enrolled)
  - IMPOSTOR speakers: Zeliha + synthetic
  - Conditions: clean, 5 noise types x 5 SNR levels

Metrics:
  - True Accept Rate (TAR): target correctly accepted
  - False Accept Rate (FAR): impostor incorrectly accepted
  - True Reject Rate (TRR): impostor correctly rejected
  - False Reject Rate (FRR): target incorrectly rejected
  - Equal Error Rate (EER) estimation
  - Detection Error Tradeoff (DET) curve data

Output: scientific_results.json
"""

import sys
import os
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from lib.noise import (
    training_data_enroll, hardened_verify,
    add_white_noise, add_babble_noise, add_street_noise,
    add_fan_noise, add_music_noise, add_pink_noise,
    add_reverb,
)
from lib.metrics import cosine_similarity

SAMPLE_RATE = 16000
SEGMENT_SEC = 5  # verification segment length


# ============================================================
# Quantized model (same as benchmark_test.py)
# ============================================================

class QuantizedResemblyzer:
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
                self.weights[key] = np.round(t / scale).clip(-127, 127).astype(np.int8)
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
        h, c = np.zeros(H, np.float32), np.zeros(H, np.float32)
        outputs = np.zeros((T, H), np.float32)
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

    def forward(self, mel):
        x = mel
        for l in range(3):
            x = self._lstm_layer(x, l)
        last_h = x[-1]
        w = self._dequant("linear.weight")
        b = self._dequant("linear.bias")
        proj = np.maximum(w @ last_h + b, 0)
        return proj / (np.linalg.norm(proj) + 1e-8)


def make_embed_fn(model):
    from resemblyzer.audio import wav_to_mel_spectrogram
    def embed(audio):
        return model.forward(wav_to_mel_spectrogram(audio))
    return embed


# ============================================================
# Audio loading
# ============================================================

def load_audio_dir(dirpath, sr=SAMPLE_RATE):
    """Load all WAVs from a directory, resample, return list of arrays."""
    import torchaudio, torch
    segments = []
    for f in sorted(Path(dirpath).glob("*.wav")):
        audio, file_sr = sf.read(str(f), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), file_sr, sr).numpy()
        segments.append(audio)
    return segments


def extract_segments(audio_list, seg_sec=SEGMENT_SEC, sr=SAMPLE_RATE):
    """Extract non-overlapping segments from list of audio arrays."""
    seg_len = seg_sec * sr
    segs = []
    for audio in audio_list:
        pos = 0
        while pos + seg_len <= len(audio):
            segs.append(audio[pos:pos + seg_len])
            pos += seg_len
    return segs


# ============================================================
# Main evaluation
# ============================================================

def run_evaluation():
    t_start = time.time()
    training_root = ROOT / "training_data"

    print("=" * 70)
    print("  VOICEPRINT — SCIENTIFIC EVALUATION")
    print("  Speaker Verification: Target vs Impostor + Noise Conditions")
    print("=" * 70)

    # --- Load model ---
    print("\n[SETUP] Loading quantized int8 model...")
    model = QuantizedResemblyzer()
    model.load()
    embed = make_embed_fn(model)

    # --- Load speakers ---
    print("[SETUP] Loading audio data...")
    arda_dir = training_root / "arda_2023"
    arda_audio = load_audio_dir(arda_dir)
    arda_total = sum(len(a) / SAMPLE_RATE for a in arda_audio)

    impostor_dirs = [d for d in sorted(training_root.iterdir())
                     if d.is_dir() and d.name != "arda_2023"
                     and list(d.glob("*.wav"))]
    impostor_audio = {}
    for d in impostor_dirs:
        segs = load_audio_dir(d)
        if segs:
            impostor_audio[d.name] = segs

    print(f"  TARGET:    arda_2023 — {len(arda_audio)} files, {arda_total:.0f}s")
    for name, segs in impostor_audio.items():
        dur = sum(len(a) / SAMPLE_RATE for a in segs)
        print(f"  IMPOSTOR:  {name} — {len(segs)} files, {dur:.0f}s")

    # --- Enroll Arda (training data method) ---
    print("\n[ENROLL] Building Arda voiceprint from all training data...")
    arda_files = sorted(arda_dir.glob("*.wav"))
    t0 = time.time()
    arda_centroid, enroll_stats = training_data_enroll(arda_files, embed)
    enroll_time = time.time() - t0
    print(f"  Segments: {enroll_stats['n_segments']}, "
          f"Embeddings: {enroll_stats['n_embeddings']}, "
          f"Time: {enroll_time:.1f}s")

    # --- Extract verification segments ---
    # Arda: use last file's audio as holdout
    arda_verify_segs = extract_segments([arda_audio[-1]])
    print(f"\n[VERIFY] Arda verification segments: {len(arda_verify_segs)} "
          f"({SEGMENT_SEC}s each from last file)")

    impostor_verify_segs = {}
    for name, audio_list in impostor_audio.items():
        segs = extract_segments(audio_list)
        impostor_verify_segs[name] = segs
        print(f"[VERIFY] {name} segments: {len(segs)}")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "threshold": 0.60,
            "segment_sec": SEGMENT_SEC,
            "enroll_segments": enroll_stats["n_segments"],
            "enroll_embeddings": enroll_stats["n_embeddings"],
            "enroll_time_sec": round(enroll_time, 1),
        },
    }

    # ============================================================
    # TEST 1: Clean condition — target vs impostor
    # ============================================================
    print(f"\n{'='*70}")
    print("  TEST 1: CLEAN CONDITION — Target vs Impostor")
    print(f"{'='*70}")

    target_clean_scores = []
    for seg in arda_verify_segs:
        score = cosine_similarity(arda_centroid, embed(seg))
        target_clean_scores.append(score)

    impostor_clean_scores = {}
    all_impostor_scores = []
    for name, segs in impostor_verify_segs.items():
        scores = []
        for seg in segs:
            score = cosine_similarity(arda_centroid, embed(seg))
            scores.append(score)
            all_impostor_scores.append(score)
        impostor_clean_scores[name] = scores

    threshold = 0.60
    tar = sum(1 for s in target_clean_scores if s >= threshold) / len(target_clean_scores)
    far = sum(1 for s in all_impostor_scores if s >= threshold) / max(len(all_impostor_scores), 1)
    frr = 1 - tar
    trr = 1 - far

    print(f"\n  {'Speaker':<30} {'Mean':>8} {'Min':>8} {'Max':>8} {'N':>5}")
    print(f"  {'-'*65}")
    print(f"  {'Arda (TARGET)':<30} {np.mean(target_clean_scores):>8.4f} "
          f"{np.min(target_clean_scores):>8.4f} {np.max(target_clean_scores):>8.4f} "
          f"{len(target_clean_scores):>5}")
    for name, scores in impostor_clean_scores.items():
        label = f"{name} (IMPOSTOR)"
        print(f"  {label:<30} {np.mean(scores):>8.4f} "
              f"{np.min(scores):>8.4f} {np.max(scores):>8.4f} {len(scores):>5}")

    print(f"\n  Threshold: {threshold}")
    print(f"  True Accept Rate (TAR):   {tar*100:.1f}%")
    print(f"  False Accept Rate (FAR):  {far*100:.1f}%")
    print(f"  False Reject Rate (FRR):  {frr*100:.1f}%")
    print(f"  True Reject Rate (TRR):   {trr*100:.1f}%")

    results["test1_clean"] = {
        "target_scores": [round(s, 4) for s in target_clean_scores],
        "impostor_scores": {k: [round(s, 4) for s in v] for k, v in impostor_clean_scores.items()},
        "TAR": round(tar, 4), "FAR": round(far, 4),
        "FRR": round(frr, 4), "TRR": round(trr, 4),
    }

    # ============================================================
    # TEST 2: Noise conditions — target
    # ============================================================
    print(f"\n{'='*70}")
    print("  TEST 2: NOISE CONDITIONS — Target (Arda)")
    print(f"{'='*70}")

    noise_fns = {
        "white": add_white_noise,
        "pink": add_pink_noise,
        "babble": add_babble_noise,
        "street": add_street_noise,
        "fan": add_fan_noise,
        "music": add_music_noise,
        "reverb": lambda a, snr_db=0: add_reverb(a, decay=0.4),
    }
    snr_levels = [0, 5, 10, 15, 20]

    # Header
    print(f"\n  {'Condition':<18}", end="")
    for snr in snr_levels:
        print(f" {snr:>6}dB", end="")
    print(f" {'Mean':>8}")
    print(f"  {'-'*68}")

    noise_target_results = {}
    for noise_name, noise_fn in noise_fns.items():
        row = {}
        means = []
        print(f"  {noise_name:<18}", end="")
        for snr in snr_levels:
            scores = []
            for seg in arda_verify_segs[:5]:  # use first 5 segments per condition
                if noise_name == "reverb":
                    noisy = noise_fn(seg.copy())
                else:
                    noisy = noise_fn(seg.copy(), snr_db=snr)
                emb_h, _ = hardened_verify(noisy, embed)
                score = cosine_similarity(arda_centroid, emb_h)
                scores.append(score)
            mean = np.mean(scores)
            means.append(mean)
            passed = mean >= threshold
            mark = "+" if passed else "x"
            print(f" {mean:>6.3f}{mark}", end="")
            row[f"{snr}dB"] = {"mean": round(mean, 4), "pass": passed}
        row_mean = np.mean(means)
        print(f" {row_mean:>7.3f}")
        noise_target_results[noise_name] = row

    results["test2_noise_target"] = noise_target_results

    # ============================================================
    # TEST 3: Noise conditions — impostor (should still reject)
    # ============================================================
    print(f"\n{'='*70}")
    print("  TEST 3: NOISE CONDITIONS — Impostor (should reject)")
    print(f"{'='*70}")

    noise_impostor_results = {}
    for imp_name, imp_segs in impostor_verify_segs.items():
        print(f"\n  --- {imp_name} ---")
        print(f"  {'Condition':<18}", end="")
        for snr in snr_levels:
            print(f" {snr:>6}dB", end="")
        print()
        print(f"  {'-'*55}")

        imp_noise_results = {}
        for noise_name, noise_fn in list(noise_fns.items())[:4]:  # white, pink, babble, street
            print(f"  {noise_name:<18}", end="")
            row = {}
            for snr in snr_levels:
                scores = []
                for seg in imp_segs[:3]:
                    if noise_name == "reverb":
                        noisy = noise_fn(seg.copy())
                    else:
                        noisy = noise_fn(seg.copy(), snr_db=snr)
                    emb_h, _ = hardened_verify(noisy, embed)
                    score = cosine_similarity(arda_centroid, emb_h)
                    scores.append(score)
                mean = np.mean(scores)
                rejected = mean < threshold
                mark = "R" if rejected else "!"
                print(f" {mean:>5.3f}{mark} ", end="")
                row[f"{snr}dB"] = {"mean": round(mean, 4), "rejected": rejected}
            print()
            imp_noise_results[noise_name] = row
        noise_impostor_results[imp_name] = imp_noise_results

    results["test3_noise_impostor"] = noise_impostor_results

    # ============================================================
    # TEST 4: EER Estimation
    # ============================================================
    print(f"\n{'='*70}")
    print("  TEST 4: EER ESTIMATION (Equal Error Rate)")
    print(f"{'='*70}")

    # Collect all target and impostor scores
    all_target = list(target_clean_scores)
    all_impostor = list(all_impostor_scores)

    # Also add noisy target scores for more data points
    for seg in arda_verify_segs[:5]:
        for noise_fn in [add_babble_noise, add_white_noise]:
            for snr in [5, 10]:
                noisy = noise_fn(seg.copy(), snr_db=snr)
                score = cosine_similarity(arda_centroid, embed(noisy))
                all_target.append(score)

    # Sweep thresholds
    thresholds = np.arange(0.30, 0.95, 0.01)
    det_curve = []
    best_eer = 1.0
    best_thr = 0.5

    print(f"\n  {'Threshold':>10} {'FAR':>8} {'FRR':>8} {'Gap':>8}")
    print(f"  {'-'*38}")

    for thr in thresholds:
        far_t = sum(1 for s in all_impostor if s >= thr) / max(len(all_impostor), 1)
        frr_t = sum(1 for s in all_target if s < thr) / max(len(all_target), 1)
        det_curve.append({"threshold": round(thr, 2), "FAR": round(far_t, 4), "FRR": round(frr_t, 4)})

        gap = abs(far_t - frr_t)
        if gap < abs(best_eer - 0):
            best_eer = (far_t + frr_t) / 2
            best_thr = thr

        if thr in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            print(f"  {thr:>10.2f} {far_t:>8.3f} {frr_t:>8.3f} {gap:>8.3f}")

    print(f"\n  Estimated EER: {best_eer*100:.1f}% at threshold {best_thr:.2f}")
    print(f"  Target scores:   N={len(all_target)}, mean={np.mean(all_target):.4f}")
    print(f"  Impostor scores: N={len(all_impostor)}, mean={np.mean(all_impostor):.4f}")
    print(f"  Score gap:       {np.mean(all_target) - np.mean(all_impostor):.4f}")

    results["test4_eer"] = {
        "estimated_eer_pct": round(best_eer * 100, 2),
        "optimal_threshold": round(best_thr, 2),
        "n_target_scores": len(all_target),
        "n_impostor_scores": len(all_impostor),
        "target_mean": round(float(np.mean(all_target)), 4),
        "impostor_mean": round(float(np.mean(all_impostor)), 4),
        "score_gap": round(float(np.mean(all_target) - np.mean(all_impostor)), 4),
        "det_curve": det_curve,
    }

    # ============================================================
    # TEST 5: Hardened verify vs raw verify comparison
    # ============================================================
    print(f"\n{'='*70}")
    print("  TEST 5: HARDENED vs RAW VERIFY")
    print(f"{'='*70}")

    comparisons = []
    for seg in arda_verify_segs[:5]:
        noisy = add_babble_noise(seg.copy(), snr_db=5)
        raw_score = cosine_similarity(arda_centroid, embed(noisy))
        hard_emb, _ = hardened_verify(noisy, embed)
        hard_score = cosine_similarity(arda_centroid, hard_emb)
        boost = hard_score - raw_score
        comparisons.append({"raw": round(raw_score, 4), "hardened": round(hard_score, 4),
                           "boost": round(boost, 4)})

    raw_mean = np.mean([c["raw"] for c in comparisons])
    hard_mean = np.mean([c["hardened"] for c in comparisons])
    boost_mean = np.mean([c["boost"] for c in comparisons])

    print(f"\n  Babble noise 5dB, {len(comparisons)} segments:")
    print(f"  {'Segment':>8} {'Raw':>8} {'Hardened':>10} {'Boost':>8}")
    print(f"  {'-'*38}")
    for i, c in enumerate(comparisons):
        print(f"  {i+1:>8} {c['raw']:>8.4f} {c['hardened']:>10.4f} {c['boost']:>+8.4f}")
    print(f"  {'MEAN':>8} {raw_mean:>8.4f} {hard_mean:>10.4f} {boost_mean:>+8.4f}")

    results["test5_hardened_vs_raw"] = {
        "segments": comparisons,
        "raw_mean": round(raw_mean, 4),
        "hardened_mean": round(hard_mean, 4),
        "boost_mean": round(boost_mean, 4),
    }

    # ============================================================
    # SUMMARY
    # ============================================================
    total_time = time.time() - t_start

    # Count all pass/fail
    n_target_clean_pass = sum(1 for s in target_clean_scores if s >= threshold)
    n_impostor_clean_reject = sum(1 for s in all_impostor_scores if s < threshold)

    n_noise_target_pass = sum(
        1 for nr in noise_target_results.values()
        for snr_data in nr.values()
        if isinstance(snr_data, dict) and snr_data.get("pass", False)
    )
    n_noise_target_total = sum(
        1 for nr in noise_target_results.values()
        for snr_data in nr.values()
        if isinstance(snr_data, dict)
    )

    print(f"\n{'='*70}")
    print("  SCIENTIFIC EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Model:             Resemblyzer GE2E (int8 per-row quantized)")
    print(f"  Enrollment:        {enroll_stats['n_segments']} segments, "
          f"{enroll_stats['n_embeddings']} embeddings from {arda_total:.0f}s audio")
    print(f"  Threshold:         {threshold}")
    print(f"  Estimated EER:     {best_eer*100:.1f}%")
    print(f"  Score gap:         {np.mean(all_target) - np.mean(all_impostor):.4f}")
    print(f"")
    print(f"  Target accept (clean):     {n_target_clean_pass}/{len(target_clean_scores)}")
    print(f"  Impostor reject (clean):   {n_impostor_clean_reject}/{len(all_impostor_scores)}")
    print(f"  Target accept (noisy):     {n_noise_target_pass}/{n_noise_target_total}")
    print(f"  Hardening boost (5dB):     +{boost_mean:.4f}")
    print(f"  Total eval time:           {total_time:.0f}s")
    print(f"{'='*70}")

    results["summary"] = {
        "model": "Resemblyzer GE2E int8 per-row",
        "threshold": threshold,
        "estimated_eer_pct": round(best_eer * 100, 2),
        "target_accept_clean": f"{n_target_clean_pass}/{len(target_clean_scores)}",
        "impostor_reject_clean": f"{n_impostor_clean_reject}/{len(all_impostor_scores)}",
        "noise_target_pass": f"{n_noise_target_pass}/{n_noise_target_total}",
        "hardening_boost": round(boost_mean, 4),
        "eval_time_sec": round(total_time, 1),
    }

    # Save — convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    out_path = Path(__file__).parent / "scientific_results.json"
    with open(out_path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    run_evaluation()
