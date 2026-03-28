"""
Impostor Rejection Test — "Will the model mistake someone else for Arda?"
==========================================================================
This is the critical real-world test:

  The device is always listening in an environment where multiple people talk.
  When ARDA speaks → ACCEPT.
  When ANYONE ELSE speaks → REJECT.

Protocol:
  1. Enroll Arda from all training data (197 segments, 2364 embeddings)
  2. For each speaker (Arda, Zeliha, ...) extract ALL possible 3s and 5s segments
  3. Score every single segment against Arda's voiceprint
  4. Sweep thresholds from 0.40 to 0.90 to find optimal operating point
  5. Test under clean + 5 noise conditions at 3 SNR levels
  6. Report: score distributions, FAR/FRR curves, confusion matrix, optimal threshold

This answers: "In a room where people are talking, how often will the device
              falsely trigger on someone who is NOT Arda?"
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
    training_data_enroll, hardened_verify,
    add_white_noise, add_babble_noise, add_street_noise,
    add_fan_noise, add_music_noise,
)
from lib.metrics import cosine_similarity

SAMPLE_RATE = 16000


# ============================================================
# Quantized model
# ============================================================

class QuantizedResemblyzer:
    def __init__(self):
        self.weights, self.scales = {}, {}

    def load(self):
        from resemblyzer import VoiceEncoder
        for key, tensor in VoiceEncoder().state_dict().items():
            t = tensor.numpy()
            if t.ndim == 2:
                s = np.array([max(np.abs(t[r]).max(), 1e-10) / 127.0 for r in range(t.shape[0])], dtype=np.float32)
                self.weights[key] = np.round(t / s[:, None]).clip(-127, 127).astype(np.int8)
                self.scales[key] = s
            else:
                s = max(np.abs(t).max(), 1e-10) / 127.0
                self.weights[key] = np.round(t / s).clip(-127, 127).astype(np.int8)
                self.scales[key] = np.array([s], dtype=np.float32)

    def _dq(self, name):
        s, w = self.scales[name], self.weights[name].astype(np.float32)
        return w * s[:, None] if (s.ndim == 1 and w.ndim == 2 and s.shape[0] == w.shape[0]) else w * s[0]

    def forward(self, mel):
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


def make_embed(model):
    from resemblyzer.audio import wav_to_mel_spectrogram
    return lambda audio: model.forward(wav_to_mel_spectrogram(audio))


# ============================================================
# Audio loading
# ============================================================

def load_speaker(dirpath, sr=SAMPLE_RATE):
    """Load all WAVs, resample, return concatenated + file list."""
    import torchaudio, torch
    chunks = []
    for f in sorted(Path(dirpath).glob("*.wav")):
        a, fsr = sf.read(str(f), dtype="float32")
        if a.ndim > 1: a = a.mean(axis=1)
        if fsr != sr:
            a = torchaudio.functional.resample(torch.from_numpy(a), fsr, sr).numpy()
        chunks.append(a)
    return chunks


def make_segments(audio_chunks, seg_sec, hop_sec, sr=SAMPLE_RATE):
    """Extract overlapping segments from list of audio arrays."""
    seg_len, hop_len = int(seg_sec * sr), int(hop_sec * sr)
    segs = []
    for audio in audio_chunks:
        pos = 0
        while pos + seg_len <= len(audio):
            segs.append(audio[pos:pos + seg_len])
            pos += hop_len
    return segs


# ============================================================
# Main
# ============================================================

def run():
    t0 = time.time()
    training_root = ROOT / "training_data"

    print("=" * 70)
    print("  IMPOSTOR REJECTION TEST")
    print("  'Will the model mistake someone else for Arda?'")
    print("=" * 70)

    # --- Setup ---
    print("\n[1] Loading model...")
    model = QuantizedResemblyzer()
    model.load()
    embed = make_embed(model)

    # --- Load speakers ---
    print("[2] Loading speakers...")
    speakers = {}
    for d in sorted(training_root.iterdir()):
        if not d.is_dir() or not list(d.glob("*.wav")):
            continue
        chunks = load_speaker(d)
        dur = sum(len(c) / SAMPLE_RATE for c in chunks)
        speakers[d.name] = {"chunks": chunks, "duration": dur}
        role = "TARGET" if "arda" in d.name.lower() else "IMPOSTOR"
        print(f"  [{role}] {d.name}: {len(chunks)} files, {dur:.0f}s")

    target_name = [k for k in speakers if "arda" in k.lower()][0]
    impostor_names = [k for k in speakers if k != target_name]

    # --- Enroll Arda ---
    print(f"\n[3] Enrolling {target_name}...")
    arda_files = sorted((training_root / target_name).glob("*.wav"))
    arda_centroid, enroll_stats = training_data_enroll(arda_files, embed)
    print(f"  {enroll_stats['n_segments']} segments, {enroll_stats['n_embeddings']} embeddings")

    # --- Extract test segments ---
    # 3s segments with 1.5s hop for maximum coverage
    print(f"\n[4] Extracting test segments...")
    target_segs = make_segments(speakers[target_name]["chunks"], seg_sec=3.0, hop_sec=1.5)
    impostor_segs = {}
    for name in impostor_names:
        segs = make_segments(speakers[name]["chunks"], seg_sec=3.0, hop_sec=1.5)
        impostor_segs[name] = segs

    print(f"  {target_name}: {len(target_segs)} segments")
    for name, segs in impostor_segs.items():
        print(f"  {name}: {len(segs)} segments")

    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ============================================================
    # CLEAN CONDITION — every segment scored
    # ============================================================
    print(f"\n{'='*70}")
    print("  CLEAN CONDITION — Score every segment")
    print(f"{'='*70}")

    target_scores = [cosine_similarity(arda_centroid, embed(seg)) for seg in target_segs]
    impostor_all_scores = []
    impostor_per_speaker = {}

    for name, segs in impostor_segs.items():
        scores = [cosine_similarity(arda_centroid, embed(seg)) for seg in segs]
        impostor_per_speaker[name] = scores
        impostor_all_scores.extend(scores)

    # Print distributions
    print(f"\n  {'Speaker':<35} {'N':>4} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
    print(f"  {'-'*70}")
    print(f"  {target_name + ' (TARGET)':<35} {len(target_scores):>4} "
          f"{np.mean(target_scores):>7.4f} {np.std(target_scores):>7.4f} "
          f"{np.min(target_scores):>7.4f} {np.max(target_scores):>7.4f}")
    for name, scores in impostor_per_speaker.items():
        print(f"  {name + ' (IMPOSTOR)':<35} {len(scores):>4} "
              f"{np.mean(scores):>7.4f} {np.std(scores):>7.4f} "
              f"{np.min(scores):>7.4f} {np.max(scores):>7.4f}")

    results["clean_scores"] = {
        "target": {"name": target_name, "scores": [round(s, 4) for s in target_scores]},
        "impostors": {name: [round(s, 4) for s in scores]
                      for name, scores in impostor_per_speaker.items()},
    }

    # ============================================================
    # THRESHOLD SWEEP — find optimal operating point
    # ============================================================
    print(f"\n{'='*70}")
    print("  THRESHOLD SWEEP — FAR / FRR / Accuracy")
    print(f"{'='*70}")

    thresholds = np.arange(0.40, 0.90, 0.005)
    sweep = []

    print(f"\n  {'Thr':>6} {'TAR':>7} {'FAR':>7} {'FRR':>7} {'TRR':>7} {'Acc':>7} {'F1':>7}")
    print(f"  {'-'*50}")

    best_f1 = 0
    best_thr = 0.60
    eer_thr = 0.60
    eer_val = 1.0

    for thr in thresholds:
        tp = sum(1 for s in target_scores if s >= thr)
        fn = sum(1 for s in target_scores if s < thr)
        fp = sum(1 for s in impostor_all_scores if s >= thr)
        tn = sum(1 for s in impostor_all_scores if s < thr)

        tar = tp / max(tp + fn, 1)
        far = fp / max(fp + tn, 1)
        frr = fn / max(tp + fn, 1)
        trr = tn / max(fp + tn, 1)
        acc = (tp + tn) / max(tp + fn + fp + tn, 1)

        precision = tp / max(tp + fp, 1)
        recall = tar
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        sweep.append({
            "threshold": round(float(thr), 3),
            "TAR": round(tar, 4), "FAR": round(far, 4),
            "FRR": round(frr, 4), "TRR": round(trr, 4),
            "accuracy": round(acc, 4), "F1": round(f1, 4),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

        if abs(far - frr) < abs(eer_val):
            eer_val = abs(far - frr)
            eer_thr = thr
            eer_rate = (far + frr) / 2

        if thr in [0.50, 0.55, 0.60, 0.625, 0.65, 0.675, 0.70, 0.75, 0.80]:
            print(f"  {thr:>6.3f} {tar:>7.1%} {far:>7.1%} {frr:>7.1%} {trr:>7.1%} {acc:>7.1%} {f1:>7.3f}")

    print(f"\n  Optimal threshold (best F1):  {best_thr:.3f}  (F1={best_f1:.3f})")
    print(f"  EER threshold:                {eer_thr:.3f}  (EER≈{eer_rate*100:.1f}%)")

    results["threshold_sweep"] = sweep
    results["optimal_threshold"] = round(float(best_thr), 3)
    results["optimal_f1"] = round(float(best_f1), 4)
    results["eer_threshold"] = round(float(eer_thr), 3)
    results["eer_pct"] = round(float(eer_rate * 100), 2)

    # ============================================================
    # NOISE CONDITIONS at optimal threshold
    # ============================================================
    OPT_THR = best_thr

    print(f"\n{'='*70}")
    print(f"  NOISE TEST @ OPTIMAL THRESHOLD {OPT_THR:.3f}")
    print(f"{'='*70}")

    noise_fns = {
        "white": add_white_noise,
        "babble": add_babble_noise,
        "street": add_street_noise,
        "fan": add_fan_noise,
        "music": add_music_noise,
    }
    snr_levels = [0, 5, 10, 15, 20]
    n_test_segs = min(15, len(target_segs))
    n_imp_segs = min(10, len(impostor_all_scores))

    noise_results = {}

    # Target under noise
    print(f"\n  TARGET ({target_name}) — should ACCEPT:")
    print(f"  {'Noise':<10}", end="")
    for snr in snr_levels:
        print(f" {snr:>5}dB", end="")
    print()
    print(f"  {'-'*45}")

    for noise_name, noise_fn in noise_fns.items():
        print(f"  {noise_name:<10}", end="")
        row = {}
        for snr in snr_levels:
            accepted = 0
            for seg in target_segs[:n_test_segs]:
                noisy = noise_fn(seg.copy(), snr_db=snr)
                emb_h, _ = hardened_verify(noisy, embed)
                score = cosine_similarity(arda_centroid, emb_h)
                if score >= OPT_THR:
                    accepted += 1
            rate = accepted / n_test_segs
            row[f"{snr}dB"] = round(rate, 3)
            print(f" {rate:>5.0%} ", end="")
        print()
        noise_results[f"target_{noise_name}"] = row

    # Impostor under noise
    for imp_name, imp_segs_list in impostor_segs.items():
        print(f"\n  IMPOSTOR ({imp_name}) — should REJECT:")
        print(f"  {'Noise':<10}", end="")
        for snr in snr_levels:
            print(f" {snr:>5}dB", end="")
        print()
        print(f"  {'-'*45}")

        n_imp = min(10, len(imp_segs_list))
        for noise_name, noise_fn in noise_fns.items():
            print(f"  {noise_name:<10}", end="")
            row = {}
            for snr in snr_levels:
                rejected = 0
                for seg in imp_segs_list[:n_imp]:
                    noisy = noise_fn(seg.copy(), snr_db=snr)
                    emb_h, _ = hardened_verify(noisy, embed)
                    score = cosine_similarity(arda_centroid, emb_h)
                    if score < OPT_THR:
                        rejected += 1
                rate = rejected / n_imp
                row[f"{snr}dB"] = round(rate, 3)
                print(f" {rate:>5.0%} ", end="")
            print()
            noise_results[f"impostor_{imp_name}_{noise_name}"] = row

    results["noise_at_optimal"] = noise_results

    # ============================================================
    # FINAL REPORT
    # ============================================================
    total_time = time.time() - t0

    # At optimal threshold, clean performance
    tp = sum(1 for s in target_scores if s >= OPT_THR)
    fn = len(target_scores) - tp
    fp = sum(1 for s in impostor_all_scores if s >= OPT_THR)
    tn = len(impostor_all_scores) - fp

    print(f"\n{'='*70}")
    print("  FINAL REPORT")
    print(f"{'='*70}")
    print(f"  Model:                Resemblyzer GE2E (int8 per-row)")
    print(f"  Enrollment:           {enroll_stats['n_embeddings']} embeddings from {enroll_stats['total_duration_sec']:.0f}s")
    print(f"  Target segments:      {len(target_scores)}")
    print(f"  Impostor segments:    {len(impostor_all_scores)}")
    print(f"")
    print(f"  Optimal threshold:    {OPT_THR:.3f} (F1={best_f1:.3f})")
    print(f"  EER:                  {eer_rate*100:.1f}% @ {eer_thr:.3f}")
    print(f"")
    print(f"  CONFUSION MATRIX (clean, threshold={OPT_THR:.3f}):")
    print(f"                        Predicted ARDA    Predicted OTHER")
    print(f"    Actual ARDA         TP={tp:<5}            FN={fn:<5}")
    print(f"    Actual OTHER        FP={fp:<5}            TN={tn:<5}")
    print(f"")
    print(f"  True Accept Rate:     {tp/(tp+fn)*100:.1f}%  ({tp}/{tp+fn})")
    print(f"  False Accept Rate:    {fp/(fp+tn)*100:.1f}%  ({fp}/{fp+tn})")
    print(f"  Precision:            {tp/max(tp+fp,1)*100:.1f}%")
    print(f"  Accuracy:             {(tp+tn)/(tp+fn+fp+tn)*100:.1f}%")
    print(f"  F1 Score:             {best_f1:.3f}")
    print(f"")
    print(f"  Score gap (target-impostor): {np.mean(target_scores)-np.mean(impostor_all_scores):.4f}")
    print(f"  Eval time:            {total_time:.0f}s")
    print(f"{'='*70}")

    results["final"] = {
        "optimal_threshold": round(float(OPT_THR), 3),
        "eer_pct": round(float(eer_rate * 100), 2),
        "f1": round(float(best_f1), 4),
        "confusion": {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)},
        "TAR_pct": round(tp/(tp+fn)*100, 1),
        "FAR_pct": round(fp/(fp+tn)*100, 1),
        "precision_pct": round(tp/max(tp+fp,1)*100, 1),
        "accuracy_pct": round((tp+tn)/(tp+fn+fp+tn)*100, 1),
        "score_gap": round(float(np.mean(target_scores)-np.mean(impostor_all_scores)), 4),
        "eval_time_sec": round(total_time, 1),
    }

    out = Path(__file__).parent / "impostor_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    run()
