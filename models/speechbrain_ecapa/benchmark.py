"""
SpeechBrain ECAPA-TDNN — Speaker Verification Benchmark
Architecture: ECAPA-TDNN with SE blocks + multi-scale
EER: 0.800% | Size: 83MB | Embedding: 192d
"""

import sys, argparse, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.audio import record_audio, save_wav, SAMPLE_RATE
from lib.metrics import cosine_similarity, measure_inference

DATA = ROOT / "data"
CACHE = ROOT / "cache"
THRESHOLD = 0.25
MODEL_INFO = {"name": "SpeechBrain ECAPA-TDNN", "eer": "0.800%", "size_mb": 83, "dim": 192}


def _fix_torchaudio():
    import torchaudio
    for attr, val in [('list_audio_backends', lambda: ['soundfile']),
                      ('get_audio_backend', lambda: 'soundfile'),
                      ('set_audio_backend', lambda x: None)]:
        if not hasattr(torchaudio, attr):
            setattr(torchaudio, attr, val)


def load_model():
    _fix_torchaudio()
    from speechbrain.inference.speaker import SpeakerRecognition
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(CACHE / "speechbrain_ecapa"),
    )


def extract_embedding(model, wav_path):
    import torch, soundfile as sf
    audio_np, fs = sf.read(str(wav_path), dtype="float32")
    if audio_np.ndim > 1: audio_np = audio_np.mean(axis=1)
    signal = torch.from_numpy(audio_np).unsqueeze(0)
    if fs != SAMPLE_RATE:
        import torchaudio
        signal = torchaudio.functional.resample(signal, fs, SAMPLE_RATE)
    return model.encode_batch(signal).squeeze().detach().numpy()


def cmd_benchmark(model, wav_path):
    print(f"\n  Model:     {MODEL_INFO['name']}")
    print(f"  EER:       {MODEL_INFO['eer']}")
    print(f"  Size:      {MODEL_INFO['size_mb']} MB")
    print(f"  Embedding: {MODEL_INFO['dim']}d")
    emb, dt, ram = measure_inference(extract_embedding, model, wav_path)
    print(f"  Inference: {dt:.0f} ms")
    print(f"  RAM delta: {ram:.1f} MB")
    print(f"  Output:    {emb.shape}")


def cmd_enroll(model, wav_path):
    DATA.mkdir(parents=True, exist_ok=True)
    emb = extract_embedding(model, wav_path)
    np.save(str(DATA / "emb_speechbrain_ecapa.npy"), emb)
    print(f"  Enrolled: {emb.shape} saved")


def cmd_verify(model, wav_path):
    emb_path = DATA / "emb_speechbrain_ecapa.npy"
    if not emb_path.exists(): print("  ERROR: enroll first"); return
    enrolled = np.load(str(emb_path))
    test_emb = extract_embedding(model, wav_path)
    score = cosine_similarity(enrolled, test_emb)
    print(f"  Score:     {score:.4f}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Result:    {'MATCH' if score >= THRESHOLD else 'REJECTED'}")


def get_audio(args, mode):
    if args.file: return Path(args.file)
    dur = 10 if mode == "enroll" else 5
    audio = record_audio(dur, mode.upper())
    p = DATA / f"{mode}_temp.wav"; save_wav(audio, p); return p


def main():
    parser = argparse.ArgumentParser(description="SpeechBrain ECAPA-TDNN Benchmark")
    parser.add_argument("command", choices=["enroll", "verify", "benchmark"])
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()
    model = load_model()
    wav = get_audio(args, args.command)
    {"benchmark": cmd_benchmark, "enroll": cmd_enroll, "verify": cmd_verify}[args.command](model, wav)

if __name__ == "__main__":
    main()
