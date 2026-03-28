"""
Resemblyzer GE2E — Speaker Verification Benchmark
Architecture: 3-layer LSTM (40→256) + Dense(256) + L2Norm
EER: ~5-7% | Size: 17MB | Embedding: 256d
"""

import sys, argparse, time, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.audio import record_audio, save_wav, load_wav, SAMPLE_RATE
from lib.metrics import cosine_similarity, measure_inference

DATA = ROOT / "data"
THRESHOLD = 0.75
MODEL_INFO = {"name": "Resemblyzer GE2E", "eer": "~5-7%", "size_mb": 17, "dim": 256}


def load_model():
    from resemblyzer import VoiceEncoder
    return VoiceEncoder()


def extract_embedding(model, wav_path):
    from resemblyzer import preprocess_wav
    wav = preprocess_wav(wav_path)
    return model.embed_utterance(wav)


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
    np.save(str(DATA / "emb_resemblyzer.npy"), emb)
    print(f"  Enrolled: {emb.shape} saved")


def cmd_verify(model, wav_path):
    emb_path = DATA / "emb_resemblyzer.npy"
    if not emb_path.exists():
        print("  ERROR: enroll first"); return
    enrolled = np.load(str(emb_path))
    test_emb = extract_embedding(model, wav_path)
    score = cosine_similarity(enrolled, test_emb)
    match = score >= THRESHOLD
    print(f"  Score:     {score:.4f}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Result:    {'MATCH' if match else 'REJECTED'}")


def get_audio(args, mode):
    if args.file:
        return Path(args.file)
    dur = 10 if mode == "enroll" else 5
    audio = record_audio(dur, mode.upper())
    p = DATA / f"{mode}_temp.wav"
    save_wav(audio, p)
    return p


def main():
    parser = argparse.ArgumentParser(description="Resemblyzer GE2E Benchmark")
    parser.add_argument("command", choices=["enroll", "verify", "benchmark"])
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    model = load_model()
    wav = get_audio(args, args.command)

    if args.command == "benchmark":
        cmd_benchmark(model, wav)
    elif args.command == "enroll":
        cmd_enroll(model, wav)
    elif args.command == "verify":
        cmd_verify(model, wav)


if __name__ == "__main__":
    main()
