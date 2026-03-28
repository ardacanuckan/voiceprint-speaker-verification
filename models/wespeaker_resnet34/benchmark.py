"""
WeSpeaker ResNet34-LM — Speaker Verification Benchmark
Architecture: ResNet34 with Large-Margin fine-tuning
EER: 0.723% | Size: 25MB | Embedding: 256d
"""

import sys, argparse, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.audio import record_audio, save_wav, SAMPLE_RATE
from lib.metrics import cosine_similarity, measure_inference

DATA = ROOT / "data"
CACHE = ROOT / "cache"
THRESHOLD = 0.5
MODEL_INFO = {"name": "WeSpeaker ResNet34-LM", "eer": "0.723%", "size_mb": 25, "dim": 256}


def load_model():
    import onnxruntime as ort
    path = CACHE / "resnet34_LM.onnx"
    if not path.exists():
        print(f"  ERROR: {path} not found. Run setup.sh first."); sys.exit(1)
    return ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])


def extract_embedding(model, wav_path):
    import torch, torchaudio, soundfile as sf
    audio_np, sr = sf.read(str(wav_path), dtype="float32")
    if audio_np.ndim > 1: audio_np = audio_np.mean(axis=1)
    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, num_mel_bins=80, sample_frequency=SAMPLE_RATE, dither=0.0)
    fbank = (fbank - fbank.mean(dim=0, keepdim=True)).unsqueeze(0).numpy()
    inp = model.get_inputs()[0].name
    out = model.get_outputs()[0].name
    return model.run([out], {inp: fbank})[0].squeeze()


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
    np.save(str(DATA / "emb_wespeaker_resnet34.npy"), emb)
    print(f"  Enrolled: {emb.shape} saved")


def cmd_verify(model, wav_path):
    emb_path = DATA / "emb_wespeaker_resnet34.npy"
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
    parser = argparse.ArgumentParser(description="WeSpeaker ResNet34-LM Benchmark")
    parser.add_argument("command", choices=["enroll", "verify", "benchmark"])
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()
    model = load_model()
    wav = get_audio(args, args.command)
    {"benchmark": cmd_benchmark, "enroll": cmd_enroll, "verify": cmd_verify}[args.command](model, wav)

if __name__ == "__main__":
    main()
