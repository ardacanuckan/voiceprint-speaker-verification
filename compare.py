"""
Speaker Verification — Multi-Model Comparison GUI
Minimal black & white tech aesthetic.
"""

import sys
import os
import time
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import psutil
import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog

# --- Config ---
SAMPLE_RATE = 16000
ENROLL_DURATION = 10
VERIFY_DURATION = 5
DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "cache"

MODELS = {
    "campplus": {
        "name": "CAM++",
        "full_name": "WeSpeaker CAM++",
        "eer": 0.654,
        "size_mb": 28,
        "embedding_dim": 512,
        "type": "onnx",
        "file": "campplus_LM.onnx",
        "threshold": 0.5,
        "tag": "BEST BALANCE",
    },
    "resnet34": {
        "name": "RESNET34",
        "full_name": "WeSpeaker ResNet34-LM",
        "eer": 0.723,
        "size_mb": 25,
        "embedding_dim": 256,
        "type": "onnx",
        "file": "resnet34_LM.onnx",
        "threshold": 0.5,
        "tag": "COMPACT",
    },
    "ecapa": {
        "name": "ECAPA-TDNN",
        "full_name": "SpeechBrain ECAPA-TDNN",
        "eer": 0.800,
        "size_mb": 83,
        "embedding_dim": 192,
        "type": "speechbrain",
        "threshold": 0.25,
        "tag": "PRODUCTION",
    },
    "resemblyzer": {
        "name": "GE2E",
        "full_name": "Resemblyzer GE2E",
        "eer": 6.0,
        "size_mb": 17,
        "embedding_dim": 256,
        "type": "resemblyzer",
        "threshold": 0.75,
        "tag": "LIGHTEST",
    },
}

# --- Palette ---
BG = "#000000"
BG_CARD = "#0A0A0A"
BG_CELL = "#111111"
BORDER = "#1A1A1A"
BORDER_LIGHT = "#2A2A2A"
WHITE = "#FFFFFF"
GRAY1 = "#E0E0E0"
GRAY2 = "#888888"
GRAY3 = "#555555"
GRAY4 = "#333333"
GRAY5 = "#1A1A1A"
OK_COLOR = "#FFFFFF"
FAIL_COLOR = "#FF3333"
ACCENT_DIM = "#444444"
MONO = "Courier"


# ============================================================
# Audio & Model Backend
# ============================================================

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _fix_torchaudio():
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ['soundfile']
    if not hasattr(torchaudio, 'get_audio_backend'):
        torchaudio.get_audio_backend = lambda: 'soundfile'
    if not hasattr(torchaudio, 'set_audio_backend'):
        torchaudio.set_audio_backend = lambda x: None


def load_model(model_key):
    info = MODELS[model_key]
    if info["type"] == "onnx":
        import onnxruntime as ort
        path = MODEL_DIR / info["file"]
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    elif info["type"] == "speechbrain":
        _fix_torchaudio()
        from speechbrain.inference.speaker import SpeakerRecognition
        return SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(MODEL_DIR / "speechbrain_ecapa"),
        )
    elif info["type"] == "resemblyzer":
        from resemblyzer import VoiceEncoder
        return VoiceEncoder()


def extract_embedding(model_key, model, wav_path):
    import torch
    import torchaudio
    info = MODELS[model_key]

    if info["type"] == "onnx":
        audio_np, sr = sf.read(str(wav_path), dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        waveform = torch.from_numpy(audio_np).unsqueeze(0)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, num_mel_bins=80, sample_frequency=SAMPLE_RATE, dither=0.0
        )
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        fbank = fbank.unsqueeze(0).numpy()
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        return model.run([output_name], {input_name: fbank})[0].squeeze()

    elif info["type"] == "speechbrain":
        audio_np, fs = sf.read(str(wav_path), dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        signal = torch.from_numpy(audio_np).unsqueeze(0)
        if fs != SAMPLE_RATE:
            signal = torchaudio.functional.resample(signal, fs, SAMPLE_RATE)
        return model.encode_batch(signal).squeeze().detach().numpy()

    elif info["type"] == "resemblyzer":
        from resemblyzer import preprocess_wav
        wav = preprocess_wav(wav_path)
        return model.embed_utterance(wav)


def measure_inference(model_key, model, wav_path):
    """Run embedding extraction with timing. Returns (embedding, time_ms, ram_mb)."""
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    t0 = time.perf_counter()
    emb = extract_embedding(model_key, model, wav_path)
    dt = (time.perf_counter() - t0) * 1000
    mem_after = process.memory_info().rss
    ram_delta = max(0, (mem_after - mem_before)) / 1024 / 1024
    return emb, dt, ram_delta


# ============================================================
# Theme setup
# ============================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# ============================================================
# Widgets
# ============================================================

class StatBox(ctk.CTkFrame):
    """Tiny stat display: label on top, value below."""

    def __init__(self, parent, label, value="--", **kwargs):
        super().__init__(parent, fg_color=BG_CELL, corner_radius=6,
                         border_width=1, border_color=BORDER, height=50, **kwargs)
        self.pack_propagate(False)

        self._label = ctk.CTkLabel(
            self, text=label.upper(),
            font=ctk.CTkFont(family=MONO, size=9),
            text_color=GRAY3,
        )
        self._label.pack(anchor="w", padx=10, pady=(8, 0))

        self._value = ctk.CTkLabel(
            self, text=value,
            font=ctk.CTkFont(family=MONO, size=13, weight="bold"),
            text_color=GRAY1,
        )
        self._value.pack(anchor="w", padx=10, pady=(0, 6))

    def set(self, value, color=GRAY1):
        self._value.configure(text=value, text_color=color)


class ModelCard(ctk.CTkFrame):
    """Single model card — minimal black/white design."""

    def __init__(self, parent, model_key, app, **kwargs):
        super().__init__(parent, fg_color=BG_CARD, corner_radius=10,
                         border_width=1, border_color=BORDER, **kwargs)
        self.key = model_key
        self.app = app
        self.info = MODELS[model_key]
        self.model_instance = None
        self.is_loaded = False
        self.enrolled = False
        self._busy = False

        self._build()

    def _build(self):
        pad = 16

        # -- Row 1: Tag + Name --
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=pad, pady=(pad, 4))

        ctk.CTkLabel(
            top, text=self.info["tag"],
            font=ctk.CTkFont(family=MONO, size=9, weight="bold"),
            text_color=GRAY3,
        ).pack(side="left")

        self.status_lbl = ctk.CTkLabel(
            top, text="OFFLINE",
            font=ctk.CTkFont(family=MONO, size=9),
            text_color=GRAY4,
        )
        self.status_lbl.pack(side="right")

        ctk.CTkLabel(
            self, text=self.info["name"],
            font=ctk.CTkFont(family=MONO, size=22, weight="bold"),
            text_color=WHITE, anchor="w",
        ).pack(fill="x", padx=pad, pady=(0, 2))

        ctk.CTkLabel(
            self, text=self.info["full_name"],
            font=ctk.CTkFont(family=MONO, size=10),
            text_color=GRAY3, anchor="w",
        ).pack(fill="x", padx=pad, pady=(0, 10))

        # -- Row 2: Static stats --
        row_static = ctk.CTkFrame(self, fg_color="transparent")
        row_static.pack(fill="x", padx=pad, pady=(0, 6))

        self.s_eer = StatBox(row_static, "EER", f"{self.info['eer']}%")
        self.s_eer.pack(side="left", expand=True, fill="x", padx=(0, 3))

        self.s_size = StatBox(row_static, "SIZE", f"{self.info['size_mb']} MB")
        self.s_size.pack(side="left", expand=True, fill="x", padx=3)

        self.s_dim = StatBox(row_static, "DIM", f"{self.info['embedding_dim']}")
        self.s_dim.pack(side="left", expand=True, fill="x", padx=(3, 0))

        # -- Row 3: Dynamic stats --
        row_dyn = ctk.CTkFrame(self, fg_color="transparent")
        row_dyn.pack(fill="x", padx=pad, pady=(0, 6))

        self.s_load = StatBox(row_dyn, "LOAD", "--")
        self.s_load.pack(side="left", expand=True, fill="x", padx=(0, 3))

        self.s_infer = StatBox(row_dyn, "INFERENCE", "--")
        self.s_infer.pack(side="left", expand=True, fill="x", padx=3)

        self.s_ram = StatBox(row_dyn, "RAM DELTA", "--")
        self.s_ram.pack(side="left", expand=True, fill="x", padx=(3, 0))

        # -- Row 4: Score result --
        self.result_frame = ctk.CTkFrame(self, fg_color=BG_CELL, corner_radius=6,
                                          border_width=1, border_color=BORDER, height=44)
        self.result_frame.pack(fill="x", padx=pad, pady=(0, 10))
        self.result_frame.pack_propagate(False)

        self.result_lbl = ctk.CTkLabel(
            self.result_frame, text="AWAITING INPUT",
            font=ctk.CTkFont(family=MONO, size=12),
            text_color=GRAY4,
        )
        self.result_lbl.pack(expand=True)

        # -- Row 5: Buttons --
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=pad, pady=(0, pad))

        self.btn_load = self._make_btn(btn_row, "LOAD", self._on_load)
        self.btn_load.pack(side="left", expand=True, fill="x", padx=(0, 3))

        self.btn_enroll = self._make_btn(btn_row, "ENROLL", self._on_enroll, disabled=True)
        self.btn_enroll.pack(side="left", expand=True, fill="x", padx=3)

        self.btn_verify = self._make_btn(btn_row, "VERIFY", self._on_verify, disabled=True)
        self.btn_verify.pack(side="left", expand=True, fill="x", padx=(3, 0))

    def _make_btn(self, parent, text, cmd, disabled=False):
        state = "disabled" if disabled else "normal"
        fg = GRAY5 if disabled else BORDER_LIGHT
        txt = GRAY4 if disabled else GRAY1
        btn = ctk.CTkButton(
            parent, text=text, height=30, corner_radius=6,
            font=ctk.CTkFont(family=MONO, size=11, weight="bold"),
            fg_color=fg, hover_color=GRAY4, text_color=txt,
            border_width=1, border_color=BORDER_LIGHT,
            command=cmd, state=state,
        )
        return btn

    def _enable_btn(self, btn, active=False):
        fg = WHITE if active else BORDER_LIGHT
        txt = BG if active else GRAY1
        btn.configure(state="normal", fg_color=fg, text_color=txt,
                      border_color=WHITE if active else BORDER_LIGHT)

    def _disable_btn(self, btn):
        btn.configure(state="disabled", fg_color=GRAY5, text_color=GRAY4,
                      border_color=BORDER)

    def _set_status(self, text, color=GRAY3):
        self.status_lbl.configure(text=text, text_color=color)

    def _set_result(self, text, color=GRAY4):
        self.result_lbl.configure(text=text, text_color=color)
        bc = BORDER if color == GRAY4 else color
        self.result_frame.configure(border_color=bc)

    # -- Load --
    def _on_load(self):
        if self._busy:
            return
        self._busy = True
        self._set_status("LOADING", GRAY2)
        self.btn_load.configure(text="...", state="disabled")
        threading.Thread(target=self._do_load, daemon=True).start()

    def _do_load(self):
        try:
            t0 = time.perf_counter()
            self.model_instance = load_model(self.key)
            dt = (time.perf_counter() - t0) * 1000
            self.is_loaded = True
            self.after(0, lambda: self._load_ok(dt))
        except Exception as e:
            self.after(0, lambda: self._load_fail(str(e)))
        finally:
            self._busy = False

    def _load_ok(self, dt):
        self.s_load.set(f"{dt:.0f} ms")
        self._set_status("READY", WHITE)
        self.btn_load.configure(text="LOADED", state="disabled",
                                fg_color=GRAY5, text_color=GRAY3)
        self._enable_btn(self.btn_enroll, active=True)
        # Check existing enrollment
        if (DATA_DIR / f"embedding_{self.key}.npy").exists():
            self.enrolled = True
            self._enable_btn(self.btn_verify)
            self._set_result("ENROLLED // VERIFY READY", GRAY3)

    def _load_fail(self, err):
        self._set_status("ERROR", FAIL_COLOR)
        self._set_result(f"ERR: {err[:50]}", FAIL_COLOR)
        self.btn_load.configure(text="RETRY", state="normal")

    # -- Enroll --
    def _on_enroll(self):
        if self._busy or not self.is_loaded:
            return
        wav = self.app.get_audio_path("enroll")
        if not wav:
            return
        self._busy = True
        self.btn_enroll.configure(text="...", state="disabled")
        self._set_status("ENROLLING", GRAY2)
        threading.Thread(target=self._do_enroll, args=(wav,), daemon=True).start()

    def _do_enroll(self, wav):
        try:
            emb, dt, ram = measure_inference(self.key, self.model_instance, wav)
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            np.save(str(DATA_DIR / f"embedding_{self.key}.npy"), emb)
            self.enrolled = True
            self.after(0, lambda: self._enroll_ok(dt, ram, emb.shape))
        except Exception as e:
            self.after(0, lambda: self._enroll_fail(str(e)))
        finally:
            self._busy = False

    def _enroll_ok(self, dt, ram, shape):
        self.s_infer.set(f"{dt:.0f} ms")
        self.s_ram.set(f"{ram:.1f} MB")
        self._set_status("ENROLLED", WHITE)
        self.btn_enroll.configure(text="RE-ENROLL", state="normal",
                                  fg_color=BORDER_LIGHT, text_color=GRAY1)
        self._enable_btn(self.btn_verify, active=True)
        self._set_result(f"ENROLLED // {shape[0]}d VECTOR STORED", GRAY2)

    def _enroll_fail(self, err):
        self._set_status("ENROLL FAILED", FAIL_COLOR)
        self._set_result(f"ERR: {err[:50]}", FAIL_COLOR)
        self._enable_btn(self.btn_enroll)

    # -- Verify --
    def _on_verify(self):
        if self._busy or not self.enrolled:
            return
        wav = self.app.get_audio_path("verify")
        if not wav:
            return
        self._busy = True
        self.btn_verify.configure(text="...", state="disabled")
        self._set_status("VERIFYING", GRAY2)
        threading.Thread(target=self._do_verify, args=(wav,), daemon=True).start()

    def _do_verify(self, wav):
        try:
            enrolled = np.load(str(DATA_DIR / f"embedding_{self.key}.npy"))
            emb, dt, ram = measure_inference(self.key, self.model_instance, wav)
            score = cosine_similarity(enrolled, emb)
            threshold = self.info["threshold"]
            match = score >= threshold
            self.after(0, lambda: self._verify_ok(score, threshold, match, dt, ram))
        except Exception as e:
            self.after(0, lambda: self._verify_fail(str(e)))
        finally:
            self._busy = False

    def _verify_ok(self, score, threshold, match, dt, ram):
        self.s_infer.set(f"{dt:.0f} ms")
        self.s_ram.set(f"{ram:.1f} MB")
        if match:
            self._set_status("MATCH", WHITE)
            self._set_result(
                f"MATCH // SCORE {score:.4f}  >  THRESHOLD {threshold:.2f}", WHITE)
        else:
            self._set_status("REJECTED", FAIL_COLOR)
            self._set_result(
                f"REJECTED // SCORE {score:.4f}  <  THRESHOLD {threshold:.2f}", FAIL_COLOR)
        self._enable_btn(self.btn_verify)

    def _verify_fail(self, err):
        self._set_status("VERIFY FAILED", FAIL_COLOR)
        self._set_result(f"ERR: {err[:50]}", FAIL_COLOR)
        self._enable_btn(self.btn_verify)


class RecordingModal(ctk.CTkToplevel):
    """Minimal recording overlay — black with white mono text."""

    def __init__(self, parent, duration, callback):
        super().__init__(parent)
        self.duration = duration
        self.callback = callback
        self._destroyed = False

        self.title("")
        self.geometry("380x240")
        self.resizable(False, False)
        self.configure(fg_color=BG)
        self.transient(parent)
        self.grab_set()
        self.overrideredirect(False)

        self.update_idletasks()
        px = parent.winfo_rootx() + parent.winfo_width() // 2 - 190
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - 120
        self.geometry(f"+{px}+{py}")

        self.state_lbl = ctk.CTkLabel(
            self, text="PREPARING",
            font=ctk.CTkFont(family=MONO, size=11, weight="bold"),
            text_color=GRAY3,
        )
        self.state_lbl.pack(pady=(30, 8))

        self.time_lbl = ctk.CTkLabel(
            self, text=f"{duration}.0",
            font=ctk.CTkFont(family=MONO, size=56, weight="bold"),
            text_color=WHITE,
        )
        self.time_lbl.pack(pady=(0, 8))

        self.bar = ctk.CTkProgressBar(
            self, width=280, height=3,
            progress_color=WHITE, fg_color=GRAY5,
        )
        self.bar.pack(pady=4)
        self.bar.set(0)

        self.hint_lbl = ctk.CTkLabel(
            self, text="",
            font=ctk.CTkFont(family=MONO, size=10),
            text_color=GRAY3,
        )
        self.hint_lbl.pack(pady=(8, 0))

        threading.Thread(target=self._run, daemon=True).start()

    def _safe_update(self, fn):
        if not self._destroyed:
            self.after(0, fn)

    def _run(self):
        # Countdown
        for i in [3, 2, 1]:
            self._safe_update(lambda n=i: self.hint_lbl.configure(text=f"STARTING IN {n}"))
            time.sleep(1)

        self._safe_update(lambda: self.state_lbl.configure(text="RECORDING", text_color=WHITE))
        self._safe_update(lambda: self.hint_lbl.configure(text="SPEAK NOW"))

        audio = sd.rec(
            int(self.duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        )

        start = time.time()
        while time.time() - start < self.duration:
            elapsed = time.time() - start
            remaining = self.duration - elapsed
            pct = elapsed / self.duration
            self._safe_update(
                lambda r=remaining, p=pct: self._tick(r, p)
            )
            time.sleep(0.05)

        sd.wait()
        result = audio.squeeze()

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = DATA_DIR / "recording_temp.wav"
        sf.write(str(wav_path), result, SAMPLE_RATE)

        self._safe_update(lambda: self._done(wav_path))

    def _tick(self, remaining, pct):
        self.time_lbl.configure(text=f"{remaining:.1f}")
        self.bar.set(pct)

    def _done(self, wav_path):
        self.state_lbl.configure(text="DONE", text_color=WHITE)
        self.time_lbl.configure(text="0.0")
        self.hint_lbl.configure(text="SAVED")
        self.bar.set(1.0)
        self.after(600, lambda: self._close(wav_path))

    def _close(self, wav_path):
        self._destroyed = True
        self.grab_release()
        self.destroy()
        self.callback(wav_path)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VOICEPRINT")
        self.geometry("1300x860")
        self.configure(fg_color=BG)
        self.minsize(1100, 720)

        self._build()

    def _build(self):
        # ===== TOP BAR =====
        top = ctk.CTkFrame(self, fg_color=BG, height=64, corner_radius=0)
        top.pack(fill="x")
        top.pack_propagate(False)

        # Left: branding
        brand_frame = ctk.CTkFrame(top, fg_color="transparent")
        brand_frame.pack(side="left", padx=24, pady=12)

        ctk.CTkLabel(
            brand_frame, text="VOICEPRINT",
            font=ctk.CTkFont(family=MONO, size=20, weight="bold"),
            text_color=WHITE,
        ).pack(side="left")

        ctk.CTkLabel(
            brand_frame, text="  //  SPEAKER VERIFICATION SYSTEM",
            font=ctk.CTkFont(family=MONO, size=10),
            text_color=GRAY4,
        ).pack(side="left", pady=(4, 0))

        # Right: audio source + system stats
        right_frame = ctk.CTkFrame(top, fg_color="transparent")
        right_frame.pack(side="right", padx=24)

        self.sys_lbl = ctk.CTkLabel(
            right_frame, text="",
            font=ctk.CTkFont(family=MONO, size=9),
            text_color=GRAY4,
        )
        self.sys_lbl.pack(side="right", padx=(16, 0))

        self.audio_mode = ctk.StringVar(value="mic")

        ctk.CTkRadioButton(
            right_frame, text="FILE", variable=self.audio_mode, value="file",
            font=ctk.CTkFont(family=MONO, size=10), text_color=GRAY2,
            fg_color=WHITE, hover_color=GRAY3, border_color=GRAY4,
        ).pack(side="right", padx=4)

        ctk.CTkRadioButton(
            right_frame, text="MIC", variable=self.audio_mode, value="mic",
            font=ctk.CTkFont(family=MONO, size=10), text_color=GRAY2,
            fg_color=WHITE, hover_color=GRAY3, border_color=GRAY4,
        ).pack(side="right", padx=4)

        ctk.CTkLabel(
            right_frame, text="INPUT:",
            font=ctk.CTkFont(family=MONO, size=9),
            text_color=GRAY4,
        ).pack(side="right", padx=(0, 8))

        # ===== DIVIDER =====
        ctk.CTkFrame(self, fg_color=BORDER, height=1, corner_radius=0).pack(fill="x")

        # ===== ACTION BAR =====
        action = ctk.CTkFrame(self, fg_color=BG, height=48, corner_radius=0)
        action.pack(fill="x")
        action.pack_propagate(False)

        btn_style = dict(
            height=28, corner_radius=4,
            font=ctk.CTkFont(family=MONO, size=10, weight="bold"),
            border_width=1, border_color=BORDER_LIGHT,
        )

        ctk.CTkButton(
            action, text="LOAD ALL", fg_color=BG, hover_color=GRAY5,
            text_color=WHITE, command=self._load_all, **btn_style,
        ).pack(side="left", padx=(24, 6), pady=10)

        ctk.CTkButton(
            action, text="ENROLL ALL", fg_color=BG, hover_color=GRAY5,
            text_color=GRAY1, command=self._enroll_all, **btn_style,
        ).pack(side="left", padx=6, pady=10)

        ctk.CTkButton(
            action, text="VERIFY ALL", fg_color=BG, hover_color=GRAY5,
            text_color=GRAY1, command=self._verify_all, **btn_style,
        ).pack(side="left", padx=6, pady=10)

        self.status_lbl = ctk.CTkLabel(
            action, text="4 MODELS AVAILABLE",
            font=ctk.CTkFont(family=MONO, size=9),
            text_color=GRAY4,
        )
        self.status_lbl.pack(side="right", padx=24)

        # ===== DIVIDER =====
        ctk.CTkFrame(self, fg_color=BORDER, height=1, corner_radius=0).pack(fill="x")

        # ===== MODEL CARDS =====
        container = ctk.CTkScrollableFrame(
            self, fg_color=BG, corner_radius=0,
            scrollbar_button_color=GRAY5,
            scrollbar_button_hover_color=GRAY4,
        )
        container.pack(fill="both", expand=True, padx=0, pady=0)

        self.cards = {}
        for i, key in enumerate(MODELS):
            card = ModelCard(container, key, self)
            card.grid(row=i // 2, column=i % 2, padx=12, pady=12, sticky="nsew")
            self.cards[key] = card

        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        # ===== BOTTOM BAR =====
        ctk.CTkFrame(self, fg_color=BORDER, height=1, corner_radius=0).pack(fill="x", side="bottom")

        bottom = ctk.CTkFrame(self, fg_color=BG, height=28, corner_radius=0)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        ctk.CTkLabel(
            bottom,
            text="EER = EQUAL ERROR RATE (LOWER IS BETTER)  //  "
                 "ALL MODELS PRETRAINED ON VOXCELEB  //  "
                 "COSINE SIMILARITY SCORING",
            font=ctk.CTkFont(family=MONO, size=9),
            text_color=GRAY5,
        ).pack(side="left", padx=24, pady=4)

        self._update_sys()

    # --- System monitor ---
    def _update_sys(self):
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0)
        self.sys_lbl.configure(
            text=f"CPU {cpu:4.0f}%  //  "
                 f"RAM {mem.used/1024/1024/1024:.1f}/{mem.total/1024/1024/1024:.0f} GB"
        )
        self.after(2000, self._update_sys)

    # --- Audio source ---
    def get_audio_path(self, mode="enroll"):
        if self.audio_mode.get() == "file":
            path = filedialog.askopenfilename(
                title=f"SELECT WAV — {mode.upper()}",
                filetypes=[("WAV", "*.wav"), ("All", "*.*")],
            )
            return Path(path) if path else None
        else:
            result = {"path": None}
            dur = ENROLL_DURATION if mode == "enroll" else VERIFY_DURATION

            def on_done(p):
                result["path"] = p

            modal = RecordingModal(self, dur, on_done)
            self.wait_window(modal)
            return result["path"]

    # --- Bulk actions ---
    def _load_all(self):
        self.status_lbl.configure(text="LOADING ALL MODELS...")
        for card in self.cards.values():
            if not card.is_loaded:
                card._on_load()

    def _enroll_all(self):
        loaded = [c for c in self.cards.values() if c.is_loaded]
        if not loaded:
            self.status_lbl.configure(text="LOAD MODELS FIRST")
            return
        wav = self.get_audio_path("enroll")
        if not wav:
            return
        self.status_lbl.configure(text="ENROLLING ALL...")
        for card in loaded:
            if not card._busy:
                card._busy = True
                card.btn_enroll.configure(text="...", state="disabled")
                card._set_status("ENROLLING", GRAY2)
                threading.Thread(target=card._do_enroll, args=(wav,), daemon=True).start()

    def _verify_all(self):
        enrolled = [c for c in self.cards.values() if c.is_loaded and c.enrolled]
        if not enrolled:
            self.status_lbl.configure(text="ENROLL FIRST")
            return
        wav = self.get_audio_path("verify")
        if not wav:
            return
        self.status_lbl.configure(text="VERIFYING ALL...")
        for card in enrolled:
            if not card._busy:
                card._busy = True
                card.btn_verify.configure(text="...", state="disabled")
                card._set_status("VERIFYING", GRAY2)
                threading.Thread(target=card._do_verify, args=(wav,), daemon=True).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
