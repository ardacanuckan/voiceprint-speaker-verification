"""
ESP32-S3 Simulator GUI — Quantized Resemblyzer
Runs the exact same int8 quantized LSTM that will run on ESP32-S3,
but on your Mac. Shows real memory usage, inference time, model size,
and side-by-side comparison with the original float32 model.
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import time
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import psutil
import customtkinter as ctk
from tkinter import filedialog

# --- Config ---
SAMPLE_RATE = 16000
ENROLL_DURATION = 10
VERIFY_DURATION = 5
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- Palette (same as main GUI) ---
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
RED = "#FF3333"
GREEN = "#44FF44"
MONO = "Courier"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# ============================================================
# QUANTIZED ENGINE — exact replica of what runs on ESP32
# ============================================================

class QuantizedResemblyzer:
    """Int8 per-row quantized Resemblyzer LSTM — identical to ESP32 C code."""

    def __init__(self):
        self.weights = {}
        self.scales = {}
        self.loaded = False
        self.weight_bytes = 0
        self.scale_bytes = 0

    def load(self):
        """Load Resemblyzer and quantize to int8 in memory."""
        from resemblyzer import VoiceEncoder
        enc = VoiceEncoder()
        state = enc.state_dict()

        self.weight_bytes = 0
        self.scale_bytes = 0

        for key, tensor in state.items():
            t = tensor.numpy()
            if t.ndim == 2:
                # Per-row quantization
                n_rows = t.shape[0]
                row_scales = np.zeros(n_rows, dtype=np.float32)
                quantized = np.zeros_like(t, dtype=np.int8)
                for r in range(n_rows):
                    absmax = np.abs(t[r]).max()
                    row_scales[r] = absmax / 127.0 if absmax > 1e-10 else 1e-10
                    quantized[r] = np.round(t[r] / row_scales[r]).clip(-127, 127).astype(np.int8)
                self.weights[key] = quantized
                self.scales[key] = row_scales
                self.weight_bytes += quantized.nbytes
                self.scale_bytes += row_scales.nbytes
            else:
                # Per-tensor quantization (biases)
                absmax = np.abs(t).max()
                scale = absmax / 127.0 if absmax > 1e-10 else 1e-10
                quantized = np.round(t / scale).clip(-127, 127).astype(np.int8)
                self.weights[key] = quantized
                self.scales[key] = np.array([scale], dtype=np.float32)
                self.weight_bytes += quantized.nbytes
                self.scale_bytes += 4

        self.loaded = True

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
        """Full forward pass with int8 quantized weights.
        mel_frames: [T, 40] numpy array
        Returns: [256] L2-normalized embedding
        """
        x = mel_frames
        for l in range(3):
            x = self._lstm_layer(x, l)
        last_h = x[-1]

        w = self._dequant("linear.weight")
        b = self._dequant("linear.bias")
        proj = np.maximum(w @ last_h + b, 0)
        return proj / (np.linalg.norm(proj) + 1e-8)

    def get_stats(self):
        return {
            "weight_bytes": self.weight_bytes,
            "scale_bytes": self.scale_bytes,
            "total_bytes": self.weight_bytes + self.scale_bytes,
            "params": sum(w.size for w in self.weights.values()),
            "n_layers": 3,
            "hidden_dim": 256,
            "embedding_dim": 256,
        }


class OriginalResemblyzer:
    """Float32 Resemblyzer for side-by-side comparison."""

    def __init__(self):
        self.encoder = None
        self.loaded = False

    def load(self):
        from resemblyzer import VoiceEncoder
        self.encoder = VoiceEncoder()
        self.loaded = True

    def forward(self, mel_frames):
        import torch
        with torch.no_grad():
            inp = torch.from_numpy(mel_frames).unsqueeze(0).float()
            out, _ = self.encoder.lstm(inp)
            last = out[:, -1, :]
            proj = torch.relu(self.encoder.linear(last))
            proj = proj / proj.norm(dim=1, keepdim=True)
            return proj.squeeze().numpy()


def compute_mel_from_audio(audio_np):
    """Extract mel spectrogram from raw audio numpy array."""
    from resemblyzer.audio import wav_to_mel_spectrogram
    return wav_to_mel_spectrogram(audio_np)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ============================================================
# GUI
# ============================================================

class StatBox(ctk.CTkFrame):
    def __init__(self, parent, label, value="--", **kw):
        super().__init__(parent, fg_color=BG_CELL, corner_radius=6,
                         border_width=1, border_color=BORDER, height=52, **kw)
        self.pack_propagate(False)
        self._lbl = ctk.CTkLabel(self, text=label.upper(),
                                  font=ctk.CTkFont(family=MONO, size=9), text_color=GRAY3)
        self._lbl.pack(anchor="w", padx=10, pady=(8, 0))
        self._val = ctk.CTkLabel(self, text=value,
                                  font=ctk.CTkFont(family=MONO, size=13, weight="bold"),
                                  text_color=GRAY1)
        self._val.pack(anchor="w", padx=10, pady=(0, 6))

    def set(self, value, color=GRAY1):
        self._val.configure(text=value, text_color=color)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VOICEPRINT // ESP32-S3 SIMULATOR")
        self.geometry("960x860")
        self.configure(fg_color=BG)
        self.minsize(900, 780)

        self.quant_model = QuantizedResemblyzer()
        self.float_model = OriginalResemblyzer()
        self.enrolled_quant = None
        self.enrolled_float = None
        self._busy = False

        self._build()

    def _build(self):
        # === TOP BAR ===
        top = ctk.CTkFrame(self, fg_color=BG, height=60, corner_radius=0)
        top.pack(fill="x")
        top.pack_propagate(False)

        ctk.CTkLabel(top, text="VOICEPRINT",
                     font=ctk.CTkFont(family=MONO, size=20, weight="bold"),
                     text_color=WHITE).pack(side="left", padx=24)
        ctk.CTkLabel(top, text="//  ESP32-S3 SIMULATOR",
                     font=ctk.CTkFont(family=MONO, size=10), text_color=GRAY4
                     ).pack(side="left", pady=(4, 0))

        self.sys_lbl = ctk.CTkLabel(top, text="",
                                     font=ctk.CTkFont(family=MONO, size=9), text_color=GRAY4)
        self.sys_lbl.pack(side="right", padx=24)

        ctk.CTkFrame(self, fg_color=BORDER, height=1).pack(fill="x")

        # === AUDIO SOURCE ===
        src_bar = ctk.CTkFrame(self, fg_color=BG, height=40)
        src_bar.pack(fill="x")
        src_bar.pack_propagate(False)

        self.audio_mode = ctk.StringVar(value="mic")
        ctk.CTkLabel(src_bar, text="INPUT:",
                     font=ctk.CTkFont(family=MONO, size=9), text_color=GRAY4
                     ).pack(side="left", padx=(24, 8))
        for val, txt in [("mic", "MIC"), ("file", "FILE")]:
            ctk.CTkRadioButton(src_bar, text=txt, variable=self.audio_mode, value=val,
                               font=ctk.CTkFont(family=MONO, size=10), text_color=GRAY2,
                               fg_color=WHITE, hover_color=GRAY3, border_color=GRAY4,
                               ).pack(side="left", padx=4)

        ctk.CTkFrame(self, fg_color=BORDER, height=1).pack(fill="x")

        # === MAIN CONTENT (scrollable) ===
        main = ctk.CTkScrollableFrame(self, fg_color=BG, corner_radius=0,
                                       scrollbar_button_color=GRAY5,
                                       scrollbar_button_hover_color=GRAY4)
        main.pack(fill="both", expand=True, padx=0, pady=0)

        # -- Section: Model Info --
        self._section(main, "MODEL")

        info_grid = ctk.CTkFrame(main, fg_color="transparent")
        info_grid.pack(fill="x", padx=24, pady=(0, 16))

        self.s_arch = StatBox(info_grid, "ARCHITECTURE", "LSTM 3x256")
        self.s_arch.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.s_params = StatBox(info_grid, "PARAMETERS", "--")
        self.s_params.pack(side="left", expand=True, fill="x", padx=3)
        self.s_quant = StatBox(info_grid, "QUANTIZATION", "INT8 PER-ROW")
        self.s_quant.pack(side="left", expand=True, fill="x", padx=3)
        self.s_emb = StatBox(info_grid, "EMBEDDING", "256d")
        self.s_emb.pack(side="left", expand=True, fill="x", padx=(3, 0))

        # -- Section: Memory --
        self._section(main, "MEMORY // ESP32-S3 FIT")

        mem_grid = ctk.CTkFrame(main, fg_color="transparent")
        mem_grid.pack(fill="x", padx=24, pady=(0, 16))

        self.s_weights = StatBox(mem_grid, "INT8 WEIGHTS", "--")
        self.s_weights.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.s_scales = StatBox(mem_grid, "SCALE FACTORS", "--")
        self.s_scales.pack(side="left", expand=True, fill="x", padx=3)
        self.s_total = StatBox(mem_grid, "TOTAL MODEL", "--")
        self.s_total.pack(side="left", expand=True, fill="x", padx=3)
        self.s_flash = StatBox(mem_grid, "ESP32 FLASH %", "--")
        self.s_flash.pack(side="left", expand=True, fill="x", padx=3)
        self.s_ram = StatBox(mem_grid, "RUNTIME RAM", "--")
        self.s_ram.pack(side="left", expand=True, fill="x", padx=(3, 0))

        # Flash usage bar
        bar_frame = ctk.CTkFrame(main, fg_color="transparent", height=30)
        bar_frame.pack(fill="x", padx=24, pady=(0, 16))
        bar_frame.pack_propagate(False)

        ctk.CTkLabel(bar_frame, text="ESP32-S3 FLASH [8MB]:",
                     font=ctk.CTkFont(family=MONO, size=9), text_color=GRAY3
                     ).pack(side="left")

        self.flash_bar = ctk.CTkProgressBar(bar_frame, width=400, height=8,
                                             progress_color=WHITE, fg_color=GRAY5)
        self.flash_bar.pack(side="left", padx=12, pady=10)
        self.flash_bar.set(0)

        self.flash_pct_lbl = ctk.CTkLabel(bar_frame, text="0%",
                                           font=ctk.CTkFont(family=MONO, size=10, weight="bold"),
                                           text_color=GRAY2)
        self.flash_pct_lbl.pack(side="left")

        # -- Section: Performance --
        self._section(main, "PERFORMANCE")

        perf_grid = ctk.CTkFrame(main, fg_color="transparent")
        perf_grid.pack(fill="x", padx=24, pady=(0, 16))

        self.s_load_time = StatBox(perf_grid, "LOAD TIME", "--")
        self.s_load_time.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.s_mel_time = StatBox(perf_grid, "MEL EXTRACT", "--")
        self.s_mel_time.pack(side="left", expand=True, fill="x", padx=3)
        self.s_infer_time = StatBox(perf_grid, "INFERENCE", "--")
        self.s_infer_time.pack(side="left", expand=True, fill="x", padx=3)
        self.s_total_time = StatBox(perf_grid, "TOTAL PIPELINE", "--")
        self.s_total_time.pack(side="left", expand=True, fill="x", padx=3)
        self.s_cpu = StatBox(perf_grid, "CPU PEAK", "--")
        self.s_cpu.pack(side="left", expand=True, fill="x", padx=(3, 0))

        # -- Section: Training Data --
        self._section(main, "TRAINING DATA ENROLLMENT")

        train_grid = ctk.CTkFrame(main, fg_color="transparent")
        train_grid.pack(fill="x", padx=24, pady=(0, 16))

        self.s_train_files = StatBox(train_grid, "FILES", "--")
        self.s_train_files.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.s_train_dur = StatBox(train_grid, "TOTAL AUDIO", "--")
        self.s_train_dur.pack(side="left", expand=True, fill="x", padx=3)
        self.s_train_segs = StatBox(train_grid, "SEGMENTS", "--")
        self.s_train_segs.pack(side="left", expand=True, fill="x", padx=3)
        self.s_train_embs = StatBox(train_grid, "EMBEDDINGS", "--")
        self.s_train_embs.pack(side="left", expand=True, fill="x", padx=3)
        self.s_train_time = StatBox(train_grid, "PROCESS TIME", "--")
        self.s_train_time.pack(side="left", expand=True, fill="x", padx=(3, 0))

        # -- Section: Noise Hardening --
        self._section(main, "NOISE HARDENING")

        noise_ctrl = ctk.CTkFrame(main, fg_color="transparent")
        noise_ctrl.pack(fill="x", padx=24, pady=(0, 8))

        self.noise_enabled = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(
            noise_ctrl, text="ENABLED", variable=self.noise_enabled,
            font=ctk.CTkFont(family=MONO, size=10), text_color=GRAY2,
            fg_color=GRAY5, progress_color=WHITE, button_color=WHITE,
            button_hover_color=GRAY1,
        ).pack(side="left")

        ctk.CTkLabel(noise_ctrl,
                     text="  //  ENROLL: 9 NOISE AUGMENTATIONS + CENTROID AVERAGING"
                          "  //  VERIFY: SPECTRAL GATING + MULTI-SEGMENT",
                     font=ctk.CTkFont(family=MONO, size=9), text_color=GRAY4
                     ).pack(side="left", padx=8)

        noise_grid = ctk.CTkFrame(main, fg_color="transparent")
        noise_grid.pack(fill="x", padx=24, pady=(0, 16))

        self.s_n_augs = StatBox(noise_grid, "AUGMENTATIONS", "--")
        self.s_n_augs.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.s_denoise = StatBox(noise_grid, "NOISE REDUCED", "--")
        self.s_denoise.pack(side="left", expand=True, fill="x", padx=3)
        self.s_segments = StatBox(noise_grid, "SEGMENTS AVG", "--")
        self.s_segments.pack(side="left", expand=True, fill="x", padx=3)
        self.s_boost = StatBox(noise_grid, "SCORE BOOST", "--")
        self.s_boost.pack(side="left", expand=True, fill="x", padx=(3, 0))

        # -- Section: Accuracy Comparison --
        self._section(main, "ACCURACY // INT8 vs FLOAT32")

        acc_grid = ctk.CTkFrame(main, fg_color="transparent")
        acc_grid.pack(fill="x", padx=24, pady=(0, 16))

        self.s_cos_sim = StatBox(acc_grid, "COSINE SIMILARITY", "--")
        self.s_cos_sim.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.s_score_q = StatBox(acc_grid, "INT8 SCORE", "--")
        self.s_score_q.pack(side="left", expand=True, fill="x", padx=3)
        self.s_score_f = StatBox(acc_grid, "FLOAT32 SCORE", "--")
        self.s_score_f.pack(side="left", expand=True, fill="x", padx=3)
        self.s_verdict = StatBox(acc_grid, "VERDICT", "--")
        self.s_verdict.pack(side="left", expand=True, fill="x", padx=(3, 0))

        # -- Result display --
        self.result_frame = ctk.CTkFrame(main, fg_color=BG_CELL, corner_radius=8,
                                          border_width=1, border_color=BORDER, height=56)
        self.result_frame.pack(fill="x", padx=24, pady=(0, 16))
        self.result_frame.pack_propagate(False)

        self.result_lbl = ctk.CTkLabel(self.result_frame, text="LOAD MODEL TO BEGIN",
                                        font=ctk.CTkFont(family=MONO, size=14),
                                        text_color=GRAY4)
        self.result_lbl.pack(expand=True)

        # -- Buttons --
        btn_frame = ctk.CTkFrame(main, fg_color="transparent")
        btn_frame.pack(fill="x", padx=24, pady=(0, 24))

        bstyle = dict(height=36, corner_radius=6,
                      font=ctk.CTkFont(family=MONO, size=12, weight="bold"),
                      border_width=1, border_color=BORDER_LIGHT)

        self.btn_load = ctk.CTkButton(btn_frame, text="LOAD MODEL",
                                       fg_color=BG, hover_color=GRAY5, text_color=WHITE,
                                       command=self._on_load, **bstyle)
        self.btn_load.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.btn_train = ctk.CTkButton(btn_frame, text="TRAIN FROM DATA",
                                        fg_color=GRAY5, hover_color=GRAY4, text_color=GRAY4,
                                        command=self._on_train, state="disabled", **bstyle)
        self.btn_train.pack(side="left", expand=True, fill="x", padx=4)

        self.btn_enroll = ctk.CTkButton(btn_frame, text="ENROLL MIC",
                                         fg_color=GRAY5, hover_color=GRAY4, text_color=GRAY4,
                                         command=self._on_enroll, state="disabled", **bstyle)
        self.btn_enroll.pack(side="left", expand=True, fill="x", padx=4)

        self.btn_verify = ctk.CTkButton(btn_frame, text="VERIFY",
                                         fg_color=GRAY5, hover_color=GRAY4, text_color=GRAY4,
                                         command=self._on_verify, state="disabled", **bstyle)
        self.btn_verify.pack(side="left", expand=True, fill="x", padx=(4, 0))

        # -- Bottom bar --
        ctk.CTkFrame(self, fg_color=BORDER, height=1).pack(fill="x", side="bottom")
        bottom = ctk.CTkFrame(self, fg_color=BG, height=28, corner_radius=0)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)
        ctk.CTkLabel(bottom,
                     text="SIMULATING ESP32-S3 (240MHz, 8MB PSRAM, 8MB FLASH)  //  "
                          "INT8 PER-ROW QUANTIZED  //  PURE C LSTM ENGINE",
                     font=ctk.CTkFont(family=MONO, size=9), text_color=GRAY5
                     ).pack(side="left", padx=24, pady=4)

        self._update_sys()

    def _section(self, parent, title):
        ctk.CTkLabel(parent, text=title,
                     font=ctk.CTkFont(family=MONO, size=10, weight="bold"),
                     text_color=GRAY3).pack(anchor="w", padx=24, pady=(16, 6))

    def _update_sys(self):
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0)
        self.sys_lbl.configure(
            text=f"CPU {cpu:4.0f}%  //  RAM {mem.used/1024**3:.1f}/{mem.total/1024**3:.0f} GB"
        )
        self.after(2000, self._update_sys)

    def _set_result(self, text, color=GRAY4):
        self.result_lbl.configure(text=text, text_color=color)
        self.result_frame.configure(border_color=color if color != GRAY4 else BORDER)

    def _get_audio(self, mode):
        if self.audio_mode.get() == "file":
            p = filedialog.askopenfilename(
                title=f"SELECT WAV — {mode.upper()}",
                filetypes=[("WAV", "*.wav"), ("All", "*.*")])
            return Path(p) if p else None
        else:
            result = {"path": None}
            dur = ENROLL_DURATION if mode == "enroll" else VERIFY_DURATION
            modal = RecordModal(self, dur, lambda p: result.update({"path": p}))
            self.wait_window(modal)
            return result["path"]

    # --- Load ---
    def _on_load(self):
        if self._busy:
            return
        self._busy = True
        self.btn_load.configure(text="LOADING...", state="disabled")
        self._set_result("LOADING QUANTIZED MODEL...", GRAY3)
        threading.Thread(target=self._do_load, daemon=True).start()

    def _do_load(self):
        try:
            proc = psutil.Process(os.getpid())
            mem_before = proc.memory_info().rss
            t0 = time.perf_counter()

            self.quant_model.load()
            self.float_model.load()

            load_time = (time.perf_counter() - t0) * 1000
            mem_after = proc.memory_info().rss
            ram_delta = (mem_after - mem_before) / 1024 / 1024

            stats = self.quant_model.get_stats()
            self.after(0, lambda: self._load_done(load_time, ram_delta, stats))
        except Exception as e:
            self.after(0, lambda: self._load_error(str(e)))
        finally:
            self._busy = False

    def _load_done(self, load_time, ram_mb, stats):
        wb = stats["weight_bytes"]
        sb = stats["scale_bytes"]
        tb = stats["total_bytes"]
        flash_pct = tb / (8 * 1024 * 1024) * 100

        self.s_params.set(f"{stats['params']:,}")
        self.s_weights.set(f"{wb/1024:.0f} KB")
        self.s_scales.set(f"{sb/1024:.1f} KB")
        self.s_total.set(f"{tb/1024:.0f} KB")
        self.s_flash.set(f"{flash_pct:.1f}%", GREEN if flash_pct < 50 else WHITE)
        self.s_ram.set(f"{ram_mb:.0f} MB")
        self.s_load_time.set(f"{load_time:.0f} ms")

        self.flash_bar.set(flash_pct / 100)
        self.flash_pct_lbl.configure(text=f"{flash_pct:.1f}%")

        self.btn_load.configure(text="LOADED", fg_color=GRAY5, text_color=GRAY3)
        self.btn_train.configure(state="normal", fg_color=WHITE, text_color=BG,
                                  border_color=WHITE)
        self.btn_enroll.configure(state="normal", fg_color=BG, text_color=WHITE,
                                  border_color=WHITE)
        self._set_result("MODEL LOADED // USE TRAIN FROM DATA FOR BEST ACCURACY", GRAY2)

        # Check existing enrollment
        if (DATA_DIR / "esp32_enrolled_q.npy").exists():
            self.enrolled_quant = np.load(str(DATA_DIR / "esp32_enrolled_q.npy"))
            self.enrolled_float = np.load(str(DATA_DIR / "esp32_enrolled_f.npy"))
            self.btn_verify.configure(state="normal", fg_color=BG, text_color=WHITE,
                                      border_color=WHITE)
            self._set_result("MODEL LOADED // ENROLLMENT FOUND // READY", GRAY2)

    def _load_error(self, err):
        self._set_result(f"LOAD ERROR: {err[:60]}", RED)
        self.btn_load.configure(text="RETRY", state="normal")

    # --- Train from data ---
    def _on_train(self):
        if self._busy or not self.quant_model.loaded:
            return
        files = filedialog.askopenfilenames(
            title="SELECT TRAINING WAV FILES",
            filetypes=[("WAV", "*.wav"), ("All", "*.*")],
            initialdir=str(ROOT / "training_data"),
        )
        if not files:
            return
        self._busy = True
        self.btn_train.configure(text="TRAINING...", state="disabled")
        self._set_result("BUILDING VOICEPRINT FROM TRAINING DATA...", GRAY3)
        wav_paths = [Path(f) for f in files]
        threading.Thread(target=self._do_train, args=(wav_paths,), daemon=True).start()

    def _do_train(self, wav_paths):
        try:
            from lib.noise import training_data_enroll

            proc = psutil.Process(os.getpid())
            proc.cpu_percent(interval=None)
            t0 = time.perf_counter()

            def q_embed_fn(audio_np):
                mel = compute_mel_from_audio(audio_np)
                return self.quant_model.forward(mel)

            def f_embed_fn(audio_np):
                mel = compute_mel_from_audio(audio_np)
                return self.float_model.forward(mel)

            emb_q, stats_q = training_data_enroll(wav_paths, q_embed_fn)
            emb_f, stats_f = training_data_enroll(wav_paths, f_embed_fn)

            train_time = (time.perf_counter() - t0) * 1000
            cpu = proc.cpu_percent(interval=0.1)
            sim = cosine_sim(emb_q, emb_f)

            np.save(str(DATA_DIR / "esp32_enrolled_q.npy"), emb_q)
            np.save(str(DATA_DIR / "esp32_enrolled_f.npy"), emb_f)
            self.enrolled_quant = emb_q
            self.enrolled_float = emb_f

            self.after(0, lambda: self._train_done(train_time, cpu, sim, stats_q))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.after(0, lambda: self._train_error(str(e)))
        finally:
            self._busy = False

    def _train_done(self, train_time, cpu, sim, stats):
        self.s_train_files.set(f"{stats['n_files']}", WHITE)
        self.s_train_dur.set(f"{stats['total_duration_sec']:.0f}s", WHITE)
        self.s_train_segs.set(f"{stats['n_segments']}", WHITE)
        self.s_train_embs.set(f"{stats['n_embeddings']}", WHITE)
        self.s_train_time.set(f"{train_time/1000:.1f}s", WHITE)

        self.s_cos_sim.set(f"{sim:.4f}", GREEN if sim > 0.98 else WHITE)
        self.s_infer_time.set(f"{train_time:.0f} ms")
        self.s_cpu.set(f"{cpu:.0f}%")

        self.btn_train.configure(text="RE-TRAIN", state="normal",
                                  fg_color=BG, text_color=GRAY1, border_color=BORDER_LIGHT)
        self.btn_verify.configure(state="normal", fg_color=WHITE, text_color=BG,
                                  border_color=WHITE)
        self._set_result(
            f"TRAINED // {stats['n_files']} FILES // {stats['n_segments']} SEGMENTS // "
            f"{stats['n_embeddings']} EMBEDDINGS // INT8~F32: {sim:.4f}", GREEN)

    def _train_error(self, err):
        self._set_result(f"TRAIN ERROR: {err[:60]}", RED)
        self.btn_train.configure(text="TRAIN FROM DATA", state="normal")

    # --- Enroll (mic) ---
    def _on_enroll(self):
        if self._busy or not self.quant_model.loaded:
            return
        wav = self._get_audio("enroll")
        if not wav:
            return
        self._busy = True
        self.btn_enroll.configure(text="...", state="disabled")
        self._set_result("ENROLLING...", GRAY3)
        threading.Thread(target=self._do_enroll, args=(wav,), daemon=True).start()

    def _do_enroll(self, wav):
        try:
            from lib.noise import hardened_enroll

            proc = psutil.Process(os.getpid())
            use_hardening = self.noise_enabled.get()

            # Load audio
            audio, sr = sf.read(str(wav), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                import torchaudio, torch
                audio = torchaudio.functional.resample(
                    torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()

            proc.cpu_percent(interval=None)
            mem_before = proc.memory_info().rss
            t0 = time.perf_counter()

            n_augs = 0
            if use_hardening:
                # Hardened enrollment: augment + centroid averaging
                def q_embed_fn(audio_np):
                    mel = compute_mel_from_audio(audio_np)
                    return self.quant_model.forward(mel)

                def f_embed_fn(audio_np):
                    mel = compute_mel_from_audio(audio_np)
                    return self.float_model.forward(mel)

                emb_q, details_q, stats_q = hardened_enroll(audio, q_embed_fn)
                emb_f, details_f, _ = hardened_enroll(audio, f_embed_fn)
                n_augs = stats_q["n_augmentations"]
            else:
                mel = compute_mel_from_audio(audio)
                emb_q = self.quant_model.forward(mel)
                emb_f = self.float_model.forward(mel)

            infer_time = (time.perf_counter() - t0) * 1000
            mem_after = proc.memory_info().rss
            cpu = proc.cpu_percent(interval=0.1)
            ram = max(0, (mem_after - mem_before)) / 1024 / 1024
            sim = cosine_sim(emb_q, emb_f)

            np.save(str(DATA_DIR / "esp32_enrolled_q.npy"), emb_q)
            np.save(str(DATA_DIR / "esp32_enrolled_f.npy"), emb_f)
            self.enrolled_quant = emb_q
            self.enrolled_float = emb_f

            self.after(0, lambda: self._enroll_done(
                infer_time, cpu, ram, sim, n_augs, use_hardening))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.after(0, lambda: self._enroll_error(str(e)))
        finally:
            self._busy = False

    def _enroll_done(self, infer_time, cpu, ram, sim, n_augs, hardened):
        self.s_infer_time.set(f"{infer_time:.0f} ms")
        self.s_total_time.set(f"{infer_time:.0f} ms")
        self.s_cpu.set(f"{cpu:.0f}%")
        if ram > 0.1:
            self.s_ram.set(f"{ram:.1f} MB")
        self.s_cos_sim.set(f"{sim:.4f}", GREEN if sim > 0.98 else WHITE)

        if hardened:
            self.s_n_augs.set(f"{n_augs}", WHITE)
            self.s_denoise.set("YES", WHITE)
        else:
            self.s_n_augs.set("OFF", GRAY4)
            self.s_denoise.set("OFF", GRAY4)

        self.btn_enroll.configure(text="RE-ENROLL", state="normal",
                                  fg_color=BG, text_color=GRAY1, border_color=BORDER_LIGHT)
        self.btn_verify.configure(state="normal", fg_color=BG, text_color=WHITE,
                                  border_color=WHITE)
        mode = "HARDENED" if hardened else "STANDARD"
        self._set_result(
            f"ENROLLED [{mode}] // INT8~FLOAT32: {sim:.4f}", WHITE)

    def _enroll_error(self, err):
        self._set_result(f"ENROLL ERROR: {err[:60]}", RED)
        self.btn_enroll.configure(text="ENROLL", state="normal")

    # --- Verify ---
    def _on_verify(self):
        if self._busy or self.enrolled_quant is None:
            return
        wav = self._get_audio("verify")
        if not wav:
            return
        self._busy = True
        self.btn_verify.configure(text="...", state="disabled")
        self._set_result("VERIFYING...", GRAY3)
        threading.Thread(target=self._do_verify, args=(wav,), daemon=True).start()

    def _do_verify(self, wav):
        try:
            from lib.noise import hardened_verify

            proc = psutil.Process(os.getpid())
            use_hardening = self.noise_enabled.get()

            audio, sr = sf.read(str(wav), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                import torchaudio, torch
                audio = torchaudio.functional.resample(
                    torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()

            proc.cpu_percent(interval=None)
            mem_before = proc.memory_info().rss
            t0 = time.perf_counter()

            n_segs = 1
            if use_hardening:
                def q_embed_fn(audio_np):
                    mel = compute_mel_from_audio(audio_np)
                    return self.quant_model.forward(mel)

                def f_embed_fn(audio_np):
                    mel = compute_mel_from_audio(audio_np)
                    return self.float_model.forward(mel)

                emb_q, vstats = hardened_verify(audio, q_embed_fn)
                emb_f, _ = hardened_verify(audio, f_embed_fn)
                n_segs = vstats["n_segments"]
            else:
                mel = compute_mel_from_audio(audio)
                emb_q = self.quant_model.forward(mel)
                emb_f = self.float_model.forward(mel)

            infer_time = (time.perf_counter() - t0) * 1000
            mem_after = proc.memory_info().rss
            cpu = proc.cpu_percent(interval=0.1)
            ram = max(0, (mem_after - mem_before)) / 1024 / 1024

            # Also compute raw (non-hardened) score for comparison
            mel_raw = compute_mel_from_audio(audio)
            emb_q_raw = self.quant_model.forward(mel_raw)
            score_raw = cosine_sim(self.enrolled_quant, emb_q_raw)

            score_q = cosine_sim(self.enrolled_quant, emb_q)
            score_f = cosine_sim(self.enrolled_float, emb_f)
            emb_sim = cosine_sim(emb_q, emb_f)
            threshold = 0.60
            match = score_q >= threshold
            boost = score_q - score_raw if use_hardening else 0.0

            self.after(0, lambda: self._verify_done(
                infer_time, cpu, ram, score_q, score_f, emb_sim,
                match, threshold, n_segs, boost, use_hardening))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.after(0, lambda: self._verify_error(str(e)))
        finally:
            self._busy = False

    def _verify_done(self, infer_time, cpu, ram, score_q, score_f, emb_sim,
                     match, threshold, n_segs, boost, hardened):
        self.s_infer_time.set(f"{infer_time:.0f} ms")
        self.s_total_time.set(f"{infer_time:.0f} ms")
        self.s_cpu.set(f"{cpu:.0f}%")
        if ram > 0.1:
            self.s_ram.set(f"{ram:.1f} MB")

        self.s_cos_sim.set(f"{emb_sim:.4f}", GREEN if emb_sim > 0.98 else WHITE)
        self.s_score_q.set(f"{score_q:.4f}", GREEN if match else RED)
        self.s_score_f.set(f"{score_f:.4f}",
                           GREEN if score_f >= threshold else RED)

        if hardened:
            self.s_segments.set(f"{n_segs}", WHITE)
            self.s_denoise.set("ADAPTIVE", WHITE)
            boost_color = GREEN if boost > 0.02 else GRAY2
            self.s_boost.set(f"+{boost:.3f}" if boost >= 0 else f"{boost:.3f}", boost_color)
        else:
            self.s_segments.set("OFF", GRAY4)
            self.s_boost.set("--", GRAY4)

        if match:
            self.s_verdict.set("MATCH", GREEN)
            self._set_result(
                f"MATCH // INT8: {score_q:.4f}  FLOAT32: {score_f:.4f}  "
                f"THRESHOLD: {threshold:.2f}", GREEN)
        else:
            self.s_verdict.set("REJECTED", RED)
            self._set_result(
                f"REJECTED // INT8: {score_q:.4f}  FLOAT32: {score_f:.4f}  "
                f"THRESHOLD: {threshold:.2f}", RED)

        self.btn_verify.configure(text="VERIFY", state="normal")

    def _verify_error(self, err):
        self._set_result(f"VERIFY ERROR: {err[:60]}", RED)
        self.btn_verify.configure(text="VERIFY", state="normal")


class RecordModal(ctk.CTkToplevel):
    def __init__(self, parent, duration, callback):
        super().__init__(parent)
        self.duration = duration
        self.callback = callback
        self._dead = False

        self.title("")
        self.geometry("380x220")
        self.resizable(False, False)
        self.configure(fg_color=BG)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        px = parent.winfo_rootx() + parent.winfo_width() // 2 - 190
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - 110
        self.geometry(f"+{px}+{py}")

        self.state_lbl = ctk.CTkLabel(self, text="PREPARING",
                                       font=ctk.CTkFont(family=MONO, size=11, weight="bold"),
                                       text_color=GRAY3)
        self.state_lbl.pack(pady=(24, 6))

        self.time_lbl = ctk.CTkLabel(self, text=f"{duration}.0",
                                      font=ctk.CTkFont(family=MONO, size=48, weight="bold"),
                                      text_color=WHITE)
        self.time_lbl.pack(pady=(0, 6))

        self.bar = ctk.CTkProgressBar(self, width=280, height=3,
                                       progress_color=WHITE, fg_color=GRAY5)
        self.bar.pack(pady=4)
        self.bar.set(0)

        self.hint_lbl = ctk.CTkLabel(self, text="",
                                      font=ctk.CTkFont(family=MONO, size=10), text_color=GRAY3)
        self.hint_lbl.pack(pady=(6, 0))

        threading.Thread(target=self._run, daemon=True).start()

    def _ui(self, fn):
        if not self._dead:
            self.after(0, fn)

    def _run(self):
        for i in [3, 2, 1]:
            self._ui(lambda n=i: self.hint_lbl.configure(text=f"STARTING IN {n}"))
            time.sleep(1)

        self._ui(lambda: self.state_lbl.configure(text="RECORDING", text_color=WHITE))
        self._ui(lambda: self.hint_lbl.configure(text="SPEAK NOW"))

        audio = sd.rec(int(self.duration * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        start = time.time()
        while time.time() - start < self.duration:
            el = time.time() - start
            r = self.duration - el
            p = el / self.duration
            self._ui(lambda r=r, p=p: (self.time_lbl.configure(text=f"{r:.1f}"),
                                        self.bar.set(p)))
            time.sleep(0.05)
        sd.wait()

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = DATA_DIR / "recording_temp.wav"
        sf.write(str(wav_path), audio.squeeze(), SAMPLE_RATE)

        self._ui(lambda: self._done(wav_path))

    def _done(self, wav_path):
        self.state_lbl.configure(text="DONE", text_color=WHITE)
        self.time_lbl.configure(text="0.0")
        self.bar.set(1.0)
        self.after(500, lambda: self._close(wav_path))

    def _close(self, wav_path):
        self._dead = True
        self.grab_release()
        self.destroy()
        self.callback(wav_path)


if __name__ == "__main__":
    app = App()
    app.mainloop()
