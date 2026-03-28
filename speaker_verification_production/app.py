"""
VOICEPRINT — Speaker Verification Production App
=================================================
Guided enrollment with text prompts, side-by-side model comparison,
real-time performance metrics.

Models:
  1. WeSpeaker CAM++     — 0.654% EER, 28MB  (best accuracy)
  2. Resemblyzer float32 — ~5-7% EER, 17MB   (full model)
  3. Resemblyzer int8    — ~5-7% EER, 1.4MB  (ESP32 version)
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

from engine import (
    FullModel, ESP32Model,
    cosine_sim, record_audio, noise_reduce, measure,
    augmented_enroll_segments,
    SAMPLE_RATE, DATA_DIR,
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Theme ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

BG = "#000000"
CARD = "#080808"
CELL = "#111111"
BORDER = "#1C1C1C"
WHITE = "#FFFFFF"
G1 = "#D0D0D0"
G2 = "#888888"
G3 = "#555555"
G4 = "#333333"
G5 = "#1A1A1A"
GREEN = "#22DD66"
RED = "#FF4444"
BLUE = "#4499FF"
MONO = "Courier"

ENROLL_DURATION = 300  # 5 minutes

READING_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank. "
    "She sells seashells by the seashore every single morning without fail.",

    "A journey of a thousand miles begins with a single step forward. "
    "Technology is best when it brings people together in meaningful ways.",

    "The rain in Spain stays mainly in the plain during autumn season. "
    "Every morning I wake up and look outside the window at the sunrise.",

    "Artificial intelligence is transforming how we interact with machines. "
    "The future of computing lies in making devices smaller and smarter.",

    "Music has the power to change the way we feel about the world. "
    "A good conversation can open doors that no amount of force ever could.",

    "Sometimes the best solution is the simplest one you can think of. "
    "Reading books expands your mind and takes you to places you have never been.",

    "The mountains were covered in a thick blanket of fresh white snow. "
    "She decided to take the longer path because the view was much better.",

    "Innovation comes from questioning everything we take for granted. "
    "The old library had thousands of books that nobody had read in years.",

    "Keep talking naturally about anything that comes to your mind. "
    "Tell a story, describe your day, or just read these sentences out loud.",

    "The more you speak, the better your voiceprint becomes. "
    "Each sentence adds more detail to your unique voice signature.",
]

# Per-model thresholds (from impostor_test.py: optimal F1 = 0.645 for Resemblyzer)
# CAM++ has better separation so can use higher threshold
THRESHOLDS = {
    "RESEMBLYZER FLOAT32": 0.80,
    "RESEMBLYZER INT8": 0.80,
}


# ============================================================
# Widgets
# ============================================================

class Stat(ctk.CTkFrame):
    def __init__(self, parent, label, value="--", wide=False, **kw):
        h = 56 if wide else 50
        super().__init__(parent, fg_color=CELL, corner_radius=6,
                         border_width=1, border_color=BORDER, height=h, **kw)
        self.pack_propagate(False)
        ctk.CTkLabel(self, text=label.upper(), font=ctk.CTkFont(family=MONO, size=8),
                     text_color=G3).pack(anchor="w", padx=10, pady=(7, 0))
        self._v = ctk.CTkLabel(self, text=value,
                               font=ctk.CTkFont(family=MONO, size=12, weight="bold"),
                               text_color=G1)
        self._v.pack(anchor="w", padx=10, pady=(0, 5))

    def set(self, val, color=G1):
        self._v.configure(text=str(val), text_color=color)


class ModelColumn(ctk.CTkFrame):
    """One column per model — shows stats + score."""

    def __init__(self, parent, model, app):
        super().__init__(parent, fg_color=CARD, corner_radius=10,
                         border_width=1, border_color=BORDER)
        self.model = model
        self.app = app
        self.enrolled_emb = None
        self._busy = False

        pad = 12

        # Header
        ctk.CTkLabel(self, text=model.tag,
                     font=ctk.CTkFont(family=MONO, size=9, weight="bold"),
                     text_color=G3).pack(anchor="w", padx=pad, pady=(pad, 2))
        ctk.CTkLabel(self, text=model.name,
                     font=ctk.CTkFont(family=MONO, size=16, weight="bold"),
                     text_color=WHITE).pack(anchor="w", padx=pad, pady=(0, 4))

        self.status = ctk.CTkLabel(self, text="NOT LOADED",
                                    font=ctk.CTkFont(family=MONO, size=9), text_color=G4)
        self.status.pack(anchor="w", padx=pad, pady=(0, 8))

        # Stats
        row1 = ctk.CTkFrame(self, fg_color="transparent")
        row1.pack(fill="x", padx=pad, pady=2)
        self.s_size = Stat(row1, "SIZE", "--")
        self.s_size.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.s_eer = Stat(row1, "EER", "--")
        self.s_eer.pack(side="left", expand=True, fill="x", padx=(2, 0))

        row2 = ctk.CTkFrame(self, fg_color="transparent")
        row2.pack(fill="x", padx=pad, pady=2)
        self.s_load = Stat(row2, "LOAD TIME", "--")
        self.s_load.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.s_infer = Stat(row2, "INFERENCE", "--")
        self.s_infer.pack(side="left", expand=True, fill="x", padx=(2, 0))

        row3 = ctk.CTkFrame(self, fg_color="transparent")
        row3.pack(fill="x", padx=pad, pady=2)
        self.s_ram = Stat(row3, "RAM", "--")
        self.s_ram.pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.s_cpu = Stat(row3, "CPU", "--")
        self.s_cpu.pack(side="left", expand=True, fill="x", padx=(2, 0))

        # Score box
        self.score_frame = ctk.CTkFrame(self, fg_color=CELL, corner_radius=6,
                                         border_width=1, border_color=BORDER, height=48)
        self.score_frame.pack(fill="x", padx=pad, pady=(8, pad))
        self.score_frame.pack_propagate(False)
        self.score_lbl = ctk.CTkLabel(self.score_frame, text="—",
                                       font=ctk.CTkFont(family=MONO, size=13), text_color=G4)
        self.score_lbl.pack(expand=True)

    def set_status(self, txt, color=G3):
        self.status.configure(text=txt, text_color=color)

    def set_score(self, score, match):
        color = GREEN if match else RED
        verdict = "MATCH" if match else "REJECTED"
        self.score_lbl.configure(
            text=f"{verdict}  {score:.4f}", text_color=color)
        self.score_frame.configure(border_color=color)

    def clear_score(self):
        self.score_lbl.configure(text="—", text_color=G4)
        self.score_frame.configure(border_color=BORDER)


class RecordOverlay(ctk.CTkToplevel):
    """Recording overlay — supports short (verify) and long (enroll) recordings.
    For long recordings, shows scrolling text prompts to keep user talking."""

    def __init__(self, parent, duration, texts, callback, mode="enroll"):
        super().__init__(parent)
        self.duration = duration
        self.texts = texts if texts else []
        self.callback = callback
        self.mode = mode
        self._dead = False

        self.title("")
        w, h = (620, 440) if duration > 30 else (520, 280)
        self.geometry(f"{w}x{h}")
        self.resizable(False, False)
        self.configure(fg_color=BG)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        px = parent.winfo_rootx() + parent.winfo_width() // 2 - w // 2
        py = parent.winfo_rooty() + parent.winfo_height() // 2 - h // 2
        self.geometry(f"+{px}+{py}")

        # Header
        dur_str = f"{duration//60}:{duration%60:02d}" if duration >= 60 else f"{duration}s"
        self.state_lbl = ctk.CTkLabel(
            self, text="PREPARING",
            font=ctk.CTkFont(family=MONO, size=12, weight="bold"), text_color=G3)
        self.state_lbl.pack(pady=(16, 4))

        # Timer row
        timer_row = ctk.CTkFrame(self, fg_color="transparent")
        timer_row.pack(fill="x", padx=24, pady=(0, 4))

        self.time_lbl = ctk.CTkLabel(
            timer_row, text=dur_str,
            font=ctk.CTkFont(family=MONO, size=36, weight="bold"), text_color=WHITE)
        self.time_lbl.pack(side="left")

        self.elapsed_lbl = ctk.CTkLabel(
            timer_row, text="",
            font=ctk.CTkFont(family=MONO, size=11), text_color=G3)
        self.elapsed_lbl.pack(side="right")

        self.bar = ctk.CTkProgressBar(self, width=w - 48, height=4,
                                       progress_color=WHITE, fg_color=G5)
        self.bar.pack(padx=24, pady=4)
        self.bar.set(0)

        # Text prompt area (for long recordings)
        self._text_idx = 0
        if self.texts:
            ctk.CTkLabel(self, text="READ ALOUD:",
                         font=ctk.CTkFont(family=MONO, size=9, weight="bold"),
                         text_color=G3).pack(anchor="w", padx=24, pady=(12, 4))

            self.text_lbl = ctk.CTkLabel(
                self, text="", wraplength=w - 60,
                font=ctk.CTkFont(size=15), text_color=WHITE, justify="left")
            self.text_lbl.pack(fill="x", padx=24, pady=(0, 6))

            self.text_counter = ctk.CTkLabel(
                self, text="",
                font=ctk.CTkFont(family=MONO, size=9), text_color=G4)
            self.text_counter.pack(anchor="w", padx=24, pady=(0, 6))

            # NEXT button
            self.next_btn = ctk.CTkButton(
                self, text="NEXT SENTENCE  >>", height=30, corner_radius=6,
                font=ctk.CTkFont(family=MONO, size=11, weight="bold"),
                fg_color=G5, hover_color=G4, text_color=WHITE,
                border_width=1, border_color=G3,
                command=self._next_text)
            self.next_btn.pack(pady=(0, 8))
            self._show_text(0)
        else:
            self.text_lbl = None
            self.next_btn = None
            ctk.CTkLabel(self, text="SPEAK NATURALLY",
                         font=ctk.CTkFont(family=MONO, size=14, weight="bold"),
                         text_color=WHITE).pack(pady=(20, 0))

        # Bottom buttons
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=(4, 12))

        if duration > 30:
            self.stop_btn = ctk.CTkButton(
                btn_row, text="STOP EARLY & SAVE", height=30, corner_radius=6,
                font=ctk.CTkFont(family=MONO, size=10, weight="bold"),
                fg_color=G5, hover_color=G4, text_color=G2,
                border_width=1, border_color=G4,
                command=self._stop_early)
            self.stop_btn.pack(side="left", padx=4)
        else:
            self.stop_btn = None

        self._stop_flag = False
        threading.Thread(target=self._run, daemon=True).start()

    def _ui(self, fn):
        if not self._dead:
            self.after(0, fn)

    def _stop_early(self):
        self._stop_flag = True

    def _next_text(self):
        self._text_idx += 1
        if self._text_idx >= len(self.texts):
            self._text_idx = 0  # loop back
        self._show_text(self._text_idx)

    def _show_text(self, idx):
        if not self.texts:
            return
        idx = min(idx, len(self.texts) - 1)
        self.text_lbl.configure(text=f'"{self.texts[idx]}"')
        self.text_counter.configure(text=f"SENTENCE {idx + 1} / {len(self.texts)}")

    def _run(self):
        # Countdown
        for i in [3, 2, 1]:
            self._ui(lambda n=i: self.state_lbl.configure(text=f"STARTING IN {n}"))
            time.sleep(1)

        self._ui(lambda: self.state_lbl.configure(text="RECORDING", text_color=RED))

        # Start recording
        audio = sd.rec(int(self.duration * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE, channels=1, dtype="float32")

        start = time.time()
        while time.time() - start < self.duration:
            if self._stop_flag:
                sd.stop()
                break
            el = time.time() - start
            remaining = self.duration - el
            mins = int(remaining) // 60
            secs = remaining % 60
            time_str = f"{mins}:{secs:04.1f}" if mins > 0 else f"{secs:.1f}"
            pct = el / self.duration

            self._ui(lambda t=time_str, p=pct, e=el:
                     (self.time_lbl.configure(text=t),
                      self.bar.set(p),
                      self.elapsed_lbl.configure(text=f"{e:.0f}s / {self.duration}s")))

            time.sleep(0.1)

        if not self._stop_flag:
            sd.wait()

        # Get actual recorded audio
        actual_duration = time.time() - start
        actual_samples = min(int(actual_duration * SAMPLE_RATE), len(audio))
        recorded = audio[:actual_samples].squeeze()

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        filename = "enrollment_5min.wav" if self.mode == "enroll" else "verify_recording.wav"
        path = DATA_DIR / filename
        sf.write(str(path), recorded, SAMPLE_RATE)

        dur_saved = len(recorded) / SAMPLE_RATE
        self._ui(lambda: self._done(path, dur_saved))

    def _done(self, path, dur):
        self.state_lbl.configure(text="DONE", text_color=GREEN)
        self.time_lbl.configure(text=f"{dur:.0f}s SAVED")
        self.bar.set(1.0)
        if self.stop_btn:
            self.stop_btn.configure(state="disabled")
        self.after(600, lambda: self._close(path))

    def _close(self, path):
        self._dead = True
        self.grab_release()
        self.destroy()
        self.callback(path)


# ============================================================
# Main App
# ============================================================

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VOICEPRINT // SPEAKER VERIFICATION")
        self.geometry("1100x900")
        self.configure(fg_color=BG)
        self.minsize(1000, 800)

        self.models_list = [FullModel(), ESP32Model()]
        self.columns = []
        self.enroll_step = 0
        self.enroll_embeddings = {m.name: [] for m in self.models_list}
        self._busy = False

        self._build()

    def _build(self):
        # === TOP BAR ===
        top = ctk.CTkFrame(self, fg_color=BG, height=50)
        top.pack(fill="x")
        top.pack_propagate(False)

        ctk.CTkLabel(top, text="VOICEPRINT",
                     font=ctk.CTkFont(family=MONO, size=18, weight="bold"),
                     text_color=WHITE).pack(side="left", padx=20)
        ctk.CTkLabel(top, text="//  SPEAKER VERIFICATION",
                     font=ctk.CTkFont(family=MONO, size=10), text_color=G4
                     ).pack(side="left", pady=(3, 0))

        self.sys_lbl = ctk.CTkLabel(top, text="",
                                     font=ctk.CTkFont(family=MONO, size=9), text_color=G4)
        self.sys_lbl.pack(side="right", padx=20)

        ctk.CTkFrame(self, fg_color=BORDER, height=1).pack(fill="x")

        # === MAIN SCROLLABLE ===
        main = ctk.CTkScrollableFrame(self, fg_color=BG, corner_radius=0,
                                       scrollbar_button_color=G5,
                                       scrollbar_button_hover_color=G4)
        main.pack(fill="both", expand=True)

        # --- Enrollment instruction ---
        self._section(main, "STEP 1 // LOAD MODELS")

        load_row = ctk.CTkFrame(main, fg_color="transparent")
        load_row.pack(fill="x", padx=20, pady=(0, 12))

        self.btn_load = ctk.CTkButton(
            load_row, text="LOAD ALL MODELS", height=36, corner_radius=6,
            font=ctk.CTkFont(family=MONO, size=12, weight="bold"),
            fg_color=WHITE, hover_color=G1, text_color=BG,
            border_width=1, border_color=WHITE,
            command=self._on_load)
        self.btn_load.pack(side="left", padx=(0, 12))

        self.load_status = ctk.CTkLabel(load_row, text="",
                                         font=ctk.CTkFont(family=MONO, size=10), text_color=G3)
        self.load_status.pack(side="left")

        # --- Model columns ---
        self._section(main, "MODELS")

        cols = ctk.CTkFrame(main, fg_color="transparent")
        cols.pack(fill="x", padx=20, pady=(0, 16))

        for i, m in enumerate(self.models_list):
            col = ModelColumn(cols, m, self)
            col.pack(side="left", expand=True, fill="both", padx=(0 if i == 0 else 6, 0))
            self.columns.append(col)

        # --- Enrollment ---
        self._section(main, "STEP 2 // ENROLL YOUR VOICE  (5 MINUTES)")

        enroll_info = ctk.CTkFrame(main, fg_color=CELL, corner_radius=8,
                                    border_width=1, border_color=BORDER)
        enroll_info.pack(fill="x", padx=20, pady=(0, 8))

        ctk.CTkLabel(enroll_info,
                     text="Record 5 minutes of your voice. Read the on-screen text aloud.\n"
                          "You can stop early (minimum 30 seconds). More audio = better accuracy.\n"
                          "Or load an existing WAV file if you already have a recording.",
                     font=ctk.CTkFont(family=MONO, size=10), text_color=G2,
                     justify="left").pack(padx=16, pady=12)

        # Augmentation toggle
        aug_row = ctk.CTkFrame(main, fg_color="transparent")
        aug_row.pack(fill="x", padx=20, pady=(0, 6))

        self.use_augmentation = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(
            aug_row, text="NOISE AUGMENTATION", variable=self.use_augmentation,
            font=ctk.CTkFont(family=MONO, size=10), text_color=G2,
            fg_color=G5, progress_color=WHITE, button_color=WHITE,
        ).pack(side="left")
        ctk.CTkLabel(aug_row, text="  OFF = clean only (stricter match)  //  ON = noise-tolerant (looser)",
                     font=ctk.CTkFont(family=MONO, size=9), text_color=G4).pack(side="left", padx=8)

        enroll_btns = ctk.CTkFrame(main, fg_color="transparent")
        enroll_btns.pack(fill="x", padx=20, pady=(0, 8))

        bstyle = dict(height=36, corner_radius=6,
                      font=ctk.CTkFont(family=MONO, size=11, weight="bold"),
                      border_width=1, border_color=BORDER)

        self.btn_enroll_rec = ctk.CTkButton(
            enroll_btns, text="RECORD 5 MIN",
            fg_color=G5, text_color=G4, state="disabled",
            command=self._on_enroll_record, **bstyle)
        self.btn_enroll_rec.pack(side="left", padx=(0, 8))

        self.btn_enroll_file = ctk.CTkButton(
            enroll_btns, text="LOAD WAV FILE",
            fg_color=G5, text_color=G4, state="disabled",
            command=self._on_enroll_file, **bstyle)
        self.btn_enroll_file.pack(side="left", padx=(0, 8))

        self.enroll_progress = ctk.CTkProgressBar(enroll_btns, width=200, height=6,
                                                    progress_color=WHITE, fg_color=G5)
        self.enroll_progress.pack(side="left", padx=12, pady=14)
        self.enroll_progress.set(0)

        self.enroll_status = ctk.CTkLabel(enroll_btns, text="NOT ENROLLED",
                                           font=ctk.CTkFont(family=MONO, size=10), text_color=G4)
        self.enroll_status.pack(side="left")

        # --- Verify ---
        self._section(main, "STEP 3 // VERIFY")

        verify_row = ctk.CTkFrame(main, fg_color="transparent")
        verify_row.pack(fill="x", padx=20, pady=(0, 8))

        self.btn_verify_mic = ctk.CTkButton(
            verify_row, text="VERIFY FROM MIC",
            fg_color=G5, text_color=G4, state="disabled",
            command=self._on_verify_mic, **bstyle)
        self.btn_verify_mic.pack(side="left", padx=(0, 8))

        self.btn_verify_file = ctk.CTkButton(
            verify_row, text="VERIFY FROM FILE",
            fg_color=G5, text_color=G4, state="disabled",
            command=self._on_verify_file, **bstyle)
        self.btn_verify_file.pack(side="left")

        self.verify_status = ctk.CTkLabel(verify_row, text="",
                                           font=ctk.CTkFont(family=MONO, size=10), text_color=G3)
        self.verify_status.pack(side="left", padx=12)

        # --- Result ---
        self.result_frame = ctk.CTkFrame(main, fg_color=CELL, corner_radius=8,
                                          border_width=1, border_color=BORDER, height=56)
        self.result_frame.pack(fill="x", padx=20, pady=(8, 20))
        self.result_frame.pack_propagate(False)
        self.result_lbl = ctk.CTkLabel(self.result_frame, text="LOAD MODELS TO BEGIN",
                                        font=ctk.CTkFont(family=MONO, size=14), text_color=G4)
        self.result_lbl.pack(expand=True)

        # --- Bottom ---
        ctk.CTkFrame(self, fg_color=BORDER, height=1).pack(fill="x", side="bottom")
        bot = ctk.CTkFrame(self, fg_color=BG, height=24)
        bot.pack(fill="x", side="bottom")
        bot.pack_propagate(False)
        ctk.CTkLabel(bot, text="THRESHOLD: 0.645  //  EER: 5.1%  //  INT8 PER-ROW QUANTIZED",
                     font=ctk.CTkFont(family=MONO, size=8), text_color=G5
                     ).pack(side="left", padx=20)

        self._update_sys()

    def _section(self, parent, title):
        ctk.CTkLabel(parent, text=title,
                     font=ctk.CTkFont(family=MONO, size=10, weight="bold"),
                     text_color=G3).pack(anchor="w", padx=20, pady=(14, 4))

    def _update_sys(self):
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0)
        self.sys_lbl.configure(
            text=f"CPU {cpu:3.0f}%  //  RAM {mem.used/1024**3:.1f}/{mem.total/1024**3:.0f}GB")
        self.after(2000, self._update_sys)

    def _set_result(self, text, color=G4):
        self.result_lbl.configure(text=text, text_color=color)
        self.result_frame.configure(border_color=color if color != G4 else BORDER)

    # --- Load ---
    def _on_load(self):
        if self._busy:
            return
        self._busy = True
        self.btn_load.configure(text="LOADING...", state="disabled")
        self.load_status.configure(text="Loading 3 models...")
        threading.Thread(target=self._do_load, daemon=True).start()

    def _do_load(self):
        try:
            for i, (m, col) in enumerate(zip(self.models_list, self.columns)):
                self.after(0, lambda c=col: c.set_status("LOADING...", G2))

                result, ms, ram, cpu = measure(m.load)
                info = m.info()

                def update(c=col, ms=ms, ram=ram, cpu=cpu, info=info):
                    c.s_size.set(info.get("size", "?"))
                    c.s_eer.set(info.get("eer", "?"))
                    c.s_load.set(f"{ms:.0f} ms")
                    c.s_ram.set(f"{ram:.0f} MB" if ram > 0.5 else "<1 MB")
                    c.s_cpu.set(f"{cpu:.0f}%")
                    c.set_status("READY", GREEN)
                self.after(0, update)

            self.after(0, self._load_done)
        except Exception as e:
            self.after(0, lambda: self._set_result(f"LOAD ERROR: {e}", RED))
        finally:
            self._busy = False

    def _load_done(self):
        self.btn_load.configure(text="LOADED", fg_color=G5, text_color=G3)
        self.load_status.configure(text="All models ready", text_color=GREEN)
        self.btn_enroll_rec.configure(state="normal", fg_color=WHITE, text_color=BG,
                                      border_color=WHITE)
        self.btn_enroll_file.configure(state="normal", fg_color=BG, text_color=G1,
                                       border_color=G3)
        self._set_result("MODELS LOADED // RECORD 5 MIN OR LOAD A WAV FILE", G2)

    # --- Enroll: record 5 minutes ---
    def _on_enroll_record(self):
        if self._busy:
            return
        result = {"path": None}
        overlay = RecordOverlay(self, ENROLL_DURATION, READING_TEXTS,
                                lambda p: result.update({"path": p}), mode="enroll")
        self.wait_window(overlay)
        if result["path"]:
            self._start_enroll_processing(result["path"])

    def _on_enroll_file(self):
        if self._busy:
            return
        path = filedialog.askopenfilename(
            title="SELECT WAV FOR ENROLLMENT",
            filetypes=[("WAV", "*.wav"), ("All", "*.*")])
        if path:
            self._start_enroll_processing(Path(path))

    def _start_enroll_processing(self, wav_path):
        self._busy = True
        self.btn_enroll_rec.configure(text="PROCESSING...", state="disabled")
        self.btn_enroll_file.configure(text="PROCESSING...", state="disabled")
        self._set_result("BUILDING VOICEPRINT — THIS MAY TAKE A MINUTE...", G2)
        threading.Thread(target=self._do_enroll, args=(wav_path,), daemon=True).start()

    def _do_enroll(self, wav_path):
        """Process enrollment audio — split into segments, embed with all models."""
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                import torchaudio, torch
                audio = torchaudio.functional.resample(
                    torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()

            total_dur = len(audio) / SAMPLE_RATE
            self.after(0, lambda: self.enroll_status.configure(
                text=f"PROCESSING {total_dur:.0f}s AUDIO...", text_color=G2))

            # Noise reduce
            audio = noise_reduce(audio, SAMPLE_RATE)

            # Split into 5s segments with 2.5s hop
            seg_len = 5 * SAMPLE_RATE
            hop_len = int(2.5 * SAMPLE_RATE)
            segments = []
            pos = 0
            while pos + seg_len <= len(audio):
                segments.append(audio[pos:pos + seg_len])
                pos += hop_len

            if not segments:
                # Audio too short — use as single segment
                segments = [audio]

            n_segs = len(segments)
            self.after(0, lambda: self.enroll_status.configure(
                text=f"0/{n_segs} SEGMENTS...", text_color=G2))

            # Embed all segments
            use_aug = self.use_augmentation.get()
            self.enroll_embeddings = {}
            total_embs = {}

            for m in self.models_list:
                if use_aug:
                    # Augmented: clean x3 + 5 noise variants per segment
                    def progress_cb(done, total, n_embs, name=m.name):
                        pct = done / max(total, 1)
                        self.after(0, lambda p=pct, d=done, t=total, n=n_embs, nm=name: (
                            self.enroll_progress.set(p),
                            self.enroll_status.configure(
                                text=f"{nm}: {d}/{t} segs, {n} emb (augmented)...")))

                    centroid, n_embs = augmented_enroll_segments(
                        segments, m.embed, progress_cb=progress_cb)
                else:
                    # Clean only: just embed each segment, average
                    embs = []
                    for i, seg in enumerate(segments):
                        embs.append(m.embed(seg))
                        if (i + 1) % 5 == 0 or i == n_segs - 1:
                            pct = (i + 1) / n_segs
                            self.after(0, lambda p=pct, n=i+1, t=n_segs, nm=m.name: (
                                self.enroll_progress.set(p),
                                self.enroll_status.configure(
                                    text=f"{nm}: {n}/{t} segs (clean)...")))
                    centroid = np.mean(embs, axis=0)
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                    n_embs = len(embs)

                self.enroll_embeddings[m.name] = centroid
                total_embs[m.name] = n_embs

            mode_str = "augmented" if use_aug else "clean"
            self.after(0, lambda: self._enroll_done(n_segs, total_dur, mode_str))
        except Exception as e:
            import traceback; traceback.print_exc()
            self.after(0, lambda: self._set_result(f"ENROLL ERROR: {e}", RED))
            self.after(0, lambda: self.btn_enroll_rec.configure(
                text="RECORD 5 MIN", state="normal"))
            self.after(0, lambda: self.btn_enroll_file.configure(
                text="LOAD WAV FILE", state="normal"))
        finally:
            self._busy = False

    def _enroll_done(self, n_segs, total_dur, mode_str):
        for m, col in zip(self.models_list, self.columns):
            centroid = self.enroll_embeddings.get(m.name)
            if centroid is not None:
                col.enrolled_emb = centroid
                col.set_status(f"ENROLLED ({mode_str})", GREEN)
                np.save(str(DATA_DIR / f"enrolled_{m.name.replace(' ', '_')}.npy"), centroid)

        self.enroll_progress.set(1.0)
        self.enroll_status.configure(
            text=f"DONE — {n_segs} SEGS / {total_dur:.0f}s / {mode_str.upper()}",
            text_color=GREEN)

        self.btn_enroll_rec.configure(text="RE-RECORD", state="normal",
                                      fg_color=BG, text_color=G1, border_color=G3)
        self.btn_enroll_file.configure(text="RE-LOAD FILE", state="normal",
                                       fg_color=BG, text_color=G1, border_color=G3)
        self.btn_verify_mic.configure(state="normal", fg_color=WHITE, text_color=BG,
                                      border_color=WHITE)
        self.btn_verify_file.configure(state="normal", fg_color=BG, text_color=G1,
                                       border_color=G3)
        self._set_result(
            f"ENROLLED // {n_segs} SEGS x 8 AUG // {total_dur:.0f}s // READY TO VERIFY", GREEN)

    # --- Verify ---
    def _on_verify_mic(self):
        if self._busy:
            return
        result = {"path": None}
        overlay = RecordOverlay(self, 5, None,
                                lambda p: result.update({"path": p}), mode="verify")
        self.wait_window(overlay)
        if result["path"]:
            self._run_verify(result["path"])

    def _on_verify_file(self):
        if self._busy:
            return
        path = filedialog.askopenfilename(
            title="SELECT WAV TO VERIFY",
            filetypes=[("WAV", "*.wav"), ("All", "*.*")])
        if path:
            self._run_verify(Path(path))

    def _run_verify(self, wav_path):
        self._busy = True
        self.verify_status.configure(text="VERIFYING...", text_color=G2)
        for col in self.columns:
            col.clear_score()
        threading.Thread(target=self._do_verify, args=(wav_path,), daemon=True).start()

    def _do_verify(self, wav_path):
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                import torchaudio, torch
                audio = torchaudio.functional.resample(
                    torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()

            audio = noise_reduce(audio, SAMPLE_RATE)

            verdicts = []
            for m, col in zip(self.models_list, self.columns):
                if col.enrolled_emb is None:
                    continue
                emb, ms, ram, cpu = measure(m.embed, audio)
                score = cosine_sim(col.enrolled_emb, emb)
                thr = THRESHOLDS.get(m.name, 0.645)
                match = score >= thr

                def update(c=col, ms=ms, cpu=cpu, score=score, match=match):
                    c.s_infer.set(f"{ms:.0f} ms")
                    c.s_cpu.set(f"{cpu:.0f}%")
                    c.set_score(score, match)
                self.after(0, update)
                verdicts.append((m.name, score, match))

            # Summary
            all_match = all(v[2] for v in verdicts)
            scores_str = "  ".join(f"{v[0].split()[0]}:{v[1]:.3f}" for v in verdicts)

            if all_match:
                self.after(0, lambda: self._set_result(f"ALL MATCH  //  {scores_str}", GREEN))
            else:
                self.after(0, lambda: self._set_result(f"MIXED  //  {scores_str}", RED))

            self.after(0, lambda: self.verify_status.configure(text="Done", text_color=G3))
        except Exception as e:
            self.after(0, lambda: self._set_result(f"VERIFY ERROR: {e}", RED))
        finally:
            self._busy = False


if __name__ == "__main__":
    app = App()
    app.mainloop()
