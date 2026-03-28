"""
Resemblyzer → ESP32-S3 Quantization Pipeline
=============================================
Exports Resemblyzer LSTM to ONNX, converts to TFLite int8,
verifies accuracy, and generates C header for ESP32-S3.

Usage:
    python quantize.py

Output:
    artifacts/
      ├── resemblyzer_float32.onnx    # ONNX export
      ├── resemblyzer_float32.tflite   # TFLite float32
      ├── resemblyzer_int8.tflite      # TFLite int8 quantized
      ├── resemblyzer_model.h          # C header for Arduino
      └── accuracy_report.txt          # Accuracy comparison
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)


def step_banner(n, title):
    print(f"\n{'='*60}")
    print(f"  STEP {n}: {title}")
    print(f"{'='*60}\n")


# ============================================================
# STEP 1: Export Resemblyzer to ONNX
# ============================================================

def export_to_onnx():
    step_banner(1, "EXPORT RESEMBLYZER → ONNX")

    import torch
    import torch.nn as nn
    from resemblyzer import VoiceEncoder

    encoder = VoiceEncoder()

    # Resemblyzer's forward pass: LSTM → take last hidden → Linear → L2 normalize
    # We need a wrapper that TFLite can handle
    class ResemblyzerExport(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.lstm = encoder.lstm
            self.linear = encoder.linear
            self.relu = nn.ReLU()

        def forward(self, mel_frames):
            # mel_frames: [1, T, 40]  (T = number of mel frames)
            lstm_out, _ = self.lstm(mel_frames)
            # Take the last timestep output
            last_hidden = lstm_out[:, -1, :]  # [1, 256]
            projection = self.linear(last_hidden)  # [1, 256]
            projection = self.relu(projection)
            # L2 normalize
            norm = torch.norm(projection, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            normalized = projection / norm  # [1, 256]
            return normalized

    model = ResemblyzerExport(encoder)
    model.eval()

    # Dummy input: 160 mel frames (~2 seconds at default settings)
    # Resemblyzer uses 40 mel channels
    dummy = torch.randn(1, 160, 40)

    onnx_path = OUTPUT_DIR / "resemblyzer_float32.onnx"

    torch.onnx.export(
        model, dummy,
        str(onnx_path),
        input_names=["mel_input"],
        output_names=["embedding"],
        opset_version=17,
        dynamo=False,  # Use legacy tracer — more reliable for LSTM
    )

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"  ONNX exported: {onnx_path}")
    print(f"  Size: {size_mb:.2f} MB")

    # Quick verification
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path))
    test_input = np.random.randn(1, 160, 40).astype(np.float32)
    result = sess.run(None, {"mel_input": test_input})[0]
    print(f"  Output shape: {result.shape}")
    print(f"  L2 norm: {np.linalg.norm(result):.4f} (should be ~1.0)")
    print(f"  ONNX export OK")

    return model, onnx_path


# ============================================================
# STEP 2: Convert to TFLite (float32 + int8)
# ============================================================

def convert_to_tflite(onnx_path):
    step_banner(2, "CONVERT ONNX → TFLITE (FLOAT32 + INT8)")

    import tensorflow as tf
    import onnx
    from onnx import numpy_helper

    # --- Method: Build TF model from scratch matching Resemblyzer architecture ---
    # This is more reliable than onnx-to-tf converters for LSTM

    print("  Building TF equivalent model...")

    # Load ONNX to get weights
    import torch
    from resemblyzer import VoiceEncoder
    encoder = VoiceEncoder()
    state = encoder.state_dict()

    # Build Keras model matching Resemblyzer architecture
    # LSTM(40→256, 3 layers, batch_first) + Linear(256→256) + ReLU + L2norm

    # Fixed input shape for TFLite compatibility
    # 160 frames = ~2s audio (16kHz, 10ms hop, 25ms window)
    # ESP32-S3 will record 3s = 300 frames, but we use 160 for smaller arena
    N_FRAMES = 160
    mel_input = tf.keras.Input(shape=(N_FRAMES, 40), name="mel_input")

    # 3-layer LSTM — unrolled for TFLite compatibility
    x = mel_input
    for layer_idx in range(3):
        return_sequences = (layer_idx < 2)
        lstm_layer = tf.keras.layers.LSTM(
            256,
            return_sequences=return_sequences,
            unroll=True,  # Required for TFLite — no dynamic tensor lists
            name=f"lstm_{layer_idx}",
        )
        x = lstm_layer(x)

    # Linear + ReLU
    x = tf.keras.layers.Dense(256, activation="relu", name="projection")(x)

    # L2 normalize
    x = tf.keras.layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=-1),
        name="l2_norm"
    )(x)

    tf_model = tf.keras.Model(inputs=mel_input, outputs=x, name="resemblyzer")
    tf_model.summary()

    # --- Copy PyTorch weights to TF model ---
    print("\n  Copying PyTorch weights to TF model...")

    for layer_idx in range(3):
        tf_lstm = tf_model.get_layer(f"lstm_{layer_idx}")

        # PyTorch LSTM stores: weight_ih [4*H, input], weight_hh [4*H, H], bias_ih, bias_hh
        # Gate order PyTorch: i, f, g, o
        # Gate order TF/Keras: i, f, c(=g), o  — same order!

        w_ih = state[f"lstm.weight_ih_l{layer_idx}"].numpy()  # [1024, input_size]
        w_hh = state[f"lstm.weight_hh_l{layer_idx}"].numpy()  # [1024, 256]
        b_ih = state[f"lstm.bias_ih_l{layer_idx}"].numpy()     # [1024]
        b_hh = state[f"lstm.bias_hh_l{layer_idx}"].numpy()     # [1024]

        # TF kernel = [input_size, 4*H] (transposed from PyTorch)
        # TF recurrent_kernel = [H, 4*H] (transposed from PyTorch)
        # TF bias = bias_ih + bias_hh (combined)

        kernel = w_ih.T          # [input_size, 1024]
        rec_kernel = w_hh.T      # [256, 1024]
        bias = b_ih + b_hh       # [1024]

        tf_lstm.set_weights([kernel, rec_kernel, bias])

    # Copy projection weights
    tf_proj = tf_model.get_layer("projection")
    w_proj = state["linear.weight"].numpy()  # [256, 256]
    b_proj = state["linear.bias"].numpy()    # [256]
    tf_proj.set_weights([w_proj.T, b_proj])  # TF expects [in, out]

    print("  Weights copied successfully")

    # --- Verify TF model matches PyTorch ---
    print("  Verifying TF model output matches PyTorch...")
    np.random.seed(42)
    test_input = np.random.randn(1, 160, 40).astype(np.float32)

    tf_output = tf_model.predict(test_input, verbose=0)

    # PyTorch reference
    import torch
    torch_input = torch.from_numpy(test_input)
    encoder.eval()
    with torch.no_grad():
        lstm_out, _ = encoder.lstm(torch_input)
        last = lstm_out[:, -1, :]
        proj = torch.relu(encoder.linear(last))
        pt_output = (proj / proj.norm(dim=1, keepdim=True)).numpy()

    cos_sim = np.dot(tf_output.flatten(), pt_output.flatten())
    print(f"  TF vs PyTorch cosine similarity: {cos_sim:.6f}")
    if cos_sim > 0.99:
        print("  MATCH — TF model is accurate")
    else:
        print(f"  WARNING — similarity is {cos_sim:.4f}, expected > 0.99")

    # --- Export float32 TFLite ---
    print("\n  Converting to TFLite float32...")
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_float = converter.convert()

    float_path = OUTPUT_DIR / "resemblyzer_float32.tflite"
    float_path.write_bytes(tflite_float)
    print(f"  Float32 TFLite: {float_path} ({len(tflite_float)/1024/1024:.2f} MB)")

    # --- Export int8 quantized TFLite ---
    print("\n  Converting to TFLite int8 (full integer quantization)...")

    def representative_dataset():
        """Generate representative mel-spectrogram data for quantization calibration."""
        np.random.seed(0)
        for _ in range(100):
            # Simulate realistic mel-spectrogram values (log-mel typically -10 to +5)
            data = np.random.randn(1, 160, 40).astype(np.float32) * 3.0
            yield [data]

    converter_int8 = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_int8.representative_dataset = representative_dataset
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_int8.inference_input_type = tf.float32   # Keep float I/O for easier use
    converter_int8.inference_output_type = tf.float32

    tflite_int8 = converter_int8.convert()

    int8_path = OUTPUT_DIR / "resemblyzer_int8.tflite"
    int8_path.write_bytes(tflite_int8)
    print(f"  Int8 TFLite: {int8_path} ({len(tflite_int8)/1024/1024:.2f} MB)")

    # --- Also try dynamic range quantization as fallback ---
    print("\n  Converting to TFLite dynamic range (hybrid)...")
    converter_dyn = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter_dyn.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_dyn = converter_dyn.convert()
    dyn_path = OUTPUT_DIR / "resemblyzer_dynamic.tflite"
    dyn_path.write_bytes(tflite_dyn)
    print(f"  Dynamic TFLite: {dyn_path} ({len(tflite_dyn)/1024/1024:.2f} MB)")

    return float_path, int8_path, dyn_path, tf_model


# ============================================================
# STEP 3: Verify quantized accuracy
# ============================================================

def verify_accuracy(float_path, int8_path, dyn_path, tf_model):
    step_banner(3, "VERIFY QUANTIZED ACCURACY")

    import tensorflow as tf

    def run_tflite(model_path, input_data):
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]

        # Resize if needed for dynamic shape
        if list(input_detail['shape']) != list(input_data.shape):
            interpreter.resize_tensor_input(input_detail['index'], input_data.shape)
            interpreter.allocate_tensors()

        interpreter.set_tensor(input_detail['index'], input_data)
        interpreter.invoke()
        return interpreter.get_tensor(output_detail['index'])

    np.random.seed(42)
    n_tests = 20
    cos_float = []
    cos_int8 = []
    cos_dyn = []

    print(f"  Running {n_tests} comparison tests...")
    print(f"  {'Test':<6} {'Keras':>10} {'Float32':>10} {'Int8':>10} {'Dynamic':>10}")
    print(f"  {'-'*50}")

    for i in range(n_tests):
        test_input = np.random.randn(1, 160, 40).astype(np.float32) * 2.0

        ref = tf_model.predict(test_input, verbose=0).flatten()
        out_f = run_tflite(float_path, test_input).flatten()
        out_i = run_tflite(int8_path, test_input).flatten()
        out_d = run_tflite(dyn_path, test_input).flatten()

        sim_f = np.dot(ref, out_f) / (np.linalg.norm(ref) * np.linalg.norm(out_f) + 1e-8)
        sim_i = np.dot(ref, out_i) / (np.linalg.norm(ref) * np.linalg.norm(out_i) + 1e-8)
        sim_d = np.dot(ref, out_d) / (np.linalg.norm(ref) * np.linalg.norm(out_d) + 1e-8)

        cos_float.append(sim_f)
        cos_int8.append(sim_i)
        cos_dyn.append(sim_d)

        if i < 5 or i == n_tests - 1:
            print(f"  {i+1:<6} {1.0:>10.4f} {sim_f:>10.4f} {sim_i:>10.4f} {sim_d:>10.4f}")

    print(f"\n  {'AVERAGE':<6} {'1.0000':>10} {np.mean(cos_float):>10.4f} "
          f"{np.mean(cos_int8):>10.4f} {np.mean(cos_dyn):>10.4f}")
    print(f"  {'MIN':<6} {'1.0000':>10} {np.min(cos_float):>10.4f} "
          f"{np.min(cos_int8):>10.4f} {np.min(cos_dyn):>10.4f}")

    # Size comparison
    f_size = float_path.stat().st_size
    i_size = int8_path.stat().st_size
    d_size = dyn_path.stat().st_size

    print(f"\n  SIZE COMPARISON:")
    print(f"  {'Float32 TFLite:':<25} {f_size/1024:.0f} KB")
    print(f"  {'Int8 TFLite:':<25} {i_size/1024:.0f} KB  ({i_size/f_size*100:.0f}% of float)")
    print(f"  {'Dynamic TFLite:':<25} {d_size/1024:.0f} KB  ({d_size/f_size*100:.0f}% of float)")

    # ESP32-S3 fit check
    esp32_psram = 8 * 1024 * 1024  # 8MB
    esp32_flash = 8 * 1024 * 1024  # 8MB
    print(f"\n  ESP32-S3 FIT CHECK:")
    print(f"  {'Int8 model:':<25} {i_size/1024:.0f} KB / {esp32_flash/1024:.0f} KB flash = "
          f"{i_size/esp32_flash*100:.1f}%")

    fits = i_size < esp32_flash
    print(f"  {'FITS IN ESP32-S3:':25} {'YES' if fits else 'NO'}")

    # Write report
    report = OUTPUT_DIR / "accuracy_report.txt"
    with open(report, "w") as f:
        f.write("RESEMBLYZER ESP32-S3 QUANTIZATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Original (float32):     {f_size/1024:.0f} KB\n")
        f.write(f"Int8 quantized:         {i_size/1024:.0f} KB\n")
        f.write(f"Dynamic range:          {d_size/1024:.0f} KB\n\n")
        f.write(f"Accuracy vs reference (cosine similarity):\n")
        f.write(f"  Float32 TFLite avg:   {np.mean(cos_float):.6f}\n")
        f.write(f"  Int8 TFLite avg:      {np.mean(cos_int8):.6f}\n")
        f.write(f"  Dynamic TFLite avg:   {np.mean(cos_dyn):.6f}\n\n")
        f.write(f"ESP32-S3 compatible:    {'YES' if fits else 'NO'}\n")

    print(f"\n  Report saved: {report}")
    return i_size


# ============================================================
# STEP 4: Generate C header for ESP32-S3
# ============================================================

def generate_c_header(int8_path):
    step_banner(4, "GENERATE C HEADER FOR ESP32-S3")

    model_bytes = int8_path.read_bytes()
    header_path = OUTPUT_DIR / "resemblyzer_model.h"

    with open(header_path, "w") as f:
        f.write("// Auto-generated — Resemblyzer int8 TFLite model for ESP32-S3\n")
        f.write(f"// Model size: {len(model_bytes)} bytes ({len(model_bytes)/1024:.0f} KB)\n")
        f.write(f"// Architecture: LSTM(40→256, 3 layers) + Dense(256) + L2Norm\n")
        f.write(f"// Input: mel-spectrogram [1, N_FRAMES, 40]\n")
        f.write(f"// Output: speaker embedding [1, 256]\n\n")
        f.write("#ifndef RESEMBLYZER_MODEL_H\n")
        f.write("#define RESEMBLYZER_MODEL_H\n\n")
        f.write(f"const unsigned int resemblyzer_model_len = {len(model_bytes)};\n\n")
        f.write("alignas(16) const unsigned char resemblyzer_model[] = {\n")

        for i in range(0, len(model_bytes), 16):
            chunk = model_bytes[i:i + 16]
            hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    {hex_str},\n")

        f.write("};\n\n")
        f.write("#endif // RESEMBLYZER_MODEL_H\n")

    print(f"  C header generated: {header_path}")
    print(f"  Array size: {len(model_bytes)} bytes")

    return header_path


# ============================================================
# STEP 5: Generate ESP32-S3 Arduino sketch
# ============================================================

def generate_esp32_sketch():
    step_banner(5, "GENERATE ESP32-S3 ARDUINO SKETCH")

    sketch_dir = OUTPUT_DIR / "speaker_verify_esp32"
    sketch_dir.mkdir(exist_ok=True)

    sketch_path = sketch_dir / "speaker_verify_esp32.ino"
    with open(sketch_path, "w") as f:
        f.write(r"""/*
 * Speaker Verification on ESP32-S3
 * ================================
 * Uses quantized Resemblyzer model (int8 TFLite) for speaker verification.
 *
 * Hardware: XIAO ESP32-S3 Sense (or any ESP32-S3 with I2S mic)
 * Model:   Resemblyzer LSTM — 3-layer LSTM(40→256) + Dense(256)
 * Input:   40-channel mel spectrogram from microphone
 * Output:  256-dim speaker embedding → cosine similarity match
 *
 * Workflow:
 *   1. Hold BOOT button → enroll (records 3s, saves embedding to flash)
 *   2. Speak normally → verify (compares against enrolled embedding)
 *   3. LED: GREEN = match, RED = no match
 */

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <driver/i2s.h>
#include <SPIFFS.h>
#include <math.h>

// Include the quantized model
#include "resemblyzer_model.h"

// ===== CONFIG =====
#define SAMPLE_RATE       16000
#define RECORD_SECONDS    3
#define NUM_SAMPLES       (SAMPLE_RATE * RECORD_SECONDS)

// Mel spectrogram params (must match training)
#define N_MEL_CHANNELS    40
#define HOP_LENGTH        160    // 10ms hop at 16kHz
#define WIN_LENGTH        400    // 25ms window at 16kHz
#define N_FFT             512
#define NUM_FRAMES        (NUM_SAMPLES / HOP_LENGTH)

// Speaker verification
#define EMBEDDING_DIM     256
#define MATCH_THRESHOLD   0.75f

// TFLite arena — adjust if OOM
#define TENSOR_ARENA_SIZE (512 * 1024)  // 512KB

// I2S pins (XIAO ESP32-S3 Sense)
#define I2S_WS   42
#define I2S_SD   41
#define I2S_SCK  -1  // PDM mode, no clock pin

// LED pin
#define LED_PIN  21

// ===== GLOBALS =====
static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));
static tflite::AllOpsResolver resolver;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;

static float enrolled_embedding[EMBEDDING_DIM];
static bool has_enrollment = false;
static int16_t audio_buffer[NUM_SAMPLES];

// ===== MEL FILTERBANK (precomputed) =====
// Simplified mel filterbank — triangular filters from 0-8000Hz
// In production, compute proper mel filterbank or use ESP-DSP
static float mel_filterbank[N_MEL_CHANNELS][N_FFT / 2 + 1];

void compute_mel_filterbank() {
    float f_min = 0.0f;
    float f_max = SAMPLE_RATE / 2.0f;

    // Mel scale conversion
    auto hz_to_mel = [](float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); };
    auto mel_to_hz = [](float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); };

    float mel_min = hz_to_mel(f_min);
    float mel_max = hz_to_mel(f_max);

    // N_MEL_CHANNELS + 2 equally spaced points in mel scale
    float mel_points[N_MEL_CHANNELS + 2];
    for (int i = 0; i < N_MEL_CHANNELS + 2; i++) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (N_MEL_CHANNELS + 1);
    }

    // Convert back to Hz and then to FFT bin indices
    float bin_points[N_MEL_CHANNELS + 2];
    for (int i = 0; i < N_MEL_CHANNELS + 2; i++) {
        float hz = mel_to_hz(mel_points[i]);
        bin_points[i] = hz * (N_FFT + 1) / SAMPLE_RATE;
    }

    // Create triangular filters
    memset(mel_filterbank, 0, sizeof(mel_filterbank));
    for (int m = 0; m < N_MEL_CHANNELS; m++) {
        int f_left   = (int)bin_points[m];
        int f_center = (int)bin_points[m + 1];
        int f_right  = (int)bin_points[m + 2];

        for (int k = f_left; k <= f_center && k <= N_FFT / 2; k++) {
            if (f_center != f_left)
                mel_filterbank[m][k] = (float)(k - f_left) / (f_center - f_left);
        }
        for (int k = f_center; k <= f_right && k <= N_FFT / 2; k++) {
            if (f_right != f_center)
                mel_filterbank[m][k] = (float)(f_right - k) / (f_right - f_center);
        }
    }
}

// ===== SIMPLE FFT (Cooley-Tukey radix-2) =====
void fft(float* real, float* imag, int n) {
    // Bit reversal
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
    }
    // FFT
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * M_PI / len;
        float wreal = cosf(ang), wimag = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cur_r = 1.0f, cur_i = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                float ur = real[i + j], ui = imag[i + j];
                float vr = real[i + j + len/2] * cur_r - imag[i + j + len/2] * cur_i;
                float vi = real[i + j + len/2] * cur_i + imag[i + j + len/2] * cur_r;
                real[i + j] = ur + vr;
                imag[i + j] = ui + vi;
                real[i + j + len/2] = ur - vr;
                imag[i + j + len/2] = ui - vi;
                float new_r = cur_r * wreal - cur_i * wimag;
                cur_i = cur_r * wimag + cur_i * wreal;
                cur_r = new_r;
            }
        }
    }
}

// ===== COMPUTE MEL SPECTROGRAM =====
void compute_mel_spectrogram(int16_t* audio, int num_samples, float* mel_output) {
    static float fft_real[N_FFT];
    static float fft_imag[N_FFT];
    static float hann_window[WIN_LENGTH];
    static bool window_init = false;

    if (!window_init) {
        for (int i = 0; i < WIN_LENGTH; i++) {
            hann_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (WIN_LENGTH - 1)));
        }
        window_init = true;
    }

    int num_frames = (num_samples - WIN_LENGTH) / HOP_LENGTH + 1;

    for (int frame = 0; frame < num_frames && frame < NUM_FRAMES; frame++) {
        int start = frame * HOP_LENGTH;

        // Apply window + zero-pad to N_FFT
        memset(fft_real, 0, sizeof(fft_real));
        memset(fft_imag, 0, sizeof(fft_imag));
        for (int i = 0; i < WIN_LENGTH && (start + i) < num_samples; i++) {
            fft_real[i] = (audio[start + i] / 32768.0f) * hann_window[i];
        }

        // FFT
        fft(fft_real, fft_imag, N_FFT);

        // Power spectrum
        float power[N_FFT / 2 + 1];
        for (int k = 0; k <= N_FFT / 2; k++) {
            power[k] = fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k];
        }

        // Apply mel filterbank
        for (int m = 0; m < N_MEL_CHANNELS; m++) {
            float sum = 0.0f;
            for (int k = 0; k <= N_FFT / 2; k++) {
                sum += mel_filterbank[m][k] * power[k];
            }
            // Log mel (with floor to avoid log(0))
            mel_output[frame * N_MEL_CHANNELS + m] = logf(fmaxf(sum, 1e-10f));
        }
    }
}

// ===== COSINE SIMILARITY =====
float cosine_similarity(float* a, float* b, int dim) {
    float dot = 0, norm_a = 0, norm_b = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-8f);
}

// ===== I2S MICROPHONE =====
void i2s_init() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 1024,
        .use_apll = false,
    };
    i2s_pin_config_t pins = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = -1,
        .data_in_num = I2S_SD,
    };
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pins);
}

void record_audio() {
    size_t bytes_read;
    int samples_read = 0;
    Serial.println("[REC] Recording...");
    while (samples_read < NUM_SAMPLES) {
        int16_t buffer[512];
        i2s_read(I2S_NUM_0, buffer, sizeof(buffer), &bytes_read, portMAX_DELAY);
        int count = bytes_read / sizeof(int16_t);
        for (int i = 0; i < count && samples_read < NUM_SAMPLES; i++) {
            audio_buffer[samples_read++] = buffer[i];
        }
    }
    Serial.printf("[REC] Done — %d samples\n", samples_read);
}

// ===== RUN INFERENCE =====
bool run_inference(float* embedding_out) {
    // Compute mel spectrogram
    static float mel_data[NUM_FRAMES * N_MEL_CHANNELS];
    compute_mel_spectrogram(audio_buffer, NUM_SAMPLES, mel_data);

    // Copy to input tensor
    TfLiteTensor* input = interpreter->input(0);
    memcpy(input->data.f, mel_data, NUM_FRAMES * N_MEL_CHANNELS * sizeof(float));

    // Run model
    unsigned long t0 = millis();
    TfLiteStatus status = interpreter->Invoke();
    unsigned long dt = millis() - t0;

    if (status != kTfLiteOk) {
        Serial.println("[ERR] Inference failed");
        return false;
    }

    Serial.printf("[INF] Inference: %lu ms\n", dt);

    // Copy output
    TfLiteTensor* output = interpreter->output(0);
    memcpy(embedding_out, output->data.f, EMBEDDING_DIM * sizeof(float));
    return true;
}

// ===== SPIFFS: Save/Load enrollment =====
void save_enrollment() {
    File f = SPIFFS.open("/enrolled.bin", FILE_WRITE);
    if (f) {
        f.write((uint8_t*)enrolled_embedding, sizeof(enrolled_embedding));
        f.close();
        Serial.println("[SAV] Enrollment saved to flash");
    }
}

bool load_enrollment() {
    File f = SPIFFS.open("/enrolled.bin", FILE_READ);
    if (f && f.size() == sizeof(enrolled_embedding)) {
        f.read((uint8_t*)enrolled_embedding, sizeof(enrolled_embedding));
        f.close();
        Serial.println("[LOD] Enrollment loaded from flash");
        return true;
    }
    return false;
}

// ===== SETUP =====
void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n========================================");
    Serial.println("  VOICEPRINT — ESP32-S3 Speaker Verify");
    Serial.println("========================================\n");

    // Init SPIFFS
    if (!SPIFFS.begin(true)) {
        Serial.println("[ERR] SPIFFS init failed");
    }

    // Init I2S mic
    i2s_init();

    // Compute mel filterbank
    compute_mel_filterbank();

    // Init TFLite
    model = tflite::GetModel(resemblyzer_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[ERR] Model version mismatch");
        return;
    }

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[ERR] AllocateTensors failed");
        return;
    }

    Serial.printf("[SYS] Arena used: %zu / %d bytes\n",
                  interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);

    // Load existing enrollment
    has_enrollment = load_enrollment();

    if (has_enrollment) {
        Serial.println("[SYS] Ready — speak to verify");
    } else {
        Serial.println("[SYS] No enrollment — hold BOOT to enroll");
    }

    // LED
    pinMode(LED_PIN, OUTPUT);
    pinMode(0, INPUT_PULLUP);  // BOOT button
}

// ===== MAIN LOOP =====
void loop() {
    // Check BOOT button for enrollment
    if (digitalRead(0) == LOW) {
        Serial.println("\n>>> ENROLLMENT MODE <<<");
        delay(500);  // Debounce

        record_audio();

        float embedding[EMBEDDING_DIM];
        if (run_inference(embedding)) {
            memcpy(enrolled_embedding, embedding, sizeof(enrolled_embedding));
            has_enrollment = true;
            save_enrollment();

            Serial.println("[ENR] Enrollment complete!");
            // Flash LED 3 times
            for (int i = 0; i < 3; i++) {
                digitalWrite(LED_PIN, HIGH); delay(200);
                digitalWrite(LED_PIN, LOW);  delay(200);
            }
        }
        return;
    }

    // Continuous verification mode
    if (!has_enrollment) {
        delay(100);
        return;
    }

    // Simple voice activity detection: check RMS of short buffer
    int16_t vad_buf[1024];
    size_t bytes_read;
    i2s_read(I2S_NUM_0, vad_buf, sizeof(vad_buf), &bytes_read, portMAX_DELAY);

    float rms = 0;
    int count = bytes_read / sizeof(int16_t);
    for (int i = 0; i < count; i++) {
        rms += (float)vad_buf[i] * vad_buf[i];
    }
    rms = sqrtf(rms / count);

    if (rms < 500) {  // Silence threshold — tune for your mic
        return;
    }

    Serial.println("\n>>> VOICE DETECTED — VERIFYING <<<");
    record_audio();

    float embedding[EMBEDDING_DIM];
    if (run_inference(embedding)) {
        float score = cosine_similarity(enrolled_embedding, embedding, EMBEDDING_DIM);
        Serial.printf("[VER] Score: %.4f (threshold: %.2f)\n", score, MATCH_THRESHOLD);

        if (score >= MATCH_THRESHOLD) {
            Serial.println("[VER] >>> MATCH <<<");
            digitalWrite(LED_PIN, HIGH);
            delay(2000);
            digitalWrite(LED_PIN, LOW);
        } else {
            Serial.println("[VER] >>> REJECTED <<<");
            // Quick blink = rejected
            for (int i = 0; i < 5; i++) {
                digitalWrite(LED_PIN, HIGH); delay(100);
                digitalWrite(LED_PIN, LOW);  delay(100);
            }
        }
    }

    delay(500);
}
""")

    print(f"  Arduino sketch: {sketch_path}")
    print(f"\n  To use:")
    print(f"  1. Copy resemblyzer_model.h into the sketch folder")
    print(f"  2. Install 'TensorFlowLite_ESP32' library in Arduino IDE")
    print(f"  3. Select board: 'XIAO ESP32S3 Sense'")
    print(f"  4. Upload and open Serial Monitor (115200 baud)")
    print(f"  5. Hold BOOT button → speak for 3s → enrollment saved")
    print(f"  6. Speak normally → LED lights up if match")

    return sketch_path


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  VOICEPRINT — RESEMBLYZER → ESP32-S3 PIPELINE")
    print("  Model: LSTM(40→256, 3 layers) + Dense(256)")
    print(f"  Original: ~5.43 MB (float32)")
    print(f"  Target:   ~1.36 MB (int8)")
    print("=" * 60)

    t_start = time.time()

    # Step 1
    model_pt, onnx_path = export_to_onnx()

    # Step 2
    float_path, int8_path, dyn_path, tf_model = convert_to_tflite(onnx_path)

    # Step 3
    verify_accuracy(float_path, int8_path, dyn_path, tf_model)

    # Step 4
    header_path = generate_c_header(int8_path)

    # Step 5
    sketch_path = generate_esp32_sketch()

    t_total = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  DONE — Total time: {t_total:.1f}s")
    print(f"{'='*60}")
    print(f"\n  Output files:")
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"    {f.relative_to(OUTPUT_DIR):<40} {size/1024:>8.0f} KB")
    print()


if __name__ == "__main__":
    main()
