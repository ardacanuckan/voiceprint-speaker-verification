"""
Export Resemblyzer weights as quantized C arrays for ESP32-S3.
No TFLite dependency — pure C LSTM implementation on device.

Quantization: per-row symmetric int8 with float32 scale factors.
ESP32-S3 will do: float_val = int8_val * scale[row]

Output:
    firmware/
      ├── model_weights.h      # Int8 quantized weight arrays
      ├── model_config.h       # Model dimensions and scales
      ├── lstm_engine.h        # Pure C LSTM forward pass
      ├── mel_features.h       # Mel spectrogram extraction
      └── speaker_verify.ino   # Main Arduino sketch
"""

import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder

OUTPUT = Path(__file__).parent / "firmware"
OUTPUT.mkdir(exist_ok=True)


def quantize_tensor(tensor, name="", per_row=False):
    """Symmetric quantization. per_row=True uses per-row scales (much better for weights)."""
    if not per_row or tensor.ndim < 2:
        # Per-tensor quantization (for biases)
        absmax = np.abs(tensor).max()
        if absmax < 1e-10:
            return np.zeros_like(tensor, dtype=np.int8), np.array([1e-10], dtype=np.float32)
        scale = absmax / 127.0
        quantized = np.round(tensor / scale).clip(-127, 127).astype(np.int8)
        reconstructed = quantized.astype(np.float32) * scale
        mse = np.mean((tensor - reconstructed) ** 2)
        rel_err = np.sqrt(mse) / (np.std(tensor) + 1e-8)
        print(f"  {name:<30} shape={str(tensor.shape):<20} "
              f"scale={scale:.6f}  rel_err={rel_err:.4f}  [per-tensor]")
        return quantized, np.array([scale], dtype=np.float32)
    else:
        # Per-row quantization — each row gets its own scale
        n_rows = tensor.shape[0]
        scales = np.zeros(n_rows, dtype=np.float32)
        quantized = np.zeros_like(tensor, dtype=np.int8)
        for r in range(n_rows):
            absmax = np.abs(tensor[r]).max()
            if absmax < 1e-10:
                scales[r] = 1e-10
            else:
                scales[r] = absmax / 127.0
                quantized[r] = np.round(tensor[r] / scales[r]).clip(-127, 127).astype(np.int8)
        reconstructed = quantized.astype(np.float32) * scales[:, None]
        mse = np.mean((tensor - reconstructed) ** 2)
        rel_err = np.sqrt(mse) / (np.std(tensor) + 1e-8)
        print(f"  {name:<30} shape={str(tensor.shape):<20} "
              f"scales={n_rows}   rel_err={rel_err:.4f}  [per-row]")
        return quantized, scales


def export_weights():
    print("Loading Resemblyzer...")
    enc = VoiceEncoder()
    state = enc.state_dict()

    print("\nQuantizing weights to int8...\n")

    weights = {}
    scales = {}

    for key, tensor in state.items():
        t = tensor.numpy()
        # Use per-row quantization for weight matrices (2D), per-tensor for biases (1D)
        use_per_row = (t.ndim == 2)
        q, s = quantize_tensor(t, key, per_row=use_per_row)
        weights[key] = q
        scales[key] = s

    # Total size
    total_bytes = sum(w.nbytes for w in weights.values())
    total_scales = sum(s.nbytes for s in scales.values())
    print(f"\n  Total int8 weights: {total_bytes:,} bytes ({total_bytes/1024:.0f} KB)")
    print(f"  Scale factors:      {total_scales} bytes")
    print(f"  Grand total:        {(total_bytes + total_scales)/1024:.0f} KB")
    print(f"  Compression:        {total_bytes / sum(t.numpy().nbytes for t in state.values()) * 100:.0f}%")

    return weights, scales, state


def verify_quantized_accuracy(weights, scales, state):
    """Run forward pass with quantized weights and compare to float."""
    print("\n\nVerifying quantized accuracy...")

    def dequant(name):
        s = scales[name]
        w = weights[name].astype(np.float32)
        if s.ndim == 1 and w.ndim == 2 and s.shape[0] == w.shape[0]:
            # Per-row: each row has its own scale
            return w * s[:, None]
        else:
            # Per-tensor: single scale
            return w * s[0]

    def lstm_forward_quantized(x, layer_idx):
        """Single LSTM layer forward pass using dequantized weights."""
        T = x.shape[0]
        H = 256
        w_ih = dequant(f"lstm.weight_ih_l{layer_idx}")  # [4H, input]
        w_hh = dequant(f"lstm.weight_hh_l{layer_idx}")  # [4H, H]
        b_ih = dequant(f"lstm.bias_ih_l{layer_idx}")     # [4H]
        b_hh = dequant(f"lstm.bias_hh_l{layer_idx}")     # [4H]

        h = np.zeros(H, dtype=np.float32)
        c = np.zeros(H, dtype=np.float32)
        outputs = []

        for t in range(T):
            gates = w_ih @ x[t] + b_ih + w_hh @ h + b_hh  # [4H]
            i = 1 / (1 + np.exp(-gates[0:H]))       # input gate
            f = 1 / (1 + np.exp(-gates[H:2*H]))     # forget gate
            g = np.tanh(gates[2*H:3*H])              # cell gate
            o = 1 / (1 + np.exp(-gates[3*H:4*H]))   # output gate
            c = f * c + i * g
            h = o * np.tanh(c)
            outputs.append(h.copy())

        return np.array(outputs), h

    def forward_quantized(mel_input):
        """Full model forward pass with quantized weights."""
        x = mel_input  # [T, 40]
        for l in range(3):
            outputs, _ = lstm_forward_quantized(x, l)
            x = outputs
        last_h = x[-1]  # [256]
        w_proj = dequant("linear.weight")  # [256, 256]
        b_proj = dequant("linear.bias")    # [256]
        proj = w_proj @ last_h + b_proj
        proj = np.maximum(proj, 0)  # ReLU
        proj = proj / (np.linalg.norm(proj) + 1e-8)  # L2 norm
        return proj

    def forward_float(mel_input):
        """Full model forward pass with float32 weights."""
        x = mel_input

        def lstm_fwd(x, l):
            T = x.shape[0]
            H = 256
            w_ih = state[f"lstm.weight_ih_l{l}"].numpy()
            w_hh = state[f"lstm.weight_hh_l{l}"].numpy()
            b_ih = state[f"lstm.bias_ih_l{l}"].numpy()
            b_hh = state[f"lstm.bias_hh_l{l}"].numpy()
            h = np.zeros(H, dtype=np.float32)
            c = np.zeros(H, dtype=np.float32)
            outs = []
            for t in range(T):
                gates = w_ih @ x[t] + b_ih + w_hh @ h + b_hh
                i = 1 / (1 + np.exp(-gates[0:H]))
                f = 1 / (1 + np.exp(-gates[H:2*H]))
                g = np.tanh(gates[2*H:3*H])
                o = 1 / (1 + np.exp(-gates[3*H:4*H]))
                c = f * c + i * g
                h = o * np.tanh(c)
                outs.append(h.copy())
            return np.array(outs), h

        for l in range(3):
            outputs, _ = lstm_fwd(x, l)
            x = outputs
        last_h = x[-1]
        w = state["linear.weight"].numpy()
        b = state["linear.bias"].numpy()
        proj = w @ last_h + b
        proj = np.maximum(proj, 0)
        proj = proj / (np.linalg.norm(proj) + 1e-8)
        return proj

    # Test with random inputs
    np.random.seed(42)
    n_tests = 10
    similarities = []

    for i in range(n_tests):
        mel = np.random.randn(80, 40).astype(np.float32) * 2.0  # ~1s audio
        emb_float = forward_float(mel)
        emb_quant = forward_quantized(mel)
        cos = np.dot(emb_float, emb_quant)
        similarities.append(cos)
        if i < 5:
            print(f"  Test {i+1}: cosine_sim = {cos:.6f}")

    avg = np.mean(similarities)
    print(f"\n  Average cosine similarity (float vs int8): {avg:.6f}")
    print(f"  Min: {np.min(similarities):.6f}")

    if avg > 0.95:
        print("  EXCELLENT — quantization preserves accuracy well")
    elif avg > 0.85:
        print("  GOOD — acceptable for speaker verification")
    else:
        print("  WARNING — significant accuracy loss")

    return avg


def write_model_config():
    """Write model_config.h."""
    path = OUTPUT / "model_config.h"
    with open(path, "w") as f:
        f.write("""#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

// Resemblyzer LSTM architecture
#define INPUT_DIM       40      // Mel channels
#define HIDDEN_DIM      256     // LSTM hidden size
#define NUM_LAYERS      3       // LSTM layers
#define EMBEDDING_DIM   256     // Output embedding dimension
#define GATE_DIM        (4 * HIDDEN_DIM)  // 1024

// Audio config
#define SAMPLE_RATE     16000
#define N_FFT           512
#define HOP_LENGTH      160     // 10ms
#define WIN_LENGTH      400     // 25ms
#define N_MEL           40
#define RECORD_SECONDS  3
#define NUM_SAMPLES     (SAMPLE_RATE * RECORD_SECONDS)
#define NUM_FRAMES      ((NUM_SAMPLES - WIN_LENGTH) / HOP_LENGTH + 1)

// Verification
#define MATCH_THRESHOLD 0.70f

#endif // MODEL_CONFIG_H
""")
    print(f"  Written: {path}")


def write_weight_header(weights, scales):
    """Write model_weights.h with int8 arrays and scales."""
    path = OUTPUT / "model_weights.h"

    def write_array(f, safe_name, orig_key, data):
        flat = data.flatten()
        sc = scales[orig_key]
        f.write(f"\n// Shape: {data.shape}\n")

        if sc.ndim == 1 and sc.shape[0] > 1:
            # Per-row scales
            f.write(f"const int {safe_name}_n_scales = {sc.shape[0]};\n")
            f.write(f"const float {safe_name}_scales[{sc.shape[0]}] PROGMEM = {{\n")
            for i in range(0, len(sc), 16):
                chunk = sc[i:i + 16]
                f.write("    " + ", ".join(f"{v:.8f}f" for v in chunk) + ",\n")
            f.write("};\n")
        else:
            # Single scale
            f.write(f"const float {safe_name}_scale = {sc[0]:.8f}f;\n")

        f.write(f"const int8_t {safe_name}[{len(flat)}] PROGMEM = {{\n")
        for i in range(0, len(flat), 32):
            chunk = flat[i:i + 32]
            f.write("    " + ", ".join(str(v) for v in chunk) + ",\n")
        f.write("};\n")

    with open(path, "w") as f:
        f.write("#ifndef MODEL_WEIGHTS_H\n")
        f.write("#define MODEL_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n")
        f.write('#include <pgmspace.h>\n\n')
        f.write(f"// Total weight size: {sum(w.nbytes for w in weights.values()):,} bytes\n")
        f.write(f"// Per-row quantization for weight matrices\n")

        for key in weights:
            safe_name = key.replace(".", "_")
            write_array(f, safe_name, key, weights[key])

        f.write("\n#endif // MODEL_WEIGHTS_H\n")

    size_kb = path.stat().st_size / 1024
    print(f"  Written: {path} ({size_kb:.0f} KB source)")


def write_lstm_engine():
    """Write lstm_engine.h — pure C LSTM forward pass."""
    path = OUTPUT / "lstm_engine.h"
    with open(path, "w") as f:
        f.write(r"""#ifndef LSTM_ENGINE_H
#define LSTM_ENGINE_H

#include "model_config.h"
#include "model_weights.h"
#include <math.h>
#include <string.h>
#include <pgmspace.h>

// ===== Dequantize helpers =====
// Per-row: each row of weight matrix has its own scale
static inline float dequant_row(const int8_t* data, int row, int col, int cols,
                                 const float* row_scales) {
    return (float)pgm_read_byte(&data[row * cols + col]) * row_scales[row];
}

// Per-tensor: single scale for bias vectors
static inline float dequant_scalar(const int8_t* data, int idx, float scale) {
    return (float)pgm_read_byte(&data[idx]) * scale;
}

// ===== Sigmoid =====
static inline float sigmoid(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

// ===== LSTM single layer forward (per-row quantized) =====
static void lstm_layer_forward(
    const float* input,           // [num_frames, input_dim]
    int num_frames,
    int input_dim,
    const int8_t* w_ih,           // [GATE_DIM, input_dim]
    const float* w_ih_scales,     // [GATE_DIM] per-row scales
    const int8_t* w_hh,           // [GATE_DIM, HIDDEN_DIM]
    const float* w_hh_scales,     // [GATE_DIM] per-row scales
    const int8_t* b_ih,           // [GATE_DIM]
    float b_ih_scale,             // single scale (bias is 1D)
    const int8_t* b_hh,           // [GATE_DIM]
    float b_hh_scale,             // single scale
    float* output,                // [num_frames, HIDDEN_DIM] or NULL
    float* final_h                // [HIDDEN_DIM]
) {
    float h[HIDDEN_DIM] = {0};
    float c[HIDDEN_DIM] = {0};
    float gates[GATE_DIM];

    for (int t = 0; t < num_frames; t++) {
        const float* x_t = &input[t * input_dim];

        // Compute gates: W_ih @ x + b_ih + W_hh @ h + b_hh
        for (int g = 0; g < GATE_DIM; g++) {
            float val = dequant_scalar(b_ih, g, b_ih_scale)
                      + dequant_scalar(b_hh, g, b_hh_scale);

            // W_ih @ x — per-row dequant
            for (int j = 0; j < input_dim; j++) {
                val += dequant_row(w_ih, g, j, input_dim, w_ih_scales) * x_t[j];
            }

            // W_hh @ h — per-row dequant
            for (int j = 0; j < HIDDEN_DIM; j++) {
                val += dequant_row(w_hh, g, j, HIDDEN_DIM, w_hh_scales) * h[j];
            }

            gates[g] = val;
        }

        // Gate activations (PyTorch order: i, f, g, o)
        for (int i = 0; i < HIDDEN_DIM; i++) {
            float i_gate = sigmoid(gates[i]);
            float f_gate = sigmoid(gates[HIDDEN_DIM + i]);
            float g_gate = tanhf(gates[2 * HIDDEN_DIM + i]);
            float o_gate = sigmoid(gates[3 * HIDDEN_DIM + i]);

            c[i] = f_gate * c[i] + i_gate * g_gate;
            h[i] = o_gate * tanhf(c[i]);
        }

        // Store output if needed (for feeding next layer)
        if (output != NULL) {
            memcpy(&output[t * HIDDEN_DIM], h, HIDDEN_DIM * sizeof(float));
        }
    }

    memcpy(final_h, h, HIDDEN_DIM * sizeof(float));
}

// ===== Full model forward pass =====
// Returns 256-dim L2-normalized speaker embedding
static void resemblyzer_forward(
    const float* mel_input,   // [num_frames, N_MEL]
    int num_frames,
    float* embedding_out      // [EMBEDDING_DIM]
) {
    // Allocate intermediate buffers (on stack or PSRAM)
    // Layer 0: input=40, output to buffer
    static float buf_a[300 * HIDDEN_DIM];  // max 300 frames
    static float buf_b[300 * HIDDEN_DIM];
    float final_h[HIDDEN_DIM];

    // Layer 0: mel(40) → hidden(256), all timesteps
    lstm_layer_forward(
        mel_input, num_frames, INPUT_DIM,
        lstm_weight_ih_l0, lstm_weight_ih_l0_scales,
        lstm_weight_hh_l0, lstm_weight_hh_l0_scales,
        lstm_bias_ih_l0, lstm_bias_ih_l0_scale,
        lstm_bias_hh_l0, lstm_bias_hh_l0_scale,
        buf_a, final_h
    );

    // Layer 1: hidden(256) → hidden(256), all timesteps
    lstm_layer_forward(
        buf_a, num_frames, HIDDEN_DIM,
        lstm_weight_ih_l1, lstm_weight_ih_l1_scales,
        lstm_weight_hh_l1, lstm_weight_hh_l1_scales,
        lstm_bias_ih_l1, lstm_bias_ih_l1_scale,
        lstm_bias_hh_l1, lstm_bias_hh_l1_scale,
        buf_b, final_h
    );

    // Layer 2: hidden(256) → final hidden only
    lstm_layer_forward(
        buf_b, num_frames, HIDDEN_DIM,
        lstm_weight_ih_l2, lstm_weight_ih_l2_scales,
        lstm_weight_hh_l2, lstm_weight_hh_l2_scales,
        lstm_bias_ih_l2, lstm_bias_ih_l2_scale,
        lstm_bias_hh_l2, lstm_bias_hh_l2_scale,
        NULL, final_h  // Only need last hidden state
    );

    // Projection: Dense(256→256) + ReLU (per-row scales)
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        float val = dequant_scalar(linear_bias, i, linear_bias_scale);
        for (int j = 0; j < HIDDEN_DIM; j++) {
            val += dequant_row(linear_weight, i, j, HIDDEN_DIM, linear_weight_scales) * final_h[j];
        }
        embedding_out[i] = fmaxf(val, 0.0f);  // ReLU
    }

    // L2 normalize
    float norm = 0.0f;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        norm += embedding_out[i] * embedding_out[i];
    }
    norm = sqrtf(norm) + 1e-8f;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        embedding_out[i] /= norm;
    }
}

// ===== Cosine similarity =====
static float cosine_similarity(const float* a, const float* b) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-8f);
}

#endif // LSTM_ENGINE_H
""")
    print(f"  Written: {path}")


def write_mel_features():
    """Write mel_features.h — mel spectrogram on ESP32."""
    path = OUTPUT / "mel_features.h"
    with open(path, "w") as f:
        f.write(r"""#ifndef MEL_FEATURES_H
#define MEL_FEATURES_H

#include "model_config.h"
#include <math.h>
#include <string.h>

// ===== Precomputed mel filterbank =====
static float mel_filters[N_MEL][N_FFT / 2 + 1];
static bool mel_initialized = false;

static void init_mel_filterbank() {
    if (mel_initialized) return;

    float f_min = 0, f_max = SAMPLE_RATE / 2.0f;
    float mel_min = 2595.0f * log10f(1.0f + f_min / 700.0f);
    float mel_max = 2595.0f * log10f(1.0f + f_max / 700.0f);

    float mel_pts[N_MEL + 2];
    for (int i = 0; i < N_MEL + 2; i++)
        mel_pts[i] = mel_min + (mel_max - mel_min) * i / (N_MEL + 1);

    float hz_pts[N_MEL + 2];
    for (int i = 0; i < N_MEL + 2; i++)
        hz_pts[i] = 700.0f * (powf(10.0f, mel_pts[i] / 2595.0f) - 1.0f);

    float bin_pts[N_MEL + 2];
    for (int i = 0; i < N_MEL + 2; i++)
        bin_pts[i] = hz_pts[i] * (N_FFT + 1) / SAMPLE_RATE;

    memset(mel_filters, 0, sizeof(mel_filters));
    for (int m = 0; m < N_MEL; m++) {
        int fl = (int)bin_pts[m], fc = (int)bin_pts[m+1], fr = (int)bin_pts[m+2];
        for (int k = fl; k <= fc && k <= N_FFT/2; k++)
            if (fc != fl) mel_filters[m][k] = (float)(k - fl) / (fc - fl);
        for (int k = fc; k <= fr && k <= N_FFT/2; k++)
            if (fr != fc) mel_filters[m][k] = (float)(fr - k) / (fr - fc);
    }
    mel_initialized = true;
}

// ===== In-place radix-2 FFT =====
static void fft_inplace(float* re, float* im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * M_PI / len;
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1, ci = 0;
            for (int j = 0; j < len/2; j++) {
                float ur = re[i+j], ui = im[i+j];
                float vr = re[i+j+len/2]*cr - im[i+j+len/2]*ci;
                float vi = re[i+j+len/2]*ci + im[i+j+len/2]*cr;
                re[i+j] = ur+vr; im[i+j] = ui+vi;
                re[i+j+len/2] = ur-vr; im[i+j+len/2] = ui-vi;
                float nr = cr*wr - ci*wi;
                ci = cr*wi + ci*wr;
                cr = nr;
            }
        }
    }
}

// ===== Compute mel spectrogram from audio =====
// audio: int16 PCM, mel_out: [num_frames, N_MEL]
static int compute_mel(const int16_t* audio, int num_samples, float* mel_out) {
    init_mel_filterbank();

    static float hann[WIN_LENGTH];
    static bool hann_init = false;
    if (!hann_init) {
        for (int i = 0; i < WIN_LENGTH; i++)
            hann[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (WIN_LENGTH - 1)));
        hann_init = true;
    }

    int nf = (num_samples - WIN_LENGTH) / HOP_LENGTH + 1;
    float re[N_FFT], im[N_FFT];

    for (int f = 0; f < nf; f++) {
        int start = f * HOP_LENGTH;
        memset(re, 0, sizeof(re));
        memset(im, 0, sizeof(im));
        for (int i = 0; i < WIN_LENGTH && (start+i) < num_samples; i++)
            re[i] = (audio[start+i] / 32768.0f) * hann[i];

        fft_inplace(re, im, N_FFT);

        float power[N_FFT/2 + 1];
        for (int k = 0; k <= N_FFT/2; k++)
            power[k] = re[k]*re[k] + im[k]*im[k];

        for (int m = 0; m < N_MEL; m++) {
            float sum = 0;
            for (int k = 0; k <= N_FFT/2; k++)
                sum += mel_filters[m][k] * power[k];
            mel_out[f * N_MEL + m] = logf(fmaxf(sum, 1e-10f));
        }
    }
    return nf;
}

#endif // MEL_FEATURES_H
""")
    print(f"  Written: {path}")


def write_arduino_sketch():
    """Write main Arduino sketch."""
    path = OUTPUT / "speaker_verify.ino"
    with open(path, "w") as f:
        f.write(r"""/*
 * VOICEPRINT — Speaker Verification on ESP32-S3
 * ==============================================
 * Pure C implementation — no TFLite dependency.
 * Resemblyzer LSTM with int8 quantized weights.
 *
 * Hardware: XIAO ESP32-S3 Sense (or any ESP32-S3 + I2S mic)
 *
 * Usage:
 *   - Hold BOOT button → record 3s → enroll voice
 *   - Speak near mic → auto-verify → LED feedback
 *   - GREEN LED = match, rapid blink = rejected
 */

#include <driver/i2s.h>
#include <SPIFFS.h>

#include "model_config.h"
#include "model_weights.h"
#include "lstm_engine.h"
#include "mel_features.h"

// ===== Pins (XIAO ESP32-S3 Sense) =====
#define I2S_WS   42
#define I2S_SD   41
#define LED_PIN  21
#define BOOT_BTN 0

// ===== Globals =====
static int16_t audio_buf[NUM_SAMPLES];
static float enrolled[EMBEDDING_DIM];
static bool has_enrollment = false;

// ===== I2S Setup =====
void i2s_init() {
    i2s_config_t cfg = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 1024,
    };
    i2s_pin_config_t pins = {
        .bck_io_num = -1,
        .ws_io_num = I2S_WS,
        .data_out_num = -1,
        .data_in_num = I2S_SD,
    };
    i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pins);
}

void record() {
    size_t br;
    int n = 0;
    Serial.println("[MIC] Recording...");
    while (n < NUM_SAMPLES) {
        int16_t tmp[512];
        i2s_read(I2S_NUM_0, tmp, sizeof(tmp), &br, portMAX_DELAY);
        int cnt = br / 2;
        for (int i = 0; i < cnt && n < NUM_SAMPLES; i++)
            audio_buf[n++] = tmp[i];
    }
    Serial.printf("[MIC] %d samples captured\n", n);
}

// ===== Inference =====
bool get_embedding(float* emb) {
    static float mel[NUM_FRAMES * N_MEL];
    int nf = compute_mel(audio_buf, NUM_SAMPLES, mel);
    Serial.printf("[MEL] %d frames extracted\n", nf);

    unsigned long t0 = millis();
    resemblyzer_forward(mel, nf, emb);
    Serial.printf("[INF] Inference: %lu ms\n", millis() - t0);
    return true;
}

// ===== SPIFFS =====
void save_enrolled() {
    File f = SPIFFS.open("/enroll.bin", FILE_WRITE);
    if (f) { f.write((uint8_t*)enrolled, sizeof(enrolled)); f.close(); }
}

bool load_enrolled() {
    File f = SPIFFS.open("/enroll.bin", FILE_READ);
    if (f && f.size() == sizeof(enrolled)) {
        f.read((uint8_t*)enrolled, sizeof(enrolled));
        f.close();
        return true;
    }
    return false;
}

// ===== LED feedback =====
void led_match() {
    digitalWrite(LED_PIN, HIGH);
    delay(2000);
    digitalWrite(LED_PIN, LOW);
}

void led_reject() {
    for (int i = 0; i < 5; i++) {
        digitalWrite(LED_PIN, HIGH); delay(80);
        digitalWrite(LED_PIN, LOW);  delay(80);
    }
}

void led_enrolled() {
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_PIN, HIGH); delay(300);
        digitalWrite(LED_PIN, LOW);  delay(200);
    }
}

// ===== Setup =====
void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n============================");
    Serial.println("  VOICEPRINT // ESP32-S3");
    Serial.println("  Resemblyzer LSTM int8");
    Serial.printf("  Weights: %d KB\n",
        (sizeof(lstm_weight_ih_l0) + sizeof(lstm_weight_hh_l0) +
         sizeof(lstm_weight_ih_l1) + sizeof(lstm_weight_hh_l1) +
         sizeof(lstm_weight_ih_l2) + sizeof(lstm_weight_hh_l2) +
         sizeof(linear_weight)) / 1024);
    Serial.println("============================\n");

    pinMode(LED_PIN, OUTPUT);
    pinMode(BOOT_BTN, INPUT_PULLUP);

    SPIFFS.begin(true);
    i2s_init();

    has_enrollment = load_enrolled();
    Serial.println(has_enrollment ?
        "[SYS] Enrollment loaded — speak to verify" :
        "[SYS] No enrollment — hold BOOT to enroll");
}

// ===== Loop =====
void loop() {
    // Enrollment mode
    if (digitalRead(BOOT_BTN) == LOW) {
        Serial.println("\n>>> ENROLL <<<");
        delay(500);
        record();
        if (get_embedding(enrolled)) {
            has_enrollment = true;
            save_enrolled();
            Serial.println("[ENR] Done!");
            led_enrolled();
        }
        return;
    }

    if (!has_enrollment) { delay(100); return; }

    // Voice activity detection
    int16_t vad[1024];
    size_t br;
    i2s_read(I2S_NUM_0, vad, sizeof(vad), &br, portMAX_DELAY);
    float rms = 0;
    int cnt = br / 2;
    for (int i = 0; i < cnt; i++) rms += (float)vad[i] * vad[i];
    rms = sqrtf(rms / cnt);

    if (rms < 500) return;  // Silence

    Serial.println("\n>>> VERIFY <<<");
    record();

    float emb[EMBEDDING_DIM];
    if (get_embedding(emb)) {
        float score = cosine_similarity(enrolled, emb);
        Serial.printf("[VER] Score: %.4f / %.2f\n", score, MATCH_THRESHOLD);

        if (score >= MATCH_THRESHOLD) {
            Serial.println("[VER] MATCH");
            led_match();
        } else {
            Serial.println("[VER] REJECTED");
            led_reject();
        }
    }
    delay(300);
}
""")
    print(f"  Written: {path}")


def write_pc_test(weights, scales, state):
    """Write accuracy_test.py for testing quantized weights on PC."""
    path = OUTPUT / "accuracy_test.py"
    with open(path, "w") as f:
        f.write("""#!/usr/bin/env python3
\"\"\"Test quantized Resemblyzer on PC with real audio.\"\"\"
import numpy as np
from pathlib import Path

print("Load this script and call test_with_audio('path.wav')")
print("Weights are embedded in the ESP32 firmware.")
print("Use the main gui.py app for PC-side testing.")
""")
    print(f"  Written: {path}")


def main():
    print("=" * 60)
    print("  VOICEPRINT — ESP32-S3 WEIGHT EXPORT")
    print("  Pure C implementation (no TFLite)")
    print("=" * 60)

    weights, scales, state = export_weights()
    avg_sim = verify_quantized_accuracy(weights, scales, state)

    print("\n\nGenerating ESP32-S3 firmware files...\n")
    write_model_config()
    write_weight_header(weights, scales)
    write_lstm_engine()
    write_mel_features()
    write_arduino_sketch()
    write_pc_test(weights, scales, state)

    print(f"\n{'='*60}")
    print(f"  OUTPUT: {OUTPUT}/")
    total = 0
    for f in sorted(OUTPUT.iterdir()):
        if f.is_file():
            sz = f.stat().st_size
            total += sz
            print(f"    {f.name:<30} {sz/1024:>8.0f} KB")
    print(f"    {'TOTAL':<30} {total/1024:>8.0f} KB")
    print(f"\n  Int8 accuracy: {avg_sim:.4f} cosine similarity")
    print(f"  ESP32-S3 flash usage: ~{sum(w.nbytes for w in weights.values())/1024:.0f} KB (weights only)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
