#ifndef LSTM_ENGINE_H
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
