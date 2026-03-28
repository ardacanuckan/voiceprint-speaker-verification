#ifndef MEL_FEATURES_H
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
