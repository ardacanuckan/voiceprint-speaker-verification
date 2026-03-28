/*
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
