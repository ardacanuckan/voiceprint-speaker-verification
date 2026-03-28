#ifndef MODEL_CONFIG_H
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
#define MATCH_THRESHOLD 0.60f

#endif // MODEL_CONFIG_H
