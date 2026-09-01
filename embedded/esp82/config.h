#ifndef ESP82_CONFIG_H_
#define ESP82_CONFIG_H_

#include <stdint.h>

// =============================================================================
// MARKUSBLUE ESP82 / ESP8266 SYSTEM CONFIGURATION
// Target: Tensilica Xtensa L106 @ 160 MHz
// =============================================================================

// Audio Sampling & Frame Parameters
#define SAMPLE_RATE_HZ       8000      // 8.0 kHz Mono (Voice Band 300 - 3400 Hz)
#define FFT_SIZE             128       // 128-point Real FFT
#define HOP_SIZE             64        // 64 samples = 8.0 ms frame duration
#define NUM_FREQ_BINS        65        // FFT_SIZE / 2 + 1 = 65 bins

// Double-Buffered I2S DMA Parameters
#define DMA_BUFFER_COUNT     2
#define DMA_BUFFER_SAMPLES   HOP_SIZE  // 64 samples per DMA block

// Memory Safety & Allocation Constraints
#define TENSOR_ARENA_BYTES   3584      // 3.5 KB static tensor workspace
#define RING_BUFFER_SAMPLES  FFT_SIZE  // 128 samples circular input window

// DSP & Gain Control Parameters
#define AGC_TARGET_RMS       0.32f     // Target voice RMS level
#define AGC_MAX_GAIN         4.0f      // +12 dB maximum gain boost
#define AGC_MIN_GAIN         0.5f      // -6 dB minimum attenuation
#define AGC_ATTACK_RATE      0.05f     // Fast attack
#define AGC_DECAY_RATE       0.005f    // Smooth decay to prevent noise breathing
#define LIMITER_THRESHOLD    0.95f     // Peak threshold before soft saturation

// Pin Mapping for ESP8266 I2S Audio
#define PIN_I2S_DATA_IN      3         // RX / I2SI_DATA
#define PIN_I2S_DATA_OUT     2         // TX / I2SO_DATA
#define PIN_I2S_BCK          15        // I2SI_BCK / I2SO_BCK
#define PIN_I2S_WS           12        // I2SI_WS / I2SO_WS

#endif // ESP82_CONFIG_H_
