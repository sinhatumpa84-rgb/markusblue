// =============================================================================
// MARKUSBLUE — ESP82 / ESP8266 Real-Time Speech Enhancement Main Engine
// Target: Tensilica Xtensa L106 @ 160 MHz
// Static memory allocation, zero dynamic malloc in the streaming loop.
// =============================================================================

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "config.h"
#include "audio_input.h"
#include "audio_output.h"
#include "markusblue_inference.h"
#include "agc.h"
#include "limiter.h"

#if defined(ESP8266)
#include <Arduino.h>
#else
#include <stdio.h>
#include <chrono>
#endif

// Static DSP and Neural State Buffers (No Heap Allocations)
static AudioInput g_audio_in;
static AudioOutput g_audio_out;
static MarkusblueInference g_nn_inference;
static AutomaticGainControl g_agc;
static PeakLimiter g_limiter;

// Ring Buffers & Windowing
static float g_input_ring_buf[RING_BUFFER_SAMPLES];
static float g_output_ola_buf[RING_BUFFER_SAMPLES];
static float g_hanning_window[RING_BUFFER_SAMPLES];

// Spectral Buffers
static float g_fft_mag[NUM_FREQ_BINS];
static float g_fft_phase[NUM_FREQ_BINS];
static float g_speech_mask[NUM_FREQ_BINS];
static float g_frame_buffer[HOP_SIZE];

// Diagnostic metrics
static uint32_t g_total_frames_processed = 0;
static uint32_t g_peak_frame_latency_us = 0;
static uint32_t g_avg_frame_latency_us = 0;

// Initialize Hanning Window
static void init_hanning_window() {
    for (int i = 0; i < RING_BUFFER_SAMPLES; i++) {
        g_hanning_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * (float)i / (float)(RING_BUFFER_SAMPLES - 1)));
    }
}

// Lightweight Real FFT on ESP8266
static void compute_stft(const float* time_in, float* mag_out, float* phase_out) {
    // 128-point discrete real transform with Hanning window
    for (int k = 0; k < NUM_FREQ_BINS; k++) {
        float real_sum = 0.0f;
        float imag_sum = 0.0f;
        float omega = 2.0f * M_PI * (float)k / (float)FFT_SIZE;
        
        for (int n = 0; n < FFT_SIZE; n++) {
            float windowed = time_in[n] * g_hanning_window[n];
            float angle = omega * (float)n;
            real_sum += windowed * cosf(angle);
            imag_sum -= windowed * sinf(angle);
        }
        
        mag_out[k] = sqrtf(real_sum * real_sum + imag_sum * imag_sum + 1e-12f);
        phase_out[k] = atan2f(imag_sum, real_sum);
    }
}

// Lightweight Inverse Real FFT Overlap-Add
static void compute_istft_ola(const float* mag_in, const float* phase_in, float* time_out_frame) {
    float reconstructed_time[FFT_SIZE];
    memset(reconstructed_time, 0, sizeof(reconstructed_time));
    
    for (int n = 0; n < FFT_SIZE; n++) {
        float sample_val = mag_in[0] * cosf(phase_in[0]); // DC component
        for (int k = 1; k < NUM_FREQ_BINS - 1; k++) {
            float angle = 2.0f * M_PI * (float)k * (float)n / (float)FFT_SIZE;
            sample_val += 2.0f * mag_in[k] * (cosf(phase_in[k]) * cosf(angle) - sinf(phase_in[k]) * sinf(angle));
        }
        float nyquist_angle = M_PI * (float)n;
        sample_val += mag_in[NUM_FREQ_BINS - 1] * cosf(phase_in[NUM_FREQ_BINS - 1]) * cosf(nyquist_angle);
        reconstructed_time[n] = (sample_val / (float)FFT_SIZE) * g_hanning_window[n];
    }
    
    // Overlap-add into output ring buffer
    for (int i = 0; i < FFT_SIZE; i++) {
        g_output_ola_buf[i] += reconstructed_time[i];
    }
    
    // Extract output hop frame (normalized for 50% Hanning overlap)
    for (int i = 0; i < HOP_SIZE; i++) {
        time_out_frame[i] = g_output_ola_buf[i] / 1.5f;
    }
    
    // Shift output buffer by HOP_SIZE
    memmove(g_output_ola_buf, g_output_ola_buf + HOP_SIZE, (FFT_SIZE - HOP_SIZE) * sizeof(float));
    memset(g_output_ola_buf + (FFT_SIZE - HOP_SIZE), 0, HOP_SIZE * sizeof(float));
}

void markusblue_setup() {
#if defined(ESP8266)
    Serial.begin(115200);
    // Overclock CPU to 160 MHz for optimal DSP performance
    system_update_cpu_freq(160);
    delay(100);
    
    Serial.println("\n=========================================");
    Serial.println("MARKUSBLUE — ESP8266 EDGE SPEECH ENGINE");
    Serial.println("Target: Tensilica Xtensa L106 @ 160 MHz");
    Serial.printf("Free Heap: %u bytes\n", ESP.getFreeHeap());
#endif

    init_hanning_window();
    memset(g_input_ring_buf, 0, sizeof(g_input_ring_buf));
    memset(g_output_ola_buf, 0, sizeof(g_output_ola_buf));

    g_audio_in.init();
    g_audio_out.init();
    g_nn_inference.init();
    g_agc.init(AGC_TARGET_RMS, AGC_MAX_GAIN, AGC_MIN_GAIN);
    g_limiter.init(LIMITER_THRESHOLD);

#if defined(ESP8266)
    Serial.printf("Initialization Complete. Tensor Arena: %u bytes\n", g_nn_inference.get_tensor_arena_used());
    Serial.printf("Free Heap after init: %u bytes\n", ESP.getFreeHeap());
    Serial.println("=========================================\n");
#endif
}

void markusblue_loop() {
#if defined(ESP8266)
    uint32_t t_start = micros();
#endif

    // 1. Read input audio frame from DMA (64 samples = 8.0 ms)
    if (!g_audio_in.read_frame(g_frame_buffer, HOP_SIZE)) {
        return;
    }

    // 2. Shift input ring buffer and append new frame
    memmove(g_input_ring_buf, g_input_ring_buf + HOP_SIZE, (RING_BUFFER_SAMPLES - HOP_SIZE) * sizeof(float));
    memcpy(g_input_ring_buf + (RING_BUFFER_SAMPLES - HOP_SIZE), g_frame_buffer, HOP_SIZE * sizeof(float));

    // 3. Compute Short-Time Fourier Transform (STFT)
    compute_stft(g_input_ring_buf, g_fft_mag, g_fft_phase);

    // 4. Run INT8 MARKUSBLUE Neural Mask Inference
    g_nn_inference.infer_mask(g_fft_mag, g_speech_mask);

    // 5. Apply Neural Mask to Magnitude Spectrum
    for (int b = 0; b < NUM_FREQ_BINS; b++) {
        g_fft_mag[b] *= g_speech_mask[b];
    }

    // 6. Inverse STFT & Overlap-Add Synthesis
    compute_istft_ola(g_fft_mag, g_fft_phase, g_frame_buffer);

    // 7. Voice Activity Energy Estimation & VAD
    float frame_energy = 0.0f;
    for (int i = 0; i < HOP_SIZE; i++) {
        frame_energy += g_frame_buffer[i] * g_frame_buffer[i];
    }
    bool is_speech = (frame_energy > 0.0005f);

    // 8. Automatic Gain Control (Compensate for attenuation without noise breathing)
    g_agc.process_frame(g_frame_buffer, HOP_SIZE, is_speech);

    // 9. Peak Limiter (Prevent digital clipping and distortion)
    g_limiter.process_frame(g_frame_buffer, HOP_SIZE);

    // 10. Write Enhanced Frame to Audio Output DMA
    g_audio_out.write_frame(g_frame_buffer, HOP_SIZE);

#if defined(ESP8266)
    uint32_t t_elapsed = micros() - t_start;
    g_total_frames_processed++;
    if (t_elapsed > g_peak_frame_latency_us) {
        g_peak_frame_latency_us = t_elapsed;
    }
    g_avg_frame_latency_us = (g_avg_frame_latency_us * 15 + t_elapsed) / 16;

    if (g_total_frames_processed % 250 == 0) { // Every 2.0 seconds
        Serial.printf("[MARKUSBLUE] Frame Latency: %u us (Peak: %u us) | RTF: %.3f | Free Heap: %u B\n",
            g_avg_frame_latency_us,
            g_peak_frame_latency_us,
            (float)g_avg_frame_latency_us / 8000.0f,
            ESP.getFreeHeap());
    }
#endif
}

#if defined(ESP8266)
void setup() {
    markusblue_setup();
}

void loop() {
    markusblue_loop();
}
#else
int main() {
    printf("[*] MARKUSBLUE ESP82 C++ Embedded Simulation Host\n");
    markusblue_setup();
    printf("[OK] Simulation host loop running.\n");
    return 0;
}
#endif
