/**
 * SIH26052 — Indigenous Edge-AI Tactical Communication & Hearing Protection System
 * ESP32-S3 Dual-Core Firmware Prototype (INMP441 Mic + MAX98357A Amp + TFLite Micro).
 */

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "model_data.h"
#include "dsp_protection.h"

// Hardware Pin Configuration for ESP32-S3
#define I2S_MIC_SCK_PIN   4   // Bit Clock
#define I2S_MIC_WS_PIN    5   // Word Select / L-R Clock
#define I2S_MIC_SD_PIN    6   // Serial Data In (INMP441)

#define I2S_SPK_SCK_PIN   15  // Amp Bit Clock
#define I2S_SPK_WS_PIN    16  // Amp Word Select
#define I2S_SPK_SD_PIN    7   // Amp Serial Data Out (MAX98357A)

#define AUDIO_SAMPLE_RATE 16000
#define DMA_BUFFER_SAMPLES 256
#define WINDOW_SAMPLES    16000
#define MEL_BINS          32
#define TIME_STEPS        32

static const char *TAG = "TACTICAL_AUDIO_AI";

// Shared Atomic State between Core 0 and Core 1
static volatile float g_impulse_probability = 0.0f;
static volatile ProtectionState g_protection_state = STATE_NORMAL;

// DSP instances
static FastTransientLimiter g_limiter(16000.0f, 0.5f, 80.0f, -28.0f);
static FastSpeechPreservationFilter g_speech_filter;

// Circular Buffer for Sliding Window Feature Extraction
static float g_audio_window[WINDOW_SAMPLES];
static int g_window_head = 0;

/**
 * Core 0 Task: Real-time I2S DMA Audio Stream & Deterministic DSP Limiter.
 * Latency budget: < 1.0 ms per audio block.
 */
void audio_dsp_task(void *pvParameters) {
    ESP_LOGI(TAG, "Audio DSP Task started on Core %d", xPortGetCoreID());
    
    int16_t mic_rx_buf[DMA_BUFFER_SAMPLES];
    int16_t spk_tx_buf[DMA_BUFFER_SAMPLES];
    
    while (1) {
        // 1. Ingest audio block from I2S Microphone (INMP441)
        // Simulated I2S read:
        // size_t bytes_read = 0;
        // i2s_channel_read(rx_chan, mic_rx_buf, sizeof(mic_rx_buf), &bytes_read, portMAX_DELAY);
        
        bool is_protecting = (g_protection_state == STATE_PROTECTION_TRIGGERED);
        
        for (int i = 0; i < DMA_BUFFER_SAMPLES; i++) {
            // Convert to normalized float [-1.0, 1.0]
            float in_sample = (float)mic_rx_buf[i] / 32768.0f;
            
            // Step A: Sub-millisecond transient limiting
            float limited = g_limiter.process_sample(in_sample, is_protecting);
            
            // Step B: Bandpass voice formant preservation
            float protected_out = g_speech_filter.process_sample(limited, is_protecting);
            
            // Write to circular window for Core 1 ML inference
            g_audio_window[g_window_head] = in_sample;
            g_window_head = (g_window_head + 1) % WINDOW_SAMPLES;
            
            // Convert back to 16-bit PCM for amplifier output (MAX98357A)
            spk_tx_buf[i] = (int16_t)(protected_out * 32767.0f);
        }
        
        // 2. Transmit to I2S Amplifier (MAX98357A)
        // size_t bytes_written = 0;
        // i2s_channel_write(tx_chan, spk_tx_buf, sizeof(spk_tx_buf), &bytes_written, portMAX_DELAY);
        
        vTaskDelay(pdMS_TO_TICKS(5)); // Yield
    }
}

/**
 * Core 1 Task: Sliding-Window Mel Filterbank & TFLite Micro INT8 Inference.
 * Latency budget: ~10-15 ms per inference window.
 */
void ml_inference_task(void *pvParameters) {
    ESP_LOGI(TAG, "ML Inference Task started on Core %d", xPortGetCoreID());
    ESP_LOGI(TAG, "Loaded Quantized INT8 Model: %u bytes", g_tactical_model_data_len);
    
    // TFLite Micro Tensor Arena (allocated in internal SRAM / PSRAM)
    // constexpr int kTensorArenaSize = 30 * 1024;
    // uint8_t tensor_arena[kTensorArenaSize];
    
    while (1) {
        int64_t t_start = esp_timer_get_time();
        
        // 1. Extract 32-bin Log-Mel Spectrogram from g_audio_window
        // (In real firmware: compute fast 512-point FFT via esp-dsp library)
        
        // 2. Run TFLite Micro Model Inference (Simulated for demonstration)
        // TfLiteStatus invoke_status = interpreter->Invoke();
        // int8_t* output = interpreter->output(0)->data.int8;
        
        // Read impulse confidence (Class 0: DANGEROUS_IMPULSE)
        float impulse_prob = 0.05f; // Ambient baseline
        
        // Update global atomic state
        g_impulse_probability = impulse_prob;
        if (impulse_prob >= 0.65f) {
            g_protection_state = STATE_PROTECTION_TRIGGERED;
        } else if (impulse_prob < 0.30f) {
            g_protection_state = STATE_NORMAL;
        }
        
        int64_t t_end = esp_timer_get_time();
        int inf_time_us = (int)(t_end - t_start);
        
        // Run at 40 Hz (every 25 ms)
        vTaskDelay(pdMS_TO_TICKS(25));
    }
}

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "==================================================");
    ESP_LOGI(TAG, "SIH26052: Indigenous Edge-AI Tactical Audio System");
    ESP_LOGI(TAG, "Indian Army Hearing Protection Research Prototype");
    ESP_LOGI(TAG, "Target: ESP32-S3 Dual-Core Xtensa LX7 @ 240 MHz");
    ESP_LOGI(TAG, "==================================================");
    
    // Launch Dual-Core Tasks with FreeRTOS
    xTaskCreatePinnedToCore(audio_dsp_task, "audio_dsp", 4096, NULL, 5, NULL, 0); // Core 0
    xTaskCreatePinnedToCore(ml_inference_task, "ml_inf", 8192, NULL, 4, NULL, 1);  // Core 1
}
