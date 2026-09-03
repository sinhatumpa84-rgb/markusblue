/**
 * PROJECT: MARKUSBLUE
 * SIH PROBLEM STATEMENT: SIH26052
 * TARGET MCU: ESP32-S3 N16R8 (Dual-Core @ 240 MHz, 16MB Flash, 8MB PSRAM)
 *
 * Real-Time Edge-AI Tactical Audio Enhancement & Active Noise Cancellation System.
 */

#include <Arduino.h>
#include <esp_task_wdt.h>

// Audio Drivers
#include "audio/i2s_config.h"
#include "audio/audio_input.h"
#include "audio/audio_output.h"
#include "audio/audio_buffer.h"

// DSP & AI
#include "dsp/stft.h"
#include "dsp/istft.h"
#include "dsp/two_mic_processor.h"
#include "dsp/vad.h"
#include "dsp/agc.h"
#include "dsp/limiter.h"
#include "ai/inference.h"

// Sensors, Storage & System
#include "sensors/oled.h"
#include "sensors/mpu6050.h"
#include "sensors/ptt.h"
#include "storage/sd_logger.h"
#include "system/power_manager.h"
#include "system/diagnostics.h"

// Global System Handles
static MarkusDisplay g_display;
static MarkusIMU g_imu;
static MarkusPTT g_ptt(1, 2);
static MarkusSDLogger g_logger(10);
static MarkusPowerManager g_power(7);
static MarkusDiagnostics g_diagnostics;

// System Shared Telemetry State
static volatile SystemDisplayState g_system_state = {
    .ai_active = true,
    .mic_ok = true,
    .enhancement_on = true,
    .recording_active = false,
    .battery_pct = 95,
    .latency_ms = 4.2f,
    .snr_db = 12.5f,
    .ptt_pressed = false
};

// =========================================================================
// CORE 1: REAL-TIME AUDIO PROCESSING PIPELINE (High Priority Task)
// =========================================================================
void AudioProcessingTask(void* parameter) {
    Serial.println("[AudioTask] Initializing Real-Time Audio Engine on Core 1...");

    const size_t frame_size = MARKUSBLUE_HOP_SIZE; // 64 samples = 4.0 ms
    const size_t n_fft = MARKUSBLUE_N_FFT;          // 256 samples
    const size_t num_bins = MARKUSBLUE_NUM_BINS;    // 129 positive bins

    AudioInput audio_in(frame_size);
    AudioOutput audio_out(frame_size);

    FastSTFT stft(n_fft, frame_size);
    FastISTFT istft(n_fft, frame_size);
    TwoMicProcessor two_mic(num_bins);
    VoiceActivityDetector vad(4.0f, 0.98f);
    AutomaticGainControl agc(0.1585f, 4.0f, 0.25f);
    PeakSafetyLimiter limiter(-0.5f, 0.2f, 50.0f, 8);
    EdgeInferenceEngine ai_engine;

    ai_engine.init();

    // Pre-allocated static buffers in SRAM (Zero dynamic memory allocation in loop)
    float ref_mic_time[frame_size];
    float ear_mic_time[frame_size];
    float stft_input_window[n_fft] = {0};

    float mag_ref[num_bins];
    float mag_ear[num_bins];
    float phase_ear[num_bins];
    float prefiltered_mag[num_bins];
    float ai_mask[num_bins];
    float enhanced_mag[num_bins];
    float output_time_frame[frame_size];

    Serial.println("[AudioTask] Real-Time DSP + AI Pipeline ACTIVE. Processing audio stream...");

    while (true) {
        uint64_t t0 = esp_timer_get_time();

        // 1. Capture DMA Frame from Dual INMP441 Microphones
        esp_err_t rx_err = audio_in.readStereoFrames(ref_mic_time, ear_mic_time, frame_size, portMAX_DELAY);
        if (rx_err != ESP_OK) {
            g_diagnostics.recordDroppedFrame();
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }
        uint64_t t1 = esp_timer_get_time();

        // Shift sliding analysis window by frame_size
        memmove(stft_input_window, stft_input_window + frame_size, (n_fft - frame_size) * sizeof(float));
        memcpy(stft_input_window + (n_fft - frame_size), ear_mic_time, frame_size * sizeof(float));

        // 2. STFT Spectral Analysis
        stft.process(stft_input_window, mag_ear, phase_ear);
        uint64_t t2 = esp_timer_get_time();

        // 3. Dual-Mic Spatial Pre-Filtering & Noise PSD Tracking
        float snr_est = 0.0f;
        two_mic.process(mag_ref, mag_ear, prefiltered_mag, &snr_est);

        // 4. Edge-AI Neural Mask Estimation
        ai_engine.inferMask(prefiltered_mag, ai_mask);
        uint64_t t3 = esp_timer_get_time();

        // 5. Apply Estimated Mask to Magnitude Spectrum: M(f, t) * Mag(f, t)
        for (size_t k = 0; k < num_bins; ++k) {
            enhanced_mag[k] = mag_ear[k] * ai_mask[k];
        }

        // 6. ISTFT Overlap-Add Reconstruction
        istft.process(enhanced_mag, phase_ear, output_time_frame);
        uint64_t t4 = esp_timer_get_time();

        // 7. Voice Activity Detection (VAD)
        bool is_speech = vad.process(enhanced_mag, num_bins);

        // 8. Speech-Aware Dynamic Range AGC
        agc.process(output_time_frame, frame_size, is_speech);

        // 9. Lookahead Peak Limiter & Output Safety Clamping
        limiter.process(output_time_frame, frame_size);
        uint64_t t5 = esp_timer_get_time();

        // 10. Transmit to MAX98357A Class-D Speaker via I2S DMA TX
        audio_out.writeMonoFrames(output_time_frame, frame_size, portMAX_DELAY);
        uint64_t t6 = esp_timer_get_time();

        // Record Telemetry
        g_diagnostics.recordFrameMetrics(
            (uint32_t)(t1 - t0),
            (uint32_t)(t2 - t1),
            (uint32_t)(t3 - t2),
            (uint32_t)(t4 - t3),
            (uint32_t)(t5 - t4),
            (uint32_t)(t6 - t5)
        );

        g_system_state.snr_db = snr_est;
        g_system_state.latency_ms = (float)(t5 - t1) / 1000.0f;
    }
}

// =========================================================================
// CORE 0: SYSTEM CONTROL, UI & STORAGE TASK (Normal Priority)
// =========================================================================
void SystemControlTask(void* parameter) {
    Serial.println("[SystemTask] Initializing UI, Sensors & Telemetry on Core 0...");

    g_display.init(8, 9);
    g_imu.init();
    g_ptt.init();
    g_logger.init(12, 13, 11);
    g_power.init();

    uint32_t last_report_ms = 0;

    while (true) {
        // 1. Poll PTT Button
        bool ptt_active = g_ptt.update();
        g_system_state.ptt_pressed = ptt_active;

        // 2. Read IMU Motion Context
        MotionData motion;
        g_imu.readMotion(motion);

        // 3. Update Battery Telemetry
        g_system_state.battery_pct = g_power.getBatteryPercentage();

        // 4. Update OLED Display
        g_display.render(const_cast<const SystemDisplayState&>(g_system_state));

        // 5. Periodic Diagnostics Output (every 2 seconds)
        if (millis() - last_report_ms >= 2000) {
            last_report_ms = millis();
            g_diagnostics.printReport();

            if (g_system_state.recording_active) {
                g_logger.logTelemetry(g_system_state.latency_ms, g_system_state.snr_db, true, g_system_state.battery_pct);
            }
        }

        vTaskDelay(pdMS_TO_TICKS(100)); // 10 Hz UI Refresh Rate
    }
}

// =========================================================================
// ARDUINO SETUP & ENTRY POINT
// =========================================================================
void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.println("\n========================================================");
    Serial.println("   PROJECT MARKUSBLUE — SIH26052 TACTICAL AUDIO ANC");
    Serial.println("   TARGET: ESP32-S3 N16R8 DUAL-CORE XTENSA @ 240 MHz");
    Serial.println("========================================================");

    // 1. Install Hardware I2S RX (Dual Mics) and TX (Speaker) Drivers
    esp_err_t mic_res = I2SConfig::initMicrophones();
    esp_err_t spk_res = I2SConfig::initSpeaker();

    if (mic_res != ESP_OK || spk_res != ESP_OK) {
        Serial.printf("[!] Fatal: I2S Initialization failed! MIC: 0x%x, SPK: 0x%x\n", mic_res, spk_res);
    } else {
        Serial.println("[+] I2S0 RX (Dual INMP441) and I2S1 TX (MAX98357A) Initialized.");
    }

    // 2. Spawn Core 1 Real-Time Audio Task (Priority 24 / High)
    xTaskCreatePinnedToCore(
        AudioProcessingTask,
        "AudioTask",
        8192,
        NULL,
        24,
        NULL,
        1 // Core 1
    );

    // 3. Spawn Core 0 System Task (Priority 5 / Normal)
    xTaskCreatePinnedToCore(
        SystemControlTask,
        "SystemTask",
        4096,
        NULL,
        5,
        NULL,
        0 // Core 0
    );

    Serial.println("[+] FreeRTOS Tasks successfully dispatched across Core 0 & Core 1.");
}

void loop() {
    vTaskDelete(NULL); // Free the Arduino default loop task
}
