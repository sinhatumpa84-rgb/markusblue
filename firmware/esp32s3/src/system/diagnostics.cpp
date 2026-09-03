#include "diagnostics.h"
#include <Arduino.h>
#include <esp_heap_caps.h>

MarkusDiagnostics::MarkusDiagnostics() {
    memset(&m_metrics, 0, sizeof(m_metrics));
}

void MarkusDiagnostics::recordFrameMetrics(
    uint32_t dma_rx,
    uint32_t stft,
    uint32_t ai,
    uint32_t istft,
    uint32_t dsp,
    uint32_t dma_tx
) {
    m_metrics.dma_capture_us = dma_rx;
    m_metrics.stft_us = stft;
    m_metrics.ai_inference_us = ai;
    m_metrics.istft_us = istft;
    m_metrics.post_dsp_us = dsp;
    m_metrics.dma_tx_us = dma_tx;
    m_metrics.total_pipeline_us = stft + ai + istft + dsp;
    m_metrics.total_latency_ms = (float)m_metrics.total_pipeline_us / 1000.0f;
}

void MarkusDiagnostics::printReport() {
    Serial.println("==================================================");
    Serial.println("MARKUSBLUE (SIH26052) — REAL-TIME DIAGNOSTICS");
    Serial.println("==================================================");
    Serial.printf("STFT Analysis:       %5lu us\n", m_metrics.stft_us);
    Serial.printf("AI Mask Inference:   %5lu us\n", m_metrics.ai_inference_us);
    Serial.printf("ISTFT Synthesis:     %5lu us\n", m_metrics.istft_us);
    Serial.printf("AGC & Peak Limiter:  %5lu us\n", m_metrics.post_dsp_us);
    Serial.printf("Total Processing:    %5lu us (%.2f ms)\n", m_metrics.total_pipeline_us, m_metrics.total_latency_ms);
    Serial.printf("Dropped DMA Frames:  %5lu\n", m_metrics.dropped_frames);
    Serial.printf("Free Internal SRAM:  %lu bytes\n", heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    Serial.printf("Free Octal PSRAM:    %lu bytes\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    Serial.println("==================================================");
}
