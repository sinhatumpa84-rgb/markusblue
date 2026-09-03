#include "audio_output.h"
#include "i2s_config.h"
#include <esp_heap_caps.h>
#include <algorithm>

AudioOutput::AudioOutput(size_t frame_size)
    : m_frame_size(frame_size) {
    // Allocate TX DMA buffer for 16-bit Mono PCM
    m_dma_tx_buffer = (int16_t*)heap_caps_malloc(frame_size * sizeof(int16_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
    if (!m_dma_tx_buffer) {
        m_dma_tx_buffer = (int16_t*)malloc(frame_size * sizeof(int16_t));
    }
}

AudioOutput::~AudioOutput() {
    if (m_dma_tx_buffer) {
        free(m_dma_tx_buffer);
        m_dma_tx_buffer = nullptr;
    }
}

esp_err_t AudioOutput::writeMonoFrames(const float* in_audio, size_t num_samples, TickType_t timeout_ticks) {
    if (!m_dma_tx_buffer || !in_audio) return ESP_ERR_INVALID_ARG;

    for (size_t i = 0; i < num_samples; ++i) {
        // Output safety clamp: strictly clamp within [-0.999f, +0.999f]
        float sample = std::max(-0.999f, std::min(0.999f, in_audio[i]));
        // Convert to 16-bit integer PCM
        m_dma_tx_buffer[i] = (int16_t)(sample * 32767.0f);
    }

    size_t bytes_to_write = num_samples * sizeof(int16_t);
    size_t bytes_written = 0;

    return i2s_write(I2S_SPK_PORT, m_dma_tx_buffer, bytes_to_write, &bytes_written, timeout_ticks);
}
