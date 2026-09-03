#include "audio_input.h"
#include "i2s_config.h"
#include <esp_heap_caps.h>
#include <string.h>

AudioInput::AudioInput(size_t frame_size)
    : m_frame_size(frame_size),
      m_dc_prev_in_left(0.0f), m_dc_prev_out_left(0.0f),
      m_dc_prev_in_right(0.0f), m_dc_prev_out_right(0.0f) {
    // Allocate raw DMA interleaved buffer (2 channels * frame_size * sizeof(int32_t))
    m_dma_raw_buffer = (int32_t*)heap_caps_malloc(frame_size * 2 * sizeof(int32_t), MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
    if (!m_dma_raw_buffer) {
        m_dma_raw_buffer = (int32_t*)malloc(frame_size * 2 * sizeof(int32_t));
    }
}

AudioInput::~AudioInput() {
    if (m_dma_raw_buffer) {
        free(m_dma_raw_buffer);
        m_dma_raw_buffer = nullptr;
    }
}

void AudioInput::resetFilters() {
    m_dc_prev_in_left = 0.0f;
    m_dc_prev_out_left = 0.0f;
    m_dc_prev_in_right = 0.0f;
    m_dc_prev_out_right = 0.0f;
}

esp_err_t AudioInput::readStereoFrames(float* out_left_ref, float* out_right_ear, size_t num_samples, TickType_t timeout_ticks) {
    size_t bytes_to_read = num_samples * 2 * sizeof(int32_t);
    size_t bytes_read = 0;

    esp_err_t res = i2s_read(I2S_MIC_PORT, m_dma_raw_buffer, bytes_to_read, &bytes_read, timeout_ticks);
    if (res != ESP_OK || bytes_read < bytes_to_read) {
        return res;
    }

    const float inv_scale = 1.0f / 2147483648.0f; // Scale 32-bit signed to [-1.0, 1.0]
    const float R = 0.995f; // DC-blocker pole

    for (size_t i = 0; i < num_samples; ++i) {
        // INMP441 left channel is top 24 bits inside 32-bit word
        int32_t raw_left = m_dma_raw_buffer[2 * i];
        int32_t raw_right = m_dma_raw_buffer[2 * i + 1];

        float norm_left = (float)raw_left * inv_scale;
        float norm_right = (float)raw_right * inv_scale;

        // DC offset blocker: y[n] = x[n] - x[n-1] + R * y[n-1]
        float dc_out_left = norm_left - m_dc_prev_in_left + R * m_dc_prev_out_left;
        m_dc_prev_in_left = norm_left;
        m_dc_prev_out_left = dc_out_left;
        out_left_ref[i] = dc_out_left;

        float dc_out_right = norm_right - m_dc_prev_in_right + R * m_dc_prev_out_right;
        m_dc_prev_in_right = norm_right;
        m_dc_prev_out_right = dc_out_right;
        out_right_ear[i] = dc_out_right;
    }

    return ESP_OK;
}
