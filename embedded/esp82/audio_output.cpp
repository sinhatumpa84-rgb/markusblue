#include "audio_output.h"
#include <string.h>

#if defined(ESP8266)
#include <Arduino.h>
#include <i2s.h>
#endif

AudioOutput::AudioOutput()
    : current_buffer_idx_(0), is_initialized_(false) {
    memset(dma_out_buffer_, 0, sizeof(dma_out_buffer_));
}

bool AudioOutput::init() {
#if defined(ESP8266)
    // Configure I2S Output transmitter DMA
    i2s_begin();
    i2s_set_rate(SAMPLE_RATE_HZ);
#endif
    is_initialized_ = true;
    return true;
}

bool AudioOutput::write_frame(const float* source_buffer, uint16_t num_samples) {
    if (!is_initialized_ || source_buffer == nullptr) {
        return false;
    }

    int16_t* current_buf = dma_out_buffer_[current_buffer_idx_];

    for (uint16_t i = 0; i < num_samples; i++) {
        // Convert float [-1.0, 1.0] to 16-bit PCM
        float clamped = source_buffer[i];
        if (clamped > 1.0f) clamped = 1.0f;
        if (clamped < -1.0f) clamped = -1.0f;
        int16_t pcm_sample = (int16_t)(clamped * 32767.0f);
        current_buf[i] = pcm_sample;

#if defined(ESP8266)
        // Write sample pair (mono replicated to stereo) to I2S DMA FIFO
        uint32_t sample_pair = ((uint32_t)(uint16_t)pcm_sample << 16) | (uint16_t)pcm_sample;
        i2s_write_sample(sample_pair);
#endif
    }

    current_buffer_idx_ = (current_buffer_idx_ + 1) % DMA_BUFFER_COUNT;
    return true;
}
