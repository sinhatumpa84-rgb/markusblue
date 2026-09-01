#include "audio_input.h"
#include <string.h>

#if defined(ESP8266)
#include <Arduino.h>
#include <i2s.h>
#endif

AudioInput::AudioInput()
    : current_buffer_idx_(0), is_initialized_(false) {
    memset(dma_ping_pong_, 0, sizeof(dma_ping_pong_));
}

bool AudioInput::init() {
#if defined(ESP8266)
    // Initialize ESP8266 I2S input DMA at configured sample rate
    i2s_rxtx_begin(true, false);
    i2s_set_rate(SAMPLE_RATE_HZ);
#endif
    is_initialized_ = true;
    return true;
}

bool AudioInput::read_frame(float* destination_buffer, uint16_t num_samples) {
    if (!is_initialized_ || destination_buffer == nullptr) {
        return false;
    }
    
    // Acquire samples from I2S DMA double-buffer and convert INT16 to normalized float [-1.0, 1.0]
    int16_t* current_buf = dma_ping_pong_[current_buffer_idx_];
    
#if defined(ESP8266)
    for (uint16_t i = 0; i < num_samples; i++) {
        uint32_t sample_pair = 0;
        i2s_read_sample(&sample_pair);
        int16_t raw_pcm = (int16_t)(sample_pair & 0xFFFF);
        current_buf[i] = raw_pcm;
        destination_buffer[i] = (float)raw_pcm / 32768.0f;
    }
#else
    // Desktop simulation fallback
    for (uint16_t i = 0; i < num_samples; i++) {
        destination_buffer[i] = (float)current_buf[i] / 32768.0f;
    }
#endif

    current_buffer_idx_ = (current_buffer_idx_ + 1) % DMA_BUFFER_COUNT;
    return true;
}
