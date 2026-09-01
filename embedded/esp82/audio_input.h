#ifndef AUDIO_INPUT_H_
#define AUDIO_INPUT_H_

#include <stdint.h>
#include "config.h"

class AudioInput {
public:
    AudioInput();
    bool init();
    bool read_frame(float* destination_buffer, uint16_t num_samples);

private:
    // Static double buffer for DMA transfers
    int16_t dma_ping_pong_[DMA_BUFFER_COUNT][DMA_BUFFER_SAMPLES];
    volatile uint8_t current_buffer_idx_;
    bool is_initialized_;
};

#endif // AUDIO_INPUT_H_
