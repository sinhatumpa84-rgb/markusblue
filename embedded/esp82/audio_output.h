#ifndef AUDIO_OUTPUT_H_
#define AUDIO_OUTPUT_H_

#include <stdint.h>
#include "config.h"

class AudioOutput {
public:
    AudioOutput();
    bool init();
    bool write_frame(const float* source_buffer, uint16_t num_samples);

private:
    int16_t dma_out_buffer_[DMA_BUFFER_COUNT][DMA_BUFFER_SAMPLES];
    volatile uint8_t current_buffer_idx_;
    bool is_initialized_;
};

#endif // AUDIO_OUTPUT_H_
