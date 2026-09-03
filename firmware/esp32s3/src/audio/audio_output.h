#ifndef AUDIO_OUTPUT_H
#define AUDIO_OUTPUT_H

#include <stdint.h>
#include <stddef.h>
#include <esp_err.h>

class AudioOutput {
private:
    int16_t* m_dma_tx_buffer;
    size_t m_frame_size;

public:
    AudioOutput(size_t frame_size);
    ~AudioOutput();

    esp_err_t writeMonoFrames(const float* in_audio, size_t num_samples, TickType_t timeout_ticks);
};

#endif // AUDIO_OUTPUT_H
