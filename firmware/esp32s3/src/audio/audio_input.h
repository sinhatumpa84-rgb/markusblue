#ifndef AUDIO_INPUT_H
#define AUDIO_INPUT_H

#include <stdint.h>
#include <stddef.h>
#include <esp_err.h>

class AudioInput {
private:
    int32_t* m_dma_raw_buffer;
    size_t m_frame_size;
    float m_dc_prev_in_left;
    float m_dc_prev_out_left;
    float m_dc_prev_in_right;
    float m_dc_prev_out_right;

public:
    AudioInput(size_t frame_size);
    ~AudioInput();

    esp_err_t readStereoFrames(float* out_left_ref, float* out_right_ear, size_t num_samples, TickType_t timeout_ticks);
    void resetFilters();
};

#endif // AUDIO_INPUT_H
