#include "limiter.h"
#include <math.h>

PeakLimiter::PeakLimiter()
    : threshold_(LIMITER_THRESHOLD) {}

void PeakLimiter::init(float threshold) {
    threshold_ = threshold;
}

void PeakLimiter::process_frame(float* buffer, uint16_t num_samples) {
    if (buffer == nullptr || num_samples == 0) return;

    for (uint16_t i = 0; i < num_samples; i++) {
        float val = buffer[i];
        float abs_val = fabsf(val);

        if (abs_val > threshold_) {
            float excess = abs_val - threshold_;
            float compressed = threshold_ + (1.0f - threshold_) * tanhf(excess / (1.0f - threshold_ + 1e-6f));
            buffer[i] = (val > 0.0f) ? compressed : -compressed;
        }

        // Hard clamp bounds
        if (buffer[i] > 1.0f) buffer[i] = 1.0f;
        if (buffer[i] < -1.0f) buffer[i] = -1.0f;
    }
}
