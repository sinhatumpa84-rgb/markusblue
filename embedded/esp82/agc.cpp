#include "agc.h"
#include <math.h>

AutomaticGainControl::AutomaticGainControl()
    : target_rms_(AGC_TARGET_RMS),
      max_gain_(AGC_MAX_GAIN),
      min_gain_(AGC_MIN_GAIN),
      attack_rate_(AGC_ATTACK_RATE),
      decay_rate_(AGC_DECAY_RATE),
      current_gain_(1.0f) {}

void AutomaticGainControl::init(float target_rms, float max_gain, float min_gain) {
    target_rms_ = target_rms;
    max_gain_ = max_gain;
    min_gain_ = min_gain;
    attack_rate_ = AGC_ATTACK_RATE;
    decay_rate_ = AGC_DECAY_RATE;
    current_gain_ = 1.0f;
}

void AutomaticGainControl::process_frame(float* buffer, uint16_t num_samples, bool is_speech) {
    if (buffer == nullptr || num_samples == 0) return;

    // Calculate frame RMS
    float energy_sum = 0.0f;
    for (uint16_t i = 0; i < num_samples; i++) {
        energy_sum += buffer[i] * buffer[i];
    }
    float frame_rms = sqrtf(energy_sum / (float)num_samples + 1e-10f);

    if (is_speech && frame_rms > 0.001f) {
        float desired_gain = target_rms_ / frame_rms;
        if (desired_gain > max_gain_) desired_gain = max_gain_;
        if (desired_gain < min_gain_) desired_gain = min_gain_;

        // Smooth gain transition
        if (desired_gain < current_gain_) {
            current_gain_ += attack_rate_ * (desired_gain - current_gain_);
        } else {
            current_gain_ += decay_rate_ * (desired_gain - current_gain_);
        }
    } else {
        // Slow decay to 1.0 to prevent pumping noise during pauses
        current_gain_ += decay_rate_ * (1.0f - current_gain_);
    }

    // Apply smooth gain to frame
    for (uint16_t i = 0; i < num_samples; i++) {
        buffer[i] *= current_gain_;
    }
}
