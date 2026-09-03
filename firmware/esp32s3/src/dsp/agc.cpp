#include "agc.h"
#include <math.h>
#include <algorithm>

AutomaticGainControl::AutomaticGainControl(
    float target_rms,
    float max_gain,
    float min_gain,
    float attack_ms,
    float release_ms,
    int sample_rate
) : m_target_rms(target_rms),
    m_max_gain(max_gain),
    m_min_gain(min_gain),
    m_current_gain(1.0f),
    m_smoothed_rms(target_rms) {
    m_alpha_attack = expf(-1.0f / (attack_ms * 1e-3f * (float)sample_rate));
    m_alpha_release = expf(-1.0f / (release_ms * 1e-3f * (float)sample_rate));
}

void AutomaticGainControl::reset() {
    m_current_gain = 1.0f;
    m_smoothed_rms = m_target_rms;
}

void AutomaticGainControl::process(float* buffer, size_t len, bool is_speech) {
    if (len == 0) return;

    if (is_speech) {
        // Calculate block RMS
        float sum_sq = 0.0f;
        for (size_t i = 0; i < len; ++i) {
            sum_sq += buffer[i] * buffer[i];
        }
        float block_rms = sqrtf(sum_sq / (float)len + 1e-8f);

        // Smooth RMS
        if (block_rms > m_smoothed_rms) {
            m_smoothed_rms = m_alpha_attack * m_smoothed_rms + (1.0f - m_alpha_attack) * block_rms;
        } else {
            m_smoothed_rms = m_alpha_release * m_smoothed_rms + (1.0f - m_alpha_release) * block_rms;
        }

        // Calculate target gain
        float target_gain = m_target_rms / (m_smoothed_rms + 1e-6f);
        target_gain = std::max(m_min_gain, std::min(m_max_gain, target_gain));

        // Smooth gain transition
        if (target_gain < m_current_gain) {
            m_current_gain = m_alpha_attack * m_current_gain + (1.0f - m_alpha_attack) * target_gain;
        } else {
            m_current_gain = m_alpha_release * m_current_gain + (1.0f - m_alpha_release) * target_gain;
        }
    }

    // Apply gain to buffer
    for (size_t i = 0; i < len; ++i) {
        buffer[i] *= m_current_gain;
    }
}
