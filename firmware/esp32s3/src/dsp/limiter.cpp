#include "limiter.h"
#include <esp_heap_caps.h>
#include <math.h>
#include <algorithm>
#include <string.h>

PeakSafetyLimiter::PeakSafetyLimiter(
    float ceiling_dbfs,
    float attack_ms,
    float release_ms,
    size_t lookahead_samples,
    int sample_rate
) : m_lookahead(lookahead_samples),
    m_gain(1.0f) {
    m_ceiling = powf(10.0f, ceiling_dbfs / 20.0f);
    m_alpha_attack = expf(-1.0f / (attack_ms * 1e-3f * (float)sample_rate));
    m_alpha_release = expf(-1.0f / (release_ms * 1e-3f * (float)sample_rate));

    m_delay_buf = (float*)heap_caps_malloc(m_lookahead * sizeof(float), MALLOC_CAP_INTERNAL);
    reset();
}

PeakSafetyLimiter::~PeakSafetyLimiter() {
    if (m_delay_buf) {
        free(m_delay_buf);
        m_delay_buf = nullptr;
    }
}

void PeakSafetyLimiter::reset() {
    m_gain = 1.0f;
    if (m_delay_buf) {
        memset(m_delay_buf, 0, m_lookahead * sizeof(float));
    }
}

void PeakSafetyLimiter::process(float* buffer, size_t len) {
    if (!m_delay_buf || len == 0) return;

    for (size_t i = 0; i < len; ++i) {
        float in_sample = buffer[i];
        float delayed_sample = m_delay_buf[0];

        // Shift lookahead buffer
        for (size_t j = 0; j < m_lookahead - 1; ++j) {
            m_delay_buf[j] = m_delay_buf[j + 1];
        }
        m_delay_buf[m_lookahead - 1] = in_sample;

        // Target gain based on peak amplitude
        float abs_val = fabsf(in_sample);
        float target_gain = (abs_val <= m_ceiling) ? 1.0f : (m_ceiling / (abs_val + 1e-12f));

        if (target_gain < m_gain) {
            m_gain = m_alpha_attack * m_gain + (1.0f - m_alpha_attack) * target_gain;
        } else {
            m_gain = m_alpha_release * m_gain + (1.0f - m_alpha_release) * target_gain;
        }

        // Apply gain to delayed sample and hard limit
        float limited = delayed_sample * m_gain;
        buffer[i] = std::max(-1.0f, std::min(1.0f, limited));
    }
}
