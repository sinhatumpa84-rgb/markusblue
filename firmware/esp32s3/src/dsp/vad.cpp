#include "vad.h"
#include <math.h>
#include <algorithm>

VoiceActivityDetector::VoiceActivityDetector(float snr_threshold_db, float alpha_noise, int hangover_limit)
    : m_noise_energy(1e-4f),
      m_alpha_noise(alpha_noise),
      m_snr_threshold_db(snr_threshold_db),
      m_hangover_count(0),
      m_hangover_limit(hangover_limit) {}

bool VoiceActivityDetector::process(const float* magnitude, size_t num_bins) {
    float frame_energy = 0.0f;
    for (size_t k = 0; k < num_bins; ++k) {
        frame_energy += magnitude[k] * magnitude[k];
    }
    frame_energy /= (float)num_bins;

    float snr_db = 10.0f * log10f((frame_energy + 1e-8f) / (m_noise_energy + 1e-8f));

    if (snr_db > m_snr_threshold_db) {
        m_hangover_count = m_hangover_limit;
        return true;
    } else {
        // Update noise floor only during silence/background
        m_noise_energy = m_alpha_noise * m_noise_energy + (1.0f - m_alpha_noise) * frame_energy;
        if (m_hangover_count > 0) {
            m_hangover_count--;
            return true;
        }
        return false;
    }
}
