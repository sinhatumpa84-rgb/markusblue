#include "two_mic_processor.h"
#include <esp_heap_caps.h>
#include <math.h>
#include <algorithm>
#include <string.h>

TwoMicProcessor::TwoMicProcessor(size_t num_bins, float alpha_smooth)
    : m_num_bins(num_bins), m_alpha_smooth(alpha_smooth), m_spatial_gain(0.85f) {
    m_noise_psd_ref = (float*)heap_caps_malloc(num_bins * sizeof(float), MALLOC_CAP_INTERNAL);
    m_noise_psd_ear = (float*)heap_caps_malloc(num_bins * sizeof(float), MALLOC_CAP_INTERNAL);
    reset();
}

TwoMicProcessor::~TwoMicProcessor() {
    if (m_noise_psd_ref) free(m_noise_psd_ref);
    if (m_noise_psd_ear) free(m_noise_psd_ear);
}

void TwoMicProcessor::reset() {
    if (m_noise_psd_ref) memset(m_noise_psd_ref, 0, m_num_bins * sizeof(float));
    if (m_noise_psd_ear) memset(m_noise_psd_ear, 0, m_num_bins * sizeof(float));
}

void TwoMicProcessor::process(const float* mag_ref, const float* mag_ear, float* out_prefiltered_mag, float* out_snr_estimate) {
    float total_speech_energy = 0.0f;
    float total_noise_energy = 0.0f;

    for (size_t k = 0; k < m_num_bins; ++k) {
        float p_ref = mag_ref[k] * mag_ref[k];
        float p_ear = mag_ear[k] * mag_ear[k];

        // Recursive power spectral density smoothing
        m_noise_psd_ref[k] = m_alpha_smooth * m_noise_psd_ref[k] + (1.0f - m_alpha_smooth) * p_ref;
        m_noise_psd_ear[k] = m_alpha_smooth * m_noise_psd_ear[k] + (1.0f - m_alpha_smooth) * p_ear;

        // Spatial coherence gain: if reference power dominates, suppress exterior noise
        float coherence_ratio = p_ear / (p_ref + 1e-6f);
        float spatial_weight = std::min(1.0f, std::max(0.15f, coherence_ratio));

        // Prefiltered magnitude passed to AI model
        out_prefiltered_mag[k] = mag_ear[k] * spatial_weight;

        total_speech_energy += out_prefiltered_mag[k];
        total_noise_energy += mag_ref[k];
    }

    if (out_snr_estimate) {
        *out_snr_estimate = 10.0f * log10f((total_speech_energy + 1e-6f) / (total_noise_energy + 1e-6f));
    }
}
