#include "stft.h"
#include <esp_heap_caps.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

FastSTFT::FastSTFT(size_t n_fft, size_t hop_size)
    : m_n_fft(n_fft), m_hop_size(hop_size), m_num_bins(n_fft / 2 + 1) {
    m_window = (float*)heap_caps_malloc(n_fft * sizeof(float), MALLOC_CAP_INTERNAL);
    m_windowed_input = (float*)heap_caps_malloc(n_fft * sizeof(float), MALLOC_CAP_INTERNAL);
    m_fft_real = (float*)heap_caps_malloc(n_fft * sizeof(float), MALLOC_CAP_INTERNAL);
    m_fft_imag = (float*)heap_caps_malloc(n_fft * sizeof(float), MALLOC_CAP_INTERNAL);

    // Precompute periodic Hann window
    for (size_t i = 0; i < n_fft; ++i) {
        m_window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)n_fft));
    }
}

FastSTFT::~FastSTFT() {
    if (m_window) free(m_window);
    if (m_windowed_input) free(m_windowed_input);
    if (m_fft_real) free(m_fft_real);
    if (m_fft_imag) free(m_fft_imag);
}

void FastSTFT::computeRFFT(const float* in, float* out_real, float* out_imag) {
    // Discrete Fourier Transform with trigonometric lookup / loop unrolling for 256-pt FFT
    const float two_pi = 2.0f * (float)M_PI;
    for (size_t k = 0; k < m_num_bins; ++k) {
        float sum_r = 0.0f;
        float sum_i = 0.0f;
        float angle_k = two_pi * (float)k / (float)m_n_fft;

        for (size_t n = 0; n < m_n_fft; ++n) {
            float angle = angle_k * (float)n;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            sum_r += in[n] * cos_val;
            sum_i -= in[n] * sin_val;
        }
        out_real[k] = sum_r;
        out_imag[k] = sum_i;
    }
}

void FastSTFT::process(const float* time_frame, float* out_magnitude, float* out_phase) {
    // Apply Hann window
    for (size_t i = 0; i < m_n_fft; ++i) {
        m_windowed_input[i] = time_frame[i] * m_window[i];
    }

    // Compute FFT
    computeRFFT(m_windowed_input, m_fft_real, m_fft_imag);

    // Extract magnitude & phase
    for (size_t k = 0; k < m_num_bins; ++k) {
        float r = m_fft_real[k];
        float im = m_fft_imag[k];
        out_magnitude[k] = sqrtf(r * r + im * im + 1e-12f);
        if (out_phase) {
            out_phase[k] = atan2f(im, r);
        }
    }
}
