#include "istft.h"
#include <esp_heap_caps.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

FastISTFT::FastISTFT(size_t n_fft, size_t hop_size)
    : m_n_fft(n_fft), m_hop_size(hop_size), m_num_bins(n_fft / 2 + 1), m_ola_size(n_fft) {
    m_window = (float*)heap_caps_malloc(n_fft * sizeof(float), MALLOC_CAP_INTERNAL);
    m_ifft_out = (float*)heap_caps_malloc(n_fft * sizeof(float), MALLOC_CAP_INTERNAL);
    m_ola_buffer = (float*)heap_caps_malloc(n_fft * sizeof(float), MALLOC_CAP_INTERNAL);

    for (size_t i = 0; i < n_fft; ++i) {
        m_window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)n_fft));
    }
    reset();
}

FastISTFT::~FastISTFT() {
    if (m_window) free(m_window);
    if (m_ifft_out) free(m_ifft_out);
    if (m_ola_buffer) free(m_ola_buffer);
}

void FastISTFT::reset() {
    if (m_ola_buffer) {
        memset(m_ola_buffer, 0, m_ola_size * sizeof(float));
    }
}

void FastISTFT::computeRIFFT(const float* in_real, const float* in_imag, float* out) {
    const float two_pi = 2.0f * (float)M_PI;
    const float inv_n = 1.0f / (float)m_n_fft;

    for (size_t n = 0; n < m_n_fft; ++n) {
        float sum = in_real[0]; // DC component (k=0)
        // Middle Nyquist component (k = N/2)
        sum += in_real[m_num_bins - 1] * cosf((float)M_PI * (float)n);

        for (size_t k = 1; k < m_num_bins - 1; ++k) {
            float angle = two_pi * (float)k * (float)n / (float)m_n_fft;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);
            sum += 2.0f * (in_real[k] * cos_val - in_imag[k] * sin_val);
        }
        out[n] = sum * inv_n;
    }
}

void FastISTFT::process(const float* magnitude, const float* phase, float* out_time_frame) {
    // 1. Reconstruct Real and Imaginary components
    float* r_spec = (float*)alloca(m_num_bins * sizeof(float));
    float* i_spec = (float*)alloca(m_num_bins * sizeof(float));

    for (size_t k = 0; k < m_num_bins; ++k) {
        float mag = magnitude[k];
        float ph = phase[k];
        r_spec[k] = mag * cosf(ph);
        i_spec[k] = mag * sinf(ph);
    }

    // 2. Inverse FFT
    computeRIFFT(r_spec, i_spec, m_ifft_out);

    // 3. Synthesis windowing & Overlap-Add
    for (size_t i = 0; i < m_n_fft; ++i) {
        m_ola_buffer[i] += m_ifft_out[i] * m_window[i];
    }

    // 4. Output the leading hop_size samples
    for (size_t i = 0; i < m_hop_size; ++i) {
        out_time_frame[i] = m_ola_buffer[i];
    }

    // 5. Shift overlap-add buffer by hop_size
    memmove(m_ola_buffer, m_ola_buffer + m_hop_size, (m_ola_size - m_hop_size) * sizeof(float));
    memset(m_ola_buffer + (m_ola_size - m_hop_size), 0, m_hop_size * sizeof(float));
}
