#ifndef STFT_H
#define STFT_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

class FastSTFT {
private:
    size_t m_n_fft;
    size_t m_hop_size;
    size_t m_num_bins;
    float* m_window;
    float* m_windowed_input;
    float* m_fft_real;
    float* m_fft_imag;

    void computeRFFT(const float* in, float* out_real, float* out_imag);

public:
    FastSTFT(size_t n_fft = 256, size_t hop_size = 64);
    ~FastSTFT();

    void process(const float* time_frame, float* out_magnitude, float* out_phase);
    size_t getNumBins() const { return m_num_bins; }
};

#endif // STFT_H
