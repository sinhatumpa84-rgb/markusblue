#ifndef ISTFT_H
#define ISTFT_H

#include <stdint.h>
#include <stddef.h>

class FastISTFT {
private:
    size_t m_n_fft;
    size_t m_hop_size;
    size_t m_num_bins;
    float* m_window;
    float* m_ifft_out;
    float* m_ola_buffer;
    size_t m_ola_size;

    void computeRIFFT(const float* in_real, const float* in_imag, float* out);

public:
    FastISTFT(size_t n_fft = 256, size_t hop_size = 64);
    ~FastISTFT();

    void process(const float* magnitude, const float* phase, float* out_time_frame);
    void reset();
};

#endif // ISTFT_H
