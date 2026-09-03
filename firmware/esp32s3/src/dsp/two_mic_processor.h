#ifndef TWO_MIC_PROCESSOR_H
#define TWO_MIC_PROCESSOR_H

#include <stdint.h>
#include <stddef.h>

class TwoMicProcessor {
private:
    size_t m_num_bins;
    float* m_noise_psd_ref;
    float* m_noise_psd_ear;
    float m_alpha_smooth;
    float m_spatial_gain;

public:
    TwoMicProcessor(size_t num_bins = 129, float alpha_smooth = 0.95f);
    ~TwoMicProcessor();

    void process(const float* mag_ref, const float* mag_ear, float* out_prefiltered_mag, float* out_snr_estimate);
    void reset();
};

#endif // TWO_MIC_PROCESSOR_H
