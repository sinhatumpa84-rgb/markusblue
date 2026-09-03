#ifndef VAD_H
#define VAD_H

#include <stdint.h>
#include <stddef.h>

class VoiceActivityDetector {
private:
    float m_noise_energy;
    float m_alpha_noise;
    float m_snr_threshold_db;
    int m_hangover_count;
    int m_hangover_limit;

public:
    VoiceActivityDetector(float snr_threshold_db = 4.0f, float alpha_noise = 0.98f, int hangover_limit = 5);
    ~VoiceActivityDetector() = default;

    bool process(const float* magnitude, size_t num_bins);
    float getNoiseFloor() const { return m_noise_energy; }
};

#endif // VAD_H
