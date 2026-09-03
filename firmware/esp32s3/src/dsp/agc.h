#ifndef AGC_H
#define AGC_H

#include <stdint.h>
#include <stddef.h>

class AutomaticGainControl {
private:
    float m_target_rms;
    float m_max_gain;
    float m_min_gain;
    float m_alpha_attack;
    float m_alpha_release;
    float m_current_gain;
    float m_smoothed_rms;

public:
    AutomaticGainControl(
        float target_rms = 0.1585f,
        float max_gain = 4.0f,
        float min_gain = 0.25f,
        float attack_ms = 10.0f,
        float release_ms = 200.0f,
        int sample_rate = 16000
    );
    ~AutomaticGainControl() = default;

    void process(float* buffer, size_t len, bool is_speech);
    void reset();
};

#endif // AGC_H
