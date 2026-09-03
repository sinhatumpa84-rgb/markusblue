#ifndef LIMITER_H
#define LIMITER_H

#include <stdint.h>
#include <stddef.h>

class PeakSafetyLimiter {
private:
    float m_ceiling;
    float m_alpha_attack;
    float m_alpha_release;
    float m_gain;
    float* m_delay_buf;
    size_t m_lookahead;

public:
    PeakSafetyLimiter(
        float ceiling_dbfs = -0.5f,
        float attack_ms = 0.2f,
        float release_ms = 50.0f,
        size_t lookahead_samples = 8,
        int sample_rate = 16000
    );
    ~PeakSafetyLimiter();

    void process(float* buffer, size_t len);
    void reset();
};

#endif // LIMITER_H
