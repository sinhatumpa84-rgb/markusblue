/**
 * SIH26052 — Indigenous Edge-AI Tactical Communication & Hearing Protection System
 * Fast deterministic DSP Limiter and Speech Preservation Filter for ESP32-S3 (C++).
 */

#ifndef DSP_PROTECTION_H_
#define DSP_PROTECTION_H_

#include <math.h>
#include <stdint.h>
#include <algorithm>

enum ProtectionState {
    STATE_NORMAL = 0,
    STATE_PROTECTION_TRIGGERED = 1,
    STATE_RECOVERY = 2
};

class FastTransientLimiter {
public:
    FastTransientLimiter(float sample_rate = 16000.0f, float attack_ms = 0.5f, float release_ms = 80.0f, float max_atten_db = -28.0f) 
        : sr(sample_rate), envelope(0.0f), current_gain(1.0f) {
        
        alpha_release = expf(-1.0f / (fmaxf(0.001f, release_ms * 1e-3f) * sr));
        attenuation_linear = powf(10.0f, max_atten_db / 20.0f); // ~0.0398 for -28dB
        threshold_linear = 0.251f; // -12 dB threshold
    }

    void reset() {
        envelope = 0.0f;
        current_gain = 1.0f;
    }

    inline float process_sample(float in_sample, bool force_protect) {
        float abs_val = fabsf(in_sample);
        
        // 1. Instant peak detection on rising edge
        if (abs_val > envelope) {
            envelope = abs_val;
        } else {
            envelope = alpha_release * envelope + (1.0f - alpha_release) * abs_val;
        }

        // 2. Determine target gain
        float target_gain = (force_protect || envelope > threshold_linear) ? attenuation_linear : 1.0f;

        // 3. Instant attack clamp, smooth exponential release
        if (target_gain < current_gain) {
            current_gain = target_gain;
        } else {
            current_gain = alpha_release * current_gain + (1.0f - alpha_release) * target_gain;
        }

        float out = in_sample * current_gain;
        // Clamp output safely within [-1.0, 1.0]
        return fmaxf(-1.0f, fminf(1.0f, out));
    }

private:
    float sr;
    float alpha_release;
    float attenuation_linear;
    float threshold_linear;
    float envelope;
    float current_gain;
};

class FastSpeechPreservationFilter {
public:
    // Direct Form II Biquad Bandpass Filter (300Hz - 3.4kHz @ 16kHz)
    FastSpeechPreservationFilter() : w1_0(0.0f), w2_0(0.0f), w1_1(0.0f), w2_1(0.0f) {
        // Precomputed Butterworth Biquad coefficients @ 16kHz
        b0 = 0.2929f; b1 = 0.0f; b2 = -0.2929f;
        a1 = -1.143f; a2 = 0.4142f;
    }

    inline float process_sample(float in_sample, bool protection_active) {
        if (!protection_active) {
            return in_sample;
        }

        // Biquad section 1
        float w0 = in_sample - a1 * w1_0 - a2 * w2_0;
        float y = b0 * w0 + b1 * w1_0 + b2 * w2_0;
        w2_0 = w1_0;
        w1_0 = w0;

        // Pass voice with slight boost, attenuate out-of-band blast shockwave
        float voice_boosted = y * 1.41f; // +3 dB
        float blast_residual = in_sample * 0.025f; // -32 dB

        return fmaxf(-1.0f, fminf(1.0f, voice_boosted + blast_residual));
    }

private:
    float b0, b1, b2, a1, a2;
    float w1_0, w2_0, w1_1, w2_1;
};

#endif // DSP_PROTECTION_H_
