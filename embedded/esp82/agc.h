#ifndef AGC_H_
#define AGC_H_

#include <stdint.h>
#include "config.h"

class AutomaticGainControl {
public:
    AutomaticGainControl();
    void init(float target_rms = AGC_TARGET_RMS, float max_gain = AGC_MAX_GAIN, float min_gain = AGC_MIN_GAIN);
    
    // Process audio frame in-place with VAD gating
    void process_frame(float* buffer, uint16_t num_samples, bool is_speech);
    
    float get_current_gain() const { return current_gain_; }

private:
    float target_rms_;
    float max_gain_;
    float min_gain_;
    float attack_rate_;
    float decay_rate_;
    float current_gain_;
};

#endif // AGC_H_
