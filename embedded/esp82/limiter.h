#ifndef LIMITER_H_
#define LIMITER_H_

#include <stdint.h>
#include "config.h"

class PeakLimiter {
public:
    PeakLimiter();
    void init(float threshold = LIMITER_THRESHOLD);
    void process_frame(float* buffer, uint16_t num_samples);

private:
    float threshold_;
};

#endif // LIMITER_H_
