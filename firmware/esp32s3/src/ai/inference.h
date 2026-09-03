#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include <stddef.h>
#include "model_config.h"

class EdgeInferenceEngine {
private:
    float m_gru_hidden[MARKUSBLUE_HIDDEN_DIM];
    float m_tcn_state1[MARKUSBLUE_HIDDEN_DIM * 3];
    float m_tcn_state2[MARKUSBLUE_HIDDEN_DIM * 3];
    float m_tcn_state3[MARKUSBLUE_HIDDEN_DIM * 3];
    uint32_t m_last_inference_us;

public:
    EdgeInferenceEngine();
    ~EdgeInferenceEngine() = default;

    bool init();
    void inferMask(const float* in_magnitude, float* out_mask);
    uint32_t getLastInferenceTimeUs() const { return m_last_inference_us; }
    void reset();
};

#endif // INFERENCE_H
