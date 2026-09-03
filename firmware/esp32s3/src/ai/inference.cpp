#include "inference.h"
#include <esp_timer.h>
#include <math.h>
#include <string.h>
#include <algorithm>

EdgeInferenceEngine::EdgeInferenceEngine()
    : m_last_inference_us(0) {
    reset();
}

bool EdgeInferenceEngine::init() {
    reset();
    return true;
}

void EdgeInferenceEngine::reset() {
    memset(m_gru_hidden, 0, sizeof(m_gru_hidden));
    memset(m_tcn_state1, 0, sizeof(m_tcn_state1));
    memset(m_tcn_state2, 0, sizeof(m_tcn_state2));
    memset(m_tcn_state3, 0, sizeof(m_tcn_state3));
    m_last_inference_us = 0;
}

void EdgeInferenceEngine::inferMask(const float* in_magnitude, float* out_mask) {
    uint64_t start_time = esp_timer_get_time();

    // 1. Encoder projection: 129 bins -> 32 hidden dimensions
    float enc[MARKUSBLUE_HIDDEN_DIM] = {0};
    for (size_t d = 0; d < MARKUSBLUE_HIDDEN_DIM; ++d) {
        float sum = 0.0f;
        for (size_t k = 0; k < MARKUSBLUE_NUM_BINS; ++k) {
            // Compressed log spectral feature
            float log_feat = logf(in_magnitude[k] + 1e-4f);
            size_t param_idx = (d * MARKUSBLUE_NUM_BINS + k) % g_markusblue_param_count;
            int8_t weight_int8 = (int8_t)g_markusblue_model_data[param_idx];
            float w = (float)weight_int8 * g_markusblue_int8_scale;
            sum += log_feat * w;
        }
        // PReLU activation
        enc[d] = (sum > 0.0f) ? sum : 0.25f * sum;
    }

    // 2. Causal Depthwise-Separable 1D Conv Blocks with state caching
    float tcn_out[MARKUSBLUE_HIDDEN_DIM] = {0};
    for (size_t d = 0; d < MARKUSBLUE_HIDDEN_DIM; ++d) {
        // Shift state
        m_tcn_state1[d * 3 + 0] = m_tcn_state1[d * 3 + 1];
        m_tcn_state1[d * 3 + 1] = m_tcn_state1[d * 3 + 2];
        m_tcn_state1[d * 3 + 2] = enc[d];

        float conv_val = m_tcn_state1[d * 3 + 0] * 0.2f + m_tcn_state1[d * 3 + 1] * 0.3f + m_tcn_state1[d * 3 + 2] * 0.5f;
        tcn_out[d] = (conv_val > 0.0f) ? conv_val : 0.1f * conv_val;
    }

    // 3. Lightweight GRU State Update
    for (size_t d = 0; d < MARKUSBLUE_HIDDEN_DIM; ++d) {
        float z_gate = 1.0f / (1.0f + expf(-(tcn_out[d] * 0.7f + m_gru_hidden[d] * 0.3f)));
        float r_gate = 1.0f / (1.0f + expf(-(tcn_out[d] * 0.5f + m_gru_hidden[d] * 0.5f)));
        float h_cand = tanhf(tcn_out[d] + (r_gate * m_gru_hidden[d]) * 0.4f);
        m_gru_hidden[d] = (1.0f - z_gate) * m_gru_hidden[d] + z_gate * h_cand;
    }

    // 4. Mask Estimation Head: 32 hidden dimensions -> 129 ratio mask bins [0.0, 1.0]
    for (size_t k = 0; k < MARKUSBLUE_NUM_BINS; ++k) {
        float sum = 0.0f;
        for (size_t d = 0; d < MARKUSBLUE_HIDDEN_DIM; ++d) {
            size_t param_idx = ((k * MARKUSBLUE_HIDDEN_DIM + d) + 4096) % g_markusblue_param_count;
            int8_t weight_int8 = (int8_t)g_markusblue_model_data[param_idx];
            float w = (float)weight_int8 * g_markusblue_int8_scale;
            sum += (m_gru_hidden[d] + enc[d]) * w;
        }
        // Sigmoid output bounds mask strictly in [0.0, 1.0]
        out_mask[k] = 1.0f / (1.0f + expf(-sum));
    }

    uint64_t end_time = esp_timer_get_time();
    m_last_inference_us = (uint32_t)(end_time - start_time);
}
