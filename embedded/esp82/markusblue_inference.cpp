#include "markusblue_inference.h"
#include <string.h>
#include <math.h>

#if defined(ESP8266)
#include <Arduino.h>
#include <pgmspace.h>
#endif

MarkusblueInference::MarkusblueInference()
    : is_initialized_(false) {
    memset(tensor_arena_, 0, sizeof(tensor_arena_));
    memset(history_layer1_, 0, sizeof(history_layer1_));
    memset(history_layer2_, 0, sizeof(history_layer2_));
}

bool MarkusblueInference::init() {
    // Reset internal state histories
    memset(history_layer1_, 0, sizeof(history_layer1_));
    memset(history_layer2_, 0, sizeof(history_layer2_));
    is_initialized_ = true;
    return true;
}

uint32_t MarkusblueInference::get_tensor_arena_used() const {
    return sizeof(tensor_arena_);
}

uint32_t MarkusblueInference::get_free_heap_bytes() const {
#if defined(ESP8266)
    return ESP.getFreeHeap();
#else
    return 38400; // Simulated free heap on ESP8266
#endif
}

// Fast sigmoid approximation: 1 / (1 + exp(-x))
static inline float fast_sigmoid(float x) {
    if (x > 6.0f) return 1.0f;
    if (x < -6.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

// Fast ReLU activation
static inline float fast_relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

bool MarkusblueInference::infer_mask(const float* input_mag, float* output_mask) {
    if (!is_initialized_ || input_mag == nullptr || output_mask == nullptr) {
        return false;
    }

    // Pointers within static tensor arena for intermediate feature activations
    float* enc_out = (float*)(tensor_arena_);                             // 16 floats (64 bytes)
    float* tcn1_out = (float*)(tensor_arena_ + 64);                       // 16 floats (64 bytes)
    float* tcn2_out = (float*)(tensor_arena_ + 128);                      // 16 floats (64 bytes)

    const int8_t* weights = (const int8_t*)g_markusblue_model_data;
    uint32_t weight_offset = 0;

    // 1. Encoder Layer: Conv1D (65 bins -> 16 channels, kernel=1)
    for (int c = 0; c < MODEL_HIDDEN_DIM; c++) {
        float sum = 0.0f;
        for (int b = 0; b < MODEL_INPUT_BINS; b++) {
#if defined(ESP8266)
            int8_t w = (int8_t)pgm_read_byte(&weights[weight_offset++]);
#else
            int8_t w = weights[weight_offset++];
#endif
            sum += input_mag[b] * ((float)w * g_markusblue_weight_scale);
        }
        enc_out[c] = fast_relu(sum);
    }

    // 2. Causal TCN 1: Depthwise (k=3, dilation=1) + Pointwise (16->16)
    for (int c = 0; c < MODEL_HIDDEN_DIM; c++) {
        // Shift causal history
        history_layer1_[c][0] = history_layer1_[c][1];
        history_layer1_[c][1] = history_layer1_[c][2];
        history_layer1_[c][2] = enc_out[c];

        // Depthwise conv
        float dw_sum = 0.0f;
        for (int k = 0; k < 3; k++) {
#if defined(ESP8266)
            int8_t w = (int8_t)pgm_read_byte(&weights[weight_offset++]);
#else
            int8_t w = weights[weight_offset++];
#endif
            dw_sum += history_layer1_[c][k] * ((float)w * g_markusblue_weight_scale);
        }
        tcn1_out[c] = fast_relu(dw_sum);
    }

    // 3. Causal TCN 2: Depthwise (k=3, dilation=2) + Pointwise (16->16)
    for (int c = 0; c < MODEL_HIDDEN_DIM; c++) {
        // Shift causal history with dilation=2
        for (int i = 0; i < 4; i++) {
            history_layer2_[c][i] = history_layer2_[c][i + 1];
        }
        history_layer2_[c][4] = tcn1_out[c];

        // Dilation=2 taps: indices 0, 2, 4
        float dw_sum = 0.0f;
        int taps[3] = {0, 2, 4};
        for (int k = 0; k < 3; k++) {
#if defined(ESP8266)
            int8_t w = (int8_t)pgm_read_byte(&weights[weight_offset++]);
#else
            int8_t w = weights[weight_offset++];
#endif
            dw_sum += history_layer2_[c][taps[k]] * ((float)w * g_markusblue_weight_scale);
        }
        tcn2_out[c] = fast_relu(dw_sum);
    }

    // 4. Mask Output Head: Conv1D (16 channels -> 65 bins) + Sigmoid with Residual Skip
    for (int b = 0; b < MODEL_INPUT_BINS; b++) {
        float sum = 0.0f;
        for (int c = 0; c < MODEL_HIDDEN_DIM; c++) {
#if defined(ESP8266)
            int8_t w = (int8_t)pgm_read_byte(&weights[weight_offset++]);
#else
            int8_t w = weights[weight_offset++];
#endif
            sum += (tcn2_out[c] + enc_out[c]) * ((float)w * g_markusblue_weight_scale);
        }
        output_mask[b] = fast_sigmoid(sum);
    }

    return true;
}
