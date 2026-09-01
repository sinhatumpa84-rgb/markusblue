#ifndef MARKUSBLUE_INFERENCE_H_
#define MARKUSBLUE_INFERENCE_H_

#include <stdint.h>
#include "config.h"
#include "markusblue_model_data.h"

class MarkusblueInference {
public:
    MarkusblueInference();
    bool init();
    
    // Process single spectral frame (65 bins)
    // Takes magnitude spectrum in, outputs speech gain mask [0.0, 1.0]
    bool infer_mask(const float* input_mag, float* output_mask);
    
    // Runtime memory diagnostics
    uint32_t get_tensor_arena_used() const;
    uint32_t get_free_heap_bytes() const;

private:
    // Statically allocated tensor memory arena (zero dynamic allocation)
    alignas(4) uint8_t tensor_arena_[TENSOR_ARENA_BYTES];
    
    // Receptive field temporal history buffers for causal depthwise convolutions
    float history_layer1_[MODEL_HIDDEN_DIM][3];
    float history_layer2_[MODEL_HIDDEN_DIM][5];
    
    bool is_initialized_;
};

#endif // MARKUSBLUE_INFERENCE_H_
