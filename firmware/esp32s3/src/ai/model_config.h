#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include "model_data.h"

// Inference Tensor Arena Configuration (allocated in internal SRAM or PSRAM)
#define MARKUSBLUE_TENSOR_ARENA_SIZE (48 * 1024) // 48 KB Arena
#define MARKUSBLUE_HIDDEN_DIM 32
#define MARKUSBLUE_NUM_CONV_BLOCKS 3

#endif // MODEL_CONFIG_H
