// MARKUSBLUE (SIH26052) — ESP32-S3 Auto-Generated Model Header
// Architecture: Causal Depthwise-Separable 1D TCN Speech Enhancement
// Quantization: INT8 Symmetric (Scale: 3.52755904, ZeroPoint: 0)
// Parameters: 18,725
#ifndef MARKUSBLUE_MODEL_DATA_H_
#define MARKUSBLUE_MODEL_DATA_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const unsigned char g_markusblue_model_data[];
extern const size_t g_markusblue_model_data_len;
extern const float g_markusblue_int8_scale;
extern const int32_t g_markusblue_int8_zero_point;
extern const size_t g_markusblue_param_count;
extern const size_t g_markusblue_num_bins;

#define MARKUSBLUE_N_FFT 256
#define MARKUSBLUE_HOP_LEN 64
#define MARKUSBLUE_NUM_BINS 129
#define MARKUSBLUE_SAMPLE_RATE 16000

#ifdef __cplusplus
}
#endif

#endif // MARKUSBLUE_MODEL_DATA_H_
