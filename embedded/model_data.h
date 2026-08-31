// Auto-generated TFLite Micro model weights array for SIH26052
// Target: ESP32-S3 N16R8 (Tactical Edge-AI Hearing Protection System)
#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const unsigned char g_tactical_model_data[];
extern const unsigned int g_tactical_model_data_len;

#define MODEL_INPUT_CHANNELS   1
#define MODEL_INPUT_MEL_BINS   32
#define MODEL_INPUT_TIME_STEPS 32
#define MODEL_NUM_CLASSES      4

#define CLASS_DANGEROUS_IMPULSE 0
#define CLASS_NORMAL_SPEECH     1
#define CLASS_BACKGROUND_NOISE  2
#define CLASS_OTHER_IMPULSE     3

#ifdef __cplusplus
}
#endif

#endif // MODEL_DATA_H_
