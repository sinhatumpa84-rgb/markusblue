#ifndef I2S_CONFIG_H
#define I2S_CONFIG_H

#include <Arduino.h>
#include "driver/i2s.h"

// Sample Rate and Framing
#define I2S_SAMPLE_RATE     16000
#define I2S_BITS_PER_SAMPLE I2S_BITS_PER_SAMPLE_32BIT
#define I2S_DMA_BUF_COUNT   8
#define I2S_DMA_BUF_LEN     64 // 64 samples = 4.0 ms buffer

// I2S0 Port: Dual INMP441 Microphones (Stereo RX)
#define I2S_MIC_PORT        I2S_NUM_0
#define I2S_MIC_BCLK_PIN    GPIO_NUM_4
#define I2S_MIC_WS_PIN      GPIO_NUM_5
#define I2S_MIC_DATA_IN_PIN GPIO_NUM_6

// I2S1 Port: MAX98357A Class-D Amplifier (Mono/Stereo TX)
#define I2S_SPK_PORT        I2S_NUM_1
#define I2S_SPK_BCLK_PIN    GPIO_NUM_15
#define I2S_SPK_LRC_PIN     GPIO_NUM_16
#define I2S_SPK_DATA_OUT_PIN GPIO_NUM_17

class I2SConfig {
public:
    static esp_err_t initMicrophones();
    static esp_err_t initSpeaker();
    static esp_err_t uninstallAll();
};

#endif // I2S_CONFIG_H
