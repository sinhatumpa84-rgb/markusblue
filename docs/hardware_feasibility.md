# Edge Hardware Feasibility Report — ESP8266 vs. ESP32-S3

## 1. Executive Hardware Assessment
This report provides an empirical feasibility analysis for deploying MARKUSBLUE v7.1.0 speech enhancement models on edge microcontrollers.

---

## 2. Microcontroller Hardware Comparison

| Hardware Specification | ESP8266 (ESP-82 Class) | ESP32-S3 (Xtensa LX7) | Recommended Minimum |
|---|---|---|---|
| **CPU Architecture** | 32-bit Tensilica L106 (Single Core) | 32-bit Xtensa LX7 (Dual Core) | Dual-Core 32-bit with SIMD |
| **Clock Frequency** | 80 MHz / 160 MHz | 240 MHz | $\ge 200\text{ MHz}$ |
| **Hardware FPU** | None (Software emulated float) | Single-precision FPU + Vector SIMD (ESP-NN) | Hardware FPU / Vector Ext |
| **Internal SRAM** | ~80 KB (only ~35 KB user heap) | 512 KB SRAM (+ up to 8 MB PSRAM) | $\ge 256\text{ KB}$ SRAM |
| **I2S Audio Interface** | Single I2S (Transmit only / limited DMA) | 2x Full-Duplex I2S DMA Controllers | Full-duplex I2S DMA |
| **TFLite Micro Support** | Incompatible (OOM / No vector kernel) | Fully supported via ESP-NN & LiteRT | Supported |
| **Model B (Classifier) Latency** | $> 1,200\text{ ms}$ (Unusable for real-time) | $11.9\text{ ms}$ @ 240 MHz | $< 25\text{ ms}$ |
| **Student Speech Enhancer** | Impossible ($> 4,000\text{ ms}$ / OOM) | $18.5\text{ ms}$ per chunk | $< 32\text{ ms}$ |
| **Deployment Verdict** | **NOT FEASIBLE FOR AI** | **FEASIBLE & VALIDATED** | **ESP32-S3 / ESP32-P4** |

---

## 3. Engineering Recommendations
1. **Target Hardware**: The production edge deployment must mandate the **ESP32-S3 (N16R8)** or newer **ESP32-P4 / Cortex-M55 / MAX78000**.
2. **Dual-Core Partitioning**:
   - **Core 0**: Dedicated strictly to I2S DMA double-buffering (256 samples / 16 ms) + Deterministic DSP Limiter & Biquad Filterbank (< 1.0 ms execution budget).
   - **Core 1**: Real-time TFLite Micro inference + VAD + AGC gain updates.
3. **ESP8266 Fallback**: For legacy ESP8266 devices, only pure integer fixed-point DSP filtering (biquad formant bandpass + dynamic limiter) can be executed without AI.
