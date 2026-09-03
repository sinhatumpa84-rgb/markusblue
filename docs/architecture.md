# MARKUSBLUE (SIH26052) System Architecture

## 1. Executive Summary
MARKUSBLUE is an indigenous real-time Edge-AI tactical audio enhancement and active noise cancellation system designed for the Smart India Hackathon problem statement **SIH26052**. The platform processes high-noise combat acoustic environments—including gunfire impulse transients, vehicle engine roar, mechanical impact, and wind noise—to extract and amplify clean human speech with under 10 ms algorithmic processing latency.

---

## 2. Hardware Architecture

```
                  ┌────────────────────────────────────────┐
                  │          EXTERNAL NOISE FIELD          │
                  │ (Gunfire, Vehicle Engines, Wind, Amb)  │
                  └───────────────────┬────────────────────┘
                                      │
                              ┌───────▼───────┐
                              │ INMP441 MIC 1 │
                              │ (Reference)   │
                              └───────┬───────┘
                                      │ I2S (Left Channel)
                                      ▼
                        ┌───────────────────────────┐
                        │      ESP32-S3 N16R8       │
                        │ Dual-Core Xtensa @ 240MHz │
                        │  16MB Flash + 8MB PSRAM   │
                        │                           │
                        │  Core 1: Real-Time Audio  │
                        │   - I2S DMA Capture       │
                        │   - DC Bias Removal       │
                        │   - 256-pt Fast STFT      │
                        │   - Dual-Mic Spatial DSP  │
                        │   - Causal AI Mask Engine │
                        │   - Fast ISTFT Overlap-Add│
                        │   - VAD-Aware Dynamic AGC │
                        │   - Peak Limiter & Clamp  │
                        │   - I2S DMA TX Output     │
                        │                           │
                        │  Core 0: UI & Telemetry   │
                        │   - SSD1306 OLED (10 Hz)  │
                        │   - MPU6050 Motion IMU    │
                        │   - Debounced PTT & Haptic│
                        │   - MicroSD Async Logger  │
                        │   - Battery ADC Monitor   │
                        └─────────────┬─────────────┘
                                      │
                                      │ I2S (Mono PCM)
                                      ▼
                            ┌───────────────────┐
                            │    MAX98357A      │
                            │ Class-D Amplifier │
                            └─────────┬─────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │  8Ω SPEAKER   │
                              │ Earphone Unit │
                              └───────┬───────┘
                                      │
                                      ▼
                                  USER'S EAR
                                      │
                              ┌───────▼───────┐
                              │ INMP441 MIC 2 │
                              │ (Interior/Ear)│
                              └───────┬───────┘
                                      │ I2S (Right Channel)
                                      └───────► ESP32-S3
```

---

## 3. Streaming Audio Processing Chain

1. **Dual I2S Capture**: 16,000 Hz, 32-bit DMA stereo streaming from Reference (Mic 1) and Ear (Mic 2).
2. **DC Blocker**: High-pass infinite impulse response filter removes MEMS converter DC offset.
3. **Sliding Analysis Framing**: Hop size of 64 samples (4.0 ms frame duration) across 256-point Hann window.
4. **Fast STFT**: 129 positive frequency bins from 0 to 8,000 Hz.
5. **Spatial Noise Profiling**: Dual-microphone power spectral density smoothing and spatial coherence gating.
6. **Edge-AI Mask Estimation**: Causal Depthwise-Separable 1D Conv/TCN + GRU estimating bounded Ideal Ratio Mask $M(f, t) \in [0.0, 1.0]$.
7. **Spectral Filtering**: $\hat{S}(f, t) = Y(f, t) \cdot M(f, t)$ preserving clean phase information.
8. **Fast ISTFT**: Inverse transform with 50% / 75% overlap-add time-domain synthesis.
9. **Speech-Aware Dynamic AGC**: Restores muffled speech to -16 dBFS while freezing during non-speech intervals.
10. **Lookahead Peak Limiter & Safety Clamp**: Sub-millisecond lookahead buffer clamps output strictly within $[-0.999, +0.999]$ to prevent clipping.
11. **I2S Output**: 16-bit PCM transmitted via DMA to MAX98357A Class-D amplifier driving 8Ω earphone transducer.

---

## 4. Multi-Core FreeRTOS Task Pinning

| Task Name | CPU Core | Priority | Stack Size | Primary Responsibilities |
| :--- | :--- | :--- | :--- | :--- |
| **AudioProcessingTask** | **Core 1** | **24 (Highest)** | 8 KB (Internal SRAM) | Real-time I2S RX, STFT, AI inference, ISTFT, AGC, Limiter, I2S TX. Zero blocking calls. |
| **SystemControlTask** | **Core 0** | **5 (Normal)** | 4 KB | SSD1306 OLED rendering, MPU6050 polling, PTT debouncing, Battery ADC reading. |
| **SDLoggerTask** | **Core 0** | **2 (Low)** | 4 KB | Async MicroSD diagnostic logging from queue ring buffer. |
