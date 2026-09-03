# MARKUSBLUE — SIH26052 Requirement Compliance Matrix

| SIH26052 Requirement Area | System Implementation / Technical Solution | Subsystem Mapping | Verification Status | Evidence / Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **Real-Time Speech Enhancement** | Causal Depthwise-Separable 1D Conv/TCN Neural Mask Estimator (129 frequency bins, 16kHz) | AI Engine + DSP STFT | **SOFTWARE VERIFIED** | `models/markusblue_esp32s3_best.pt`, `evaluate_esp32s3.py` |
| **Tactical Impulse / Gunfire Noise Suppression** | Multi-resolution STFT loss trained on 6,000+ gunshot audio bursts across -15dB to +10dB SNR | AI Model + Spatial Pre-filter | **SOFTWARE VERIFIED** | `reports/esp32s3_evaluation_summary.json` |
| **Anti-Audio-Blanking (Speech Preservation)** | Instant transient recovery AGC + Lookahead Peak Limiter preventing mute/gate artifact | DSP AGC & Limiter | **SOFTWARE VERIFIED** | `reports/audio_samples/speech_enhanced_output.wav` (Preservation ratio: 1.79x) |
| **Dual-Microphone Strategy** | Mic 1 (Exterior Ref) + Mic 2 (Interior Ear) spatial coherence gating & PSD tracking | `TwoMicProcessor` | **SOFTWARE VERIFIED** | `firmware/esp32s3/src/dsp/two_mic_processor.cpp` |
| **Edge Hardware Architecture** | ESP32-S3 N16R8 (Dual-core 240MHz Xtensa LX7, 16MB Flash, 8MB PSRAM) | MCU Hardware | **VERIFIED (Architecture & Pinout)** | `hardware/pinout.md`, `firmware/esp32s3/platformio.ini` |
| **Audio I/O Interfaces** | 2 × INMP441 I2S MEMS Mics (RX) + MAX98357A I2S Class-D Amp (TX) driving 8Ω Speaker | I2S0 RX & I2S1 TX DMA | **VERIFIED (Code & Pin Map)** | `firmware/esp32s3/src/audio/i2s_config.cpp` |
| **Low Latency Target (< 20 ms end-to-end)** | 4.0 ms frame hop (64 samples @ 16kHz), ~3.2 ms processing pipeline latency | Real-Time Engine (Core 1) | **SOFTWARE VERIFIED (3.2 ms sim)** | `reports/esp32s3_evaluation_summary.json`, `firmware/esp32s3/src/system/diagnostics.cpp` |
| **Audio Safety & Clipping Prevention** | Sub-millisecond lookahead peak limiter with -0.5 dBFS ceiling & brickwall safety clamp | `PeakSafetyLimiter` | **SOFTWARE VERIFIED** | `firmware/esp32s3/src/dsp/limiter.cpp`, `src/limiter/peak_limiter.py` |
| **Portable Power System** | 3.7V 2500mAh Li-Po + TP4056 + 3.3V LDO + 5.0V Boost Converter (~11.8h active runtime) | Power Architecture | **ESTIMATED (Calculated Budget)** | `hardware/power_tree.md` |
| **Tactical User Interface** | 0.96" I2C OLED (SSD1306) + MPU6050 6-DOF IMU + Debounced PTT + Haptic Motor Alert | Core 0 Telemetry Task | **VERIFIED (Driver & Pin Map)** | `firmware/esp32s3/src/sensors/` |
| **Acoustic Feedback Isolation** | 3D-printed sealed ear cup with acoustic foam barrier and firmware gain clamping | Mechanical & Firmware | **PENDING HARDWARE VALIDATION** | `hardware/wiring.md` |
| **Hardware-in-the-Loop Field Test** | Physical PCB bench test with oscilloscope and calibrated acoustic mannequin ear | Complete Prototype | **PENDING HARDWARE VALIDATION** | Physical HIL Benchmarking |
