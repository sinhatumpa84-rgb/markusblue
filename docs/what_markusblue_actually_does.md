# What MARKUSBLUE Actually Does — Architectural Analysis

## 1. Evolutionary Lineage of MARKUSBLUE

1. **Acoustic Classifier (`models/tactical_edge_model_best.pt`)**:
   - 4-class Conv2D Spectrogram Classifier. Threat detection only.

2. **Large Neural Speech Enhancer (`models/markusblue_final.pt`)**:
   - 129-bin TCN + GRU Causal Mask Estimator (~8,400 params) for ESP32-S3.

3. **ESP82 Ultra-Lightweight Speech Enhancer (`models/markusblue_esp82_student_best.pt`)**:
   - 65-bin Causal Depthwise-Separable 1D TCN Mask Estimator (2,948 params, 2.88 KB INT8) for **ESP82 / ESP8266**.
   - Pipeline: Mic -> I2S DMA -> STFT -> INT8 Mask -> IFFT -> VAD -> AGC -> Limiter -> Speaker.

## 2. Registered-Speaker Voice Isolation
- **Status**: **NOT IMPLEMENTED**. Enhances universal human speech without voiceprint enrollment.
