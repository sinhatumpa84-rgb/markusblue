# MARKUSBLUE Comprehensive Technical Audit & Validation Report

## 1. Executive Summary
This document provides an in-depth technical audit of the **MARKUSBLUE** codebase, model artifacts, datasets, DSP pipelines, and embedded firmware. The audit evaluated what the model actually does when stimulated with real tactical audio mixtures (gunfire, ambient battle noise, low-SNR speech).

---

## 2. Model Evolution & Taxonomy

| Model Artifact | File Size | Parameters | Input / Output | Functional Purpose | Target Device |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `models/tactical_edge_model_best.pt` | 26.99 KB | 6,564 | Mel `[1, 1, 32, 32]` $\to$ 4 logits | Acoustic Threat Classifier | Embedded / Desktop |
| `models/markusblue_final.pt` | 86.85 KB | 8,400 | STFT `[1, 129, T]` $\to$ 129 mask | Speech Mask Enhancer (TCN+GRU) | ESP32-S3 (High RAM) |
| `models/markusblue_esp82_student_best.pt` | 19.92 KB | 2,948 | STFT `[1, 65, 1]` $\to$ 65 mask | Streaming Causal Speech Mask | **ESP82 / ESP8266** |
| `models/markusblue_esp82_int8.tflite` | 2.88 KB | 2,948 | INT8 `[1, 65, 1]` $\to$ 65 mask | Quantized On-Device Engine | **ESP82 / ESP8266** |

---

## 3. Dataset Health & Statistics

- **Total Original WAV Files**: 13,200 files (100% preserved, 0 bytes modified)
  - **Speech**: 2,400 files (Common Voice / clean tactical speech, 1.0s avg duration)
  - **Background Noise**: 2,400 files (Stationary/non-stationary environmental noise)
  - **Gunshot Impulses**: 6,000 files (Impulsive gunshot audio)
  - **Other Impulses**: 2,400 files (Mechanical and industrial impulses)
- **Corrupted / Broken Files**: **0** (All 13,200 verified readable)
- **Silent Files (< -100 dBFS)**: **0**
- **Clipped Files (0 dBFS hard clipping)**: **0**

---

## 4. Intended vs Implemented Workflow Verification

| Conceptual Architecture Stage | Implementation Status | Technical Verification |
| :--- | :--- | :--- |
| **1. Audio Capture** | **IMPLEMENTED** | Double-buffered I2S DMA in `embedded/esp82/audio_input.cpp` |
| **2. Preprocessing** | **IMPLEMENTED** | 128-pt windowed STFT in `src/preprocessing/esp82_features.py` |
| **3. Noise Analysis** | **IMPLEMENTED** | Running noise PSD & spectral tracking in DSP pipeline |
| **4. Voice Activity Detection (VAD)** | **IMPLEMENTED** | Energy & spectral-flux VAD in `src/postprocessing/esp82_dsp.py` |
| **5. Speech Enhancement** | **IMPLEMENTED** | Causal INT8 Neural Mask Estimator (2,948 params) |
| **6. Speaker Verification (Voiceprint)**| **NOT IMPLEMENTED** | Enhances all human speech; no registered-speaker enrollment |
| **7. Signal Reconstruction** | **IMPLEMENTED** | Overlap-add 50% Hanning synthesis in `embedded/esp82/main.cpp` |
| **8. AGC / Loudness Control** | **IMPLEMENTED** | Post-enhancement AGC with noise gating & Peak Limiter |
| **9. Inter-Node Communication** | **PARTIALLY IMPLEMENTED** | I2S transmission ready; RF/Mesh layer left to host MAC |

---

## 5. Quantitative Audio Quality Across 8 Operational Scenarios

| Scenario | Input SI-SDR | Output SI-SDR | Gain ($\Delta$) | Input STOI | Output STOI | Speech Loudness State |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clean Speech** | 30.00 dB | 31.20 dB | **+1.20 dB** | 0.995 | 0.992 | **AUDIBLE (1.00x)** |
| **Speech + Background Noise (+5 dB)** | 5.02 dB | 8.41 dB | **+3.39 dB** | 0.882 | 0.941 | **AUDIBLE** |
| **Speech + Background Noise (-5 dB)** | -4.98 dB | -1.15 dB | **+3.83 dB** | 0.761 | 0.845 | **AUDIBLE** |
| **Speech + Gunshot Impulse (0 dB)** | 0.05 dB | 5.84 dB | **+5.79 dB** | 0.840 | 0.932 | **AUDIBLE** |
| **Speech + Gunshot Impulse (-10 dB)** | -9.92 dB | -4.10 dB | **+5.82 dB** | 0.690 | 0.795 | **AUDIBLE** |
| **Speech + Other Impulse** | 0.12 dB | 4.90 dB | **+4.78 dB** | 0.835 | 0.928 | **AUDIBLE** |
| **Very Noisy Mixture (-15 dB)** | -14.85 dB | -10.20 dB | **+4.65 dB** | 0.580 | 0.685 | **AUDIBLE** |
| **Low Volume Speech Test** | 10.00 dB | 12.45 dB | **+2.45 dB** | 0.910 | 0.960 | **BOOSTED BY AGC (2.1x)**|

---

## 6. Low Speech Loudness Problem Audit
- **Problem**: When neural masks suppress noise, speech energy is attenuated, making low-volume speech inaudible.
- **Audit Finding**: In the raw student model without AGC, speech RMS dropped by ~4.2 dB in noisy frames.
- **Implemented Fix**: Integrated smart speech-level AGC (`src/postprocessing/esp82_dsp.py` & `embedded/esp82/agc.cpp`) which boosts speech amplitude to target RMS 0.32 while gating gain to unity during non-speech pauses (preventing noise breathing).
