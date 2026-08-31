# MARKUSBLUE v7.0.00 — Technical Model & Pipeline Audit

## 1. Executive Summary
This document provides an exhaustive, empirical baseline audit of the existing MARKUSBLUE v7.0.00 repository prior to the Speech Enhancement Student-Teacher Neural Redesign.

---

## 2. Technical Audit Parameters

| Item | Property | Current Repository State |
|---|---|---|
| **1** | **Model Architecture** | `ESP32EdgeCNN` (Model B: Depthwise-Separable 2D CNN with 3 blocks + GAP + Linear) & `BaselineCNN` (Model A: 2D Residual CNN). |
| **2** | **Input Format** | 2D Log-Mel Spectrogram tensor `[Batch, 1, 32, 32]` (32 Mel bins $\times$ 32 time steps). |
| **3** | **Output Format** | 4-class classification logits `[Batch, 4]`. |
| **4** | **Audio Sample Rate** | $16,000\text{ Hz}$ (16 kHz, single-channel mono PCM). |
| **5** | **Frame / Window Size** | DMA Block: 256–512 samples ($16-32\text{ ms}$); Sliding Spectrogram Window: 16,000 samples ($1.0\text{ s}$); Hop: 400 samples ($25\text{ ms}$). |
| **6** | **Feature Extraction** | Short-Time Fourier Transform (STFT) with 512-point FFT, 32 Mel filterbanks, logarithmic dynamic range compression. |
| **7** | **Class Labels** | `0: DANGEROUS_IMPULSE` (gunshot), `1: NORMAL_SPEECH` (voice), `2: BACKGROUND_NOISE` (ambient), `3: OTHER_IMPULSE` (non-lethal). |
| **8** | **Training Objective** | Acoustic scene and impulse event classification. |
| **9** | **Loss Function** | Cross-Entropy Loss with label smoothing ($\alpha=0.05$) + AdamW optimizer. |
| **10** | **Dataset Structure** | 40,807 total WAV files across `datasets/` (13,200 standard benchmark files: 6,000 gunshot, 2,400 speech, 2,400 background, 2,400 other_impulse), `data/processed/`, and `data/extracted/`. |
| **11** | **Preprocessing Pipeline** | Peak normalization, dynamic time-domain cropping, SpecAugment (frequency/time masking), Gaussian white noise injection. |
| **12** | **Inference Pipeline** | Streaming Mel-spectrogram buffer $\rightarrow$ CNN forward pass $\rightarrow$ State Machine update $\rightarrow$ Deterministic Limiter & Voice Bandpass DSP Filter. |
| **13** | **TFLite Model** | INT8 quantized flatbuffer `model/markusblue_v7.0.00_int8.tflite` (4,160 bytes). |
| **14** | **ESP Target Hardware** | **ESP32-S3** Dual-Core Xtensa LX7 @ 240 MHz (INMP441 I2S Mic + MAX98357A I2S Amp). *Note: ESP8266 lacks FPU/RAM for neural models*. |
| **15** | **Model Parameter Count & Size** | 3,916 parameters; Float32: $14.9\text{ KB}$; INT8: $3.82\text{ KB}$. |
| **16** | **Inference RAM Footprint** | Peak SRAM during inference: $< 24.8\text{ KB}$ (TFLite Micro Arena $\approx 30\text{ KB}$). |
| **17** | **Inference Latency** | Desktop GPU: $0.61\text{ ms}$; Desktop CPU: $1.73\text{ ms}$; ESP32-S3 estimated: $11.93\text{ ms}$ @ 240 MHz. |

---

## 3. Key Findings & The "Weak Loudness" Problem
1. **Classification vs. Waveform Enhancement**: The legacy system classified audio into discrete labels and applied static IIR bandpass filters (300 Hz–3.4 kHz) rather than directly reconstructing the clean speech waveform $\hat{s}(t)$.
2. **Downward Safety Attenuation**: When the transient limiter triggers, it attenuates the entire signal by up to $-28\text{ dB}$, dropping voice RMS to $<0.03$ ($-30\text{ dBFS}$) without a post-filtering Automatic Gain Control (AGC) or Dynamic Range Compressor (DRC) stage.
3. **Redesign Mandate**: The student model must directly estimate clean speech masks/waveforms, paired with a causal VAD-aware AGC and lookahead limiter.
