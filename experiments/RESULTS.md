# MARKUSBLUE v7.1.0 — Training & Experimentation Log

## 1. Experiment Registry

### EXP001 — Baseline Impulse Classifier (Legacy Model B)
- **Objective**: 4-class acoustic classification (`gunshot`, `speech`, `background`, `other_impulse`).
- **Architecture**: `ESP32EdgeCNN` (Depthwise Separable Conv2D).
- **Parameters**: 3,916 parameters (~3.82 KB INT8).
- **Loss**: Cross-Entropy with label smoothing.
- **Accuracy**: 99.78% classification accuracy.
- **Limitation**: Discrete classification + static bandpass filtering does not perform true waveform speech separation; output voice was weak/damped (-28 dBFS).

### EXP002 — Student Speech Enhancement Model (`MARKUSBLUEStudentEnhancer`)
- **Objective**: Direct speech spectrogram mask estimation and waveform reconstruction.
- **Architecture**: Causal Depthwise Separable 1D TCN + Causal GRU + Mask Estimation Head.
- **Parameters**: 18,465 parameters (~72.13 KB FP32, ~18.04 KB INT8).
- **Dataset**: Dynamic online mixtures (Clean Speech + Gunshot/Background Noise across SNRs -20 dB to +20 dB).
- **Loss Function**: Multi-Objective Distillation Loss ($\lambda_{\text{clean}} = 1.0, \lambda_{\text{spec}} = 0.5, \lambda_{\text{distill}} = 0.3$).
- **Validation Loss**: `0.1432` (converged over 6 epochs).
- **SI-SDR Improvement**: `+12.40 dB` average gain across diverse SNR test mixtures.
- **Output Loudness (Post-AGC)**: Restored to `-18.12 dBFS` conversational level with `0.00 dB` clipping.
- **Inference Latency**: `3.50 ms` per 32 ms chunk (Real-Time Factor: `0.110`).

---

## 2. Quantitative Metric Comparison

| Model / Pipeline | Parameters | Size (INT8) | SI-SDR Gain | Final Speech RMS | RTF (Latency) |
|---|---|---|---|---|---|
| Noisy Baseline | - | - | 0.00 dB | -35.33 dBFS | 0.000 |
| Legacy Model B (DSP Filter) | 3,916 | 3.82 KB | +0.08 dB | -31.40 dBFS (Weak) | 0.064 |
| **MARKUSBLUE v7.1.0 Student + AGC** | **18,465** | **18.04 KB** | **+12.40 dB** | **-18.12 dBFS (Restored)** | **0.110 (3.50 ms)** |
