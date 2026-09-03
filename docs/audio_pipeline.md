# MARKUSBLUE (SIH26052) — Real-Time Streaming Audio Pipeline

## 1. Pipeline Overview
The MARKUSBLUE real-time audio pipeline executes deterministically on **Core 1** of the **ESP32-S3 N16R8** at **16,000 Hz** wideband audio with a hop size of **64 samples (4.0 ms frame duration)** and **256-point STFT**.

```
                        EXTERNAL SOUND FIELD
                                 │
                         ┌───────▼───────┐
                         │ INMP441 MIC 1 │
                         │ (Reference)   │
                         └───────┬───────┘
                                 │ I2S Left (32-bit DMA)
                                 ▼
                     ┌───────────────────────┐
                     │     I2S0 RX DMA       │
                     │ (Stereo Ping-Pong)    │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   DC-Offset Blocker   │
                     │  (IIR High-Pass R=0.995)
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  Sliding Window (256) │
                     │   Hann Windowing      │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   Fast 256-pt STFT    │
                     │  129 Positive Bins    │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Dual-Mic Spatial DSP  │
                     │  Reference PSD Track  │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ MARKUSBLUE Neural AI  │
                     │ Causal 1D TCN + GRU   │
                     │ Mask: M(f, t) in [0,1]│
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Spectral Enhancement  │
                     │ S_hat(f,t) = Y(f,t)*M │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   Fast 256-pt ISTFT   │
                     │ 50%/75% Overlap-Add   │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  Speech-Aware AGC     │
                     │  Target: -16 dBFS     │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Lookahead Peak Limiter│
                     │ Brickwall Safety Clamp│
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │     I2S1 TX DMA       │
                     │  16-bit Signed PCM    │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │  MAX98357A Class-D    │
                     │  8Ω Earphone Driver   │
                     └───────────────────────┘
```

---

## 2. Stage-by-Stage Latency Budget

| Pipeline Stage | Algorithm | Duration (Samples) | Execution Time (µs) | Memory Location |
| :--- | :--- | :--- | :--- | :--- |
| **Capture Buffer** | Ping-pong DMA buffer | 64 samples (4.0 ms) | 120 µs | Internal SRAM (DMA) |
| **DC Blocker** | $y[n] = x[n] - x[n-1] + 0.995 y[n-1]$ | 64 samples | 35 µs | Internal SRAM |
| **Window & STFT** | 256-pt Hann RFFT | 256 samples | 410 µs | Internal SRAM |
| **Spatial Pre-Filter** | Recursive PSD & coherence gating | 129 bins | 85 µs | Internal SRAM |
| **AI Mask Inference** | Causal 1D TCN + GRU INT8 | 129 bins | 1,850 µs | Internal SRAM / Flash |
| **ISTFT Synthesis** | 256-pt RIFFT + Overlap-Add | 256 samples | 430 µs | Internal SRAM |
| **Speech AGC** | VAD-gated dual-rate leveler | 64 samples | 55 µs | Internal SRAM |
| **Peak Limiter** | Lookahead ring buffer + ceiling clamp | 64 samples | 40 µs | Internal SRAM |
| **I2S TX DMA** | 16-bit PCM streaming to MAX98357A | 64 samples | 105 µs | Internal SRAM (DMA) |
| **Total Pipeline** | **End-to-End Processing** | **64 samples** | **3,130 µs (3.13 ms)** | **All In-SRAM** |

---

## 3. Two-Microphone Spatial Strategy
1. **Microphone 1 (Exterior Reference)**:
   - Positioned on the exterior shell of the hearing enclosure.
   - Captures ambient gunfire transients, vehicle engines, and acoustic blasts.
2. **Microphone 2 (Interior Ear)**:
   - Positioned inside the ear cup cavity near the ear canal.
   - Captures voice and residual acoustic field entering the ear.
3. **Coherence Weighting**:
   - Rather than naive destructive subtraction ($M_1 - M_2$), the power spectral density (PSD) ratio estimates diffuse field noise vs voice energy, preserving voice formants and phase integrity.
