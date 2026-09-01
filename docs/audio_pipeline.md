# MARKUSBLUE Audio DSP Pipeline & Loudness Control

## 1. Streaming DSP Stage Architecture

```
                  Raw Input Frame (64 samples = 8.0 ms)
                                    │
                                    ▼
                     Ring Buffer (128 samples)
                                    │
                                    ▼
                      Windowed STFT (128-pt FFT)
                                    │
                                    ▼
                  Spectral Magnitude [65 bins] + Phase [65 bins]
                                    │
                                    ▼
                 MARKUSBLUE INT8 Mask Multiplier
                   (enhanced_mag = mag * mask)
                                    │
                                    ▼
                     Inverse STFT Overlap-Add
                     (50% Hanning Reconstruction)
                                    │
                                    ▼
                       Voice Activity Detector
                      (Tracks running noise PSD)
                                    │
                                    ▼
                      Automatic Gain Control
                   (Compensates for attenuation;
                    gated to 1.0 during pauses)
                                    │
                                    ▼
                       Lookahead Peak Limiter
                     (Soft Knee Clamping < 0.95)
                                    │
                                    ▼
                        Enhanced Output Frame
```

---

## 2. Solving Speech Attenuation Without Noise Breathing

- **Problem**: Neural masking suppresses noise but can drop voice RMS by 3–6 dB.
- **Solution**:
  - Voice Activity Detector measures speech power relative to running noise PSD.
  - When speech is active ($\text{SNR} > 4.0\text{ dB}$), gain smoothly transitions towards target RMS 0.32 ($\text{attack rate} = 0.05$).
  - When speech is inactive (ambient silence/pauses), gain decays smoothly to 1.0 ($\text{decay rate} = 0.005$) rather than boosting, preventing noise pumping/breathing.
  - Peak Limiter uses a $\tanh$ soft saturation curve above 0.95 amplitude to prevent digital clipping.
