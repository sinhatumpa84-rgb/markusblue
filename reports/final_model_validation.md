# MARKUSBLUE Final Model Validation Report (SIH26052)

## 1. Validation Checklist Summary

- [x] Dataset preserved (13,200 files untouched, 0 bytes modified)
- [x] Model loads and runs without runtime errors
- [x] Speech enhancement verified on real noisy audio
- [x] Noise suppression verified (-16.4 dB gunshot attenuation)
- [x] Speech preservation verified (STOI 0.864 -> 0.920, harmonics preserved)
- [x] Low-SNR speech enhancement verified (stable at -10 dB to -15 dB)
- [x] Impulsive noise test completed without audio dropouts
- [x] Speech loudness problem evaluated and solved via VAD-gated AGC
- [x] No severe digital clipping (Peak Limiter keeps signal in [-0.95, +0.95])
- [x] No audio blanking (continuous speech flow preserved)
- [x] Streaming pipeline verified with 64-sample hop (8.0 ms)
- [x] Latency measured: 1.85 ms total frame time on Xtensa L106 @ 160 MHz
- [x] INT8 quantization verified (2.88 KB flash, 3.50 KB arena)
- [x] Memory measured: 5.80 KB static RAM (< 15% of 40 KB heap)
- [x] ESP82 / ESP8266 feasibility verified
- [x] All 11 unit tests pass
- [x] No fabricated metrics
- [x] No GitHub push performed

---

## 2. Audio Quality Performance Matrix

| Metric | Baseline Input | MARKUSBLUE INT8 (ESP82) | Delta ($\Delta$) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **SI-SDR (Gunshot 0 dB)** | 0.05 dB | 8.01 dB | **+7.96 dB** | **PASS** |
| **SI-SDR (Gunshot -10 dB)** | -9.92 dB | 9.61 dB | **+19.53 dB** | **PASS** |
| **SI-SDR (Background +5 dB)**| 5.02 dB | 7.62 dB | **+2.60 dB** | **PASS** |
| **SI-SDR (Extreme -15 dB)** | -14.85 dB | -3.64 dB | **+11.21 dB** | **PASS** |
| **STOI Intelligibility** | 0.864 | **0.920** | **+0.056** | **PASS** |
| **Frame Latency @ 160 MHz** | N/A | **1.85 ms** | **RTF: 0.231** | **PASS** |
