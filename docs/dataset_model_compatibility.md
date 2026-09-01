# MARKUSBLUE Dataset ↔ Model Compatibility Audit

## 1. Dimensional & Processing Verification Matrix

| Parameter | Dataset Native | Training Pipeline | Inference (ESP82) | Status | Notes / Adaptations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sample Rate** | 16,000 Hz | 8,000 Hz (Decimated) | 8,000 Hz | **PASS** | 2:1 decimation preserves telephony voice-band 300–3,400 Hz |
| **Audio Channels** | Mono (1 ch) | Mono (1 ch) | Mono (1 ch) | **PASS** | 100% matched |
| **Bit Depth** | 16-bit PCM | Float32 normalized | 16-bit PCM / Float32 | **PASS** | Direct I2S INT16 compatibility |
| **Window / FFT Size**| N/A | 128 samples (16.0 ms) | 128 samples (16.0 ms)| **PASS** | 65 real frequency bins |
| **Hop Size** | N/A | 64 samples (8.0 ms) | 64 samples (8.0 ms) | **PASS** | 50% Hanning overlap-add |
| **Dynamic Range** | [-1.0, 1.0] | Normalized RMS | Normalized + AGC | **PASS** | Continuous gain tracking |
| **SNR Distribution** | Diverse | [-20 dB, +20 dB] | Real tactical range | **PASS** | Generalized across clean & heavy noise |

---

## 2. Leakage & Contamination Audit
- **Train / Validation / Test Split**: Partitioned by unique file IDs (85% train, 15% validation/test).
- **Data Leakage**: **None detected**. Clean speech files and noise samples are strictly isolated between training and evaluation splits.
- **Normalization Integrity**: Dynamic peak normalization during online mixing preserves relative SNR scaling.
