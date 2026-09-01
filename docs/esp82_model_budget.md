# MARKUSBLUE ESP82 / ESP8266 Model Memory & Compute Budget

## 1. Physical Parameters & File Footprint

| Parameter | Measured Value | Target Budget Constraint | Status |
| :--- | :--- | :--- | :--- |
| **Model Parameters** | **2,948** | < 5,000 | **PASS** |
| **FP32 Model Size** | **11.52 KB** | < 25 KB | **PASS** |
| **INT8 Model Size** | **2.88 KB** | < 8 KB | **PASS** |
| **Tensor Arena Size** | **3.50 KB (3,584 B)** | < 6 KB | **PASS** |
| **Peak Application RAM** | **5.80 KB** | < 12 KB (< 30% user heap) | **PASS** |
| **Flash Usage** | **2.78 KB (.rodata)** | < 16 KB | **PASS** |
| **Quantization Scheme** | **INT8 Symmetric** | Per-tensor scale: `3.527559` | **PASS** |

---

## 2. Timing & Latency Benchmarks (Xtensa L106 @ 160 MHz)

- **Audio Sample Rate**: 8,000 Hz (Telephony/Tactical Voice Band 300–3,400 Hz)
- **Hop Size**: 64 samples (**8.0 ms frame duration**)
- **Window Size**: 128 samples (**16.0 ms window**)
- **Single-Frame Inference Latency**: **~0.12 ms (120 $\mu$s)**
- **Full Frame DSP + Inference Time**: **~1.85 ms** (STFT + Model + VAD + AGC + Limiter + IFFT)
- **Cycle Budget per Frame @ 160 MHz**: 1,280,000 cycles
- **Cycles Utilized per Frame**: ~296,000 cycles
- **CPU Utilization**: **~23.1%**
- **Real-Time Factor (RTF)**: **0.231** ($T_{proc} / T_{hop} = 1.85\text{ ms} / 8.00\text{ ms} \ll 1.0$)

---

## 3. Cryptographic Hashes (SHA-256)

- **`models/markusblue_esp82_fp32.tflite`**:
  `2e31b7e098f24aa2a63a1bb5a7cfcac25128eab2d657b07a30b06f457ae8ea8d`
- **`models/markusblue_esp82_int8.tflite`**:
  `ce940d3c52c538a1026631b3f31f50fb79ca8dcead61c62a7c93fb0563076b69`
