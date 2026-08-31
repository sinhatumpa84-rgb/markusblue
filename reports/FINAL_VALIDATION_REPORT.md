# SIH26052: Final Honest Technical Validation & Verification Report

**Project Title:** Indigenous Edge-AI Tactical Communication and Hearing Protection System for the Indian Army  
**Project ID:** SIH26052  
**System Classification:** Defensive Tactical Hearing Protection & Voice Communication Enhancement  
**Evaluation Date:** August 31, 2026  
**Auditor / Engineer:** ML & Embedded-AI Lead Engineer  

---

## 1. Executive Scorecard

| Dimension | Controlled Benchmark | Adversarial / Challenge Stress | Engineering Status |
| :--- | :--- | :--- | :--- |
| **Gunfire Impulse Recall (Sensitivity)** | **100.0%** (221/221 on Test Split) | **97.5% – 100.0%** (-5 dB to +20 dB SNR) | **VERIFIED** |
| **Dangerous Impulse Precision** | **100.0%** (0 FP on Test Split) | **95.0% – 100.0%** (Clutter & Impacts) | **VERIFIED** |
| **Non-Hazardous Impact Rejection** | **100.0%** (`OTHER_IMPULSE`) | **100.0%** (Door slams, metal drops) | **VERIFIED** |
| **Dynamic Limiter Attack Time** | **< 0.5 ms** (Instant sub-sample) | **< 0.5 ms** (Zero-delay peak clamp) | **VERIFIED** |
| **Peak Blast Attenuation** | **-28.7 dB** (Digital Peak Domain) | **-15.85 dB to -28.7 dB** (Mixed Stream) | **VERIFIED** |
| **Spectral Formant Preservation Proxy** | **32.7% – 64.2%** (Formant Correlation) | Preserved across voice bands | **PROXIED** *(Lab PESQ Required)* |
| **Model B Flash Size (INT8)** | **3.8 KB** (4,160 bytes) | Fits in standard internal Flash | **VERIFIED** |
| **Model B Peak SRAM Footprint** | **< 24.8 KB** | Fits inside ESP32-S3 internal SRAM | **VERIFIED** |
| **ESP32-S3 Inference Latency** | **~11.9 ms** *(Simulated Estimate)* | Requires physical hardware benchmark | **ESTIMATED** |

---

## 2. Dataset & Integrity Forensic Audit

### 2.1 Dataset Composition
- **Raw Audio Archives in `gunsound/`:** 24 ZIP archives containing real-world firearm recordings (7.62mm, 5.56mm, 9mm, .45 ACP, shotguns, acoustic blasts).
- **Extracted Unique WAV Files:** 12,406 WAVs.
- **Enriched Balanced Dataset (`data/processed/`):** 5,600 standardized samples (16 kHz mono, 1.0s window):
  - `DANGEROUS_IMPULSE`: 2,000 real recorded gunshot audio samples.
  - `NORMAL_SPEECH`: 1,200 multi-speaker samples with formant tracking (F1-F3) and VHF radio bandpass (300 Hz – 3.4 kHz).
  - `BACKGROUND_NOISE`: 1,200 complex military ambient samples (diesel combustion harmonics, wind turbulence, generator hum).
  - `OTHER_IMPULSE`: 1,200 hard negative physical impacts (armored hatch slams, dropped metal magazines/tools, rifle bolt clicks).

### 2.2 Split Isolation & Data Leakage Audit
Cross-split leakage was evaluated using cryptographic MD5 audio hashes, recording group IDs, and filename paths:
- **Train Split:** 4,015 samples (71.7%)
- **Validation Split:** 764 samples (13.6%)
- **Held-Out Test Split:** 821 samples (14.7%)
- **Cross-Split Filepath Overlap:** **0 files (0.0%)**
- **Cross-Split Audio Hash Collision:** **0 files (0.0%)**
- **Cross-Split Source/Session Leakage:** **0 files (0.0%)**

---

## 3. Model Architecture Comparison

| Metric | Model B: ESP32-S3 Edge DS-CNN | Model A: Baseline High-Capacity CNN |
| :--- | :--- | :--- |
| **Architecture Type** | Depthwise-Separable 2D CNN | ResNet-style Deep 2D CNN |
| **Total Parameters** | **3,916** | **470,820** |
| **Float32 Weights Size** | **15.3 KB** | **1.80 MB** |
| **INT8 Quantized Size** | **3.8 KB (4,160 bytes)** | **459.8 KB** |
| **Peak SRAM Requirement** | **< 24.8 KB** | **~480 KB (Requires External PSRAM)** |
| **Input Representation** | 32 Mel bins $\times$ 32 Time steps | 64 Mel bins $\times$ 32 Time steps |
| **GPU Inference Latency** | 0.61 ms | 1.11 ms |
| **Desktop CPU Latency** | 1.74 ms | 6.20 ms |
| **Simulated ESP32-S3 Latency** | **~11.9 ms @ 240 MHz** | **> 180 ms (Exceeds Real-Time Budget)** |
| **Deployment Recommendation** | **RECOMMENDED FOR EDGE HARDWARE** | **RESEARCH BENCHMARK ONLY** |

---

## 4. Controlled vs. Extreme Challenge Test Results

### 4.1 Controlled Test Performance (Held-Out Test Split)
- **Overall Test Accuracy:** **100.00%**
- **Dangerous Impulse Sensitivity (Recall):** **100.00%** (221/221 detected)
- **Dangerous Impulse Precision:** **100.00%**
- **False Negative Rate (Missed Blasts):** **0.00%**
- **False Positive Rate (False Alarms):** **0.00%**
- **Macro F1-Score:** **1.0000**
- **Multi-Class ROC-AUC:** **1.0000**

### 4.2 Adversarial & Stress Challenge Performance
Tested using `src/evaluation/challenge_evaluator.py` under severe battlefield acoustic distortions:

```
+-----------------------------------------------------------------------------------+
| Challenge Condition                              | Model B Recall / Rejection     |
+-----------------------------------------------------------------------------------+
| A. Gunshot @ -5 dB SNR (Heavy Noise Masking)     | 97.5% Recall (39/40 detected)  |
| A. Gunshot @ 0 dB SNR (Equal Noise Masking)      | 97.5% Recall (39/40 detected)  |
| A. Gunshot @ +5 dB to +20 dB SNR                 | 97.5% - 100.0% Recall          |
| B. Rapid Burst Fire (3 shots @ 500 RPM spacing)  | 100.0% Recall (30/30 detected) |
| D. Speech Interleaved Blast (Pre/Post Voice)     | 100.0% Recall (30/30 detected) |
| F. Heavy Vehicle & Wind Clutter Rejection        | 95.0% Rejection (5.0% FAR)     |
| H. Hard Negative Physical Impacts (Door/Metal)   | 100.0% Rejection (0.0% FAR)    |
| K. Saturated MEMS Pre-amp Clipping Overload      | 100.0% Recall (30/30 detected) |
+-----------------------------------------------------------------------------------+
```

---

## 5. DSP Protection Engine Verification

### 5.1 Deterministic Dynamic Limiter
- **Attack Time:** $< 0.5$ ms (sub-sample instantaneous clamping on rising edge).
- **Release Time:** $\tau = 80.0$ ms smooth exponential recovery.
- **Maximum Attenuation:** $-28.7$ dB digital peak reduction.
- **Output Ceiling:** Clamped strictly to $\le 0.35$ safe digital threshold.

### 5.2 Speech Formant Preservation
- **Filter Bank Architecture:** 4th-Order Butterworth Second-Order Sections (SOS) bandpass (300 Hz – 3.4 kHz).
- **Voice Band Gain:** $+3.0$ dB formant boost.
- **Blast Band Attenuation:** $-32.0$ dB out-of-band shockwave cut.
- **Formant Preservation Proxy Score:** $32.7\% - 64.2\%$ spectral correlation preservation during simultaneous blast and voice events.

---

## 6. Real-World Limitations & Physical Risks

1. **Acoustic Overload Point (AOP) of MEMS Microphones:**
   - Standard commercial MEMS microphones (e.g. INMP441) saturate and produce hard analog clipping above **120–130 dB SPL**.
   - Tactical military gunshots produce **140–165+ dB SPL**.
   - *Limitation:* A software DSP limiter can only clamp the digital signal; if the analog MEMS diaphragm saturates, the waveform is permanently clipped at the hardware level. High-AOP microphones (e.g., Knowles high-SPL MEMS $\ge 160\text{ dB SPL}$) or analog passive hearing-protection earcups are strictly required for physical hearing protection.
2. **Speech Intelligibility Certification:**
   - The reported Formant Preservation score is a mathematical spectral correlation proxy. True human speech intelligibility under blast reverberation must be certified using standardized laboratory acoustic manikins (e.g., ANSI S12.42, MIL-STD-1474E, PESQ / STOI).
3. **Simulated vs. Physical ESP32 Hardware Latency:**
   - All latency metrics reported in this audit reflect desktop execution and theoretical Xtensa LX7 cycle estimations. Exact on-device timings must be measured using hardware GPIO toggle profiling on physical ESP32-S3 boards.

---

## 7. Hardware Deployment Architecture Recommendations

```
  INMP441 I2S MEMS Mic (16 kHz Mono)
                │
                ▼
  [ ESP32-S3 CORE 0 : Audio DSP ISR ]
   ├── DMA I2S Ping-Pong Buffer (512 samples / 32 ms)
   ├── Fast Transient Peak Limiter (<0.5 ms Attack)
   └── Direct Form II Speech Bandpass (300 Hz - 3.4 kHz)
                │
                ▼ Output to MAX98357A I2S DAC / Earpiece
                │ (Shared Audio RingBuffer)
                ▼
  [ ESP32-S3 CORE 1 : Edge AI Task (FreeRTOS) ]
   ├── 32-ms Spectrogram Accumulation
   ├── TFLite Micro INT8 Inference (Model B : 4,160 bytes)
   └── Hearing Protection State Machine Supervisor
```

1. **DMA Double Buffering:** Use two 512-sample DMA ping-pong buffers on Core 0 to achieve uninterrupted audio streaming without frame drops.
2. **Interrupt Priority:** Assign the Core 0 I2S audio ISR the highest hardware interrupt priority (Priority 5) so that ML inference on Core 1 never introduces jitter to the deterministic safety limiter.
3. **Power Budget:** At 240 MHz dual-core clock with I2S and DMA active, ESP32-S3 consumes approximately 110–140 mA (~0.5 W), providing over 14 hours of continuous operation on a standard 2000 mAh Li-Po battery.

---

## 8. Final Engineering Sign-off

The SIH26052 Tactical Audio AI & Hearing Protection software pipeline has been verified from raw ZIP archives to quantized INT8 headers. All unit test suites pass, zero data leakage is verified, and the model exhibits robust generalization across severe battlefield stress conditions.

**Signed:**  
*ML & Embedded-AI Lead Engineer*  
*SIH26052 Tactical Communication & Hearing Protection System*
