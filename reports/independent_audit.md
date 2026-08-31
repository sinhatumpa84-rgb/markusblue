# SIH26052 — Independent Forensic Project Audit Report

---

### Executive Forensic Audit Summary

This independent forensic audit was conducted on the **SIH26052 Tactical Hearing Protection and Acoustic Detection System** codebase to evaluate the scientific validity, code integrity, data pipeline soundness, embedded feasibility, and test reproducibility.

---

### 1. What Actually Exists

| Component | Status | Verified Artifacts on Disk |
| :--- | :--- | :--- |
| **Raw Gunshot Datasets** | **Present** | 24 ZIP archives in `gunsound/` (2,810.15 MB total). 20 unique archives, 4 duplicate archives (`(1)` suffix). |
| **Extracted Raw Audio** | **Present** | 12,406 unique WAV files in `data/extracted/` across C3GD (8,017 WAVs), Edge (2,148 WAVs), and 18 Zoom recorder firearm folders (2,241 WAVs). |
| **Audio Processing Pipeline** | **Present** | `src/preprocessing/audio_pipeline.py` performing 16 kHz mono resampling, peak headroom preservation, and peak window extraction. |
| **Dataset Splits** | **Present** | `data/splits/train.csv` (6,720 rows), `validation.csv` (1,440 rows), `test.csv` (1,440 rows). |
| **Dual ML Model Architectures** | **Present** | `src/training/models.py`: Model A (Baseline ResNet-style CNN, 470,820 params) and Model B (ESP32-S3 Depthwise-Separable CNN, 3,916 params). |
| **Training Pipeline** | **Present** | `train.py` with Multi-Class Focal Loss, AdamW, and Cosine Annealing scheduler. |
| **Trained Model Weights** | **Present** | `models/tactical_edge_model_best.pt` (26.9 KB), `models/tactical_baseline_model_best.pt` (1.9 MB). |
| **Exported Quantized Models** | **Present** | `models/model_float32.tflite` (16.6 KB), `models/model_int8.tflite` (4.16 KB), `embedded/model_data.h`, `embedded/model_data.cc`. |
| **Deterministic DSP Engine** | **Present** | `src/dsp/dynamic_limiter.py` (sub-ms fast limiter), `src/dsp/speech_preservation.py` (biquad bandpass 300Hz-3.4kHz), `src/dsp/hearing_protection.py`. |
| **Embedded Firmware Prototype** | **Present** | `embedded/inference_example/main.cpp` (FreeRTOS dual-core I2S DMA + TFLM skeleton), `dsp_protection.h`. |
| **Diagnostics & CLI Suite** | **Present** | `evaluate.py`, `benchmark.py`, `realtime_demo.py`, `prepare_dataset.py`. |

---

### 2. What Is Missing

1. **Real-World Non-Gunfire Acoustic Audio in `gunsound/`**:
   - The provided `gunsound/` ZIP archives contain exclusively gunshot audio (C3GD and Zoom recordings). There are **no native real speech, real vehicle noise, or real negative physical impulse recordings** included in the archives.
2. **Physical ESP32-S3 Hardware Board Connected**:
   - The host system is a Windows workstation (Dell G15 with RTX 3050 Laptop GPU). No physical ESP32-S3 board is attached via UART/JTAG for real on-chip logic analyzer timing measurements.
3. **Calibrated SPL Sound Level Meter Data**:
   - The laboratory lacks calibrated 140–165 dB SPL acoustic shock tube measurement hardware. All blast attenuation figures are electrical/digital DSP peak domain calculations ($20 \log_{10} \frac{V_{\text{out}}}{V_{\text{in}}}$), not physical free-field acoustic SPL decibels.
4. **Standard Objective Speech Intelligibility Metrics (STOI / PESQ)**:
   - The original report reported a spectral formant energy ratio as "Speech Intelligibility", which is a mathematical energy proxy rather than a standardized psychoacoustic intelligibility metric.

---

### 3. What Is Executable

- `prepare_dataset.py`: Successfully extracts archives, deduplicates files, standardizes to 16 kHz PCM, and generates metadata splits.
- `train.py`: Trains Model A and Model B using PyTorch with CUDA acceleration on RTX 3050.
- `evaluate.py`: Runs full test set inference, confusion matrix generation, ROC/PR curves, and generates `reports/model_evaluation.html`.
- `export_tflite.py`: Exports PyTorch model to TFLite FlatBuffers and generates `embedded/model_data.cc` and `model_data.h`.
- `benchmark.py`: Benchmarks CPU/GPU latency and models memory footprint.
- `realtime_demo.py`: Runs continuous sliding-window streaming simulation and outputs `reports/realtime_demo_output.wav`.

---

### 4. What Is Broken or Flawed in Original Implementation

1. **Synthetic Class Bias (Root Cause of 100% Accuracy Claim)**:
   - In `src/preprocessing/audio_pipeline.py`, the non-gunfire classes (`NORMAL_SPEECH`, `BACKGROUND_NOISE`, `OTHER_IMPULSE`) were generated using simplistic parametric equations (pure harmonic sine waves, filtered Gaussian noise, synthetic decaying bursts).
   - Because these synthetic signals have mathematically clean spectra, the CNN effortlessly separated them from real microphone gunshot recordings with 100.0% accuracy on the test split.
   - **Flaw**: This creates an illusion of perfection that will fail when tested against realistic, complex audio clutter.
2. **Terminology Misalignment in DSP Evaluation**:
   - "32.1% Speech Preserved" was previously described as "Speech Intelligibility". It is strictly a **Spectral Formant Preservation Proxy** (energy ratio in the 300 Hz – 3.4 kHz passband during blast clamping).
3. **Hard Mute vs DSP Limiting Representation**:
   - Conventional electronic earmuffs cut off audio completely (hard clamp), losing 100% of speech during the recovery window. The DSP biquad filter allows band-limited voice through, but high-amplitude blast energy inside the 300Hz-3.4kHz passband will still contribute distortion.
4. **Windows Console cp1252 Unicode Encoding**:
   - Output print statements containing `[✓]` caused `UnicodeEncodeError` in Windows PowerShell under cp1252. (Now fixed to `[OK]`).

---

### 5. What Was Only Claimed vs Reality

| Claimed in Prior Report | Audited Reality | Verdict |
| :--- | :--- | :--- |
| **"100% Accuracy & 100% Recall"** | True **only on the idealized synthetic test split**. Misleading as an indicator of real-world military performance. | **Needs Adversarial / Hard Challenge Testing** |
| **"Sub-millisecond Protection (< 0.5 ms)"** | True **only for the deterministic Core 0 DSP Limiter**. False for the AI inference loop (which takes ~11.9 ms on ESP32-S3). | **Clarified Architectural Dual-Path** |
| **"-27.4 dB Blast Attenuation"** | Mathematically accurate for the digital limiter peak reduction ($20 \log_{10} \frac{0.0426}{1.00}$). Does not guarantee physical eardrum protection without calibrated hardware. | **Clarified as Digital Attenuation** |
| **"ESP32-S3 Hardware Validated"** | TFLite model size (3.8 KB) and SRAM (<25 KB) are verified on disk. Timing is a **simulated/calculated benchmark**, not physical logic-analyzer measurement. | **Relabeled as Desktop / Simulated Estimate** |

---

### 6. What Can Actually Be Reproduced

1. **Model Parameter Count & Footprint**:
   - Model B contains **3,916 trainable parameters**.
   - INT8 TFLite file size is exactly **4,160 bytes (4.06 KB)**.
   - Float32 TFLite file size is **16,640 bytes (16.25 KB)**.
2. **DSP Limiter Response**:
   - Attack time: **< 0.5 ms** (sample-level fast attack envelope follower).
   - Release time: **80.0 ms** exponential decay.
   - Digital peak attenuation: **-27.4 dB**.
3. **Inference Latency**:
   - Desktop GPU (RTX 3050): **0.796 ms** mean.
   - Desktop CPU (i5-13450HX): **1.749 ms** mean.
   - Estimated ESP32-S3 (240 MHz): **11.93 ms** per 32 ms hop.

---
