# SIH26052: Engineering Audit & Comprehensive Fixes Log

## Executive Summary
This document provides a line-by-line accounting of all forensic audit findings, vulnerabilities, architectural bugs, and corrections implemented across the SIH26052 Tactical Acoustic Protection codebase.

---

## 1. Summary of Forensic Audit Findings

| Category | Initial Claimed Status | Discovered Reality / Vulnerability | Remediation Applied |
| :--- | :--- | :--- | :--- |
| **Negative Dataset Classes** | Realistic battlefield acoustics | Simplistic synthetic sine waves and white noise bursts in memory. | Re-engineered with Liljencrants-LF multi-speaker vocal model, dynamic biquad formants (F1-F3), syllabic rhythm, VHF radio bandpass (300Hz-3.4kHz), complex military diesel/wind/generator noise, and hard negative physical impacts (`OTHER_IMPULSE`). |
| **Data Leakage & Splitting** | Zero leakage claimed | Audio files lacked rigorous hash-based tracking across splits. | Implemented MD5 audio payload hash tracking, sub-session grouping, and verified 0 cross-split leakage in `reports/data_leakage_report.json`. |
| **DSP Limiter Attack Envelope** | Instant sub-ms attack | Envelope follower filtered twice (envelope filter + gain filter), allowing the initial blast sample to pass completely unattenuated (0 dB attenuation on attack). | Implemented instantaneous zero-delay peak detection on rising edge with sub-sample clamp (<0.1 ms) and smooth exponential 80ms release in Python and C++ header. |
| **Empty Audio Robustness** | Unhandled | `normalize_audio_preserving_dynamics` threw `ValueError` on empty inputs due to unprotected `np.max(np.abs(audio))`. | Added empty array and None checks across all preprocessing and DSP routines. |
| **Windows Console Compatibility** | Crashed on cp1252 | Non-ASCII Unicode characters (`[✓]`) caused `UnicodeEncodeError` on Windows command terminals. | Sanitized all visualizer, benchmark, export, and trainer outputs to use robust ASCII logging (`[OK]`). |
| **Speech Preservation Terminology** | "100% Speech Intelligibility" | Formant energy ratio was over-claimed as human speech intelligibility without PESQ/STOI. | Relabeled to "Spectral Formant Preservation Proxy" with scientific disclaimers on physical acoustic testing. |
| **Hardware Performance Claims** | ESP32-S3 verified | Latency benchmarks were run on desktop Intel CPU/GPU without hardware labeling. | Explicitly labeled as "DESKTOP / SIMULATED EMBEDDED ESTIMATE" with hardware profile breakdown. |

---

## 2. File-by-File Detailed Fix Log

### A. Preprocessing & Dataset
- **`src/preprocessing/audio_pipeline.py`**:
  - Replaced simplistic synthesis with 3 rich tactical acoustic generators (`_synthesize_realistic_speech`, `_synthesize_realistic_background`, `_synthesize_realistic_other_impulse`).
  - Added multi-speaker pitch modeling (85-280 Hz), Liljencrants-Fant glottal pulse approximation, syllabic modulation (3.5-6.5 Hz), and consonant frication.
  - Added military diesel engine rumble (harmonic combustion tones at 18, 36, 72 Hz), high-altitude wind gust turbulence with low-frequency sensor flutter, and generator hum (50/60 Hz).
  - Added hard negative physical impacts: armored vehicle door slams, dropped steel tool/magazine resonances (1.5, 2.8, 4.2 kHz), and rifle bolt rack/charging handle double clicks.
  - Added empty audio array guards to `normalize_audio_preserving_dynamics`.
  - Fixed parameter name mismatch in `process_raw_audio`.

- **`prepare_dataset.py`**:
  - Configured balanced dataset curation (2,000 Gunshot, 1,200 Background, 1,200 Speech, 1,200 Other Impulse = 5,600 balanced samples).
  - Ensured source-isolated train/val/test splits (Train=4,015, Val=764, Test=821).

### B. DSP Protection Engine
- **`src/dsp/dynamic_limiter.py`**:
  - Fixed double-filter delay bug in transient envelope tracking.
  - Implemented instantaneous peak jump on rising edge (`if abs_val > self.envelope: self.envelope = abs_val`), guaranteeing zero-delay attenuation (< 0.1 ms attack) on high-energy blasts.
  - Maintained smooth exponential release ($\tau = 80$ ms) for hearing comfort.

- **`embedded/inference_example/dsp_protection.h`**:
  - Synced C++ embedded limiter with instant attack peak detection and exponential decay.

- **`src/dsp/speech_preservation.py`**:
  - Implemented 4th-order Butterworth Second-Order Sections (SOS) bandpass filter (300 Hz - 3.4 kHz) with +3 dB speech formant preservation and -32 dB wideband blast cut.

### C. Training & Quantization
- **`train.py`**:
  - Updated pipeline to retrain Model A and Model B from scratch with fixed deterministic random seed (42).
  - Configured checkpointing and metadata persistence to `models/retrained_model_a/` and `models/retrained_model_b/`.
- **`export_tflite.py`**:
  - Verified full integer INT8 quantization, input/output tensors, quantization scale/zero-point, and generated `embedded/model_data.h` and `embedded/model_data.cc` (4,160 bytes).

### D. Testing & Evaluation Suites
- **`tests/test_dsp.py`**:
  - Unit tests for filter stability, impulse response decay, biquad frequency response, limiter attack attenuation, and exponential release.
- **`tests/test_pipeline_failures.py`**:
  - Robustness tests for empty audio, NaN/Inf values, extreme over-amplitude clipping (+20 dB), stereo-to-mono downmixing, variable sample rates, and batch tensor sizes.
- **`src/evaluation/challenge_evaluator.py`**:
  - Evaluates Model B under 11 battlefield stress conditions (Variable SNR, burst fire, speech interleaving, clutter rejection, hard negative impacts, MEMS saturation).
  - Generates `reports/challenge_test_report.html` and `reports/challenge_test_results.json`.
- **`src/inference/realtime_benchmark.py`**:
  - Dual-core streaming benchmark simulating I2S DMA buffers, Core 0 DSP ISR, Core 1 AI inference, and outputs `reports/realtime_benchmark.json`.

---

## 3. Verification Summary
All unit tests, failure suites, training runs, quantization exports, challenge evaluations, and real-time streaming simulations pass with 100% test reproducibility.
