# MARKUSBLUE Audio Dataset Audit Report

## 1. Executive Summary
The MARKUSBLUE dataset contains audio for training and evaluating speech enhancement and tactical noise suppression on edge microcontrollers.

- **Total Audio Files**: 13,200
- **Speech Files**: 2,400 (0.7 hours estimated)
- **Noise / Disturbance Files**: 10,800
  - **Background Noise**: 2,400
  - **Gunshot Impulses**: 6,000
  - **Other Impulses**: 2,400

---

## 2. Dataset Distribution & Characteristics

| Category | File Count | Mean Duration (s) | Min/Max Duration (s) | Primary Sample Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Clean Speech** | 2,400 | 1.00 s | 1.00s / 1.00s | 16,000 Hz / 8,000 Hz |
| **Background Noise** | 2,400 | 1.00 s | 1.00s / 1.00s | 16,000 Hz |
| **Gunshot Impulses** | 6,000 | 1.00 s | 1.00s / 1.00s | 16,000 Hz |
| **Other Impulses** | 2,400 | 1.00 s | 1.00s / 1.00s | 16,000 Hz |

---

## 3. Training Mixture Generation Strategy
During model training and distillation:
- **Clean speech** is dynamically combined on-the-fly with **environmental noise, gunfire, and acoustic impulse sounds**.
- **SNR Distribution**: Sampled uniformly from `[-20 dB, -15 dB, -10 dB, -5 dB, 0 dB, +5 dB, +10 dB, +15 dB, +20 dB]` to ensure robustness under heavy noise and near-clean conditions.
- **Normalization**: Dynamic RMS normalization with floating-point to INT8 scaling consistency.
