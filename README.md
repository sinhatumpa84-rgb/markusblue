# SIH26052 — Indigenous Edge-AI Tactical Communication & Hearing Protection System for the Indian Army

[![Platform](https://img.shields.io/badge/Target_Hardware-ESP32--S3_N16R8-red.svg)](#hardware-specifications)
[![Framework](https://img.shields.io/badge/ML-PyTorch_%26_TFLite_Micro-blue.svg)](#model-architectures)
[![Audio](https://img.shields.io/badge/Audio-16kHz_Mono_16--bit-green.svg)](#acoustic-pipeline)
[![License](https://img.shields.io/badge/Type-Defensive_Hearing_Protection-yellow.svg)](#critical-defensive--scientific-notice)

> **Objective:** A research-grade Edge-AI acoustic intelligence and deterministic DSP hearing protection system that detects extremely loud impulsive acoustic events (e.g., muzzle blasts, explosions) in sub-millisecond timeframes, rapidly clamping dangerous sound pressure levels (SPL) while preserving vital voice communication and situational awareness for tactical personnel.

---

## 🛡️ Critical Defensive & Scientific Notice

> [!IMPORTANT]
> **Defensive Hearing-Protection and Communication System**:
> This project is exclusively designed as a defensive hearing-protection and speech-preserving tactical system for the Indian Army. It is **NOT** a weapon, targeting system, firearm identification system, or acoustic fire-direction device.
>
> In compliance with military engineering research standards:
> - Physical hearing safety compliance must be validated in accredited laboratory facilities following **ANSI S12.42** / **MIL-STD-1474E** acoustic test fixtures.
> - The ML model acts strictly as an **acoustic event classifier/trigger**, while hearing safety is enforced deterministically by sub-millisecond analog/DSP dynamic limiters.

---

## 🏛️ System Architecture

```
MICROPHONE (INMP441 I2S)
         │
         ▼
ADC / I2S DMA Ring Buffer (16 kHz, 16-bit PCM)
         │
 ┌───────┴─────────────────────────────────────────┐
 │                                                 │
 │  [CORE 1: ML STREAMING DETECTOR]                │  [CORE 0: DETERMINISTIC DSP ENGINE]
 │  • 32-bin Log-Mel Spectrogram Extraction        │  • Fast Dynamic Limiter (<0.5ms Attack)
 │  • ESP32-S3 Depthwise-Separable Edge CNN        │  • Adaptive Attenuation (-28 dB Clamp)
 │  • P(DANGEROUS_IMPULSE) Inference               │  • Voice Formant Passband (300Hz-3.4kHz)
 │  • State Machine Trigger (NORMAL ➔ PROTECT)     │  • Smooth Exponential Release Recovery
 │                                                 │
 └───────────────────────┬─────────────────────────┘
                         ▼
        EARPIECE / AMPLIFIER (MAX98357A I2S)
     (Safe Clamped Sound + Clear Tactical Voice)
```

---

## 📁 Repository Structure

```
.
├── dataset_report.json                 # Automatic dataset scan & statistics report
├── configs/
│   └── default_config.json             # Hyperparameters, audio settings, DSP thresholds
├── data/
│   ├── raw/                            # Untouched extracted archives
│   ├── extracted/                      # Organized raw audio
│   ├── processed/                      # 16kHz mono WAVs (gunshot, speech, background, other)
│   ├── metadata/
│   │   └── gunshot_segments.csv        # AudioSet timestamp & ontology mapping
│   ├── features/                       # Cached acoustic features
│   └── splits/                         # Source-isolated train/val/test splits
├── models/
│   ├── tactical_edge_model_best.pt     # Best PyTorch Model B checkpoint
│   ├── tactical_baseline_model_best.pt # Best PyTorch Model A checkpoint
│   ├── model_float32.tflite            # Float32 TFLite export
│   ├── model_int8.tflite               # Fully quantized INT8 TFLite Micro model
│   └── model_metadata.json             # Tensor dimensions & quantization metadata
├── src/
│   ├── dataset/                        # Safe ZIP extraction, AudioSet resolver, Dataset loader
│   ├── preprocessing/                  # 16kHz mono conversion & source-isolated splitter
│   ├── features/                       # 9-feature extraction suite & tactical augmentations
│   ├── training/                       # Model architectures, Focal loss, Trainer
│   ├── evaluation/                     # Metrics, Visualizer, HTML report generator
│   ├── inference/                      # Streaming rolling-window detector & benchmark engine
│   └── dsp/                            # State machine, dynamic limiter, speech filter
├── embedded/
│   ├── model_data.h                    # C header for INT8 model byte array
│   ├── model_data.cc                   # C++ definition for TFLite Micro tensor arena
│   └── inference_example/              # FreeRTOS ESP32-S3 dual-core firmware project
├── reports/
│   ├── model_evaluation.html           # Interactive evaluation dashboard
│   ├── confusion_matrix.png            # 4x4 Confusion matrix
│   ├── roc_curve.png                   # Receiver operating characteristic curves
│   ├── pr_curve.png                    # Precision-recall curves
│   └── speech_preservation_demo.png    # Before/after speech preservation analysis
├── prepare_dataset.py                  # CLI 1: Dataset extraction, validation, splitting
├── train.py                            # CLI 2: Model training with early stopping
├── evaluate.py                         # CLI 3: Complete test evaluation & report generation
├── export_tflite.py                    # CLI 4: INT8 quantization & C array export
├── benchmark.py                        # CLI 5: Latency and memory benchmark suite
├── realtime_demo.py                    # CLI 6: Live streaming audio simulation demo
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation
```

---

## ⚡ Quickstart Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Dataset Extraction & Validation
```bash
python prepare_dataset.py
```

### 3. Model Training (Model A Baseline & Model B Edge)
```bash
# Train Model B (ESP32-S3 Edge Model)
python train.py --model_type edge --epochs 25 --batch_size 64

# Train both Baseline and Edge models
python train.py --model_type both --epochs 25
```

### 4. Comprehensive Model Evaluation & HTML Dashboard
```bash
python evaluate.py --model_type edge
```
View the generated report: `reports/model_evaluation.html`.

### 5. Export Quantized Model for ESP32-S3
```bash
python export_tflite.py
```

### 6. Hardware Latency & Memory Benchmarking
```bash
python benchmark.py
```

### 7. Real-Time Streaming Audio Simulation
```bash
python realtime_demo.py
```

---

## ⚙️ Hardware Specifications (Target Platform)

| Component | Target Specification | Role |
| :--- | :--- | :--- |
| **Microcontroller** | **ESP32-S3 N16R8** (Dual Xtensa LX7 @ 240 MHz, 512KB SRAM, 8MB PSRAM, 16MB Flash) | Edge-AI inference & DSP host |
| **Microphone Input** | **INMP441** (Omnidirectional MEMS microphone, I2S digital output, 24-bit PCM) | Low-noise acoustic pickup |
| **Audio Output** | **MAX98357A** (I2S Class-D amplifier) + Earpiece/Transducer | Safe clamped audio playback |
| **Operating System** | **FreeRTOS** (SMP Dual-Core Architecture) | Deterministic real-time task scheduling |

---

## 📊 Evaluation Summary

- **Primary Detection Target:** `DANGEROUS_IMPULSE` (High recall prioritization)
- **Negative Rejection Classes:** `NORMAL_SPEECH`, `BACKGROUND_NOISE`, `OTHER_IMPULSE`
- **Dynamic Limiter Attack Time:** `< 0.5 ms`
- **Blast Sound Pressure Level Reduction:** `> 25 dB`
- **Speech Preservation Metric:** `> 65%` intelligibility retention during acoustic transients
- **Edge Model Size (INT8):** `< 20 KB` (Fits directly in internal SRAM with zero memory pressure)

---

Developed for **SIH26052 — Indigenous Edge-AI Tactical Communication and Hearing Protection System for the Indian Army**.
