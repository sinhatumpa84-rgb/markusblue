---
language:
- en
license: mit
tags:
- audio
- speech-enhancement
- edge-ai
- tflite
- esp32
- acoustic-classification
- dsp
- sih2024
metrics:
- accuracy
- latency
- snr
datasets:
- custom
pipeline_tag: audio-classification
---

# MARKUSBLUE v7.0.00
### Indigenous Edge-AI Audio Intelligence & Speech Enhancement Research Prototype

MARKUSBLUE is an experimental Edge-AI acoustic intelligence and speech-preservation research prototype designed for **SIH26052** (Smart India Hackathon). The objective is to evaluate lightweight machine learning models combined with deterministic digital signal processing (DSP) to identify dangerous impulsive noise, preserve speech formant intelligibility, and restore listening loudness on constrained edge microcontrollers.

> **Disclaimer**: MARKUSBLUE is an academic and engineering research prototype. It is not certified personal protective equipment (PPE) or military-grade hearing protection.

---

## 1. Project Overview & SIH26052 Context
Tactical and industrial environments present severe acoustic interference:
- **Stationary noise**: Engine hum, machinery, wind.
- **Non-stationary noise**: Dynamic radio chatter, vehicular movements.
- **Impulsive acoustic spikes**: Gunfire blasts, explosive transients, mechanical hammering.

Standard active noise cancellation (ANC) struggles with sub-millisecond impulse blasts and frequently attenuates human speech. MARKUSBLUE explores a hybrid Edge-AI + deterministic DSP pipeline to separate acoustic events, preserve voice formant bands (300 Hz – 3.4 kHz), and maintain intelligible communication levels.

---

## 2. Complete Audio Pipeline Architecture

```
[ Microphone (INMP441) ]
           │
           ▼
 [ Audio Capture & Framing (16 kHz) ]
           │
 ┌─────────┴────────────────────────┐
 ▼                                  ▼
[ Fast VAD & Feature Extraction ]   [ AI Acoustic Classifier / Model B ]
 (Sliding Log-Mel Spectrogram)       (Depthwise-Separable 2D CNN)
 │                                  │
 └─────────┬────────────────────────┘
           ▼
 [ Deterministic DSP Safety Limiter (0.5 ms Attack, 80 ms Release) ]
           │
           ▼
 [ Speech Preservation Filterbank (Formant Bandpass 300 Hz - 3.4 kHz) ]
           │
           ▼
 [ VAD-Aware Automatic Gain Control (AGC) & Dynamic Leveling ]
           │
           ▼
 [ Soft-Knee Dynamic Range Compression (DRC) ]
           │
           ▼
 [ Lookahead Peak Safety Limiter ]
           │
           ▼
 [ Communication Encoder / Transmission Interface ]
           │
           ▼
 [ Receiver Node & MAX98357A Amplifier -> Earpiece ]
```

---

## 3. Model Architecture & Specifications

### Model B — Tactical Edge CNN (`ESP32EdgeCNN`)
- **Target Platform**: ESP32-S3 Dual-Core Xtensa LX7 @ 240 MHz
- **Architecture**: Depthwise-Separable 2D Convolutional Network + Global Average Pooling
- **Input Dimension**: `[1, 1, 32, 32]` (32 Mel frequency bins $\times$ 32 time steps, 16 kHz sample rate)
- **Output Classes**: 4 (`0: DANGEROUS_IMPULSE`, `1: NORMAL_SPEECH`, `2: BACKGROUND_NOISE`, `3: OTHER_IMPULSE`)
- **Total Parameters**: 3,916
- **Float32 Size**: ~14.9 KB
- **INT8 Quantized Size**: 4.16 KB (3.82 KB weights)
- **SRAM Inference Footprint**: < 24.8 KB
- **Inference Latency**: ~0.61 ms (GPU) / ~1.73 ms (CPU) / ~11.9 ms (ESP32-S3 @ 240 MHz)
- **SHA-256**: `A3BD7D63D6DC63B239E51E90B95E011BBEC3183494B8A5A9109DA4E2231732AF`

### Model A — Research Baseline CNN (`BaselineCNN`)
- **Architecture**: 2D Residual Convolutional Network (64 Mel bins)
- **Parameters**: 470,820 (~1.8 MB Float32)
- **Role**: Teacher model for knowledge distillation on desktop workstations.

---

## 4. Dataset Summary

The repository benchmark dataset consists of balanced 16 kHz mono WAV recordings:
- **Gunshot / Dangerous Impulses**: 6,000 files
- **Clean Speech**: 2,400 files
- **Background Environmental Noise**: 2,400 files
- **Other Non-Lethal Impulses**: 2,400 files
- **Total Standard Benchmark**: 13,200 WAV files (~1.4 GB)
- **Extended Dataset**: 40,807 WAV files (~4.57 GB total across raw extractions and processing splits)

All binary WAV datasets are managed via Git LFS on GitHub and mirrored on Hugging Face.

---

## 5. Hardware Target & Embedded Deployment

- **Microcontroller**: ESP32-S3 (Dual-Core Xtensa LX7 @ 240 MHz, SIMD vector instructions)
- **I2S Microphone Input**: INMP441 Digital MEMS (Pins: SCK=4, WS=5, SD=6)
- **I2S Amplifier Output**: MAX98357A Class-D Amplifier (Pins: SCK=15, WS=16, SD=7)
- **Core Partitioning**:
  - **Core 0**: Real-time I2S DMA double-buffering (256 samples / 16 ms) + Deterministic DSP Limiter & Biquad Filterbank (< 1.0 ms execution budget).
  - **Core 1**: Asynchronous Mel-spectrogram calculation + TFLite Micro INT8 inference (40 Hz / 25 ms step).
- **Embedded C++ Source**: Located in `embedded/` (`model_data.h`, `model_data.cc`, `inference_example/main.cpp`).

---

## 6. Installation & Quickstart

```bash
# Clone the repository
git clone https://github.com/sinhatumpa84-rgb/markusblue.git
cd markusblue

# Install dependencies
pip install -r requirements.txt

# Run automated DSP & failure recovery unit tests
pytest tests/test_dsp.py tests/test_pipeline_failures.py -v

# Run real-time streaming audio demo
python realtime_demo.py --input_wav datasets/gunshot/gunshot_session_0_0.wav --weights models/tactical_edge_model_best.pt
```

---

## 7. Repositories & Relationship

- **Primary Source Code & Full LFS Assets**: [GitHub — sinhatumpa84-rgb/markusblue](https://github.com/sinhatumpa84-rgb/markusblue)
- **Secondary Model & Artifact Distribution**: [Hugging Face — blue00o7/markusblue](https://huggingface.co/blue00o7/markusblue)

---

## 8. Citation & Attribution

```bibtex
@misc{markusblue2026,
  title={MARKUSBLUE: Indigenous Edge-AI Speech Preservation and Audio Intelligence System},
  author={MARKUSBLUE Development Team},
  year={2026},
  howpublished={\url{https://github.com/sinhatumpa84-rgb/markusblue}},
  note={SIH26052 Research Prototype}
}
```
