# MARKUSBLUE — Real-Time Edge-AI Tactical Audio Enhancement System
**SIH Problem Statement**: SIH26052  
**Target Hardware Platform**: Espressif ESP32-S3 N16R8 (Dual-Core Xtensa® LX7 @ 240 MHz, 16MB Flash, 8MB PSRAM)  
**System Type**: Indigenous Edge-AI Tactical Audio Enhancement & Active Noise Cancellation (ANC)

---

## 1. Problem Statement & Mission Objective
Tactical combat environments subject personnel to severe acoustic interference:
- High-energy gunfire impulse transients and mechanical impacts.
- Continuous engine roar from combat vehicles and armored personnel carriers.
- High-velocity wind turbulence and environmental acoustic noise.
- Low signal-to-noise ratio (SNR) speech mixtures (-15 dB to +10 dB).

Traditional ear defenders either attenuate sound passively (causing hearing loss of tactical communications) or employ simple energy threshold gates that **blank/mute speech** immediately following a gunshot.

**MARKUSBLUE** solves this critical challenge by deploying a real-time, low-latency streaming Edge-AI pipeline on an embedded **ESP32-S3 N16R8** microcontroller. The system captures noisy acoustic signals via dual I2S MEMS microphones, performs spatial noise estimation and causal neural mask prediction, and restores clear human speech without audio blanking.

---

## 2. Hardware Architecture

```
                  ┌────────────────────────────────────────┐
                  │          EXTERNAL NOISE FIELD          │
                  │ (Gunfire, Vehicle Engines, Wind, Amb)  │
                  └───────────────────┬────────────────────┘
                                      │
                              ┌───────▼───────┐
                              │ INMP441 MIC 1 │
                              │ (Reference)   │
                              └───────┬───────┘
                                      │ I2S (Left Channel)
                                      ▼
                        ┌───────────────────────────┐
                        │      ESP32-S3 N16R8       │
                        │ Dual-Core Xtensa @ 240MHz │
                        │  16MB Flash + 8MB PSRAM   │
                        │                           │
                        │  Core 1: Real-Time Audio  │
                        │   - I2S DMA Capture       │
                        │   - DC Bias Removal       │
                        │   - 256-pt Fast STFT      │
                        │   - Dual-Mic Spatial DSP  │
                        │   - Causal AI Mask Engine │
                        │   - Fast ISTFT Overlap-Add│
                        │   - VAD-Aware Dynamic AGC │
                        │   - Peak Limiter & Clamp  │
                        │   - I2S DMA TX Output     │
                        │                           │
                        │  Core 0: UI & Telemetry   │
                        │   - SSD1306 OLED (10 Hz)  │
                        │   - MPU6050 Motion IMU    │
                        │   - Debounced PTT & Haptic│
                        │   - MicroSD Async Logger  │
                        │   - Battery ADC Monitor   │
                        └─────────────┬─────────────┘
                                      │
                                      │ I2S (Mono PCM)
                                      ▼
                            ┌───────────────────┐
                            │    MAX98357A      │
                            │ Class-D Amplifier │
                            └─────────┬─────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │  8Ω SPEAKER   │
                              │ Earphone Unit │
                              └───────┬───────┘
                                      │
                                      ▼
                                  USER'S EAR
                                      │
                              ┌───────▼───────┐
                              │ INMP441 MIC 2 │
                              │ (Interior/Ear)│
                              └───────┬───────┘
                                      │ I2S (Right Channel)
                                      └───────► ESP32-S3
```

### Component Breakdown
1. **Central Processor**: ESP32-S3 N16R8 (Dual Xtensa LX7 @ 240 MHz, 512 KB internal SRAM, 16 MB Flash, 8 MB Octal PSRAM).
2. **Audio Input**: 2 × INMP441 I2S MEMS Microphones:
   - **Mic 1 (Exterior Reference)**: Captures external gunfire blasts and environmental noise.
   - **Mic 2 (Interior Ear)**: Captures voice and the acoustic conditions inside the ear cup.
3. **Audio Output**: MAX98357A I2S Class-D amplifier driving an 8Ω speaker / earphone driver inside an acoustically isolated ear cup.
4. **Peripherals**:
   - 0.96" I2C OLED (SSD1306): Real-time telemetry (SNR, Latency, Battery %, Status).
   - MPU6050 6-DOF IMU: Rapid head-motion context detection.
   - PTT Button: Debounced manual communication activation.
   - Haptic Vibration Motor: Silent tactile alert on event.
   - MicroSD Module: Non-blocking diagnostic and telemetry logging.
5. **Power Subsystem**: 3.7V 2500 mAh Li-Po battery + TP4056 charging module + 3.3V LDO regulator + 5.0V synchronous boost converter (~11.85 hours active runtime).

---

## 3. Real-Time Streaming Audio Pipeline

The real-time audio pipeline executes deterministically on **Core 1** at **16,000 Hz** wideband audio with a hop size of **64 samples (4.0 ms frame duration)**:

$$\text{I2S DMA Capture} \to \text{DC Blocker} \to \text{256-pt Hann STFT} \to \text{Spatial Pre-Filter} \to \text{AI Mask Inference} \to \text{Fast ISTFT} \to \text{AGC} \to \text{Peak Limiter} \to \text{I2S TX DMA}$$

### Latency Budget
- **Algorithmic Processing Latency**: **3.13 ms (3,130 µs)**.
- **Total End-to-End Latency**: **~7.13 ms** (Well within the < 20.0 ms tactical requirement).
- **Real-Time Factor (RTF)**: **0.7825** (< 1.00 sustainable real-time operation).

---

## 4. Edge-AI Model Architecture

- **Model Identity**: `MARKUSBLUEStudentEnhancer`
- **Network Structure**: Causal Depthwise-Separable 1D Conv/TCN (with dilations $d = 1, 2, 4$) coupled with a 32-dim GRU recurrent cell.
- **Input**: 129 positive frequency bins (log magnitude STFT).
- **Output**: 129-bin Ideal Ratio Mask $M(f, t) \in [0.0, 1.0]$.
- **Parameters**: 18,725 parameters.
- **Quantization**: INT8 Symmetric quantization.
- **Flash Footprint**: **18.29 KB** (< 0.12% of 16 MB Flash).
- **RAM Footprint**: **12.0 KB** internal SRAM tensor arena.

---

## 5. Repository Structure

```
MARKUSBLUE/
├── firmware/
│   └── esp32s3/
│       ├── src/
│       │   ├── main.cpp                 # Dual-core FreeRTOS application entry point
│       │   ├── audio/                   # I2S0 RX, I2S1 TX, ping-pong DMA buffers
│       │   ├── dsp/                     # Fast STFT, ISTFT, spatial filter, AGC, limiter, VAD
│       │   ├── ai/                      # Streaming neural inference & compiled model data
│       │   ├── sensors/                 # OLED, MPU6050, PTT & haptic drivers
│       │   ├── storage/                 # Non-blocking MicroSD telemetry logger
│       │   └── system/                  # Battery ADC & microsecond latency profiler
│       └── platformio.ini               # PlatformIO build configuration
│
├── models/
│   ├── markusblue_esp32s3_int8.tflite   # Quantized INT8 deployment flatbuffer (18.29 KB)
│   ├── markusblue_esp32s3_fp32.tflite   # Float32 benchmark model
│   ├── markusblue_esp32s3_best.pt       # PyTorch master checkpoint
│   └── esp32s3_model_metadata.json      # Model checksums & hyperparameter specification
│
├── datasets/                            # 100% Protected raw dataset assets
│   ├── speech/                          # 2,400 clean speech utterances
│   ├── gunshot/                         # 6,000 gunfire impulse recordings
│   ├── background_noise/                # 2,400 vehicle/ambient noise files
│   └── other_impulse/                   # 2,400 mechanical impact recordings
│
├── hardware/
│   ├── pinout.md                        # Complete GPIO allocation matrix
│   ├── power_tree.md                    # Power architecture & battery runtime budget
│   └── wiring.md                        # Electrical interconnects & acoustic isolation
│
├── docs/
│   ├── architecture.md                  # Comprehensive technical architecture
│   ├── audio_pipeline.md                # Streaming DSP pipeline details
│   ├── model_deployment.md              # Model quantization & PROGMEM deployment
│   ├── hardware_validation.md           # Multi-level testing strategy
│   ├── sih26052_mapping.md              # SIH problem statement compliance matrix
│   └── validation_report.md             # 25-point final system validation report
│
├── tests/
│   ├── test_dsp.py                      # Unit tests for STFT, ISTFT, AGC, limiter
│   └── test_enhancement_pipeline.py     # Pipeline integration & blanking recovery tests
│
├── tools/
│   ├── sih_demo_suite.py                # Interactive tactical demonstration suite
│   ├── verify_datasets.py               # SHA-256 dataset integrity verification tool
│   ├── train_esp32s3_student.py         # PyTorch student model trainer
│   └── export_esp32s3_model.py          # Quantization & C++ header generation tool
│
└── requirements.txt                     # Clean minimal Python dependencies
```

---

## 6. Build & Test Instructions

### 1. Python Environment Setup
```powershell
pip install -r requirements.txt
```

### 2. Run Dataset Integrity Verification
```powershell
python tools/verify_datasets.py
```

### 3. Run Unit Tests
```powershell
python -m unittest discover tests
```

### 4. Run Interactive Demonstration
```powershell
python tools/sih_demo_suite.py
```

### 5. Build ESP32-S3 Firmware (PlatformIO)
```powershell
cd firmware/esp32s3
pio run -e esp32-s3-devkitc-1
```

---

## 7. Verification Results Summary

- **Audio Blanking Defect**: **PASSED (0.0 ms speech mute on 4.0x gunshot burst; 0.94x speech RMS preservation)**.
- **Clipping Prevention**: **PASSED (Zero numeric wrap-around / DAC overflow)**.
- **Model Footprint**: **18.29 KB INT8**.
- **Algorithmic Latency**: **3.13 ms**.
- **Unit Test Suite**: **11 of 11 Tests Passed**.
