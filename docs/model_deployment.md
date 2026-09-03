# MARKUSBLUE (SIH26052) — Model Deployment Specification

## 1. Deployed Model Architecture
- **Model Identity**: `MARKUSBLUEStudentEnhancer` (Causal Neural Mask Estimator)
- **Input Feature**: 129-bin log magnitude STFT spectrum (16 kHz, 256 N_FFT, 64-hop).
- **Network Layers**:
  - Input Projection: 129 bins $\to$ 32 hidden dimensions (Pointwise Conv1D + PReLU).
  - Causal Temporal Convolutional Network (TCN): 3 Depthwise-Separable 1D blocks with increasing dilation ($d = 1, 2, 4$) and left causal padding.
  - Recurrent Cell: 32-dim GRU cell for pitch harmonic and formant tracking.
  - Mask Head: 32 hidden dimensions $\to$ 129 frequency bins with Sigmoid activation.
- **Output**: Ideal Ratio Mask $M(f, t) \in [0.0, 1.0]$.
- **Total Parameters**: 18,725 parameters.

---

## 2. Quantization & Formats

| Format | File Path | Size | Target Environment |
| :--- | :--- | :--- | :--- |
| **PyTorch FP32** | `models/markusblue_esp32s3_best.pt` | 76.70 KB | Research / Offline Training |
| **TFLite FP32** | `models/markusblue_esp32s3_fp32.tflite` | 73.14 KB | Simulation / Host Benchmark |
| **TFLite INT8** | `models/markusblue_esp32s3_int8.tflite` | 18.29 KB | Embedded Testing |
| **C++ Header Array**| `firmware/esp32s3/src/ai/model_data.cc` | **18.29 KB** | **ESP32-S3 N16R8 PROGMEM** |

---

## 3. Embedded Inference on ESP32-S3
- **Memory Footprint**:
  - Model weights stored in PROGMEM flash: **18.29 KB** (< 0.12% of 16 MB Flash).
  - Tensor Scratch / Arena: **12.0 KB** internal SRAM.
- **Execution Performance**:
  - Hardware: Dual Xtensa LX7 @ 240 MHz.
  - Execution Time per Frame: **1.85 ms (1,850 µs)**.
  - Hop Time Budget: **4.00 ms (4,000 µs)**.
  - Real-Time Margin: **> 53% idle headroom**.
