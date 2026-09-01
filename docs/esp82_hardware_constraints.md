# ESP82 / ESP8266 Hardware Constraints & Architecture Specifications

## 1. Overview & Core Target
The **ESP82 / ESP8266 / ESP-12** family represents ultra-low-cost, resource-constrained 32-bit Wi-Fi microcontrollers based on the **Tensilica Xtensa L106 Diamond Standard core**. Unlike the modern ESP32 and ESP32-S3 series, the ESP8266 features **no hardware floating-point unit (FPU)**, **no vector SIMD instructions**, and strict memory segmentation.

All audio processing, feature extraction, neural mask inference, and dynamic range control must be engineered within these rigid constraints.

---

## 2. Hardware Specifications

| Parameter | Specification | Real-World Constraint on Audio / AI |
| :--- | :--- | :--- |
| **CPU Core** | Tensilica Xtensa L106 32-bit RISC | Single-core, non-superscalar |
| **Clock Frequency** | 80 MHz (default) / 160 MHz (turbo) | At 160 MHz, ~160,000 clock cycles per 1 ms |
| **Floating Point** | **None (Software Emulation)** | FP32 math is ~10-40x slower than INT32/INT8 |
| **Total Internal RAM** | ~80 KB (IRAM + DRAM) | Physical SRAM partition |
| **Instruction RAM (IRAM)** | ~32 KB (partially used by boot/Wi-Fi) | Critical ISRs & fast inner loops only |
| **Data RAM (DRAM)** | ~48 KB total | Holds stack, static globals, BSS |
| **Usable User Heap** | **~36 KB – 42 KB** | Maximum dynamic memory ceiling |
| **Flash Memory** | 1 MB / 2 MB / 4 MB SPI Flash (Quad SPI) | Mapped via 1 MB MMU cache window (IRAM cache) |
| **Flash Read Overhead** | ~4–8 wait cycles on cache miss | Execution in flash is slower than IRAM |
| **Hardware DMA** | I2S DMA Controller (Rx & Tx) | Supports double-buffered circular streaming |
| **Audio I/O** | I2S (GPIO 2, 3, 12, 15) / Sigma-Delta DAC / ADC | Mono 8 kHz / 16 kHz 16-bit PCM streaming |

---

## 3. Real-Time Audio Budget & Timing Feasibility

For continuous, non-blocking real-time audio enhancement, the total processing latency per audio frame must be strictly lower than the frame hop duration:

$$\text{Processing Time per Frame } (T_{\text{proc}}) \ll \text{Frame Hop Duration } (T_{\text{hop}})$$

### Frame Configuration Trade-offs:

1. **Option A: 8 kHz Sample Rate (Telephony / Tactical Voice Band 300 Hz – 3,400 Hz)**
   - **Hop size**: 64 samples = **8.0 ms**
   - **Frame size**: 128 samples = **16.0 ms** (50% overlap)
   - **STFT Bins**: 65 bins ($N_{\text{fft}} = 128$) or 16-band Mel filterbank
   - **Cycle budget @ 160 MHz**: $8.0\text{ ms} \times 160,000\text{ cycles/ms} = \mathbf{1,280,000\text{ cycles}}$
   - **Target Processing Time**: $\le 3.5\text{ ms}$ ($\text{Real-Time Factor (RTF)} \le 0.44$)

2. **Option B: 16 kHz Sample Rate (Wideband Audio)**
   - **Hop size**: 128 samples = **8.0 ms**
   - **Frame size**: 256 samples = **16.0 ms** (50% overlap)
   - **STFT Bins**: 129 bins ($N_{\text{fft}} = 256$)
   - **Target Processing Time**: $\le 5.0\text{ ms}$ ($\text{RTF} \le 0.62$)

*Selected Default*: 8 kHz / 16 kHz compatible architecture with low-dimensional feature space (16–32 spectral bands or 65 bins) to guarantee real-time execution with ample headroom for Wi-Fi/networking tasks.

---

## 4. Memory Allocations & Safe Budget

To prevent heap fragmentation and watchdog timer resets (WDT), the system uses **100% static memory allocation**:

| Memory Component | Size Budget | Purpose |
| :--- | :--- | :--- |
| **DMA Audio Ping-Pong Buffers** | 1,024 bytes | Double-buffered I2S Rx/Tx (2x 256 words) |
| **Ring Buffer & Window Overlap** | 512 bytes | 128-sample circular input and output overlap-add |
| **Feature Extraction (STFT/Filterbank)** | 512 bytes | Windowed FFT magnitude & phase preservation |
| **TFLite Micro Tensor Arena** | 3,584 bytes (3.5 KB) | Static tensor workspace for INT8 activations |
| **INT8 Model Weights (Flash)** | 4,096–8,192 bytes (4–8 KB) | Stored in Flash (.rodata / PROGMEM) |
| **DSP State (VAD, AGC, Limiter)** | 256 bytes | Attack/decay filters, running noise variance, RMS |
| **Total Static RAM Footprint** | **~5.8 KB** | **< 15% of total 40 KB free heap** |

---

## 5. Summary of Engineering Directives
1. **Zero Dynamic Allocation**: No `malloc()`, `new`, or standard dynamic vectors during streaming inference.
2. **Strict INT8 Quantization**: Neural network weights and activations must be quantized to INT8 with integer arithmetic where possible to bypass software FP32 slowdowns.
3. **Causal Depthwise-Separable Architecture**: Minimize multiply-accumulate (MAC) count while maintaining voice formant preservation.
4. **Independent Post-Enhancement Loudness DSP**: Incorporate fixed-rate AGC and Limiter to restore speech volume without amplifying residual noise floor.
