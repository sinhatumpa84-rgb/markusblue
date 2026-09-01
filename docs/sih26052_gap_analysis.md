# SIH26052 Technical Review & Research Gap Analysis

## 1. Executive Assessment Against SIH26052 Problem Statement

The SIH26052 challenge calls for **real-time edge noise cancellation and speech enhancement for tactical communications** operating under extreme acoustic disturbance (gunfire, explosions, vehicle engines) with low power and strict memory ceilings.

---

## 2. Status of Capabilities

| Architectural Capability | Status | Implementation Details |
| :--- | :--- | :--- |
| **Real-Time Edge Noise Cancellation** | **IMPLEMENTED** | Causal INT8 Depthwise-Separable TCN mask estimator (2,948 params) |
| **Tactical Gunshot Suppression** | **IMPLEMENTED** | -16.4 dB impulse attenuation within 8.0 ms frame |
| **Speech Preservation & Intelligibility** | **IMPLEMENTED** | Voice formant band protection; STOI improved from 0.864 to 0.920 |
| **Low-Loudness Compensation** | **IMPLEMENTED** | VAD-gated speech-level AGC + Peak Limiter |
| **Zero Dynamic Allocation Runtime** | **IMPLEMENTED** | 100% static memory buffers in `embedded/esp82/` |
| **Offline Teacher Distillation** | **IMPLEMENTED** | Demucs / Wiener spectral targets distilled to INT8 student |
| **Dual-Mic Beamforming** | **NOT IMPLEMENTED** | Single-channel I2S DMA pipeline; spatial filtering not yet integrated |
| **Registered-Speaker Isolation** | **NOT IMPLEMENTED** | Universal speech enhancement; voiceprint enrollment not present |
| **Inter-Node Mesh Networking** | **PARTIALLY IMPLEMENTED**| I2S transmission ready; ESP-NOW MAC framing left to host |

---

## 3. Prioritized SIH26052 Feature Roadmap

### A. REQUIRED FOR SIH26052 (Core Objective)
1. **Real-Time Streaming Edge Execution**
   - **Why**: Microcontrollers must process audio continuously without buffering entire files.
   - **Expected Benefit**: Continuous bidirectional communication.
   - **Implementation Cost**: Low (Implemented in `embedded/esp82/`).
   - **Hardware Impact**: 5.80 KB RAM, 2.88 KB Flash.
   - **Research Value**: High (Demonstrates TinyML streaming feasibility).

2. **Loudness Recovery Post-Suppression**
   - **Why**: Noise reduction inherently attenuates low-amplitude speech energy.
   - **Expected Benefit**: Prevents voice dropouts in combat communication.
   - **Implementation Cost**: Low (Implemented via AGC).
   - **Hardware Impact**: < 200 bytes RAM, < 0.2 ms CPU cycles.
   - **Research Value**: High (Solves key industry limitation in edge speech processing).

---

### B. STRONGLY RECOMMENDED (Substantial Prototype Enhancements)
1. **ESP-NOW Low-Latency Peer-to-Peer Audio Mesh**
   - **Why**: Battlefield nodes must transmit enhanced audio wirelessly without Wi-Fi routers.
   - **Expected Benefit**: Direct soldier-to-soldier voice link (< 15 ms transmission latency).
   - **Implementation Cost**: Medium (Add ESP-NOW broadcast wrapper in `embedded/esp82/`).
   - **Hardware Impact**: ~4 KB DRAM for Wi-Fi stack.
   - **Research Value**: High (End-to-end tactical prototype).

2. **Dual-Microphone Differential Noise Cancellation**
   - **Why**: External environmental noise can be sampled via a secondary ambient microphone.
   - **Expected Benefit**: Additional 6–12 dB coherent low-frequency noise reduction.
   - **Implementation Cost**: Medium (Requires secondary I2S/ADC channel).
   - **Hardware Impact**: ~1.5 KB extra DMA buffer.
   - **Research Value**: High (Hardware-DSP hybrid novelty).

---

### C. OPTIONAL (Research Extensions)
1. **Lightweight Speaker Enrollment (Voiceprint Verification)**
   - **Why**: Restricts audio transmission strictly to the enrolled soldier's voice.
   - **Expected Benefit**: Rejects background chatter from enemy/bystanders.
   - **Implementation Cost**: High (Requires 16-dim speaker embedding network).
   - **Hardware Impact**: ~4–6 KB additional Flash and ~2 KB RAM.
   - **Research Value**: High (Tactical voice authentication).

---

### D. NOT REQUIRED (Unnecessary Complexity)
1. **Transformer / Large Attention Models on Edge**
   - **Why**: Exceeds ESP8266 RAM/cycle budgets by 100x; causes fatal crashes.
   - **Hardware Impact**: Unfeasible (> 1 MB RAM required).
2. **Cloud / LLM in the Real-Time Audio Loop**
   - **Why**: Introduces 500–2,000 ms latency and requires constant internet connectivity.
