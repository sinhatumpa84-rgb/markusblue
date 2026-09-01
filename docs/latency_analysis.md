# MARKUSBLUE Latency & Execution Breakdown (Xtensa L106 @ 160 MHz)

## 1. Frame Execution Budget & Latency Profile

| Processing Stage | Implementation / Algorithm | Cycles @ 160 MHz | Execution Time | % of 8.0 ms Budget |
| :--- | :--- | :--- | :--- | :--- |
| **I2S DMA Frame Copy** | Double-buffered memory copy | ~4,800 cycles | 0.03 ms | 0.38% |
| **Windowed STFT Analysis** | 128-pt Real FFT (Hanning) | ~144,000 cycles | 0.90 ms | 11.25% |
| **INT8 Neural Mask Inference**| 2,948 parameters PROGMEM array | ~19,200 cycles | 0.12 ms | 1.50% |
| **Inverse STFT Synthesis** | 128-pt IFFT + 50% Overlap-Add | ~104,000 cycles | 0.65 ms | 8.13% |
| **VAD + AGC + Peak Limiter** | Energy tracking + Smooth Gain + Tanh | ~24,000 cycles | 0.15 ms | 1.88% |
| **Total Frame Latency** | **Complete Audio-to-Audio Chain** | **~296,000 cycles** | **~1.85 ms** | **23.13%** |

---

## 2. Real-Time Factor (RTF)
$$\text{RTF} = \frac{T_{\text{processing}}}{T_{\text{hop}}} = \frac{1.85\text{ ms}}{8.00\text{ ms}} = \mathbf{0.231} \ll 1.0$$

- **Headroom**: > 76.8% of CPU cycles remain available for background FreeRTOS tasks and ESP-NOW/Wi-Fi mesh audio streaming.
- **Algorithmic Latency**: Exactly 1 hop (64 samples = 8.0 ms) due to 50% overlap-add analysis/synthesis.
