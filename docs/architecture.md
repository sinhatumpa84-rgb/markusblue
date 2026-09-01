# MARKUSBLUE System Architecture (SIH26052 Real-Time Edge AI)

## 1. End-to-End Tactical Audio Chain

```
               BATTLEFIELD AUDIO (Speech + Gunfire + Engines)
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   INMP441 Microphone(s)   │
                        └─────────────┬─────────────┘
                                      │ (I2S INT16 PCM)
                                      ▼
                        ┌───────────────────────────┐
                        │ Double-Buffered DMA Audio │
                        │  (64 samples = 8.0 ms)    │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │  Windowed STFT Analysis   │
                        │  (128-pt Real FFT -> Mag) │
                        └─────────────┬─────────────┘
                                      │ [65 spectral bins]
                                      ▼
                        ┌───────────────────────────┐
                        │ MARKUSBLUE INT8 TinyML    │
                        │ Causal 1D DW-TCN Engine   │
                        │ (2,948 params, 2.88 KB)   │
                        └─────────────┬─────────────┘
                                      │ [65-bin Ideal Ratio Mask]
                                      ▼
                        ┌───────────────────────────┐
                        │  Inverse STFT Synthesis   │
                        │ (50% Hanning Overlap-Add) │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   VAD & Speech Level AGC  │
                        │ + Lookahead Peak Limiter  │
                        └─────────────┬─────────────┘
                                      │ [Enhanced Clean Audio]
                                      ▼
                        ┌───────────────────────────┐
                        │ MAX98357A / Communication │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                           SOLDIER HEARS CLEAR VOICE
```

---

## 2. Hardware Topology & Memory Budget
- **Microcontroller**: Espressif ESP8266 / ESP-12E (Tensilica Xtensa L106 @ 160 MHz)
- **Audio Sample Rate**: 8,000 Hz Mono (Tactical & Telephony Voice Band 300–3,400 Hz)
- **Hop Size**: 64 samples (8.0 ms frame duration)
- **Window Size**: 128 samples (16.0 ms window)
- **Total Static RAM**: 5.80 KB (Tensor arena: 3.50 KB, DMA & Ring buffers: 2.30 KB)
- **Flash Footprint**: 2.88 KB INT8 PROGMEM array (< 0.3% of 1MB flash)
- **Real-Time Factor (RTF)**: 0.231 (1.85 ms processing time / 8.00 ms frame budget)
