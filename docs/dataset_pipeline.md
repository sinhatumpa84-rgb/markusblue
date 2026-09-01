# MARKUSBLUE Dataset Pipeline & SNR Mixture Synthesis

## 1. Dataset Directory Organization (100% Read-Only Protected)

```
datasets/
├── speech/             (2,400 clean speech WAV files @ 16 kHz)
├── background_noise/   (2,400 stationary/environmental noise WAV files @ 16 kHz)
├── gunshot/            (6,000 tactical gunshot impulse WAV files @ 16 kHz)
└── other_impulse/      (2,400 industrial/mechanical impulse WAV files @ 16 kHz)
```

- **Total Files**: 13,200 audio files (100% preserved, 0 bytes modified)
- **Decimation**: On-the-fly 2:1 decimation ($16\text{ kHz} \to 8\text{ kHz}$) in memory to match the ESP8266 telecommunication voice-band (300–3,400 Hz) without modifying raw disk files.

---

## 2. Dynamic Online Mixture Generation

For every training/validation step:
1. Clean speech sample $s(t)$ and noise sample $n(t)$ are loaded into memory.
2. Signal powers $P_s = \mathbb{E}[s^2]$ and $P_n = \mathbb{E}[n^2]$ are computed.
3. Target SNR $\text{SNR}_{\text{target}}$ is randomly selected from `[-20 dB, +20 dB]`.
4. Noise scaling factor is calculated:
   $$\alpha = \sqrt{\frac{P_s}{10^{\text{SNR}_{\text{target}} / 10} \cdot P_n}}$$
5. Noisy mixture $x(t) = s(t) + \alpha \cdot n(t)$ is generated and normalized.
