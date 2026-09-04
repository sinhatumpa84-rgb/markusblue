# MARKUSBLUE (SIH26052) — Dataset Expansion Audit Report

## 1. Existing Dataset Audit
- **Speech Data**: `datasets/speech/` (2,400 clean speech utterances, 16 kHz mono WAV).
- **Gunfire Impulses**: `datasets/gunshot/` (6,000 gunfire recordings from 22LR to 12-gauge).
- **Existing Background Noise**: `datasets/background_noise/` (2,400 ambient and engine recordings).
- **Mechanical Impulses**: `datasets/other_impulse/` (2,400 door slams and hammer drops).
- **Extended Raw Archives**: `data/` (27,626 files) and `gunsound/` (26 archive zips).
- **Status of Original Assets**: **READ-ONLY & 100% PROTECTED** (Will not be renamed, moved, normalized, or resampled in place).

---

## 2. Identified Acoustic Gaps (Missing Noise Categories)
Tactical combat and realistic field operations encounter diverse acoustic interference beyond clean speech and close-range gunfire:
1. **Aviation & Rotorcraft**:
   - High-speed turbofan scream, jet engine afterburners, propeller aircraft, helicopter blade slap, rotorcraft hover and approach flybys.
2. **Heavy Ground & Armored-Vehicle Proxies**:
   - Heavy tracked machinery (excavator, bulldozer, tank-like track squeal), large industrial diesel engines, heavy trucks, buses, tractors.
3. **Industrial & Mechanical Machinery**:
   - Diesel generators, air compressors, high-speed rotary drills, angle grinders, chainsaws, ventilation fans, hydraulic pumps.
4. **Environmental & Atmospheric**:
   - Severe wind turbulence, rain downpours, thunder cracks, urban road ambience, building echo.
5. **Tactical Communications & Human Chatter**:
   - Competing crowd chatter, running combat boots, tactical radio static, walkie-talkie squelch bursts, electrical 50Hz mains hum and harmonic hum.
6. **Sudden Non-Gunfire Transients**:
   - Metal-on-metal impacts, breach drops, distant explosive thuds.

---

## 3. Audio Standardization Specification
- **Sampling Rate**: **16,000 Hz** (standard wideband tactical speech for ESP32-S3).
- **Channels**: **1 (Mono)**.
- **Bit Depth**: **16-bit Signed Linear PCM**.
- **STFT Configuration**: $N_{FFT} = 256$, Hop Size = 64 samples (4.0 ms frame duration), 129 positive frequency bins.
- **Storage Strategy**:
  - Raw downloaded audio preserved in `datasets/external_noise/<category>/`.
  - Standardized derived clips generated in `datasets/derived/train/`, `datasets/derived/validation/`, and `datasets/derived/test/`.
  - Zero modifications to `datasets/original/`.

---

## 4. Augmentation & Mixing Strategy
- **Multi-Noise Mixing**: Dynamic combination of 1 to 4 independent acoustic sources per speech utterance.
- **SNR Randomization**: -15 dB to +20 dB in randomized steps.
- **Temporal Movement**: Stationary background, approaching flybys, receding flybys, transient bursts, and smooth crossfades.
- **Data Leakage Safeguards**: Strict source-level grouping across train, validation, and test splits.
