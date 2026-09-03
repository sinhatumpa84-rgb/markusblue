# MARKUSBLUE (SIH26052) — Hardware Wiring & Acoustic Design

## 1. Schematic Interconnect Diagram

```
                 ESP32-S3 N16R8                  PERIPHERALS
             ┌─────────────────────┐
             │                     │
             │       GPIO 4 (BCLK) ├────────┬──────────── INMP441 #1 (SCK)
             │                     │        └──────────── INMP441 #2 (SCK)
             │                     │
             │         GPIO 5 (WS) ├────────┬──────────── INMP441 #1 (WS)
             │                     │        └──────────── INMP441 #2 (WS)
             │                     │
             │         GPIO 6 (SD) ├────────┬──────────── INMP441 #1 (SD)
             │                     │        └──────────── INMP441 #2 (SD)
             │                     │
             │                     │  [3.3V] ──────────── INMP441 #1 (L/R - Left)
             │                     │  [GND]  ──────────── INMP441 #2 (L/R - Right)
             │                     │
             │       GPIO 15 (BCK) ├───────────────────── MAX98357A (BCLK)
             │       GPIO 16 (LRC) ├───────────────────── MAX98357A (LRC)
             │       GPIO 17 (DIN) ├───────────────────── MAX98357A (DIN)
             │                     │  [GND]  ──────────── MAX98357A (GAIN -> 12dB)
             │                     │
             │        GPIO 8 (SDA) ├────────┬──────────── OLED 128x64 (SDA)
             │                     │        └──────────── MPU6050 (SDA)
             │        GPIO 9 (SCL) ├────────┬──────────── OLED 128x64 (SCL)
             │                     │        └──────────── MPU6050 (SCL)
             │                     │
             │        GPIO 10 (CS) ├───────────────────── MicroSD (CS)
             │      GPIO 11 (MOSI) ├───────────────────── MicroSD (MOSI)
             │       GPIO 12 (SCK) ├───────────────────── MicroSD (SCK)
             │      GPIO 13 (MISO) ├───────────────────── MicroSD (MISO)
             │                     │
             │        GPIO 1 (PTT) ├───────────────────── PTT Switch (to GND)
             │     GPIO 2 (HAPTIC) ├───────────────────── 2N7002 Gate -> Motor
             │        GPIO 7 (ADC) ├───────────────────── 100k/100k Divider (VBATT)
             │                     │
             │                3.3V ├───────────────────── VDD (Sensors, SD, Mics)
             │                 GND ├───────────────────── Common Ground Plane
             └─────────────────────┘
```

---

## 2. Acoustic Enclosure & Mechanical Isolation Design

### Problem: Acoustic Feedback Loop
A major defect in real-time hearing enhancement headsets is acoustic feedback:
$$\text{Speaker Output} \xrightarrow{\text{Acoustic Leakage}} \text{Internal Mic 2} \xrightarrow{\text{AI Gain / AGC}} \text{Speaker Output}$$
This can cause howling oscillations, phase smearing, and severe speech degradation.

### Mechanical Mitigation Strategy (3D-Printed Ear Cup)
1. **Physical Baffle & Acoustic Foam Barrier**:
   - The 8Ω speaker driver is enclosed in a sealed acoustic chamber with front-damping mesh directing sound exclusively toward the ear canal.
   - Acoustic dense memory foam (closed-cell polyurethane) seals the perimeter against the skull, providing > 22 dB passive noise isolation.
2. **Microphone Spatial Separation**:
   - **Mic 1 (Exterior Reference)** is positioned on the outer surface of the ear cup, shielded by a sintered bronze wind/debris mesh.
   - **Mic 2 (Interior Ear Mic)** is positioned inside the ear cup cavity facing the ear canal to capture the acoustic field at the eardrum.
   - High-density silicone gaskets isolate both microphones mechanically from PCB body-borne vibrations.
3. **Firmware Safeguard**:
   - Continuous feedback oscillation detector monitors high-energy single-bin spectral peaks.
   - Instant 18 dB attenuation clamp applied if closed-loop gain approaches unit stability threshold.
