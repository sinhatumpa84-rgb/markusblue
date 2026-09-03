# MARKUSBLUE (SIH26052) — Power Architecture & Battery Tree

## 1. Power Topology Overview

```
                      ┌────────────────────────────────────────┐
                      │    3.7V Li-Po Battery (2500 mAh)       │
                      │   (Nominal: 3.7V, Full: 4.2V, End: 3.2V)│
                      └───────────────────┬────────────────────┘
                                          │
                      ┌───────────────────▼────────────────────┐
                      │  TP4056 Charging + DW01A Protection    │
                      │ (1A Charge, Overcharge/Discharge Cut)  │
                      └─────────────┬──────────────────────────┘
                                    │ V_BATT (3.2V - 4.2V)
             ┌──────────────────────┴──────────────────────┐
             │                                             │
             ▼                                             ▼
┌─────────────────────────┐                   ┌─────────────────────────┐
│ Ultra-Low Dropout (LDO) │                   │ Synchronous Boost 5.0V  │
│  AP2112K-3.3 / ME6211   │                   │    (TPS61023 / MT3608)  │
│  3.3V System Rail (1A)  │                   │  5.0V Audio Rail (1.5A) │
└────────────┬────────────┘                   └────────────┬────────────┘
             │ 3.3V Clean Rail                             │ 5.0V Boost Rail
   ┌─────────┼─────────┬─────────┬─────────┐               │
   │         │         │         │         │               │
   ▼         ▼         ▼         ▼         ▼               ▼
┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐         ┌───────────┐
│ESP32│   │2x   │   │OLED │   │MPU  │   │Micro│         │ MAX98357A │
│ -S3 │   │INMP │   │ SSD │   │6050 │   │ SD  │         │  Class-D  │
│N16R8│   │ 441 │   │1306 │   │ IMU │   │ Card│         │ Amplifier │
└─────┘   └─────┘   └─────┘   └─────┘   └─────┘         └─────┬─────┘
                                                              │
                                                              ▼
                                                        ┌───────────┐
                                                        │ 8Ω Driver │
                                                        └───────────┘
```

---

## 2. Voltage & Rail Requirements by Subsystem

| Subsystem | Input Rail | Voltage Tolerance | Quiescent Current | Active Current (Typ) | Peak Current |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ESP32-S3 N16R8** | 3.3V Rail | 3.0V – 3.6V | 15 mA (Light sleep) | 72 mA (240MHz DSP+AI) | 120 mA |
| **INMP441 Mic #1** | 3.3V Rail | 1.8V – 3.3V | 0.01 mA (Standby) | 1.4 mA | 1.6 mA |
| **INMP441 Mic #2** | 3.3V Rail | 1.8V – 3.3V | 0.01 mA (Standby) | 1.4 mA | 1.6 mA |
| **OLED (SSD1306)** | 3.3V Rail | 3.0V – 5.0V | 0.05 mA | 18.0 mA | 25.0 mA |
| **MPU6050 IMU** | 3.3V Rail | 2.375V – 3.46V | 0.005 mA | 3.8 mA | 4.2 mA |
| **MicroSD Module**| 3.3V Rail | 2.7V – 3.6V | 0.15 mA | 28.0 mA (Burst write)| 65.0 mA |
| **MAX98357A Amp** | 5.0V Rail | 2.5V – 5.5V | 2.4 mA (No signal) | 85.0 mA (Normal voice)| 380.0 mA (Max peak) |
| **Haptic Motor** | V_BATT | 3.0V – 4.2V | 0 mA | 75.0 mA (100ms pulse)| 95.0 mA |

---

## 3. Power Budget Scenarios & Battery Runtime

Calculations based on **3.7V 2500 mAh Li-Po Battery (9.25 Watt-hours)**:

### Scenario A: Standby / Passive Hearing (Audio Pipeline Idling, Mics Active)
- Average Current: **95 mA** @ 3.7V (~0.35 W)
- Estimated Runtime: **26.3 Hours**

### Scenario B: Active AI Speech Enhancement (Real-Time DSP + AI Inference + Voice Output)
- ESP32-S3 (240 MHz Dual-Core): 75 mA
- 2x INMP441: 2.8 mA
- MAX98357A (75 dB SPL Voice): 85 mA @ 5V -> ~115 mA @ 3.7V equivalent
- OLED Status: 18 mA
- Total Current: **210.8 mA** @ 3.7V (~0.78 W)
- Estimated Runtime: **11.85 Hours** Continuous

### Scenario C: Tactical Field Operations (AI + Audio + OLED + Periodic SD Logging)
- Average Current: **235 mA** @ 3.7V (~0.87 W)
- Estimated Runtime: **10.64 Hours** Continuous

---

## 4. Hardware Power Safety Rules
1. **Wi-Fi & Bluetooth RF Subsystem Disabled**: Radios are held in power-down mode during tactical audio operation to eliminate RF noise injection into high-impedance MEMS microphone lines.
2. **Brownout Detector Threshold**: ESP32-S3 brownout reset voltage configured to **2.8V** (Level 7) to guarantee orderly shutdown prior to battery protection trip.
3. **Decoupling Strategy**:
   - 10 µF + 100 nF ceramic capacitors placed within 5 mm of ESP32-S3 VDD pins.
   - 220 µF low-ESR electrolytic capacitor on 5.0V Class-D amplifier power supply to suppress audio ground bounce.
