# MARKUSBLUE (SIH26052) — Power Consumption & Battery Benchmark Report
**Hardware Target**: Espressif ESP32-S3 N16R8  
**Verification Standard**: Theoretical electrical engineering power budget calculations.

---

## 1. Executive Summary

> [!WARNING]
> **Bench Measurement Status**: **ESTIMATED — NOT HARDWARE VERIFIED**  
> Physical battery discharge curves and current draws have **NOT** been measured using a hardware DC bench power supply or digital multimeter because physical hardware is not connected in this local development environment.  
> All figures below represent component-level datasheet calculations: **ESTIMATED — NOT HARDWARE VERIFIED**.

---

## 2. Power Tree Current Draw Breakdown (3.7V Nominal Bus)

| Subsystem / Peripheral | Operating Voltage | Active Current Draw | Quiescent / Standby | Power Consumption (mW) | Source / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ESP32-S3 SoC (Dual Core @ 240MHz)** | 3.3V | **95.0 mA** | 25.0 mA | **313.5 mW** | Both Xtensa LX7 cores active, Wi-Fi/BLE radios powered down |
| **Dual INMP441 Microphones (Mic 1 & 2)** | 3.3V | **2.8 mA** (1.4 mA each) | 0.02 mA | **9.2 mW** | Continuous 16 kHz stereo I2S sampling |
| **MAX98357A Class-D Power Amplifier** | 5.0V (Boosted) | **45.0 mA** (Average speech) | 0.01 mA | **225.0 mW** | 8Ω load, 12 dB gain, typical conversational listening level |
| **0.96" SSD1306 OLED Display** | 3.3V | **12.0 mA** | 0.05 mA | **39.6 mW** | 128x64 monochrome, 25% pixel fill factor, no animations |
| **MPU6050 6-Axis IMU** | 3.3V | **3.8 mA** | 0.01 mA | **12.5 mW** | Low-power accelerometer & gyro polling @ 50 Hz |
| **MicroSD SPI Logging Interface** | 3.3V | **2.5 mA** (Average write)| 0.10 mA | **8.3 mW** | 512-byte telemetry buffer flushed every 2.0 seconds |
| **Haptic Vibration Motor (When Active)** | 3.3V (Switched)| **75.0 mA** (Peak 150ms)| 0.00 mA | **247.5 mW** (Pulsed) | 150 ms tactile alert pulses (average < 1.0 mW) |
| **LDO & 5V Boost Regulators Quiescent** | 3.7V | **4.0 mA** | 0.05 mA | **14.8 mW** | Conversion efficiency losses (~88% efficiency) |
| **TOTAL AVERAGE SYSTEM LOAD** | **3.7V** | **~165.1 mA** | **~25.2 mA** | **~622.9 mW** | **ESTIMATED — NOT HARDWARE VERIFIED** |

---

## 3. Battery Runtime Estimates (3.7V 2500 mAh Li-Po Cell)

| Operational Mode | System Current | Estimated Runtime | Verification Status | Operational Scenario |
| :--- | :--- | :--- | :--- | :--- |
| **Full Operational Mode** | **165.1 mA** | **~15.1 Hours** | **ESTIMATED** | Continuous AI speech enhancement + OLED on + SD logging |
| **Tactical Stealth Mode (OLED Off)** | **153.1 mA** | **~16.3 Hours** | **ESTIMATED** | AI enhancement active, display blanked for night operations |
| **High Ambient Noise / Max Audio Mode**| **210.0 mA** | **~11.9 Hours** | **ESTIMATED** | Continuous high-volume playback (MAX98357A @ 1.5W output) |
| **Standby / Ambient Monitoring Mode** | **38.0 mA** | **~65.7 Hours** | **ESTIMATED** | Low-power VAD listening, AI inference sleeping |

---

## 4. Power Integrity & Safety Measures
1. **Low-Voltage Cutoff**: Hardware DW01 battery protection chip cuts power at **3.0V**, preventing deep discharge and permanent lithium degradation.
2. **Bulk Decoupling**: 220 µF electrolytic capacitor on the 5V booster rail absorbs Class-D switching spikes, preventing resets on loud gunfire impulses.
3. **Thermal Safety**: TP4056 charge controller is limited to **500 mA charge rate** to ensure safe temperature within the sealed ear-cup enclosure.
