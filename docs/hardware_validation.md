# MARKUSBLUE (SIH26052) — Hardware Validation Strategy

## 1. Multi-Level Testing Hierarchy

### Level 1 — Unit Tests (Algorithm & Component Correctness)
- Executed via `python -m unittest discover tests`.
- Tests STFT / ISTFT reconstruction, Overlap-Add energy conservation, Voice Activity Detection (VAD) state transitions, Automatic Gain Control (AGC) attack/release rates, and Peak Safety Limiter lookahead ceiling compliance.
- Status: **VERIFIED (11 of 11 Unit Tests Passing)**.

### Level 2 — Offline Audio Tests (Tactical Scenarios)
- Executed via `python evaluate_esp32s3.py`.
- Evaluated on 100 tactical speech utterances mixed with gunfire blasts, vehicle engines, and background noise across -15dB, -10dB, -5dB, 0dB, +5dB, +10dB SNR.
- Critical Audio Blanking Test: Verifies zero speech gating or muting when high-energy gunshot impulse occurs during continuous speech.
- Status: **SOFTWARE VERIFIED**.

### Level 3 — ESP32-S3 Firmware Compilation & Static Profiling
- Compiled via PlatformIO for `esp32-s3-devkitc-1`.
- Memory profiling verifies:
  - SRAM footprint: ~30.1 KB (< 6.0% of 512 KB SRAM).
  - Flash footprint: < 1.2 MB (< 7.5% of 16 MB Flash).
  - PSRAM allocation: 64 KB ring-buffer.
- Status: **VERIFIED (Static Analysis & Architecture Mapping)**.

### Level 4 — Hardware-in-the-Loop (HIL) Testing
- Physical connection of ESP32-S3-DevKitC-1 N16R8, 2 × INMP441 microphones, MAX98357A Class-D amplifier, 8Ω speaker, 0.96" OLED display, MPU6050, and 3.7V Li-Po battery.
- Acoustic chamber feedback measurement and oscilloscope latency timing.
- Status: **PENDING HARDWARE VALIDATION** (Firmware, pinout, power tree, and wiring diagrams completed and ready for physical bench assembly).
