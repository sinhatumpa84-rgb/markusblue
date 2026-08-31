# MARKUSBLUE Model Card

## Model Details
- **Model Name**: MARKUSBLUE Student Speech Enhancer
- **Version**: 7.1.0
- **Framework**: PyTorch / TFLite Micro / LiteRT
- **Target Hardware**: ESP32-S3 Dual-Core Xtensa LX7 @ 240 MHz
- **License**: MIT
- **Contact**: SIH26052 Engineering Team

## Intended Use
- Real-time speech separation and acoustic noise suppression for defense, industrial, and tactical communications.
- Restoration of weak/damped voice levels via VAD-aware Automatic Gain Control and Dynamic Range Compression.

## Factors & Limitations
- **Academic Prototype**: Not certified as medical hearing protection or formal military PPE.
- **Microphone Linearity**: Designed for standard 16 kHz mono digital I2S microphones (e.g. INMP441 / ICS-43434).

## Model Architecture & Training Metrics
- **Parameters**: 18,465
- **INT8 Quantized Size**: 18.04 KB
- **Input Dimension**: `[Batch, 129, 32]` (129 FFT bins, 32 time frames)
- **Output**: Ideal Ratio Mask (IRM) `[Batch, 129, 32]`
- **SI-SDR Improvement**: +12.40 dB average gain over noisy mixtures.
- **Latency**: 3.50 ms per 32 ms chunk (Real-Time Factor: 0.110).
