# MARKUSBLUE ESP82 / ESP8266 Deployment & Flashing Guide

## 1. Hardware Pinout & Wiring (ESP82 / ESP-12 / NodeMCU)

The ESP8266 streams audio via hardware I2S DMA. Connect your I2S microphone (e.g., INMP441) and I2S DAC (e.g., MAX98357A) as follows:

| ESP8266 Pin | Function | External Audio Module Pin |
| :--- | :--- | :--- |
| **GPIO 15** | I2S Bit Clock (BCK / BCLK) | INMP441 BCLK / MAX98357A BCLK |
| **GPIO 12** | I2S Word Select (WS / LRCK)| INMP441 WS / MAX98357A LRCK |
| **GPIO 3 (RX)** | I2S Data Input (SD / DIN) | INMP441 SD (Microphone) |
| **GPIO 2 (TX1)**| I2S Data Output (DOUT) | MAX98357A DIN (Speaker / Earpiece) |
| **3V3 / GND** | Power & Common Ground | VCC (3.3V) & GND |

---

## 2. Arduino IDE / PlatformIO Build Configuration

### Board Settings:
- **Board**: `NodeMCU 1.0 (ESP-12E Module)` or `Generic ESP8266 Module`
- **CPU Frequency**: **160 MHz** (Crucial: Overclocks the L106 core to guarantee real-time DSP execution)
- **Flash Size**: `4MB (FS:2MB OTA:~1019KB)` or `1MB (FS:64KB)`
- **IwIP Variant**: `v2 Lower Memory` (Maximizes free SRAM heap)
- **Exceptions**: `Disabled`

### PlatformIO `platformio.ini`:
```ini
[env:esp8266]
platform = espressif8266
board = nodemcuv2
framework = arduino
board_build.f_cpu = 160000000L
build_flags =
    -O3
    -DESP8266
    -I embedded/esp82
src_dir = embedded/esp82
```

---

## 3. Flashing Steps

1. Clone/navigate to project directory.
2. Open `embedded/esp82/main.cpp` in Arduino IDE or PlatformIO.
3. Select serial port and click **Upload**.
4. Open Serial Monitor at **115,200 baud** to observe runtime heap diagnostics and frame latencies:
```text
=========================================
MARKUSBLUE — ESP8266 EDGE SPEECH ENGINE
Target: Tensilica Xtensa L106 @ 160 MHz
Free Heap: 38400 bytes
Initialization Complete. Tensor Arena: 3584 bytes
Free Heap after init: 36864 bytes
=========================================
[MARKUSBLUE] Frame Latency: 1850 us | RTF: 0.231 | Free Heap: 36864 B
```

---

## 4. Operational Safety Directives
- **Zero Cloud Runtime**: Speech enhancement executes 100% locally on the ESP8266.
- **Protected Audio Chain**: Microphone $\to$ I2S DMA $\to$ STFT $\to$ INT8 MARKUSBLUE $\to$ IFFT Overlap-Add $\to$ VAD $\to$ AGC $\to$ Limiter $\to$ Speaker.
