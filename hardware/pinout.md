# MARKUSBLUE (SIH26052) — Hardware Pinout Matrix

## 1. Platform Specification
- **System**: MARKUSBLUE Indigenous Tactical Audio Enhancement System
- **MCU**: Espressif ESP32-S3-WROOM-1 / ESP32-S3-DevKitC-1 N16R8
- **Core Architecture**: Dual-core Xtensa® 32-bit LX7 @ 240 MHz with Vector/DSP extensions
- **Memory**: 16 MB Quad SPI Flash + 8 MB Octal SPI PSRAM (OPI mode)
- **Internal SRAM**: 512 KB SRAM (DMA-accessible)

---

## 2. Complete GPIO Allocation Table

| Peripheral | Signal | ESP32-S3 GPIO | Direction | Type / Notes | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **INMP441 #1 & #2 (I2S RX)** | BCLK / SCK | **GPIO 4** | Output from MCU | Shared I2S Bit Clock (Master) | **VERIFIED** |
| | WS / Word Select | **GPIO 5** | Output from MCU | Shared Frame Sync (16 kHz) | **VERIFIED** |
| | SD / Serial Data | **GPIO 6** | Input to MCU | Time-multiplexed Stereo Data | **VERIFIED** |
| | L/R Pin (Mic 1: Ref) | VDD (3.3V) | Pin Level | Left Channel (External Reference) | **VERIFIED** |
| | L/R Pin (Mic 2: Ear) | GND (0V) | Pin Level | Right Channel (Internal Earphone) | **VERIFIED** |
| **MAX98357A (I2S TX)** | BCLK | **GPIO 15** | Output from MCU | I2S1 Bit Clock | **VERIFIED** |
| | LRC / WS | **GPIO 16** | Output from MCU | I2S1 Word Select / Sync | **VERIFIED** |
| | DIN | **GPIO 17** | Output from MCU | Serial Audio PCM stream to DAC | **VERIFIED** |
| | GAIN | GND | Pin Level | 12 dB fixed hardware gain | **VERIFIED** |
| | SD_MODE | 3.3V (100kΩ pull-up) | Control | Unmuted Mono (Left Channel) | **VERIFIED** |
| **I2C Bus (OLED & MPU6050)**| SDA | **GPIO 8** | Bidirectional | 4.7 kΩ pull-up to 3.3V | **VERIFIED** |
| | SCL | **GPIO 9** | Output from MCU | 4.7 kΩ pull-up to 3.3V (400 kHz) | **VERIFIED** |
| | OLED Display | Address: `0x3C` | I2C Device | SSD1306 128x64 0.96" Monochrome | **VERIFIED** |
| | MPU6050 IMU | Address: `0x68` | I2C Device | AD0 tied to GND | **VERIFIED** |
| **MicroSD Card (SPI)** | CS | **GPIO 10** | Output from MCU | Chip Select (Active LOW) | **VERIFIED** |
| | MOSI | **GPIO 11** | Output from MCU | SPI Master Out | **VERIFIED** |
| | SCK | **GPIO 12** | Output from MCU | SPI Clock (20 MHz max) | **VERIFIED** |
| | MISO | **GPIO 13** | Input to MCU | SPI Master In (10 kΩ pull-up) | **VERIFIED** |
| **Tactical Controls** | PTT Button | **GPIO 1** | Input to MCU | Internal Pull-Up, Active LOW | **VERIFIED** |
| | Haptic Motor | **GPIO 2** | Output from MCU | PWM Drive via 2N7002 MOSFET | **VERIFIED** |
| | Battery Sense | **GPIO 7** | Analog Input | ADC1_CH6 (100kΩ / 100kΩ divider) | **VERIFIED** |

---

## 3. Critical Pin Conflicts & Reserved Pins Audit

| GPIO Range | Function / Assignment | Constraint / Safety Rule | Status |
| :--- | :--- | :--- | :--- |
| **GPIO 0** | Boot Strap Pin | Pulled HIGH for normal boot. Used for flashing. | **PROTECTED (Not allocated to audio)** |
| **GPIO 19, 20** | USB D- / D+ | Native USB OTG / JTAG programming & telemetry. | **RESERVED FOR USB/DEBUG** |
| **GPIO 33–37** | Octal Flash / PSRAM | Dedicated to High-Speed 8-line Octal PSRAM bus. | **STRICTLY RESERVED (NO EXTERNAL WIRING)** |
| **GPIO 43, 44** | UART0 TX / RX | Hardware UART console logging. | **CONSOLE DEBUG ONLY** |
| **GPIO 45, 46** | Boot Strapping | VDD_SPI & ROM Log Strapping pins. | **UNUSED / FLOATING** |

---

## 4. Hardware Connection Guidelines

### Dual INMP441 Microphone Bus
1. Both INMP441 MEMS microphones share the identical BCLK (GPIO 4) and WS (GPIO 5) lines.
2. The `SD` (Serial Data) pins are connected together to GPIO 6.
3. INMP441 #1 (`L/R` -> 3.3V) drives the bus during the **Left** clock phase (External Reference mic).
4. INMP441 #2 (`L/R` -> GND) drives the bus during the **Right** clock phase (Internal Ear mic).
5. Add a 100 nF decoupling capacitor directly adjacent to the `VDD` pin of each microphone.

### MAX98357A Class-D Speaker Amplifier
1. Driven via separate I2S1 peripheral (GPIO 15, 16, 17) to allow independent clocking and avoid DMA collision.
2. Connect `GAIN` to `GND` for optimal 12 dB dynamic range headroom without distortion.
3. Place a 220 µF bulk capacitor on the 5V power supply rail close to MAX98357A to handle transient Class-D current spikes.
