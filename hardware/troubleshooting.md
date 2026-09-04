# MARKUSBLUE (SIH26052) — Hardware Troubleshooting & Diagnostic Runbook

## 1. Electrical & Power Subsystem Diagnostics

| Symptom | Root Cause Analysis | Diagnostic Check | Remedial Action |
| :--- | :--- | :--- | :--- |
| **System fails to boot / No LED on ESP32-S3** | Battery voltage below 3.0V cut-off or loose JST connector | Measure battery terminals with multimeter. Must be $\ge 3.4\text{ V}$. | Plug in USB-C charger. Verify red charging LED illuminates on TP4056. |
| **ESP32-S3 resets repeatedly during audio output** | Class-D amplifier drawing transient current, causing 3.3V LDO brownout | Monitor 3.3V rail with oscilloscope during loud impulses. | Verify 220 µF bulk capacitor is present on 5V booster output; ensure LDO input is fed directly from battery. |
| **Excessive hum / 50Hz electrical buzz in speaker** | Ground loop between USB charger and audio electronics | Disconnect USB-C charging cable. Test on pure battery power. | Never operate headset while charging from noisy ungrounded AC mains adapters. |

---

## 2. Audio Capture & I2S Diagnostics

| Symptom | Root Cause Analysis | Diagnostic Check | Remedial Action |
| :--- | :--- | :--- | :--- |
| **Only one microphone captures audio (Mono only)** | L/R pin floating or tied to identical logic level | Inspect voltage on L/R pin of both INMP441 modules. | Mic 1 L/R must be 3.3V (Left); Mic 2 L/R must be GND (Right). |
| **Complete silence / Zero samples in I2S buffer** | BCLK (GPIO 4) or WS (GPIO 5) shorted to ground | Measure clock pins with multimeter (frequency mode: BCLK=1.024MHz, WS=16kHz). | Verify wiring harness integrity; check for cold solder joints on GPIO 4, 5, 6. |
| **Loud high-frequency feedback squeal** | Speaker output acoustically leaking into internal microphone | Inspect ear-cup perimeter seal and speaker baffle gasket. | Tighten baffle mounting screws; ensure closed-cell foam damping is in place. |
| **Digital distortion / harsh crackling** | I2S DMA buffer underrun or digital clipping | Check OLED diagnostic screen for `BUF_UNDERRUN` counter. | Ensure SD logging does not block Core 0 audio task; verify DMA buffer count $\ge 4$. |

---

## 3. Peripheral Bus Diagnostics

| Symptom | Root Cause Analysis | Diagnostic Check | Remedial Action |
| :--- | :--- | :--- | :--- |
| **OLED display remains blank** | I2C address mismatch or missing pull-up resistors | Run I2C scanner (`tools/diagnose_i2c.py`). Verify address `0x3C`. | Solder 4.7 kΩ pull-up resistors on SDA (GPIO 8) and SCL (GPIO 9). |
| **MicroSD card initialization fails** | SPI bus speed exceeds card capability or loose wiring | Check serial log for `SD_MOUNT_FAIL`. | Verify 10 kΩ pull-up on MISO (GPIO 13); format card with FAT32 / 32KB clusters. |
| **Haptic motor does not vibrate** | 2N7002 gate threshold voltage not reached or reversed diode | Test GPIO 2 voltage when triggered. Must pulse to 3.3V. | Check orientation of flyback diode across motor terminals (cathode to 3.3V, anode to drain). |
