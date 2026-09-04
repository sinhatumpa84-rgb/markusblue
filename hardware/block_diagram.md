# MARKUSBLUE (SIH26052) — Hardware Block Diagram & System Interconnect

## 1. System Block Diagram (Mermaid)

```mermaid
graph TD
    subgraph Acoustic_Environment [Acoustic Environment]
        EXT_SOUND["External Noise / Speech / Alarms"]
        EAR_SOUND["Ear Canal Residual Sound"]
    end

    subgraph Audio_Capture [Dual MEMS Microphone Capture]
        MIC1["INMP441 #1 (External Reference)<br/>L/R=VDD (Left Channel)"]
        MIC2["INMP441 #2 (Ear-Side Error)<br/>L/R=GND (Right Channel)"]
    end

    subgraph MCU_ESP32S3 [ESP32-S3 N16R8 SoC (Dual Core LX7 @ 240 MHz)]
        I2S0_RX["I2S0 Peripheral (RX)<br/>BCLK: GPIO4 | WS: GPIO5 | SD: GPIO6<br/>Stereo DMA Ping-Pong Buffer"]
        CORE0["Core 0: Audio Pipeline<br/>STFT (256-pt) -> Spatial Filter -> VAD"]
        CORE1["Core 1: Edge-AI Engine<br/>MARKUSBLUEStudentEnhancer (INT8)<br/>Mask Generation (129 bins)"]
        SYNTH["Synthesis & Protection<br/>ISTFT -> AGC -> Lookahead Limiter"]
        I2S1_TX["I2S1 Peripheral (TX)<br/>BCLK: GPIO15 | WS: GPIO16 | DOUT: GPIO17<br/>Mono 16-bit DMA Buffer"]
    end

    subgraph Audio_Output [Audio Output Stage]
        AMP["MAX98357A I2S Class-D DAC/Amp<br/>Gain: 12 dB (GND)"]
        SPEAKER["8Ω 2W Tactical Headset Speaker / Earphone"]
    end

    subgraph Peripherals_Telemetry [Control & Telemetry]
        PTT["PTT Tactical Button (GPIO 1)"]
        HAPTIC["Haptic Feedback Motor (GPIO 2 / 2N7002)"]
        BATT_SENSE["Battery Voltage Divider (GPIO 7)"]
        I2C_BUS["I2C Bus (SDA: GPIO8, SCL: GPIO9)"]
        OLED["0.96 inch SSD1306 OLED (0x3C)"]
        IMU["MPU6050 Motion Sensor (0x68)"]
        SD_SPI["SPI Bus (GPIO 10,11,12,13)<br/>MicroSD Telemetry Logger"]
    end

    subgraph Power_Subsystem [Power Management]
        BATT["3.7V 2500mAh Li-Po Cell"]
        CHARGER["TP4056 USB-C Charger & DW01 Protection"]
        BOOST["5V Boost Converter (for MAX98357A & Analog)"]
        LDO["3.3V High-PSRR LDO (for ESP32-S3 & INMP441s)"]
    end

    %% Audio Connections
    EXT_SOUND --> MIC1
    EAR_SOUND --> MIC2
    MIC1 --> I2S0_RX
    MIC2 --> I2S0_RX
    I2S0_RX --> CORE0
    CORE0 --> CORE1
    CORE1 --> SYNTH
    SYNTH --> I2S1_TX
    I2S1_TX --> AMP
    AMP --> SPEAKER

    %% Control & Telemetry Connections
    PTT --> CORE0
    CORE0 --> HAPTIC
    BATT_SENSE --> CORE0
    CORE0 --> I2C_BUS
    I2C_BUS --> OLED
    I2C_BUS --> IMU
    CORE0 --> SD_SPI

    %% Power Routing
    BATT --> CHARGER
    CHARGER --> BOOST
    CHARGER --> LDO
    BOOST --> AMP
    LDO --> MCU_ESP32S3
    LDO --> MIC1
    LDO --> MIC2
    LDO --> OLED
    LDO --> IMU
    LDO --> SD_SPI
```

---

## 2. Bus Architecture Summary

| Bus Interface | Peripheral Devices | Pins Allocated | Operating Speed | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| **I2S0 (RX)** | INMP441 Ref & Error | GPIO 4, 5, 6 | 1.024 MHz BCLK | 16 kHz 16-bit Stereo (Time-Multiplexed) |
| **I2S1 (TX)** | MAX98357A Amp | GPIO 15, 16, 17 | 1.024 MHz BCLK | 16 kHz 16-bit Mono Linear PCM |
| **I2C0** | SSD1306 OLED, MPU6050 | GPIO 8 (SDA), 9 (SCL)| 400 kHz Fast Mode | 7-bit addressing (`0x3C`, `0x68`) |
| **SPI2** | MicroSD Card Slot | GPIO 10, 11, 12, 13 | 20 MHz Max | SPI Mode 0, FAT32 non-blocking write |
| **Discrete ADC** | Battery Voltage Divider | GPIO 7 (ADC1_CH6) | 1 kHz Polling | 12-bit ADC (0 to 3.3V mapped to 3.0-4.2V) |
| **GPIO Out/In** | PTT, Haptic Motor | GPIO 1 (PTT), 2 (Haptic)| Asynchronous/PWM | Active LOW input, PWM gate drive |
