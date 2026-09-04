# MARKUSBLUE (SIH26052) — Hardware Bill of Materials (BOM)

## 1. System Components & Procurement Matrix

| Item # | Component Description | Manufacturer / Part Number | Package / Footprint | Qty | Unit Price (INR) | Subtotal (INR) | Primary Function |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | ESP32-S3 Dual-Core LX7 MCU | Espressif Systems / ESP32-S3-WROOM-1-N16R8 | SMD Module / DevKit | 1 | ₹680.00 | ₹680.00 | Main Edge-AI SoC (240MHz, 16MB Flash, 8MB PSRAM) |
| **2** | Omnidirectional MEMS Microphones | InvenSense / INMP441 | Surface Mount Module | 2 | ₹195.00 | ₹390.00 | Digital I2S Microphones (Mic 1: Ref, Mic 2: Ear) |
| **3** | Mono Class-D Audio Amplifier | Maxim Integrated / MAX98357A | QFN Module | 1 | ₹165.00 | ₹165.00 | I2S Digital DAC & 3.2W Class-D Power Amp |
| **4** | 8Ω 2W Tactical Speaker Transducer | PUI Audio / AS04008PR-R | 40mm Round Transducer | 1 | ₹180.00 | ₹180.00 | Circumaural Ear-Cup Acoustic Transducer |
| **5** | Li-Po Battery 3.7V 2500mAh | Generic / 103450 Li-Po Cell | Pouch Cell with JST-PH2.0 | 1 | ₹420.00 | ₹420.00 | Primary Rechargeable Power Reservoir |
| **6** | Li-Po Charge & Protection Board | TP4056 + DW01 + FS8205A | SMD PCB Module (USB-C) | 1 | ₹45.00 | ₹45.00 | CC/CV 1A Charger with Low-Voltage Cutoff |
| **7** | 5V Synchronous Boost Converter | MT3608 or TPS61023 | SOT-23 / Mini Module | 1 | ₹55.00 | ₹55.00 | Regulates 3.7V battery to clean 5.0V for MAX98357A |
| **8** | 3.3V Low-Dropout Regulator (LDO) | Texas Instruments / TLV75533PDBVR | SOT-23-5 (500mA, Low Noise) | 1 | ₹35.00 | ₹35.00 | Clean low-noise 3.3V rail for INMP441 and ESP32-S3 |
| **9** | 0.96" Monochrome I2C OLED Display | Solomon Systech / SSD1306 | 128x64 I2C Module | 1 | ₹160.00 | ₹160.00 | Operator Visual Status (Battery, Mode, Health) |
| **10** | 6-Axis Inertial Measurement Unit (IMU)| TDK InvenSense / MPU6050 | QFN-24 Module | 1 | ₹145.00 | ₹145.00 | Head-Pose & Blast Shock Dynamic Profiling |
| **11** | MicroSD Card Breakout Module | Molex / Generic SPI SD Socket | MicroSD Push-Push Socket | 1 | ₹40.00 | ₹40.00 | Real-Time Telemetry & Acoustic Logger |
| **12** | Tactical PTT Pushbutton | C&K / Sealed Tactile Switch | IP67 Panel Mount | 1 | ₹65.00 | ₹65.00 | Push-to-Talk and Audio Mode Selector |
| **13** | ERM Vibration Haptic Motor | Generic / 1027 Coin Motor | 10mm Coin Vibrator | 1 | ₹30.00 | ₹30.00 | Tactile Operator Warning & Low Battery Alert |
| **14** | MOSFET Gate Driver for Haptics | ON Semi / 2N7002 N-Channel | SOT-23 | 1 | ₹8.00 | ₹8.00 | Drives Haptic Motor from GPIO 2 |
| **15** | Passive RLC Components | Yageo / TDK / Murata | 0805 SMD / Through-Hole | 1 Kit | ₹120.00 | ₹120.00 | Decoupling caps (100nF, 10µF, 220µF), pullups |
| **16** | Circumaural Ear-Cup & Headband | 3D Printed PETG / ABS Ear-Cup | Custom Industrial CAD | 1 | ₹650.00 | ₹650.00 | Ruggedized Acoustic Chamber with Seal Cushion |
| **TOTAL** | **COMPLETE PROTOTYPE BOM** | — | — | **17 Items** | — | **₹3,188.00** | **Total Prototype Cost (~$38.20 USD)** |

---

## 2. Component Availability & Supply Chain Resilience
- **All components** are standard commercial-off-the-shelf (COTS) components readily available in Indian electronics distribution hubs (SP Road Bengaluru, Lamington Road Mumbai, Chandni Chowk Delhi, Robu.in, element14 India).
- **Zero Proprietary Locked Silicon**: No restricted FPGA/ASIC chips.
- **Zero Foreign Cloud Dependency**: 100% locally programmed and processed.
