# MARKUSBLUE (SIH26052) — Technical & Environmental Limitations

## 1. Algorithmic & Acoustic Limitations

| Feature / Acoustic Factor | Current Capability | Known Limitation | Mitigation / Planned Upgrade |
| :--- | :--- | :--- | :--- |
| **Sample Rate** | 16,000 Hz (16 kHz) | Maximum audio bandwidth is 8 kHz (Nyquist cutoff). High-frequency overtones above 8 kHz are not preserved. | 16 kHz is standard for military tactical voice (STANAG 4204/4285). Higher rates exceed ESP32-S3 SRAM budget. |
| **STFT Resolution** | 256-pt (125 Hz bin resolution) | Low-frequency pitch bins below 250 Hz have coarse frequency resolution. | Causal 1D TCN with dilated receptive fields provides temporal modeling of pitch fundamentals. |
| **Competing Babble** | Multi-talker diffuse crowd | Secondary human voices talking at equal volume directly into the mic may not be fully suppressed. | Two-microphone spatial beamforming attenuates off-axis competing speakers by $>6\text{ dB}$. |
| **Wind Turbulence** | Simulated gusts | Extreme direct gale winds (>40 km/h) into bare MEMS port cause turbulent acoustic overload. | Physical open-cell foam windscreen and hydrophobic acoustic membrane must be mounted over mic port. |

---

## 2. Hardware & Mechanical Limitations

| Subsystem | Current Laboratory Status | Known Limitation | Required Industrial Upgrade |
| :--- | :--- | :--- | :--- |
| **Enclosure** | 3D-Printed PETG / ABS | Not certified for MIL-STD-810H immersion, dust, or high-velocity impact. | Injection-molded polycarbonate / ABS blend with continuous silicone gasketing. |
| **Passive Noise Attenuation** | Generic ear-cups (~12 dB NRR) | Insufficient passive attenuation in extreme (>110 dB SPL) tank or jet engine noise. | Circumaural gel ear-cushions with high-density polyurethane acoustic foam ($\ge 26\text{ dB}$ NRR). |
| **Microcontroller Platform** | ESP32-S3-WROOM-1 DevKit | Breadboard/perfboard headers subject to mechanical vibration fatigue. | Fabricate unified 4-layer PCB with SMD components and conformal coating. |
| **Power Interconnects** | JST-PH2.0 & commercial USB-C | Commercial connectors are not dust-proof or waterproof when unmated. | Military bayonet connectors or IP68 magnetic pogo-pin charging interface. |
