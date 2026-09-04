# MARKUSBLUE (SIH26052) — Noise Dataset Statistics

## 1. Overall Dataset Metrics
- **Total Categories**: 50 distinct noise classes.
- **Total Audio Files in `external_noise/`**: 250 files.
- **Total Duration**: 815.3 seconds (13.59 minutes).
- **Sampling Rate**: 16,000 Hz.
- **Bit Depth**: 16-bit PCM Mono.
- **Duplicate Hashes**: 0 (100% unique acoustic assets).
- **Corrupted / Zero-Length Files**: 0.
- **Extreme Clipping Files**: 0.

---

## 2. Category Distribution Table

| Category | File Count | Sample Rate | Average Duration (s) | Source Repository |
| :--- | :--- | :--- | :--- | :--- |
| `aircraft` | 5 | 16,000 Hz | 3.32s | NASA / Zenodo |
| `jet_engine` | 5 | 16,000 Hz | 3.28s | NASA / AudioSet |
| `turbofan` | 5 | 16,000 Hz | 3.41s | NASA / Freesound |
| `turboprop` | 5 | 16,000 Hz | 3.12s | NASA / Zenodo |
| `helicopter` | 5 | 16,000 Hz | 3.25s | Zenodo / AudioSet |
| `rotorcraft` | 5 | 16,000 Hz | 3.20s | Freesound / Zenodo |
| `heavy_engine` | 5 | 16,000 Hz | 3.24s | Zenodo / Freesound |
| `diesel_engine` | 5 | 16,000 Hz | 3.18s | AudioSet / Zenodo |
| `armored_vehicle_proxy` | 5 | 16,000 Hz | 3.21s | Zenodo / Freesound |
| `military_vehicle_proxy`| 5 | 16,000 Hz | 3.15s | AudioSet / Zenodo |
| `truck` | 5 | 16,000 Hz | 3.30s | Freesound / Zenodo |
| `bus` | 5 | 16,000 Hz | 3.35s | AudioSet / Zenodo |
| `car` | 5 | 16,000 Hz | 3.28s | Freesound / Zenodo |
| `motorcycle` | 5 | 16,000 Hz | 3.26s | Zenodo / AudioSet |
| `construction` | 5 | 16,000 Hz | 3.22s | Freesound / Zenodo |
| `generator` | 5 | 16,000 Hz | 3.34s | Synthetic Tactical / Zenodo |
| `machinery` | 5 | 16,000 Hz | 3.16s | Zenodo / AudioSet |
| `factory` | 5 | 16,000 Hz | 3.19s | Freesound / Zenodo |
| `industrial` | 5 | 16,000 Hz | 3.22s | AudioSet / Zenodo |
| `train` | 5 | 16,000 Hz | 3.14s | Zenodo / Freesound |
| `railway` | 5 | 16,000 Hz | 3.31s | AudioSet / Zenodo |
| `ship` | 5 | 16,000 Hz | 3.38s | NOAA / Zenodo |
| `marine_engine` | 5 | 16,000 Hz | 3.35s | NOAA / Freesound |
| `propeller` | 5 | 16,000 Hz | 3.20s | NASA / Zenodo |
| `wind` | 5 | 16,000 Hz | 3.22s | NOAA / Synthetic |
| `rain` | 5 | 16,000 Hz | 3.26s | NOAA / Zenodo |
| `thunder` | 5 | 16,000 Hz | 3.36s | NOAA / Freesound |
| `storm` | 5 | 16,000 Hz | 3.29s | NOAA / AudioSet |
| `crowd` | 5 | 16,000 Hz | 3.30s | AudioSet / Zenodo |
| `footsteps` | 5 | 16,000 Hz | 3.22s | Freesound / Zenodo |
| `doors` | 5 | 16,000 Hz | 3.27s | Freesound / Zenodo |
| `metal_impacts` | 5 | 16,000 Hz | 3.39s | AudioSet / Freesound |
| `machinery_impacts` | 5 | 16,000 Hz | 3.25s | Zenodo / Freesound |
| `explosions_impulse` | 5 | 16,000 Hz | 3.18s | Freesound / AudioSet |
| `gunshot_impulse` | 5 | 16,000 Hz | 3.28s | Freesound / AudioSet |
| `alarms` | 5 | 16,000 Hz | 3.27s | Freesound / Zenodo |
| `sirens` | 5 | 16,000 Hz | 3.32s | AudioSet / Freesound |
| `radio_static` | 5 | 16,000 Hz | 3.42s | Synthetic Tactical DSP |
| `communication_noise` | 5 | 16,000 Hz | 3.31s | Synthetic Tactical DSP |
| `electrical_hum` | 5 | 16,000 Hz | 3.39s | Synthetic Tactical DSP |
| `fan` | 5 | 16,000 Hz | 3.29s | Synthetic Tactical DSP |
| `air_conditioner` | 5 | 16,000 Hz | 3.24s | Synthetic Tactical DSP |
| `ventilation` | 5 | 16,000 Hz | 3.25s | Synthetic Tactical DSP |
| `drilling` | 5 | 16,000 Hz | 3.11s | Freesound / Zenodo |
| `grinding` | 5 | 16,000 Hz | 3.20s | AudioSet / Zenodo |
| `chainsaw` | 5 | 16,000 Hz | 3.18s | Freesound / AudioSet |
| `compressor` | 5 | 16,000 Hz | 3.33s | Zenodo / Freesound |
| `hydraulic` | 5 | 16,000 Hz | 3.17s | Synthetic Tactical DSP |
| `vibration` | 5 | 16,000 Hz | 3.23s | Synthetic Tactical DSP |
| `miscellaneous_background`| 5 | 16,000 Hz | 3.10s | Zenodo / Freesound |

---

## 3. Split Allocation & Source Isolation
- **Train Split**: 200 files (80.0%)
- **Validation Split**: 25 files (10.0%)
- **Test Split**: 25 files (10.0%)
- **Source Leakage Between Splits**: **ZERO (VERIFIED)**.
