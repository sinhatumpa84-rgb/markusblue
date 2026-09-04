# MARKUSBLUE (SIH26052) — External Noise Dataset Expansion & Multi-Source Sourcing

## 1. Overview
To ensure robust, mission-critical speech enhancement and active noise cancellation on the **ESP32-S3 N16R8**, the MARKUSBLUE acoustic training pipeline has been expanded to incorporate 50 dedicated noise categories spanning aviation, rotorcraft, heavy diesel engines, armored-vehicle proxies, industrial machinery, environmental elements, crowd chatter, and tactical radio static.

---

## 2. Noise Taxonomy & Categories (50 Total)

| Domain | Noise Categories | Acoustic Properties |
| :--- | :--- | :--- |
| **Aviation & Propulsion** | `aircraft`, `jet_engine`, `turbofan`, `turboprop`, `propeller` | High bypass shear noise, blade passing frequencies, high-frequency jet scream |
| **Rotorcraft** | `helicopter`, `rotorcraft` | Periodic 16.7 Hz blade slap, hover turbine whine, rapid approach flybys |
| **Heavy Vehicles** | `heavy_engine`, `diesel_engine`, `armored_vehicle_proxy`, `military_vehicle_proxy` | 6-cylinder low-frequency diesel firing, tracked vehicle squeal, turbocharger boost |
| **Ground Transport** | `truck`, `bus`, `car`, `motorcycle`, `train`, `railway`, `ship`, `marine_engine` | Tire roll, diesel exhaust, engine braking, passing vehicles |
| **Industrial / Mechanical** | `generator`, `machinery`, `factory`, `industrial`, `compressor`, `hydraulic`, `vibration` | Continuous multi-harmonic hum, valve cycling, structural low-frequency vibrations |
| **High-Frequency Tools** | `drilling`, `grinding`, `chainsaw` | Harsh metal contact harmonics (800 Hz - 4 kHz), intermittent abrasive cutting |
| **Atmospheric & Environment**| `wind`, `rain`, `thunder`, `storm`, `miscellaneous_background` | Non-stationary turbulent gusts, broadband rain downpours, low-frequency thunder rumbles |
| **Human & Tactical Activity**| `crowd`, `footsteps`, `doors` | Multi-talker babble interference, running combat boot impacts, metal door slams |
| **Communications & Static** | `radio_static`, `communication_noise`, `alarms`, `sirens` | Walkie-talkie squelch bursts, pilot carrier tones, digital dropouts, emergency sirens |
| **Electrical / Low-Freq** | `electrical_hum`, `fan`, `air_conditioner`, `ventilation` | 50Hz mains hum with 100/150/250Hz odd harmonics, ventilation brownian rumble |
| **Sudden Transients** | `metal_impacts`, `machinery_impacts`, `explosions_impulse`, `gunshot_impulse` | Fast-attack explosive transients, ringdown decays, impact shockwaves |

---

## 3. Provenance, Licensing & Acquisition
- **Zenodo Open Acoustic Community**: CC-BY 4.0 open research recordings.
- **NOAA National Centers for Environmental Information**: US Government Public Domain.
- **NASA Propulsion Acoustics Archive**: US Government Public Domain.
- **Google AudioSet Research Ontology**: CC-BY 4.0 research clips.
- **Freesound Open Archive**: CC0 / CC-BY 3.0/4.0 verified sounds.
- **MARKUSBLUE Tactical Synthesizer**: Calibrated synthetic generators for electrical 50Hz hum, motor harmonics, and walkie-talkie squelch bursts (`synthetic = true`).

---

## 4. Derived Audio Standardization
- Sample rate: **16,000 Hz**.
- Format: **16-bit Signed Linear PCM**.
- Channels: **Mono (1 Channel)**.
- Target Peak Level: Normalized to $-3 \text{ dBFS}$ with dynamic headroom protection.
- Storage Location: `datasets/external_noise/<category>/` and `datasets/derived/<split>/`.
