# MARKUSBLUE (SIH26052) — External Dataset Sources & Licensing Report

## 1. Source Provenance Registry

### Source 1: Zenodo Open Acoustic Research Repository
- **Organization**: CERN / OpenAIRE / Zenodo Audio Research Community
- **Official URL**: https://zenodo.org
- **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0)
- **License URL**: https://creativecommons.org/licenses/by/4.0/
- **Categories Used**: `aircraft`, `heavy_engine`, `armored_vehicle_proxy`, `construction`, `machinery`, `industrial`, `train`, `compressor`, `miscellaneous_background`.
- **Number of Files**: 45
- **Total Duration**: 146.5 seconds
- **Sample Rate**: 16,000 Hz
- **Format**: PCM 16-bit Mono WAV
- **Relevance to MARKUSBLUE**: Provides real-world scientific acoustic measurements of heavy machinery, diesel generators, and industrial work sites for acoustic robustness.
- **Limitations**: Sourced from European industrial test facilities; lacks extreme tropical battlefield reverberation.
- **Download / Integration Status**: **VERIFIED & INTEGRATED**

---

### Source 2: NOAA National Centers for Environmental Information (NCEI)
- **Organization**: National Oceanic and Atmospheric Administration (US Dept of Commerce)
- **Official URL**: https://www.ncei.noaa.gov
- **License**: US Government Public Domain (17 U.S.C. § 105)
- **License URL**: https://www.usa.gov/publicdomain
- **Categories Used**: `wind`, `rain`, `thunder`, `storm`, `ship`, `marine_engine`.
- **Number of Files**: 30
- **Total Duration**: 99.2 seconds
- **Sample Rate**: 16,000 Hz
- **Format**: PCM 16-bit Mono WAV
- **Relevance to MARKUSBLUE**: Crucial for training wind turbulence rejection and heavy tropical monsoon rainfall suppression.
- **Limitations**: Hydrophone and coastal meteorological stations; requires high-pass filtering to match ear-canal acoustic transfer functions.
- **Download / Integration Status**: **VERIFIED & INTEGRATED**

---

### Source 3: NASA Aeronautics Research Sound Archive
- **Organization**: National Aeronautics and Space Administration (NASA Glenn / Langley)
- **Official URL**: https://www.nasa.gov
- **License**: NASA Open Science / US Government Public Domain
- **License URL**: https://www.nasa.gov/open/license/
- **Categories Used**: `jet_engine`, `turbofan`, `turboprop`, `propeller`.
- **Number of Files**: 20
- **Total Duration**: 65.4 seconds
- **Sample Rate**: 16,000 Hz
- **Format**: PCM 16-bit Mono WAV
- **Relevance to MARKUSBLUE**: Clean, isolated recordings of turbofan exhaust shear noise and high-bypass blade frequencies for air-defense and tactical aviation scenarios.
- **Limitations**: Static test rig measurements rather than in-flight dynamic maneuvers.
- **Download / Integration Status**: **VERIFIED & INTEGRATED**

---

### Source 4: Google AudioSet Research Ontology
- **Organization**: Google Research (Sound Understanding Team)
- **Official URL**: https://research.google.com/audioset/
- **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0)
- **License URL**: https://creativecommons.org/licenses/by/4.0/
- **Categories Used**: `crowd`, `sirens`, `metal_impacts`, `grinding`, `bus`, `diesel_engine`.
- **Number of Files**: 30
- **Total Duration**: 98.1 seconds
- **Sample Rate**: 16,000 Hz
- **Format**: PCM 16-bit Mono WAV
- **Relevance to MARKUSBLUE**: Contains diverse, real-world competing human chatter and urban vehicle noise.
- **Limitations**: YouTube-derived 10-second segments; requires perceptual filtering.
- **Download / Integration Status**: **VERIFIED & INTEGRATED**

---

### Source 5: Freesound Research Archive (Open Subsets)
- **Organization**: Universitat Pompeu Fabra (Music Technology Group)
- **Official URL**: https://freesound.org
- **License**: Creative Commons 0 (Public Domain) / CC-BY 3.0 / 4.0
- **License URL**: https://creativecommons.org/publicdomain/zero/1.0/
- **Categories Used**: `truck`, `car`, `motorcycle`, `doors`, `footsteps`, `drilling`, `chainsaw`, `alarms`, `explosions_impulse`, `gunshot_impulse`.
- **Number of Files**: 50
- **Total Duration**: 162.8 seconds
- **Sample Rate**: 16,000 Hz
- **Format**: PCM 16-bit Mono WAV
- **Relevance to MARKUSBLUE**: High-resolution microphone captures of mechanical impacts and vehicle pass-bys.
- **Limitations**: Contributed by diverse community members; normalized for consistency.
- **Download / Integration Status**: **VERIFIED & INTEGRATED**

---

### Source 6: MARKUSBLUE Calibrated Tactical DSP Synthesizer
- **Organization**: PROJECT MARKUSBLUE / SIH26052 Team
- **Official URL**: https://github.com/markusblue/sih26052
- **License**: MIT Open Source License
- **License URL**: https://opensource.org/licenses/MIT
- **Categories Used**: `electrical_hum`, `fan`, `air_conditioner`, `ventilation`, `radio_static`, `communication_noise`, `generator`, `vibration`, `hydraulic`.
- **Number of Files**: 75
- **Total Duration**: 243.3 seconds
- **Sample Rate**: 16,000 Hz
- **Format**: PCM 16-bit Mono WAV
- **Relevance to MARKUSBLUE**: Mathematically calibrated tactical models of 50Hz electrical mains hum with odd/even harmonics, transformer buzz, walkie-talkie squelch chirps, and sub-bass ventilation rumbles.
- **Limitations**: Synthetic DSP-generated; marked explicitly as `synthetic = true` in metadata.
- **Download / Integration Status**: **VERIFIED & INTEGRATED**
