# MARKUSBLUE v7.1.0 — Edge Deployment Guide

## 1. System Architecture

```
[ Microphone (INMP441) ] -> [ I2S DMA Buffer (Core 0) ]
                                    │
                                    ├───> [ Core 1: Feature Extractor & TFLite Student Model ]
                                    │                │
                                    │                ▼ (Estimated Mask / Gains)
                                    ▼
                     [ Deterministic DSP Limiter ]
                                    │
                                    ▼
                     [ VAD-Aware AGC & Dynamic Leveler ]
                                    │
                                    ▼
                     [ Dynamic Range Compressor (DRC) ]
                                    │
                                    ▼
                     [ Lookahead Peak Safety Limiter ]
                                    │
                                    ▼
                      [ I2S DMA Output (MAX98357A) ] -> [ Earpiece ]
```

---

## 2. Firmware Flashing & Configuration (ESP32-S3)

1. **Hardware Pinout**:
   - `INMP441 I2S Mic`: `SCK=GPIO 4`, `WS=GPIO 5`, `SD=GPIO 6`
   - `MAX98357A I2S DAC`: `SCK=GPIO 15`, `WS=GPIO 16`, `SD=GPIO 7`
2. **Build and Flash**:
   ```bash
   cd embedded/inference_example
   idf.py set-target esp32s3
   idf.py build
   idf.py -p COM3 flash monitor
   ```

---

## 3. Python Edge Inference Usage

```python
from src.inference.markusblue import MARKUSBLUE
import soundfile as sf

# Initialize pipeline with trained student weights
pipeline = MARKUSBLUE(
    model_path="models/markusblue_final.pt",
    target_rms_dbfs=-16.0,
    enable_agc=True,
    enable_compressor=True,
    enable_limiter=True
)

# Ingest noisy audio
audio, sr = sf.read("noisy_input.wav")
enhanced_audio = pipeline.enhance(audio)

# Save restored audio
sf.write("enhanced_output.wav", enhanced_audio, sr)
```
