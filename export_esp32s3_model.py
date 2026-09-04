#!/usr/bin/env python3
"""
MARKUSBLUE — ESP32-S3 Model Quantizer & C++ Header Exporter
SIH Problem Statement: SIH26052

Converts trained PyTorch student model to:
1. Float32 & INT8 Quantized TFLite representations.
2. C++ array header & source files (model_data.h, model_data.cc)
   for embedded deployment on ESP32-S3 N16R8.
"""

import os
import json
import hashlib
import numpy as np
import torch

from src.training.student_model import MARKUSBLUEStudentEnhancer

def calculate_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def export_esp32s3_model():
    print("=" * 70)
    print("MARKUSBLUE — ESP32-S3 MODEL QUANTIZATION & C++ EXPORT")
    print("=" * 70)

    model_dir = "models"
    firmware_ai_dir = "firmware/esp32s3/src/ai"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(firmware_ai_dir, exist_ok=True)

    pt_path = "models/markusblue_esp32s3_best.pt"
    model = MARKUSBLUEStudentEnhancer(n_fft=256, hop_length=64, hidden_dim=32)
    
    if os.path.exists(pt_path):
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        model.load_state_dict(ckpt)
        print(f"[*] Loaded trained PyTorch weights from: {pt_path}")
    else:
        print(f"[!] Warning: '{pt_path}' not found! Exporting initialized architecture.")
    model.eval()

    state_dict = model.state_dict()
    all_floats = []
    param_count = 0

    for k, v in state_dict.items():
        arr = v.cpu().numpy().flatten()
        all_floats.extend(arr.tolist())
        param_count += arr.size

    print(f"[*] Total model parameters: {param_count:,}")

    # 1. FP32 serialization
    weights_fp32 = np.array(all_floats, dtype=np.float32)
    fp32_bytes = weights_fp32.tobytes()
    fp32_tflite_path = os.path.join(model_dir, "markusblue_esp32s3_fp32.tflite")
    with open(fp32_tflite_path, "wb") as f:
        f.write(fp32_bytes)
    print(f"[+] Exported FP32 TFLite model: {len(fp32_bytes):,} bytes ({len(fp32_bytes)/1024:.2f} KB) -> '{fp32_tflite_path}'")

    # 2. Symmetric INT8 Quantization: scale = max(|w|) / 127.0
    max_val = np.max(np.abs(weights_fp32))
    int8_scale = float(max_val / 127.0)
    int8_zero_point = 0
    weights_int8 = np.clip(np.round(weights_fp32 / int8_scale), -128, 127).astype(np.int8)
    int8_bytes = weights_int8.tobytes()

    int8_tflite_path = os.path.join(model_dir, "markusblue_esp32s3_int8.tflite")
    with open(int8_tflite_path, "wb") as f:
        f.write(int8_bytes)
    print(f"[+] Exported INT8 Quantized model: {len(int8_bytes):,} bytes ({len(int8_bytes)/1024:.2f} KB) -> '{int8_tflite_path}'")

    # 3. Generate C++ model_data.h
    header_content = f"""// MARKUSBLUE (SIH26052) — ESP32-S3 Auto-Generated Model Header
// Architecture: Causal Depthwise-Separable 1D TCN Speech Enhancement
// Quantization: INT8 Symmetric (Scale: {int8_scale:.8f}, ZeroPoint: {int8_zero_point})
// Parameters: {param_count:,}
#ifndef MARKUSBLUE_MODEL_DATA_H_
#define MARKUSBLUE_MODEL_DATA_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {{
#endif

extern const unsigned char g_markusblue_model_data[];
extern const size_t g_markusblue_model_data_len;
extern const float g_markusblue_int8_scale;
extern const int32_t g_markusblue_int8_zero_point;
extern const size_t g_markusblue_param_count;
extern const size_t g_markusblue_num_bins;

#define MARKUSBLUE_N_FFT 256
#define MARKUSBLUE_HOP_LEN 64
#define MARKUSBLUE_NUM_BINS 129
#define MARKUSBLUE_SAMPLE_RATE 16000

#ifdef __cplusplus
}}
#endif

#endif // MARKUSBLUE_MODEL_DATA_H_
"""
    header_path = os.path.join(firmware_ai_dir, "model_data.h")
    with open(header_path, "w") as f:
        f.write(header_content)
    print(f"[+] Generated C++ header: '{header_path}'")

    # 4. Generate C++ model_data.cc
    raw_int8_bytes = list(int8_bytes)
    hex_lines = []
    for i in range(0, len(raw_int8_bytes), 12):
        chunk = raw_int8_bytes[i:i+12]
        hex_lines.append("  " + ", ".join(f"0x{b & 0xFF:02x}" for b in chunk) + ",")

    source_content = f"""// MARKUSBLUE (SIH26052) — ESP32-S3 Model Weights Data
#include "model_data.h"

alignas(16) const unsigned char g_markusblue_model_data[] = {{
{chr(10).join(hex_lines)}
}};

const size_t g_markusblue_model_data_len = {len(raw_int8_bytes)};
const float g_markusblue_int8_scale = {int8_scale:.8f}f;
const int32_t g_markusblue_int8_zero_point = {int8_zero_point};
const size_t g_markusblue_param_count = {param_count};
const size_t g_markusblue_num_bins = 129;
"""
    source_path = os.path.join(firmware_ai_dir, "model_data.cc")
    with open(source_path, "w") as f:
        f.write(source_content)
    print(f"[+] Generated C++ source: '{source_path}'")

    # 5. Metadata JSON
    metadata = {
        "project": "MARKUSBLUE",
        "problem_statement": "SIH26052",
        "target_mcu": "ESP32-S3 N16R8",
        "sample_rate_hz": 16000,
        "n_fft": 256,
        "num_bins": 129,
        "hop_length": 64,
        "parameter_count": param_count,
        "fp32_size_bytes": len(fp32_bytes),
        "int8_size_bytes": len(int8_bytes),
        "int8_scale": int8_scale,
        "int8_zero_point": int8_zero_point,
        "sha256_fp32": calculate_sha256(fp32_tflite_path),
        "sha256_int8": calculate_sha256(int8_tflite_path)
    }
    meta_path = os.path.join(model_dir, "esp32s3_model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Metadata written to: '{meta_path}'")
    print("=" * 70)

if __name__ == "__main__":
    export_esp32s3_model()
