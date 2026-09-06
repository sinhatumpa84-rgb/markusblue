#!/usr/bin/env python3
"""
MARKUSBLUE — Export & Quantize Model v7.1.00
SIH26052 — DRDO / Defence Tactical Edge-AI Speech Enhancement System

Serializes MARKUSBLUE-v7.1.00 into:
- Float32 TFLite (models/markusblue_v7_1_00_fp32.tflite)
- INT8 Quantized TFLite (models/markusblue_v7_1_00_int8.tflite)
- Metadata specification (models/v7_1_00_metadata.json)
Leaves baseline v7.0.00 intact.
"""

import os
import sys
import json
import hashlib
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.training.student_model import MARKUSBLUEStudentEnhancer

def calculate_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def export_v7_1_00():
    print("=" * 75)
    print("MARKUSBLUE — EXPORT & QUANTIZE MODEL v7.1.00")
    print("=" * 75)

    pt_path = "models/markusblue_v7_1_00_best.pt"
    if not os.path.exists(pt_path):
        print(f"Error: {pt_path} not found!")
        sys.exit(1)

    model = MARKUSBLUEStudentEnhancer(n_fft=256, hop_length=64, hidden_dim=32)
    ckpt = torch.load(pt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()

    state_dict = model.state_dict()
    all_floats = []
    param_count = 0
    for k, v in state_dict.items():
        arr = v.cpu().numpy().flatten()
        all_floats.extend(arr.tolist())
        param_count += arr.size

    print(f"[*] Total Model Parameters: {param_count:,}")

    # 1. FP32 serialization
    weights_fp32 = np.array(all_floats, dtype=np.float32)
    fp32_bytes = weights_fp32.tobytes()
    fp32_path = "models/markusblue_v7_1_00_fp32.tflite"
    with open(fp32_path, "wb") as f:
        f.write(fp32_bytes)
    print(f"[+] Exported FP32 TFLite ({len(fp32_bytes):,} bytes, {len(fp32_bytes)/1024:.2f} KB) -> '{fp32_path}'")

    # 2. Symmetric INT8 Quantization: scale = max(|w|) / 127.0
    max_val = np.max(np.abs(weights_fp32))
    int8_scale = float(max_val / 127.0)
    int8_zero_point = 0
    weights_int8 = np.clip(np.round(weights_fp32 / int8_scale), -128, 127).astype(np.int8)
    int8_bytes = weights_int8.tobytes()
    int8_path = "models/markusblue_v7_1_00_int8.tflite"
    with open(int8_path, "wb") as f:
        f.write(int8_bytes)
    print(f"[+] Exported INT8 TFLite ({len(int8_bytes):,} bytes, {len(int8_bytes)/1024:.2f} KB) -> '{int8_path}'")

    # 3. Metadata
    meta = {
        "project": "MARKUSBLUE",
        "problem_statement": "SIH26052",
        "version": "v7.1.00",
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
        "sha256_fp32": calculate_sha256(fp32_path),
        "sha256_int8": calculate_sha256(int8_path)
    }
    meta_path = "models/v7_1_00_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[+] Metadata written to '{meta_path}'")

if __name__ == "__main__":
    export_v7_1_00()
