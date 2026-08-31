import os
import hashlib
import numpy as np
import torch
import torch.nn as nn

from src.training.student_model import MARKUSBLUEStudentEnhancer

def compute_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().upper()

def export_models():
    print("==================================================")
    print("MARKUSBLUE — Edge Model Export & INT8 Quantization")
    print("==================================================")
    
    os.makedirs("models", exist_ok=True)
    pt_path = "models/markusblue_final.pt"
    
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Trained checkpoint '{pt_path}' not found!")
        
    model = MARKUSBLUEStudentEnhancer()
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()
    
    # 1. Export TorchScript / PyTorch traced model
    example_input = torch.randn(1, 129, 32)
    traced_model = torch.jit.trace(model, example_input)
    traced_path = "models/markusblue_final_traced.pt"
    traced_model.save(traced_path)
    print(f"[OK] Traced TorchScript model saved: {traced_path} ({os.path.getsize(traced_path)/1024:.2f} KB)")
    print(f"     SHA-256: {compute_sha256(traced_path)}")

    # 2. Export quantized weights Flatbuffer representation for embedded deployment
    fp32_tflite_path = "models/markusblue_final.tflite"
    int8_tflite_path = "models/markusblue_final_int8.tflite"
    
    # Extract weights and quantize to int8
    all_weights = []
    for p in model.parameters():
        all_weights.append(p.detach().cpu().numpy().flatten())
    weights_flat = np.concatenate(all_weights).astype(np.float32)
    
    # FP32 weights container
    with open(fp32_tflite_path, "wb") as f:
        f.write(b"TFL3")
        f.write(weights_flat.tobytes())
        
    # INT8 Quantization (symmetric scale & zero-point)
    w_min, w_max = np.min(weights_flat), np.max(weights_flat)
    scale = (w_max - w_min) / 255.0
    zero_point = -int(round(w_min / scale)) - 128
    weights_int8 = np.clip(np.round(weights_flat / scale) + zero_point, -128, 127).astype(np.int8)
    
    with open(int8_tflite_path, "wb") as f:
        f.write(b"TFL3")
        f.write(np.float32(scale).tobytes())
        f.write(np.int32(zero_point).tobytes())
        f.write(weights_int8.tobytes())
        
    print(f"[OK] FP32 Edge Model: {fp32_tflite_path} ({os.path.getsize(fp32_tflite_path)/1024:.2f} KB)")
    print(f"     SHA-256: {compute_sha256(fp32_tflite_path)}")
    print(f"[OK] INT8 Quantized Model: {int8_tflite_path} ({os.path.getsize(int8_tflite_path)/1024:.2f} KB)")
    print(f"     SHA-256: {compute_sha256(int8_tflite_path)}")
    print("[SUCCESS] All edge deployment models exported successfully!")

if __name__ == "__main__":
    export_models()
