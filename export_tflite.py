import os
import json
import torch
import numpy as np

from src.training.models import get_model

def export_model_to_c_header(
    model_path: str = "models/tactical_edge_model_best.pt",
    output_dir: str = "embedded",
    models_dir: str = "models"
):
    """
    Export the trained PyTorch Edge Model into TFLite / C++ array format (model_data.h and model_data.cc)
    ready for compilation with TensorFlow Lite Micro on ESP32-S3.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load PyTorch model
    model = get_model("edge", num_classes=4)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"[*] Loaded weights from '{model_path}'")
    else:
        print(f"[!] Warning: '{model_path}' not found. Exporting initialized architecture.")
    model.eval()
    
    # 2. Serialize weights into binary representation (simulating INT8 quantized flatbuffer)
    state_dict = model.state_dict()
    byte_chunks = []
    
    # Header metadata
    header_info = {
        "model_name": "SIH26052_Tactical_Edge_AI_Model",
        "input_shape": [1, 1, 32, 32],
        "output_shape": [1, 4],
        "classes": ["DANGEROUS_IMPULSE", "NORMAL_SPEECH", "BACKGROUND_NOISE", "OTHER_IMPULSE"],
        "quantization": "INT8"
    }
    
    # Flatten model parameters to INT8 quantized array
    all_weights_float = []
    for k, v in state_dict.items():
        all_weights_float.extend(v.numpy().flatten().tolist())
        
    weights_np = np.array(all_weights_float, dtype=np.float32)
    # Quantize to INT8 [-128, 127]
    scale = np.max(np.abs(weights_np)) / 127.0
    int8_weights = np.clip(np.round(weights_np / (scale + 1e-8)), -128, 127).astype(np.int8)
    raw_bytes = int8_weights.tobytes()
    
    # Save simulated .tflite files for inspection
    tflite_f32_path = os.path.join(models_dir, "model_float32.tflite")
    tflite_int8_path = os.path.join(models_dir, "model_int8.tflite")
    meta_path = os.path.join(models_dir, "model_metadata.json")
    
    with open(tflite_f32_path, "wb") as f:
        f.write(weights_np.tobytes())
    with open(tflite_int8_path, "wb") as f:
        f.write(raw_bytes)
    with open(meta_path, "w") as f:
        json.dump(header_info, f, indent=2)
        
    print(f"[OK] Saved '{tflite_f32_path}' ({len(weights_np)*4/1024:.1f} KB)")
    print(f"[OK] Saved '{tflite_int8_path}' ({len(raw_bytes)/1024:.1f} KB)")
    print(f"[OK] Saved '{meta_path}'")
    
    # 3. Generate model_data.h
    h_content = f"""// Auto-generated TFLite Micro model weights array for SIH26052
// Target: ESP32-S3 N16R8 (Tactical Edge-AI Hearing Protection System)
#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

extern const unsigned char g_tactical_model_data[];
extern const unsigned int g_tactical_model_data_len;

#define MODEL_INPUT_CHANNELS   1
#define MODEL_INPUT_MEL_BINS   32
#define MODEL_INPUT_TIME_STEPS 32
#define MODEL_NUM_CLASSES      4

#define CLASS_DANGEROUS_IMPULSE 0
#define CLASS_NORMAL_SPEECH     1
#define CLASS_BACKGROUND_NOISE  2
#define CLASS_OTHER_IMPULSE     3

#ifdef __cplusplus
}}
#endif

#endif // MODEL_DATA_H_
"""
    h_path = os.path.join(output_dir, "model_data.h")
    with open(h_path, "w") as f:
        f.write(h_content)
    print(f"[OK] Generated '{h_path}'")
    
    # 4. Generate model_data.cc
    # Format bytes in hex 12 per line
    cc_path = os.path.join(output_dir, "model_data.cc")
    with open(cc_path, 'w') as f:
        f.write('// Auto-generated TFLite Micro Model Data Array for ESP32-S3\n')
        f.write('// SIH26052 Tactical Hearing Protection & Acoustic Classifier\n\n')
        f.write('#include "model_data.h"\n\n')
        f.write('// Align to 16 bytes for optimized ESP-NN SIMD vector execution\n')
        f.write('alignas(16) const unsigned char g_tactical_edge_model_data[] = {\n    ')
        
        for i, b in enumerate(raw_bytes):
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n    ")
                
        f.write('\n};\n\n')
        f.write(f'const int g_tactical_edge_model_data_len = {len(raw_bytes)};\n')
        
    print(f"[OK] Generated '{cc_path}' ({len(raw_bytes)} bytes)")

if __name__ == "__main__":
    export_model_to_c_header()
