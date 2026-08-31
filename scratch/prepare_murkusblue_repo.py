"""
SIH26052 — MURKUSBLUE Repository Dataset & Model Verification Engine
Scans and verifies existing MARKAS LOOP model and audio datasets with SHA-256 integrity.
"""

import os
import sys
import hashlib
import json
import glob
import soundfile as sf
import numpy as np

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def scan_model(model_path="models/model_int8.tflite"):
    print("=== MODEL VERIFICATION ===")
    if not os.path.exists(model_path):
        print(f"[!] Model not found at {model_path}")
        return None
    
    size_bytes = os.path.getsize(model_path)
    file_sha256 = sha256_file(model_path)
    
    # Inspect with tensorflow / tflite runtime if available (read-only)
    model_info = {
        "model_name": "MARKAS LOOP",
        "original_path": model_path,
        "target_filename": "markas_loop_model_int8.tflite",
        "size_bytes": size_bytes,
        "size_kb": round(size_bytes / 1024.0, 2),
        "sha256": file_sha256,
        "format": "TensorFlow Lite (FlatBuffers INT8 Quantized)"
    }
    
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        model_info["input_shape"] = input_details[0]["shape"].tolist()
        model_info["input_dtype"] = str(input_details[0]["dtype"])
        model_info["input_quantization"] = {
            "scale": float(input_details[0]["quantization_parameters"]["scales"][0]) if len(input_details[0]["quantization_parameters"]["scales"]) > 0 else None,
            "zero_point": int(input_details[0]["quantization_parameters"]["zero_points"][0]) if len(input_details[0]["quantization_parameters"]["zero_points"]) > 0 else None
        }
        model_info["output_shape"] = output_details[0]["shape"].tolist()
        model_info["output_dtype"] = str(output_details[0]["dtype"])
        model_info["output_quantization"] = {
            "scale": float(output_details[0]["quantization_parameters"]["scales"][0]) if len(output_details[0]["quantization_parameters"]["scales"]) > 0 else None,
            "zero_point": int(output_details[0]["quantization_parameters"]["zero_points"][0]) if len(output_details[0]["quantization_parameters"]["zero_points"]) > 0 else None
        }
    except Exception as e:
        model_info["interpreter_error"] = str(e)
        
    print(json.dumps(model_info, indent=2))
    return model_info

def scan_audio_datasets():
    print("\n=== AUDIO DATASETS FORENSIC SCAN ===")
    
    # Potential audio locations
    dataset_dirs = {
        "gunshot": "data/processed/gunshot",
        "speech": "data/processed/speech",
        "background_noise": "data/processed/background",
        "other_impulse": "data/processed/other_impulse"
    }
    
    stats = {}
    unique_hashes = {}
    duplicates = []
    
    total_files = 0
    total_size = 0
    total_duration_sec = 0.0
    largest_file = {"path": None, "size": 0}
    
    for category, dir_path in dataset_dirs.items():
        if not os.path.exists(dir_path):
            print(f"[-] Category '{category}' not found at '{dir_path}' (skipping)")
            continue
            
        wav_files = glob.glob(os.path.join(dir_path, "*.wav"))
        if not wav_files:
            print(f"[-] Category '{category}' has 0 wav files (skipping)")
            continue
            
        cat_size = 0
        cat_duration = 0.0
        cat_unique_files = []
        
        for w in wav_files:
            sz = os.path.getsize(w)
            if sz > largest_file["size"]:
                largest_file = {"path": w, "size": sz}
                
            file_h = sha256_file(w)
            if file_h in unique_hashes:
                duplicates.append((w, unique_hashes[file_h]))
                continue
                
            unique_hashes[file_h] = w
            cat_unique_files.append(w)
            cat_size += sz
            
            try:
                info = sf.info(w)
                cat_duration += info.duration
            except Exception:
                cat_duration += 1.0 # default 1s
                
        stats[category] = {
            "directory": dir_path,
            "raw_file_count": len(wav_files),
            "unique_file_count": len(cat_unique_files),
            "total_size_bytes": cat_size,
            "total_size_mb": round(cat_size / (1024.0 * 1024.0), 2),
            "total_duration_sec": round(cat_duration, 2),
            "total_duration_formatted": f"{int(cat_duration//60)}m {int(cat_duration%60)}s",
            "sample_files": cat_unique_files[:3]
        }
        
        total_files += len(cat_unique_files)
        total_size += cat_size
        total_duration_sec += cat_duration
        
    print("\n--- DATASET BREAKDOWN ---")
    for cat, data in stats.items():
        print(f"[{cat.upper()}] Files: {data['unique_file_count']} | Size: {data['total_size_mb']} MB | Duration: {data['total_duration_formatted']}")
        
    summary = {
        "categories_present": list(stats.keys()),
        "total_unique_audio_files": total_files,
        "total_dataset_size_bytes": total_size,
        "total_dataset_size_mb": round(total_size / (1024.0 * 1024.0), 2),
        "total_dataset_duration_sec": round(total_duration_sec, 2),
        "total_dataset_duration_formatted": f"{int(total_duration_sec//3600)}h {int((total_duration_sec%3600)//60)}m {int(total_duration_sec%60)}s",
        "largest_audio_file": largest_file,
        "duplicate_count_found": len(duplicates),
        "category_details": stats
    }
    
    print("\n--- SUMMARY ---")
    print(f"Total Audio Files: {summary['total_unique_audio_files']}")
    print(f"Total Dataset Size: {summary['total_dataset_size_mb']} MB")
    print(f"Total Duration: {summary['total_dataset_duration_formatted']}")
    print(f"Largest File: {largest_file['path']} ({round(largest_file['size']/1024, 1)} KB)")
    print(f"Exact Duplicates Detected: {len(duplicates)}")
    
    return summary

if __name__ == "__main__":
    model_res = scan_model()
    dataset_res = scan_audio_datasets()
    
    out_manifest = {
        "project": "MURKUSBLUE",
        "model": model_res,
        "dataset": dataset_res
    }
    
    os.makedirs("scratch", exist_ok=True)
    with open("scratch/murkusblue_scan_manifest.json", "w") as f:
        json.dump(out_manifest, f, indent=2)
    print("\n[OK] Scan manifest saved to 'scratch/murkusblue_scan_manifest.json'")
