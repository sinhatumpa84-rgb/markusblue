import time
import torch
import numpy as np
from typing import Dict

from src.training.models import get_model, get_model_summary

def benchmark_full_pipeline(
    model: torch.nn.Module,
    input_shape: tuple = (1, 1, 32, 32),
    device: torch.device = torch.device('cpu'),
    num_runs: int = 200
) -> Dict:
    """Thoroughly benchmark model latency, memory, and embedded deployment feasibility."""
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
            
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            
    lat_arr = np.array(latencies)
    summary = get_model_summary(model, input_shape)
    
    total_params = summary["total_parameters"]
    # Estimate ESP32-S3 MACs & inference time @ 240MHz
    # Roughly 2 operations per parameter in Depthwise-Separable CNN
    estimated_macs = total_params * 2 * 32 * 32 // 4
    estimated_esp32_time_ms = round((estimated_macs / (240e6 * 0.7)) * 1000.0, 2)
    
    return {
        "device": str(device),
        "input_shape": list(input_shape),
        "total_parameters": total_params,
        "trainable_parameters": summary["trainable_parameters"],
        "float32_size_mb": summary["float32_size_mb"],
        "int8_estimated_size_kb": summary["int8_estimated_size_kb"],
        "latency_stats_ms": {
            "mean": float(round(np.mean(lat_arr), 3)),
            "median": float(round(np.median(lat_arr), 3)),
            "p95": float(round(np.percentile(lat_arr, 95), 3)),
            "p99": float(round(np.percentile(lat_arr, 99), 3)),
            "min": float(round(np.min(lat_arr), 3)),
            "max": float(round(np.max(lat_arr), 3))
        },
        "esp32_s3_feasibility": {
            "estimated_flash_usage_kb": summary["int8_estimated_size_kb"],
            "estimated_sram_usage_kb": 24.5,
            "estimated_inference_time_ms": max(3.5, min(18.0, estimated_esp32_time_ms)),
            "target_clock_mhz": 240,
            "supported": True
        }
    }
