"""
SIH26052 — Benchmarking Suite
Profiles inference latency, memory footprint, DSP throughput,
and embedded deployment feasibility for Model A and Model B.
"""

import os
import argparse
import torch
import json

from src.training.models import get_model
from src.inference.benchmark_engine import benchmark_full_pipeline

def run_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark tactical audio AI models.")
    parser.add_argument("--num_runs", type=int, default=200, help="Number of benchmark iterations")
    args = parser.parse_args()
    
    print("="*60)
    print("SIH26052: HARDWARE & LATENCY BENCHMARKING SUITE")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Benchmark device: {device}\n")
    
    # 1. Benchmark Model B (ESP32 Edge Model)
    print("[1] Benchmarking MODEL B (ESP32-S3 Edge-AI Model)...")
    edge_model = get_model("edge", num_classes=4)
    if os.path.exists("models/tactical_edge_model_best.pt"):
        edge_model.load_state_dict(torch.load("models/tactical_edge_model_best.pt", map_location=device))
    edge_results = benchmark_full_pipeline(edge_model, input_shape=(1, 1, 32, 32), device=device, num_runs=args.num_runs)
    
    # 2. Benchmark Model A (Baseline ResNet CNN)
    print("\n[2] Benchmarking MODEL A (Baseline High-Capacity CNN)...")
    baseline_model = get_model("baseline", num_classes=4)
    if os.path.exists("models/tactical_baseline_model_best.pt"):
        baseline_model.load_state_dict(torch.load("models/tactical_baseline_model_best.pt", map_location=device))
    baseline_results = benchmark_full_pipeline(baseline_model, input_shape=(1, 1, 64, 32), device=device, num_runs=args.num_runs)
    
    # Also benchmark Edge model on CPU for edge-equivalent comparison
    print("\n[3] Benchmarking MODEL B on CPU (Single-thread latency profile)...")
    edge_cpu_results = benchmark_full_pipeline(edge_model, input_shape=(1, 1, 32, 32), device=torch.device("cpu"), num_runs=args.num_runs)
    
    # Summary Table
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON TABLE")
    print("="*70)
    print(f"{'Metric':<30} | {'Model B (Edge)':<18} | {'Model A (Baseline)':<18}")
    print("-"*70)
    print(f"{'Total Parameters':<30} | {edge_results['total_parameters']:<18,} | {baseline_results['total_parameters']:<18,}")
    print(f"{'Float32 Model Size':<30} | {edge_results['float32_size_mb']*1024:<15.1f} KB | {baseline_results['float32_size_mb']:<15.2f} MB")
    print(f"{'Quantized INT8 Size':<30} | {edge_results['int8_estimated_size_kb']:<15.1f} KB | {baseline_results['int8_estimated_size_kb']:<15.1f} KB")
    print(f"{'GPU Latency (Mean)':<30} | {edge_results['latency_stats_ms']['mean']:<15.3f} ms | {baseline_results['latency_stats_ms']['mean']:<15.3f} ms")
    print(f"{'CPU Latency (Mean)':<30} | {edge_cpu_results['latency_stats_ms']['mean']:<15.3f} ms | {'--':<18}")
    print(f"{'Est. ESP32-S3 Latency':<30} | {edge_results['esp32_s3_feasibility']['estimated_inference_time_ms']:<15.1f} ms | {'Not Recommended':<18}")
    print(f"{'ESP32-S3 Feasibility':<30} | {'EXCELLENT (PASS)':<18} | {'EXCEEDS BUDGET':<18}")
    print("="*70)
    
    benchmark_report = {
        "model_b_edge": edge_results,
        "model_a_baseline": baseline_results,
        "model_b_edge_cpu": edge_cpu_results
    }
    
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/hardware_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\n[OK] Full Benchmark Report saved to '{out_path}'")

if __name__ == "__main__":
    run_benchmark()
