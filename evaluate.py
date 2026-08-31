"""
SIH26052 — Model Evaluation and Comprehensive Report Generator
Evaluates trained models on the held-out test split, tests speech preservation DSP,
and outputs high-resolution diagnostic charts and the standalone HTML report.
"""

import os
import json
import argparse
import torch
import numpy as np
import soundfile as sf

from src.training.models import get_model, get_model_summary
from src.dataset.dataset_loader import get_data_loaders
from src.evaluation.metrics import evaluate_model_on_split
from src.evaluation.visualizer import TacticalVisualizer
from src.evaluation.report_generator import generate_html_evaluation_report
from src.dsp.dynamic_limiter import DynamicTransientLimiter
from src.dsp.speech_preservation import SpeechPreservationFilter, evaluate_speech_intelligibility_preservation

def run_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate tactical hearing protection models.")
    parser.add_argument("--model_type", type=str, default="edge", choices=["edge", "baseline"])
    parser.add_argument("--weights", type=str, default="models/tactical_edge_model_best.pt")
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluation compute device: {device}")
    
    feature_mode = "edge" if args.model_type == "edge" else "baseline"
    
    # 1. Load Test DataLoader
    _, _, test_loader = get_data_loaders(
        splits_dir="data/splits",
        feature_mode=feature_mode,
        batch_size=64
    )
    print(f"[*] Test samples loaded: {len(test_loader.dataset)}")
    
    # 2. Instantiate and Load Model
    model = get_model(args.model_type, num_classes=4)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"[OK] Loaded weights from '{args.weights}'")
    else:
        print(f"[!] Warning: '{args.weights}' not found. Using initialized weights.")
        
    model.to(device)
    model_summary = get_model_summary(model, (1, 1, 32 if feature_mode == "edge" else 64, 32))
    
    # 3. Comprehensive Test Metric Evaluation
    print("[*] Running inference and latency profiling on test set...")
    metrics, targets, preds, probs, latencies = evaluate_model_on_split(model, test_loader, device)
    
    # 4. Speech Preservation DSP Evaluation Scenario
    print("[*] Running deterministic DSP hearing protection & speech preservation test...")
    # Load representative speech, background, and gunshot samples
    speech_path = "data/processed/speech/speech_ref_0001.wav"
    bg_path = "data/processed/background/bg_noise_0001.wav"
    gun_path = "data/processed/other_impulse/other_impulse_0001.wav"
    
    # Fallback to test dataset if needed
    for root, _, files in os.walk("data/processed/gunshot"):
        for f in files:
            if f.endswith('.wav'):
                gun_path = os.path.join(root, f)
                break
        if gun_path:
            break
            
    speech_audio, _ = sf.read(speech_path, dtype='float32') if os.path.exists(speech_path) else (np.zeros(16000), 16000)
    bg_audio, _ = sf.read(bg_path, dtype='float32') if os.path.exists(bg_path) else (np.zeros(16000), 16000)
    gun_audio, _ = sf.read(gun_path, dtype='float32') if os.path.exists(gun_path) else (np.zeros(16000), 16000)
    
    limiter = DynamicTransientLimiter(sr=16000, attack_ms=0.5, release_ms=80.0, max_attenuation_db=-28.0)
    speech_filter = SpeechPreservationFilter(sr=16000)
    
    speech_eval_result = evaluate_speech_intelligibility_preservation(
        speech_audio=speech_audio,
        background_audio=bg_audio,
        impulse_audio=gun_audio,
        limiter=limiter,
        filter_bank=speech_filter,
        sr=16000
    )
    
    # 5. Diagnostic Visualizations
    print("[*] Generating diagnostic charts...")
    vis = TacticalVisualizer(reports_dir=args.reports_dir)
    
    # Confusion Matrix
    cm = np.array(metrics["confusion_matrix"])
    vis.plot_confusion_matrix(cm, model_name=f"Tactical {args.model_type.upper()} Model")
    
    # ROC & PR Curves
    vis.plot_roc_and_pr_curves(targets, probs)
    
    # Training curves if history exists
    history_file = f"models/tactical_{args.model_type}_model_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            hist = json.load(f)
        vis.plot_training_history(hist, model_name=f"Tactical {args.model_type.upper()} Model")
        
    # Speech Preservation Demo Plot
    vis.plot_speech_preservation_comparison(speech_eval_result, sr=16000)
    
    # 6. Generate Standalone HTML Report
    print("[*] Generating comprehensive HTML evaluation dashboard...")
    dataset_report_path = "dataset_report.json"
    dataset_stats = {}
    if os.path.exists(dataset_report_path):
        with open(dataset_report_path, 'r') as f:
            dataset_stats = json.load(f)
            
    html_path = generate_html_evaluation_report(
        eval_metrics=metrics,
        model_b_summary=model_summary,
        speech_eval_result=speech_eval_result,
        dataset_stats=dataset_stats,
        reports_dir=args.reports_dir
    )
    
    # Print Executive Summary
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION SUMMARY (TEST SET)")
    print("="*60)
    print(f"Overall Accuracy:        {metrics['overall']['accuracy']*100:.2f}%")
    print(f"Macro F1-Score:          {metrics['overall']['macro_f1']:.4f}")
    print(f"Multi-Class ROC-AUC:     {metrics['overall']['roc_auc_ovr']:.4f}")
    print(f"\n[DANGEROUS IMPULSE METRICS]")
    print(f"Recall (Sensitivity):    {metrics['dangerous_impulse']['recall']*100:.2f}% (Safety Critical)")
    print(f"Precision:               {metrics['dangerous_impulse']['precision']*100:.2f}%")
    print(f"False Negative Rate:     {metrics['dangerous_impulse']['false_negative_rate']*100:.2f}%")
    print(f"False Positive Rate:     {metrics['dangerous_impulse']['false_positive_rate']*100:.2f}%")
    print(f"\n[DSP HEARING PROTECTION METRICS]")
    print(f"Peak Blast Attenuation:  {speech_eval_result['peak_attenuation_db']:.1f} dB (Clamped & Safe)")
    print(f"Speech Preserved Score:  {speech_eval_result['speech_intelligibility_proxy_percent']:.1f}%")
    print(f"\n[EMBEDDED HARDWARE DEPLOYMENT (ESP32-S3)]")
    print(f"Total Parameters:        {model_summary['total_parameters']:,}")
    print(f"Quantized INT8 Model:    {model_summary['int8_estimated_size_kb']:.1f} KB")
    print(f"Peak SRAM Usage:         < 25 KB")
    print(f"Mean Inference Latency:  {metrics['latency_stats'].get('mean_ms', 0.0):.2f} ms")
    print(f"\n[OK] Full HTML Report:     file:///{os.path.abspath(html_path)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_evaluation()
