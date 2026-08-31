"""
SIH26052 — Real-Time Streaming Audio Demo
Simulates streaming microphone input through rolling-window ML inference,
triggers deterministic hearing protection, and preserves speech intelligibility.
"""

import os
import time
import argparse
import numpy as np
import soundfile as sf
import torch

from src.training.models import get_model
from src.inference.streaming_detector import StreamingImpulseDetector

def run_realtime_demo():
    parser = argparse.ArgumentParser(description="Real-time streaming tactical audio demonstration.")
    parser.add_argument("--input_wav", type=str, default=None, help="Input WAV file for simulation")
    parser.add_argument("--weights", type=str, default="models/tactical_edge_model_best.pt", help="Path to edge model weights")
    parser.add_argument("--hop_ms", type=int, default=25, help="Streaming step in milliseconds (default 25ms)")
    parser.add_argument("--threshold", type=float, default=0.65, help="Impulse detection threshold")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print("SIH26052: REAL-TIME STREAMING AUDIO DEMO")
    print("="*60)
    print(f"[*] Hardware Execution Device: {device}")
    
    # 1. Prepare Audio Stream
    sr = 16000
    if args.input_wav and os.path.exists(args.input_wav):
        audio, _ = sf.read(args.input_wav, dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        print(f"[*] Loaded input file: '{args.input_wav}' ({len(audio)/sr:.2f}s)")
    else:
        # Create synthetic 3-second tactical test stream (Speech -> Gunshot Impulse -> Speech)
        print("[*] Generating simulated tactical acoustic sequence (Speech -> Gunshot -> Speech)...")
        t = np.linspace(0, 3.0, 3 * sr, endpoint=False)
        
        # Continuous speech
        speech = 0.4 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
        # Background hum
        bg = 0.1 * np.random.randn(len(t))
        # Sudden gunshot impulse at t = 1.2s
        impulse = np.zeros_like(t)
        imp_start = int(1.2 * sr)
        imp_len = int(0.15 * sr)
        decay = np.exp(-np.linspace(0, 8, imp_len))
        impulse[imp_start:imp_start + imp_len] = 0.95 * np.sin(2 * np.pi * 1200 * np.linspace(0, 0.15, imp_len)) * decay
        
        audio = speech + bg + impulse
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        
    # 2. Load Model & Streaming Detector
    model = get_model("edge", num_classes=4)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"[OK] Loaded Edge Model from '{args.weights}'")
    else:
        print(f"[!] Running with default weights (Model not trained yet).")
        
    chunk_samples = int((args.hop_ms / 1000.0) * sr) # e.g. 400 samples = 25ms
    detector = StreamingImpulseDetector(
        model=model,
        device=device,
        sr=sr,
        hop_size_samples=chunk_samples,
        feature_mode="edge",
        detection_threshold=args.threshold
    )
    
    # 3. Process Stream with Console Animation
    print("\n" + "-"*75)
    print(f"{'Time (s)':<10} | {'State':<22} | {'P(Impulse)':<12} | {'Latency':<10} | {'Live Level'}")
    print("-"*75)
    
    num_chunks = len(audio) // chunk_samples
    out_chunks = []
    
    for i in range(num_chunks):
        chunk = audio[i * chunk_samples : (i + 1) * chunk_samples]
        curr_time = (i * chunk_samples) / sr
        
        out_chunk, state, prob, lat = detector.process_chunk(chunk)
        out_chunks.append(out_chunk)
        
        # Audio level bar
        raw_level = np.max(np.abs(chunk))
        bar_len = int(raw_level * 20)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        
        # Highlight state transitions
        state_str = state.value
        if state_str == "PROTECTION_TRIGGERED":
            state_str = f">> {state_str} <<"
            
        print(f"{curr_time:<10.2f} | {state_str:<22} | {prob*100:<10.1f}% | {lat:<8.2f}ms | [{bar}]")
        
    processed_audio = np.concatenate(out_chunks)
    
    # Save output
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/realtime_demo_output.wav"
    sf.write(out_path, processed_audio, sr, subtype='PCM_16')
    print("-"*75)
    print(f"\n[OK] Real-time simulation complete!")
    print(f"[OK] Saved protected & speech-preserved audio to: '{out_path}'")
    print(f"[OK] Mean inference latency: {np.mean(detector.inference_latencies_ms):.2f} ms")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_realtime_demo()
