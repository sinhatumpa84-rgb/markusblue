#!/usr/bin/env python3
"""
MARKUSBLUE — Real-Time Battlefield Streaming Simulation Demonstration
SIH26052 — DRDO / Defence Tactical Edge-AI Speech Enhancement System

Simulates the real-time embedded streaming pipeline of the ESP32-S3 N16R8.
Processes streaming audio in blocks of 64 samples (4.0 ms frame duration at 16 kHz)
through the complete edge DSP pipeline:
  [DMA Stream] -> [256-pt Hann STFT] -> [AI Mask Inference] -> [ISTFT] -> [AGC] -> [Peak Limiter] -> [Output]

Demonstrates real-time speech enhancement under severe simulated battlefield conditions
(human voice + heavy vehicle engine noise + sudden gunfire transient impulses).
"""

import os
import sys
import time
import json
import random
import numpy as np
import soundfile as sf
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.training.student_model import MARKUSBLUEStudentEnhancer
from src.agc.automatic_gain_control import AutomaticGainControl
from src.limiter.peak_limiter import PeakSafetyLimiter

try:
    from pystoi import stoi
except ImportError:
    stoi = None

def run_simulation(
    speech_path="datasets/speech/speech_0001.wav",
    noise_path="datasets/external_noise/suppressible/heavy_engine/heavy_engine_001.wav",
    gunshot_path="datasets/gunshot/gunshot_0001.wav",
    out_dir="reports/audio_demonstrations",
    model_path="models/markusblue_esp32s3_best.pt"
):
    os.chdir(REPO_ROOT)
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 80)
    print("MARKUSBLUE — REAL-TIME BATTLEFIELD STREAMING SIMULATION DEMONSTRATION")
    print("Controlled Laboratory Simulation (SIH26052 Tactical Prototype)")
    print("=" * 80)

    sr = 16000
    hop_size = 64        # 4.0 ms per frame (64 / 16000 s)
    n_fft = 256          # 16.0 ms window (256 / 16000 s)
    frame_budget_us = (hop_size / sr) * 1e6 # 4,000 µs (4.0 ms)

    # 1. Load Model
    device = torch.device("cpu")
    model = MARKUSBLUEStudentEnhancer(n_fft=n_fft, hop_length=hop_size, hidden_dim=32).to(device)
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        model.load_state_dict(ckpt)
        print(f"[*] Loaded Edge-AI Model Checkpoint: '{model_path}'")
    model.eval()

    agc = AutomaticGainControl(sr=sr, target_rms_dbfs=-16.0)
    limiter = PeakSafetyLimiter(sr=sr, ceiling_dbfs=-0.5)

    # 2. Create Realistic Battlefield Audio Stream (5 seconds = 80,000 samples)
    stream_len = 80000
    
    def read_clip(p, target_len):
        if not os.path.exists(p):
            # Fallback to any valid file in the folder
            folder = os.path.dirname(p)
            wavs = glob.glob(os.path.join(folder, "*.wav")) if os.path.exists(folder) else []
            p = wavs[0] if wavs else None
        if p and os.path.exists(p):
            d, _ = sf.read(p)
            if len(d.shape) > 1: d = np.mean(d, axis=1)
            if len(d) < target_len:
                d = np.tile(d, int(np.ceil(target_len / len(d))))
            return d[:target_len].astype(np.float32)
        return np.zeros(target_len, dtype=np.float32)

    import glob
    speech = read_clip(speech_path, stream_len)
    engine = read_clip(noise_path, stream_len)
    gunshot = read_clip(gunshot_path, stream_len)

    # Synthetic Battlefield Mix:
    # 0s - 1s: Ambient vehicle noise
    # 1s - 4s: Human speech under heavy engine noise (-5 dB SNR)
    # 2.5s: Sudden loud gunfire impulse burst
    # 4s - 5s: Continued ambient vehicle noise
    p_speech = np.mean(speech[16000:64000]**2) + 1e-8
    p_engine = np.mean(engine**2) + 1e-8
    engine_scaled = engine * np.sqrt((p_speech / (10.0 ** (-5.0 / 10.0))) / p_engine)
    
    # Place gunfire transient at sample 40,000 (2.5 seconds in)
    gunfire_event = np.zeros(stream_len, dtype=np.float32)
    gunfire_event[40000:40000 + len(gunshot[:16000])] = gunshot[:16000] * 1.5

    battlefield_input = engine_scaled + gunfire_event
    battlefield_input[16000:64000] += speech[16000:64000]

    # Normalize input slightly to avoid hard master digital clipping before entry
    max_peak = np.max(np.abs(battlefield_input))
    if max_peak > 0.98:
        battlefield_input = battlefield_input * (0.95 / max_peak)

    # 3. Simulate Streaming Real-Time Processing (Block by Block: 64 samples / 4.0 ms)
    print(f"\n[*] Commencing Frame-by-Frame Streaming Simulation:")
    print(f"    - Frame Duration / Hop: {hop_size} samples ({frame_budget_us:.0f} µs)")
    print(f"    - Total Audio Duration: {stream_len / sr:.2f} seconds ({stream_len // hop_size:,} frames)")
    print(f"    - Real-Time Latency Target: < 4,000 µs per frame (RTF < 1.000)")

    output_stream = np.zeros(stream_len, dtype=np.float32)
    frame_latencies_us = []
    buffer_overruns = 0

    # Causal STFT FIFO buffer (256 samples)
    stft_fifo = np.zeros(n_fft, dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    
    # Overlap-add reconstruction buffer
    ola_buffer = np.zeros(n_fft, dtype=np.float32)

    total_frames = stream_len // hop_size

    for f_idx in range(total_frames):
        start_sample = f_idx * hop_size
        end_sample = start_sample + hop_size
        frame_input = battlefield_input[start_sample:end_sample]

        t_start = time.perf_counter()

        # Step 1: Slide FIFO window
        stft_fifo[:-hop_size] = stft_fifo[hop_size:]
        stft_fifo[-hop_size:] = frame_input

        # Step 2: Causal STFT
        windowed = stft_fifo * window
        rfft_out = np.fft.rfft(windowed, n=n_fft)
        mag = np.abs(rfft_out).astype(np.float32)
        phase = np.angle(rfft_out)

        # Step 3: Neural Mask Inference
        mag_t = torch.tensor(mag, dtype=torch.float32).view(1, 129, 1).to(device)
        with torch.no_grad():
            mask_t = model(mag_t)
        mask = mask_t.squeeze().cpu().numpy()

        # Step 4: Spectral Filtering & Inverse FFT
        filtered_mag = mag * mask
        recon_complex = filtered_mag * np.exp(1j * phase)
        irfft_out = np.fft.irfft(recon_complex, n=n_fft) * window # Synthesis Hann

        # Step 5: Overlap-Add
        ola_buffer += irfft_out
        reconstructed_block = ola_buffer[:hop_size].copy()
        ola_buffer[:-hop_size] = ola_buffer[hop_size:]
        ola_buffer[-hop_size:] = 0.0

        # Step 6: Post-DSP (AGC & Peak Limiter)
        agc_block = agc.process_frame(reconstructed_block, is_speech=True)
        final_block = limiter.process_frame(agc_block)

        t_elapsed = (time.perf_counter() - t_start) * 1e6 # microseconds
        frame_latencies_us.append(t_elapsed)
        if t_elapsed > frame_budget_us:
            buffer_overruns += 1

        output_stream[start_sample:end_sample] = final_block

    mean_lat_us = float(np.mean(frame_latencies_us))
    max_lat_us = float(np.max(frame_latencies_us))
    p95_lat_us = float(np.percentile(frame_latencies_us, 95))
    rtf = mean_lat_us / frame_budget_us

    print(f"\n[+] Streaming Simulation Complete:")
    print(f"    - Mean Frame Latency: {mean_lat_us:.1f} µs ({mean_lat_us/1000:.3f} ms)")
    print(f"    - 95th Percentile Latency: {p95_lat_us:.1f} µs ({p95_lat_us/1000:.3f} ms)")
    print(f"    - Maximum Peak Frame Latency: {max_lat_us:.1f} µs ({max_lat_us/1000:.3f} ms)")
    print(f"    - Real-Time Factor (RTF): {rtf:.4f} (Sustainable Real-Time: {'YES' if rtf < 1.0 else 'NO'})")
    print(f"    - Buffer Overruns / Frame Drops: {buffer_overruns} / {total_frames} ({buffer_overruns/total_frames*100:.2f}%)")

    # Speech intelligibility & enhancement metrics
    clean_speech_stream = np.zeros_like(battlefield_input)
    clean_speech_stream[16000:64000] = speech[16000:64000]
    
    stoi_in = float(stoi(clean_speech_stream[16000:64000], battlefield_input[16000:64000], 16000)) if stoi else 0.0
    stoi_out = float(stoi(clean_speech_stream[16000:64000], output_stream[16000:64000], 16000)) if stoi else 0.0

    print(f"\n[*] Intelligibility & Quality in Active Speech Zone (1.0s - 4.0s):")
    print(f"    - Input STOI: {stoi_in:.3f}")
    print(f"    - Enhanced Output STOI: {stoi_out:.3f} ({stoi_out - stoi_in:+.3f} Intelligibility Gain)")
    
    # Save Audio Demonstrations
    path_ref = os.path.join(out_dir, "battlefield_sim_reference_speech.wav")
    path_in = os.path.join(out_dir, "battlefield_sim_input_noisy.wav")
    path_out = os.path.join(out_dir, "battlefield_sim_output_clean.wav")

    sf.write(path_ref, clean_speech_stream, sr)
    sf.write(path_in, battlefield_input, sr)
    sf.write(path_out, output_stream, sr)

    print(f"\n[+] Audio Artifacts Saved to '{out_dir}':")
    print(f"    1. Reference Clean Speech: {path_ref}")
    print(f"    2. Corrupted Battlefield Input: {path_in}")
    print(f"    3. MARKUSBLUE Enhanced Output: {path_out}")

    report = {
        "mean_latency_us": mean_lat_us,
        "max_latency_us": max_lat_us,
        "p95_latency_us": p95_lat_us,
        "frame_budget_us": frame_budget_us,
        "rtf": rtf,
        "buffer_overruns": buffer_overruns,
        "total_frames": total_frames,
        "stoi_in": stoi_in,
        "stoi_out": stoi_out,
        "delta_stoi": stoi_out - stoi_in,
        "audio_files": {
            "reference": path_ref,
            "input": path_in,
            "output": path_out
        }
    }
    
    with open("reports/realtime_battlefield_simulation.json", "w") as f:
        json.dump(report, f, indent=2)

    return report

if __name__ == "__main__":
    run_simulation()
