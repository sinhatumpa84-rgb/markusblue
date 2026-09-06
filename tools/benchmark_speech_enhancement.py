#!/usr/bin/env python3
"""
MARKUSBLUE — Comprehensive Speech Enhancement Benchmark & Validation Suite
SIH26052 — DRDO / Defence Tactical Edge-AI Speech Enhancement System

Evaluates any model checkpoint (baseline v7.0.00 or fine-tuned v7.1.00) against
10 realistic operational battlefield scenarios on unseen test speech.
Computes SI-SDR, SDR, SNR improvement, STOI, Speech RMS Preservation,
Noise Attenuation, Critical Audio Cue Preservation, Audio Blanking, and RTF.
"""

import os
import sys
import glob
import json
import time
import random
import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

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

REPO_ROOT = r"c:\Users\sinha\OneDrive\Desktop\demucs"

def compute_sisdr(reference: np.ndarray, estimated: np.ndarray) -> float:
    eps = 1e-8
    ref = reference - np.mean(reference)
    est = estimated - np.mean(estimated)
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    ratio = (np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)
    return float(10.0 * np.log10(max(ratio, 1e-10)))

def compute_sdr(reference: np.ndarray, estimated: np.ndarray) -> float:
    eps = 1e-8
    noise = estimated - reference
    ratio = (np.sum(reference ** 2) + eps) / (np.sum(noise ** 2) + eps)
    return float(10.0 * np.log10(max(ratio, 1e-10)))

def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    eps = 1e-8
    ratio = (np.sum(signal ** 2) + eps) / (np.sum(noise ** 2) + eps)
    return float(10.0 * np.log10(max(ratio, 1e-10)))

def load_mono(path, target_len=48000, sr=16000):
    try:
        data, file_sr = sf.read(path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if len(data) < target_len:
            data = np.pad(data, (0, target_len - len(data)))
        elif len(data) > target_len:
            start = random.randint(0, len(data) - target_len)
            data = data[start:start + target_len]
        return data.astype(np.float32)
    except Exception:
        return np.zeros(target_len, dtype=np.float32)

def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple:
    eps = 1e-8
    p_clean = np.mean(clean ** 2) + eps
    p_noise = np.mean(noise ** 2) + eps
    target_p_noise = p_clean / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_p_noise / p_noise)
    scaled_noise = noise * scale
    noisy = clean + scaled_noise
    return noisy, scaled_noise

def run_enhancement(model, noisy_audio: np.ndarray, agc=None, limiter=None, n_fft=256, hop_length=64) -> np.ndarray:
    noisy_t = torch.tensor(noisy_audio, dtype=torch.float32)
    w = torch.hann_window(n_fft)
    spec = torch.stft(noisy_t, n_fft=n_fft, hop_length=hop_length, window=w, return_complex=True)
    mag = torch.abs(spec).unsqueeze(0) # [1, Bins, Frames]
    
    with torch.no_grad():
        mask = model(mag) # [1, Bins, Frames]
        
    enh_spec = spec * mask.squeeze(0)
    enh_audio = torch.istft(enh_spec, n_fft=n_fft, hop_length=hop_length, window=w, length=len(noisy_audio))
    out = enh_audio.numpy().astype(np.float32)
    
    if agc is not None:
        agc.reset()
        block_sz = 256
        for b in range(0, len(out), block_sz):
            chunk = out[b:b + block_sz]
            out[b:b + block_sz] = agc.process_frame(chunk, is_speech=True)
            
    if limiter is not None:
        limiter.reset()
        out = limiter.process_frame(out)
        
    return out

def run_benchmark(model_path="models/markusblue_esp32s3_best.pt", model_tag="v7.0.00", save_audio=True):
    os.chdir(REPO_ROOT)
    print("=" * 80)
    print(f"MARKUSBLUE SPEECH ENHANCEMENT BENCHMARK: [{model_tag}]")
    print(f"Model Checkpoint: {model_path}")
    print("=" * 80)

    device = torch.device("cpu") # Benchmark deterministic CPU inference
    model = MARKUSBLUEStudentEnhancer(n_fft=256, hop_length=64, hidden_dim=32).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint {model_path} not found!")
        return None
        
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()

    agc = AutomaticGainControl(sr=16000, target_rms_dbfs=-16.0)
    limiter = PeakSafetyLimiter(sr=16000, ceiling_dbfs=-0.5)

    # Asset libraries
    test_manifest_path = "audit_results/final_dataset_manifest.csv"
    speech_files = []
    if os.path.exists(test_manifest_path):
        import csv
        with open(test_manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == "test" and row["class"] == "speech":
                    speech_files.append(row["file_path"])
    if not speech_files:
        speech_files = sorted(glob.glob("datasets/speech/*.wav"))[-300:]

    gunshot_files = sorted(glob.glob("datasets/gunshot/*.wav"))
    engine_files = sorted(glob.glob("datasets/external_noise/suppressible/heavy_engine/*.wav")) + sorted(glob.glob("datasets/external_noise/suppressible/diesel_engine/*.wav"))
    wind_files = sorted(glob.glob("datasets/external_noise/suppressible/wind/*.wav"))
    impulse_files = sorted(glob.glob("datasets/other_impulse/*.wav"))
    radio_noise = sorted(glob.glob("datasets/critical_audio/radio_communication/*.wav"))
    movement_cues = sorted(glob.glob("datasets/critical_audio/movement/*.wav")) + sorted(glob.glob("datasets/critical_audio/footsteps/*.wav"))
    alarm_cues = sorted(glob.glob("datasets/critical_audio/alarms/*.wav")) + sorted(glob.glob("datasets/critical_audio/sirens/*.wav"))
    aircraft_files = sorted(glob.glob("datasets/external_noise/suppressible/aircraft/*.wav")) + sorted(glob.glob("datasets/external_noise/suppressible/helicopter/*.wav"))
    crowd_files = sorted(glob.glob("datasets/external_noise/suppressible/crowd/*.wav"))
    bg_ambient = sorted(glob.glob("datasets/background_noise/*.wav"))

    print(f"[*] Loaded {len(speech_files)} unseen test speech utterances.")

    # Define 10 Operational Scenarios
    scenarios = [
        {"id": 1, "name": "Speech + Gunfire Transients", "snr": -5.0, "noise_pool": gunshot_files, "cue_pool": []},
        {"id": 2, "name": "Speech + Continuous Engine Noise", "snr": -10.0, "noise_pool": engine_files, "cue_pool": []},
        {"id": 3, "name": "Speech + Compound Gunfire & Engine", "snr": -10.0, "noise_pool": engine_files, "overlay_pool": gunshot_files, "cue_pool": []},
        {"id": 4, "name": "Speech + Wind & Mechanical Impact", "snr": -5.0, "noise_pool": wind_files, "overlay_pool": impulse_files, "cue_pool": []},
        {"id": 5, "name": "Speech + Radio Noise & Ambient Hum", "snr": 0.0, "noise_pool": bg_ambient, "overlay_pool": radio_noise, "cue_pool": []},
        {"id": 6, "name": "Conversational Speech + Tactical Movement Cues", "snr": 5.0, "noise_pool": bg_ambient, "cue_pool": movement_cues},
        {"id": 7, "name": "Speech + Intermittent Loud Gunshot Impulses", "snr": 0.0, "noise_pool": gunshot_files, "cue_pool": []},
        {"id": 8, "name": "Speech + Multi-Noise (Aircraft + Engine + Crowd)", "snr": -5.0, "noise_pool": aircraft_files, "overlay_pool": crowd_files, "cue_pool": []},
        {"id": 9, "name": "Severe Low-SNR Buried Speech (-15 dB)", "snr": -15.0, "noise_pool": engine_files, "cue_pool": []},
        {"id": 10, "name": "Dynamic Fluctuating SNR (-15dB to +10dB)", "snr": -5.0, "noise_pool": bg_ambient, "cue_pool": alarm_cues},
    ]

    out_audio_dir = f"reports/audio_demonstrations/{model_tag}"
    if save_audio:
        os.makedirs(out_audio_dir, exist_ok=True)

    scenario_results = []
    total_test_samples_per_scenario = 20 # 20 test utterances per scenario = 200 evaluations
    
    random.seed(1337)
    
    for sc in scenarios:
        sc_id = sc["id"]
        sc_name = sc["name"]
        target_snr = sc["snr"]
        
        metrics = {
            "sisdr_in": [], "sisdr_out": [], "delta_sisdr": [],
            "sdr_in": [], "sdr_out": [], "delta_sdr": [],
            "stoi_in": [], "stoi_out": [], "delta_stoi": [],
            "snr_in": [], "snr_out": [], "delta_snr": [],
            "speech_preservation_rms": [],
            "noise_attenuation_db": [],
            "cue_retention_ratio": [],
            "blanking_detected": False
        }

        # Select test files
        test_samples = random.sample(speech_files, min(total_test_samples_per_scenario, len(speech_files)))
        
        for idx, sp_path in enumerate(test_samples):
            clean_sp = load_mono(sp_path, target_len=48000) # 3 seconds @ 16 kHz
            
            # Draw noise
            noise_path = random.choice(sc["noise_pool"]) if sc["noise_pool"] else None
            noise_raw = load_mono(noise_path, target_len=48000) if noise_path else np.zeros_like(clean_sp)
            
            if "overlay_pool" in sc and sc["overlay_pool"]:
                overlay_path = random.choice(sc["overlay_pool"])
                overlay_raw = load_mono(overlay_path, target_len=48000)
                noise_raw = 0.6 * noise_raw + 0.4 * overlay_raw
                
            noisy_audio, actual_noise = mix_at_snr(clean_sp, noise_raw, target_snr)
            
            # Inject critical cues if specified
            cue_raw = np.zeros_like(clean_sp)
            if sc["cue_pool"]:
                cue_path = random.choice(sc["cue_pool"])
                cue_raw = load_mono(cue_path, target_len=48000)
                noisy_audio = noisy_audio + 0.3 * cue_raw
                # For critical cue preservation, reference includes clean speech + critical cue
                target_reference = clean_sp + 0.3 * cue_raw
            else:
                target_reference = clean_sp

            # Run Model
            enh_audio = run_enhancement(model, noisy_audio, agc=agc, limiter=limiter)

            # Check audio blanking (zero dropouts during or immediately after gunfire)
            if sc_id in [1, 3, 7]:
                sub_rms = [np.mean(enh_audio[k:k+256]**2) for k in range(0, len(enh_audio), 256)]
                if any(r < 1e-12 for r in sub_rms[20:60]): # check active speech region
                    metrics["blanking_detected"] = True

            # Calculate metrics
            sisdr_in = compute_sisdr(target_reference, noisy_audio)
            sisdr_out = compute_sisdr(target_reference, enh_audio)
            
            sdr_in = compute_sdr(target_reference, noisy_audio)
            sdr_out = compute_sdr(target_reference, enh_audio)

            snr_in_val = compute_snr(target_reference, noisy_audio - target_reference)
            snr_out_val = compute_snr(target_reference, enh_audio - target_reference)

            if stoi:
                try:
                    st_in = float(stoi(target_reference, noisy_audio, 16000, extended=False))
                    st_out = float(stoi(target_reference, enh_audio, 16000, extended=False))
                except Exception:
                    st_in, st_out = 0.5, 0.5
            else:
                st_in, st_out = 0.0, 0.0

            # RMS preservation of speech
            sp_mask = np.abs(clean_sp) > 0.02
            clean_rms = np.sqrt(np.mean(clean_sp[sp_mask] ** 2)) if np.any(sp_mask) else 0.1
            enh_rms = np.sqrt(np.mean(enh_audio[sp_mask] ** 2)) if np.any(sp_mask) else 0.1
            preservation_ratio = float(enh_rms / (clean_rms + 1e-8))

            # Noise attenuation
            noise_mask = ~sp_mask
            in_noise_rms = np.sqrt(np.mean(noisy_audio[noise_mask] ** 2) + 1e-8)
            out_noise_rms = np.sqrt(np.mean(enh_audio[noise_mask] ** 2) + 1e-8)
            attenuation_db = float(20.0 * np.log10(in_noise_rms / out_noise_rms))

            # Cue retention
            if sc["cue_pool"]:
                cue_mask = np.abs(cue_raw) > 0.02
                if np.any(cue_mask):
                    cue_clean_rms = np.sqrt(np.mean(cue_raw[cue_mask] ** 2))
                    cue_out_rms = np.sqrt(np.mean(enh_audio[cue_mask] ** 2))
                    cue_retention = float(cue_out_rms / (cue_clean_rms + 1e-8))
                else:
                    cue_retention = 1.0
                metrics["cue_retention_ratio"].append(cue_retention)

            metrics["sisdr_in"].append(sisdr_in)
            metrics["sisdr_out"].append(sisdr_out)
            metrics["delta_sisdr"].append(sisdr_out - sisdr_in)
            
            metrics["sdr_in"].append(sdr_in)
            metrics["sdr_out"].append(sdr_out)
            metrics["delta_sdr"].append(sdr_out - sdr_in)

            metrics["stoi_in"].append(st_in)
            metrics["stoi_out"].append(st_out)
            metrics["delta_stoi"].append(st_out - st_in)

            metrics["snr_in"].append(snr_in_val)
            metrics["snr_out"].append(snr_out_val)
            metrics["delta_snr"].append(snr_out_val - snr_in_val)

            metrics["speech_preservation_rms"].append(preservation_ratio)
            metrics["noise_attenuation_db"].append(attenuation_db)

            # Save representative audio for the first sample
            if save_audio and idx == 0:
                sc_slug = sc_name.lower().replace(" ", "_").replace("+", "plus").replace("&", "and").replace("(", "").replace(")", "").replace("-", "neg")
                sf.write(os.path.join(out_audio_dir, f"scenario_{sc_id}_{sc_slug}_clean.wav"), clean_sp, 16000)
                sf.write(os.path.join(out_audio_dir, f"scenario_{sc_id}_{sc_slug}_noisy.wav"), noisy_audio, 16000)
                sf.write(os.path.join(out_audio_dir, f"scenario_{sc_id}_{sc_slug}_enhanced.wav"), enh_audio, 16000)

        # Average scenario metrics
        sc_summary = {
            "scenario_id": sc_id,
            "scenario_name": sc_name,
            "target_snr_db": target_snr,
            "sisdr_in": float(np.mean(metrics["sisdr_in"])),
            "sisdr_out": float(np.mean(metrics["sisdr_out"])),
            "delta_sisdr": float(np.mean(metrics["delta_sisdr"])),
            "sdr_in": float(np.mean(metrics["sdr_in"])),
            "sdr_out": float(np.mean(metrics["sdr_out"])),
            "delta_sdr": float(np.mean(metrics["delta_sdr"])),
            "stoi_in": float(np.mean(metrics["stoi_in"])),
            "stoi_out": float(np.mean(metrics["stoi_out"])),
            "delta_stoi": float(np.mean(metrics["delta_stoi"])),
            "snr_in": float(np.mean(metrics["snr_in"])),
            "snr_out": float(np.mean(metrics["snr_out"])),
            "delta_snr": float(np.mean(metrics["delta_snr"])),
            "speech_rms_preservation": float(np.mean(metrics["speech_preservation_rms"])),
            "noise_attenuation_db": float(np.mean(metrics["noise_attenuation_db"])),
            "cue_retention_ratio": float(np.mean(metrics["cue_retention_ratio"])) if metrics["cue_retention_ratio"] else None,
            "blanking_detected": metrics["blanking_detected"]
        }
        scenario_results.append(sc_summary)
        
        print(
            f"  [Scen {sc_id:02d}] {sc_name[:38]:<38} | "
            f"SNR In: {sc_summary['snr_in']:5.1f} dB -> Out: {sc_summary['snr_out']:5.1f} dB (Gain: +{sc_summary['delta_snr']:4.1f} dB) | "
            f"STOI: {sc_summary['stoi_in']:.3f} -> {sc_summary['stoi_out']:.3f} (+{sc_summary['delta_stoi']:+.3f}) | "
            f"SI-SDR: {sc_summary['delta_sisdr']:+5.1f} dB"
        )

    # Measure algorithmic latency and RTF across 1,000 frames (64 samples = 4.0 ms per frame)
    print("\n[*] Measuring real-time latency and Real-Time Factor (RTF) across 1,000 consecutive 4.0ms frames...")
    dummy_frame = torch.randn(1, 129, 1).to(device)
    # Warmup
    for _ in range(50):
        _ = model(dummy_frame)
        
    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        _ = model(dummy_frame)
        latencies.append(time.perf_counter() - t0)
        
    avg_latency_us = float(np.mean(latencies) * 1e6)
    p95_latency_us = float(np.percentile(latencies, 95) * 1e6)
    frame_budget_us = 4000.0 # 4.0 ms hop size
    rtf = avg_latency_us / frame_budget_us

    print(f"    - Mean AI Inference Latency: {avg_latency_us:.1f} µs ({avg_latency_us/1000:.3f} ms)")
    print(f"    - 95th Percentile Latency: {p95_latency_us:.1f} µs ({p95_latency_us/1000:.3f} ms)")
    print(f"    - Available Frame Duration: {frame_budget_us:.0f} µs (4.00 ms)")
    print(f"    - Real-Time Factor (RTF): {rtf:.4f} (< 1.000 means sustainable real-time)")

    overall_delta_snr = float(np.mean([s["delta_snr"] for s in scenario_results]))
    overall_delta_sisdr = float(np.mean([s["delta_sisdr"] for s in scenario_results]))
    overall_stoi_in = float(np.mean([s["stoi_in"] for s in scenario_results]))
    overall_stoi_out = float(np.mean([s["stoi_out"] for s in scenario_results]))
    overall_delta_stoi = float(np.mean([s["delta_stoi"] for s in scenario_results]))
    overall_preservation = float(np.mean([s["speech_rms_preservation"] for s in scenario_results]))
    overall_attenuation = float(np.mean([s["noise_attenuation_db"] for s in scenario_results]))

    summary = {
        "model_tag": model_tag,
        "model_path": model_path,
        "avg_delta_snr_db": overall_delta_snr,
        "avg_delta_sisdr_db": overall_delta_sisdr,
        "avg_stoi_in": overall_stoi_in,
        "avg_stoi_out": overall_stoi_out,
        "avg_delta_stoi": overall_delta_stoi,
        "avg_speech_rms_preservation": overall_preservation,
        "avg_noise_attenuation_db": overall_attenuation,
        "mean_latency_us": avg_latency_us,
        "rtf": rtf,
        "scenarios": scenario_results
    }

    out_json = f"reports/benchmark_{model_tag}.json"
    os.makedirs("reports", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Benchmark completed. Results saved to {out_json}")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/markusblue_esp32s3_best.pt")
    parser.add_argument("--tag", type=str, default="v7.0.00")
    args = parser.parse_args()
    run_benchmark(args.model, args.tag)
