#!/usr/bin/env python3
"""
MARKUSBLUE — ESP32-S3 Student Model Fine-Tuner & Full-Data Trainer (v7.1.00)
SIH26052 — DRDO / Defence Tactical Edge-AI Speech Enhancement System

Performs a controlled fine-tuning cycle on the complete valid dataset pool (42,670 training files)
leveraging NVIDIA GeForce RTX 3050 CUDA GPU.
Incorporates:
  - Base Speech Utterances (Train Split)
  - Full Gunshot & Impulse Libraries (datasets/gunshot, datasets/other_impulse, data/ deduplicated)
  - Full Suppressible Noise Library (datasets/external_noise, datasets/background_noise)
  - Critical Audio Cue Preservation (alarms, sirens, radio, footsteps)
Preserves baseline v7.0.00 intact and saves new model as MARKUSBLUE-v7.1.00.
"""

import os
import sys
import time
import json
import random
import csv
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.training.student_model import MARKUSBLUEStudentEnhancer

class FullTacticalSpeechDataset(Dataset):
    """
    High-Performance Multi-Noise Tactical Speech Dataset.
    Synthesizes clean speech + 1-3 noise sources + critical cue preservation targets.
    """
    def __init__(self, speech_files, noise_files, critical_files, sr=16000, duration_samples=32000, snr_range=(-15.0, 15.0)):
        self.speech_files = speech_files
        self.noise_files = noise_files
        self.critical_files = critical_files
        self.sr = sr
        self.duration = duration_samples
        self.snr_min, self.snr_max = snr_range
        self.n_fft = 256
        self.hop_length = 64
        self.window = torch.hann_window(self.n_fft)

    def __len__(self):
        return len(self.speech_files)

    def _load_audio(self, path):
        try:
            data, _ = sf.read(path)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            if len(data) < self.duration:
                data = np.pad(data, (0, self.duration - len(data)))
            else:
                max_start = len(data) - self.duration
                start = random.randint(0, max_start) if max_start > 0 else 0
                data = data[start:start + self.duration]
            return data.astype(np.float32)
        except Exception:
            return np.zeros(self.duration, dtype=np.float32)

    def __getitem__(self, idx):
        clean = self._load_audio(self.speech_files[idx])
        
        # Draw 1 to 2 random noise files
        num_noises = random.randint(1, 2)
        noise = np.zeros_like(clean)
        for _ in range(num_noises):
            n_path = random.choice(self.noise_files)
            noise += self._load_audio(n_path)
            
        # Draw 1 critical cue in 50% of samples to reinforce preservation
        cue = np.zeros_like(clean)
        has_cue = random.random() < 0.5
        if has_cue and self.critical_files:
            c_path = random.choice(self.critical_files)
            cue = self._load_audio(c_path) * random.uniform(0.2, 0.5)

        # Scale noise to target SNR
        p_clean = np.mean(clean ** 2) + 1e-8
        p_noise = np.mean(noise ** 2) + 1e-8
        target_snr = random.uniform(self.snr_min, self.snr_max)
        target_p_noise = p_clean / (10.0 ** (target_snr / 10.0))
        noise_scaled = noise * np.sqrt(target_p_noise / p_noise)

        # Composite noisy input and preservation target
        noisy = clean + noise_scaled + cue
        target_audio = clean + cue # Both clean voice and critical cue must be preserved

        # Compute STFT
        noisy_t = torch.tensor(noisy, dtype=torch.float32)
        target_t = torch.tensor(target_audio, dtype=torch.float32)

        w = torch.hann_window(self.n_fft)
        noisy_stft = torch.stft(noisy_t, n_fft=self.n_fft, hop_length=self.hop_length, window=w, return_complex=True)
        target_stft = torch.stft(target_t, n_fft=self.n_fft, hop_length=self.hop_length, window=w, return_complex=True)

        noisy_mag = torch.abs(noisy_stft) # [Bins, Frames]
        target_mag = torch.abs(target_stft)

        # Ideal Ratio Mask target
        irm = target_mag / (target_mag + torch.abs(noisy_stft - target_stft) + 1e-8)
        irm = torch.clamp(irm, 0.0, 1.0)

        return {
            "noisy_mag": noisy_mag,
            "target_mag": target_mag,
            "target_mask": irm,
            "noisy_audio": noisy_t,
            "target_audio": target_t
        }

def train_v7_1_00(epochs=5, batch_size=32, lr=2e-4):
    print("=" * 80)
    print("MARKUSBLUE — CONTROLLED FINE-TUNING & FULL-DATA TRAINING (v7.1.00)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load dataset paths from manifest
    manifest_path = "audit_results/final_dataset_manifest.csv"
    train_speech = []
    val_speech = []
    noise_pool = []
    critical_pool = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row["file_path"]
            split = row["split"]
            cls = row["class"]
            
            if cls == "speech":
                if split == "train":
                    train_speech.append(p)
                elif split == "validation":
                    val_speech.append(p)
            elif cls.startswith("critical_"):
                critical_pool.append(p)
            elif split == "train" and cls != "derived_train":
                noise_pool.append(p)

    print(f"[*] Manifest Data Indexed:")
    print(f"    - Clean Speech Training Utterances: {len(train_speech):,}")
    print(f"    - Clean Speech Validation Utterances: {len(val_speech):,}")
    print(f"    - Suppressible Noise Library Assets: {len(noise_pool):,}")
    print(f"    - Critical Cue Preservation Assets: {len(critical_pool):,}")

    train_ds = FullTacticalSpeechDataset(train_speech, noise_pool, critical_pool, duration_samples=32000)
    val_ds = FullTacticalSpeechDataset(val_speech, noise_pool, critical_pool, duration_samples=32000)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize model from baseline v7.0.00 checkpoint
    baseline_path = "models/markusblue_esp32s3_best.pt"
    model = MARKUSBLUEStudentEnhancer(n_fft=256, hop_length=64, hidden_dim=32).to(device)
    
    if os.path.exists(baseline_path):
        ckpt = torch.load(baseline_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        model.load_state_dict(ckpt)
        print(f"[*] Successfully initialized from baseline weights: {baseline_path}")
    else:
        print(f"[!] Warning: Baseline {baseline_path} not found. Training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    mask_loss_fn = nn.MSELoss()
    
    v7_1_pt_path = "models/markusblue_v7_1_00_best.pt"
    best_val_loss = float("inf")
    history = []

    print(f"\n[*] Starting {epochs} Epochs of Fine-Tuning on RTX 3050 CUDA...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            noisy_mag = batch["noisy_mag"].to(device)
            target_mag = batch["target_mag"].to(device)
            target_mask = batch["target_mask"].to(device)

            optimizer.zero_grad()
            
            pred_mask = model(noisy_mag)
            est_mag = noisy_mag * pred_mask
            
            l_mask = mask_loss_fn(pred_mask, target_mask)
            l_mag = F.l1_loss(est_mag, target_mag)
            
            # Penalize under-estimating voice or critical cues (preservation loss)
            active_target = (target_mag > 0.03).float()
            under_est = F.relu(target_mask - pred_mask) * active_target
            l_preserve = torch.mean(under_est ** 2)
            
            # Penalize mask leakage in deep noise areas
            noise_only = (target_mag < 0.01).float()
            l_noise_leak = torch.mean((pred_mask * noise_only) ** 2)
            
            total_loss = l_mask + 2.0 * l_mag + 4.0 * l_preserve + 1.5 * l_noise_leak
            total_loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()
            
        scheduler.step()
        train_loss /= len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                noisy_mag = batch["noisy_mag"].to(device)
                target_mag = batch["target_mag"].to(device)
                target_mask = batch["target_mask"].to(device)
                
                pred_mask = model(noisy_mag)
                est_mag = noisy_mag * pred_mask
                
                l_mask = mask_loss_fn(pred_mask, target_mask)
                l_mag = F.l1_loss(est_mag, target_mag)
                active_target = (target_mag > 0.03).float()
                under_est = F.relu(target_mask - pred_mask) * active_target
                l_preserve = torch.mean(under_est ** 2)
                
                total_loss = l_mask + 2.0 * l_mag + 4.0 * l_preserve
                val_loss += total_loss.item()
                
        val_loss /= len(val_loader)
        elapsed = time.time() - t0
        
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": elapsed
        }
        history.append(record)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "version": "v7.1.00",
                "param_count": 18725
            }, v7_1_pt_path)
            print(f"  --> Saved new best checkpoint: {v7_1_pt_path} (Val Loss: {best_val_loss:.4f})")

    # Save history
    hist_path = "models/markusblue_v7_1_00_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[+] Fine-tuning completed. History saved to {hist_path}")
    return v7_1_pt_path

if __name__ == "__main__":
    train_v7_1_00(epochs=5, batch_size=32, lr=2e-4)
