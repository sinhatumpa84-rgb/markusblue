#!/usr/bin/env python3
"""
MARKUSBLUE — ESP32-S3 Student Speech Enhancement Model Trainer
SIH Problem Statement: SIH26052 — DRDO / Defence Speech-Enhancement System

Trains the causal neural mask estimator (MARKUSBLUEStudentEnhancer) for the ESP32-S3 N16R8.
Evaluates on dynamic multi-noise mixtures (-15dB to +20dB), temporal flybys, bursts,
and sudden gunfire impulse transients across 50+ noise categories while ACTIVELY
PRESERVING critical audio cues (alarms, sirens, footsteps, radio communication).
"""

import os
import time
import json
import random
import argparse
import glob
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.training.student_model import MARKUSBLUEStudentEnhancer
from src.training.losses import MultiResolutionSTFTLoss, SISDRLoss
from src.dataset.multi_noise_mixer import MultiNoiseMixer

class TacticalSpeechNoiseDataset(Dataset):
    """
    Dynamic Online Multi-Noise Mixture Dataset for Tactical Speech Enhancement.
    Generates realistic combinations of:
      Critical Target (Clean Speech + Alarms / Sirens / Footsteps / Radio)
      + 1-3 Suppressible Noise Sources (Aviation, Engines, Environmental, Industrial)
    Across specified SNR range (-15 dB to +20 dB) with dynamic temporal movement.
    """
    def __init__(
        self,
        speech_files: list,
        suppressible_files: list,
        critical_files: list,
        sr: int = 16000,
        duration_samples: int = 16000,
        snr_range: tuple = (-15.0, 20.0),
        n_fft: int = 256,
        hop_length: int = 64
    ):
        self.speech_files = speech_files
        self.suppressible_files = suppressible_files
        self.critical_files = critical_files
        self.sr = sr
        self.duration = duration_samples
        self.snr_min, self.snr_max = snr_range
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)
        self.mixer = MultiNoiseMixer(
            suppressible_files=suppressible_files,
            critical_files=critical_files,
            sr=sr,
            duration_samples=duration_samples,
            snr_range=snr_range,
            max_noise_sources=3
        )

    def __len__(self):
        return len(self.speech_files)

    def _load_audio(self, path: str) -> np.ndarray:
        try:
            data, _ = sf.read(path)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1) # Mono
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
        # 1. Load clean speech
        speech = self._load_audio(self.speech_files[idx])
        
        # 2. Dynamic multi-noise mixing (mix with suppressible noise + critical cue to preserve)
        target_snr = random.uniform(self.snr_min, self.snr_max)
        noisy, target_critical, _ = self.mixer.mix_operational(
            speech, target_snr_db=target_snr, inject_critical_cue=True
        )

        # 3. Compute STFTs (Target to preserve is target_critical: Speech + Alarm/Siren/Footsteps)
        speech_t = torch.tensor(target_critical, dtype=torch.float32)
        noisy_t = torch.tensor(noisy, dtype=torch.float32)
        
        w = torch.hann_window(self.n_fft)
        sp_stft = torch.stft(speech_t, n_fft=self.n_fft, hop_length=self.hop_length, window=w, return_complex=True)
        no_stft = torch.stft(noisy_t, n_fft=self.n_fft, hop_length=self.hop_length, window=w, return_complex=True)
        
        sp_mag = torch.abs(sp_stft) # [Bins, Frames]
        no_mag = torch.abs(no_stft)
        
        # 4. Compute Ideal Ratio Mask (IRM) Target
        irm = sp_mag / (sp_mag + torch.abs(no_stft - sp_stft) + 1e-8)
        irm = torch.clamp(irm, 0.0, 1.0)
        
        return {
            "noisy_mag": no_mag,
            "speech_mag": sp_mag,
            "target_mask": irm,
            "clean_audio": speech_t,
            "noisy_audio": noisy_t,
            "snr": target_snr
        }

def train(args):
    print("=" * 70)
    print("MARKUSBLUE — ESP32-S3 OPERATIONAL STUDENT MODEL TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on compute device: {device}")

    # Dataset file collection
    speech_files = glob.glob("datasets/speech/*.wav") + glob.glob("datasets/critical_audio/speech/*.wav")
    suppressible_files = (
        glob.glob("datasets/external_noise/suppressible/*/*.wav") +
        glob.glob("datasets/background_noise/*.wav") +
        glob.glob("datasets/gunshot/*.wav") +
        glob.glob("datasets/other_impulse/*.wav")
    )
    critical_cues = (
        glob.glob("datasets/critical_audio/alarms/*.wav") +
        glob.glob("datasets/critical_audio/sirens/*.wav") +
        glob.glob("datasets/critical_audio/footsteps/*.wav") +
        glob.glob("datasets/critical_audio/movement/*.wav") +
        glob.glob("datasets/critical_audio/radio_communication/*.wav") +
        glob.glob("datasets/critical_audio/environmental_cues/*.wav")
    )
    
    print(f"[*] Found {len(speech_files)} speech files, {len(critical_cues)} critical audio cues, and {len(suppressible_files)} suppressible noise assets.")
    
    # Train / Val Split (85% / 15%)
    random.seed(42)
    random.shuffle(speech_files)
    split_idx = int(0.85 * len(speech_files))
    train_speech = speech_files[:split_idx]
    val_speech = speech_files[split_idx:]
    
    train_dataset = TacticalSpeechNoiseDataset(train_speech, suppressible_files, critical_cues, sr=args.sr, snr_range=(-15.0, 20.0))
    val_dataset = TacticalSpeechNoiseDataset(val_speech, suppressible_files, critical_cues, sr=args.sr, snr_range=(-15.0, 20.0))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Instantiate Model & Loss
    model = MARKUSBLUEStudentEnhancer(n_fft=256, hop_length=64, hidden_dim=32).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Initialized MARKUSBLUEStudentEnhancer: {param_count:,} trainable parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    mask_loss_fn = nn.MSELoss()
    
    os.makedirs("models", exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch in train_loader:
            noisy_mag = batch["noisy_mag"].to(device)
            speech_mag = batch["speech_mag"].to(device)
            target_mask = batch["target_mask"].to(device)
            
            optimizer.zero_grad()
            
            pred_mask = model(noisy_mag) # [Batch, Bins, Frames]
            est_mag = noisy_mag * pred_mask
            
            l_mask = mask_loss_fn(pred_mask, target_mask)
            l_mag = F.l1_loss(est_mag, speech_mag)
            
            # Speech & Critical Cue Preservation Penalty
            speech_active = (speech_mag > 0.04).float()
            under_est = F.relu(target_mask - pred_mask) * speech_active
            l_preserve = torch.mean(under_est ** 2)
            
            total_loss = l_mask + 2.0 * l_mag + 4.0 * l_preserve
            total_loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()
            
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                noisy_mag = batch["noisy_mag"].to(device)
                speech_mag = batch["speech_mag"].to(device)
                target_mask = batch["target_mask"].to(device)
                
                pred_mask = model(noisy_mag)
                est_mag = noisy_mag * pred_mask
                
                l_mask = mask_loss_fn(pred_mask, target_mask)
                l_mag = F.l1_loss(est_mag, speech_mag)
                
                loss = l_mask + 2.0 * l_mag
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (LR: {scheduler.get_last_lr()[0]:.6f})")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0]
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "n_fft": 256,
                    "hop_length": 64,
                    "hidden_dim": 32,
                    "sr": args.sr
                }
            }, "models/markusblue_esp32s3_best.pt")
            print(f"  --> Saved best model checkpoint to 'models/markusblue_esp32s3_best.pt' (Val Loss: {best_val_loss:.4f})")

    with open("models/markusblue_esp32s3_history.json", "w") as fp:
        json.dump(history, fp, indent=2)

    print("\n[+] Operational Training Complete!")
    print(f"[+] Best Validation Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MARKUSBLUE Operational Student Model for ESP32-S3")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    args = parser.parse_args()
    
    train(args)
