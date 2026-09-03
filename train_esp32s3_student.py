#!/usr/bin/env python3
"""
MARKUSBLUE — ESP32-S3 Student Speech Enhancement Model Trainer
SIH Problem Statement: SIH26052

Trains the causal neural mask estimator (MARKUSBLUEStudentEnhancer) for the ESP32-S3 N16R8.
Evaluates on dynamic SNR mixtures (-15dB to +10dB) and sudden gunshot impulse transients.
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

class TacticalSpeechNoiseDataset(Dataset):
    """
    Dynamic Online Mixture Dataset for Tactical Speech Enhancement.
    Generates realistic combinations of:
      Clean Speech + Background Noise + Gunshot Transients + Mechanical Impact
    Across specified SNR range (-15 dB to +10 dB).
    """
    def __init__(
        self,
        speech_files: list,
        noise_files: list,
        sr: int = 16000,
        duration_samples: int = 16000,
        snr_levels: list = [-15, -10, -5, 0, 5, 10],
        n_fft: int = 256,
        hop_length: int = 64
    ):
        self.speech_files = speech_files
        self.noise_files = noise_files
        self.sr = sr
        self.duration = duration_samples
        self.snr_levels = snr_levels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)

    def __len__(self):
        return len(self.speech_files)

    def _load_audio(self, path: str) -> np.ndarray:
        try:
            data, sr = sf.read(path)
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
        
        # 2. Pick random noise (background or gunshot impulse)
        noise_path = random.choice(self.noise_files)
        noise = self._load_audio(noise_path)
        
        # 3. Calculate RMS & Mix at target SNR
        sp_rms = np.sqrt(np.mean(speech ** 2) + 1e-8)
        no_rms = np.sqrt(np.mean(noise ** 2) + 1e-8)
        
        target_snr = random.choice(self.snr_levels)
        scale = (sp_rms / (no_rms + 1e-8)) / (10.0 ** (target_snr / 20.0))
        scaled_noise = noise * scale
        
        # Optionally add a sudden transient impulse spike (20% chance)
        if random.random() < 0.20:
            impulse_pos = random.randint(2000, 12000)
            impulse_len = min(800, self.duration - impulse_pos)
            spike = (np.random.rand(impulse_len) * 2.0 - 1.0) * sp_rms * 4.0
            scaled_noise[impulse_pos:impulse_pos + impulse_len] += spike
        
        noisy = speech + scaled_noise
        
        # Peak normalization safety
        peak = max(np.max(np.abs(noisy)), 1e-6)
        if peak > 0.95:
            norm_factor = 0.95 / peak
            speech = speech * norm_factor
            noisy = noisy * norm_factor

        # 4. Compute STFTs
        speech_t = torch.tensor(speech, dtype=torch.float32)
        noisy_t = torch.tensor(noisy, dtype=torch.float32)
        
        w = torch.hann_window(self.n_fft)
        sp_stft = torch.stft(speech_t, n_fft=self.n_fft, hop_length=self.hop_length, window=w, return_complex=True)
        no_stft = torch.stft(noisy_t, n_fft=self.n_fft, hop_length=self.hop_length, window=w, return_complex=True)
        
        sp_mag = torch.abs(sp_stft) # [Bins, Frames]
        no_mag = torch.abs(no_stft)
        
        # 5. Compute Ideal Ratio Mask (IRM) Target
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
    print("MARKUSBLUE — ESP32-S3 STUDENT SPEECH ENHANCEMENT TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on compute device: {device}")

    # Dataset file collection
    speech_files = glob.glob("datasets/speech/*.wav")
    gunshot_files = glob.glob("datasets/gunshot/*.wav")
    bg_files = glob.glob("datasets/background_noise/*.wav")
    other_files = glob.glob("datasets/other_impulse/*.wav")
    
    noise_files = gunshot_files + bg_files + other_files
    print(f"[*] Found {len(speech_files)} speech files and {len(noise_files)} noise assets.")
    
    # Train / Val Split (85% / 15%)
    random.seed(42)
    random.shuffle(speech_files)
    split_idx = int(0.85 * len(speech_files))
    train_speech = speech_files[:split_idx]
    val_speech = speech_files[split_idx:]
    
    train_dataset = TacticalSpeechNoiseDataset(train_speech, noise_files, sr=args.sr, snr_levels=[-15, -10, -5, 0, 5, 10])
    val_dataset = TacticalSpeechNoiseDataset(val_speech, noise_files, sr=args.sr, snr_levels=[-15, -10, -5, 0, 5, 10])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Instantiate Model & Loss
    model = MARKUSBLUEStudentEnhancer(n_fft=256, hop_length=64, hidden_dim=32).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Initialized MARKUSBLUEStudentEnhancer: {param_count:,} trainable parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)
    sisdr_loss_fn = SISDRLoss().to(device)
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
            
            # Predict mask
            pred_mask = model(noisy_mag)
            enhanced_mag = noisy_mag * pred_mask
            
            # Multi-objective Loss: Mask MSE + Magnitude L1 + Spectral Loss
            l_mask = mask_loss_fn(pred_mask, target_mask)
            l_spec = F.l1_loss(enhanced_mag, speech_mag)
            
            # Speech preservation penalty: penalize over-attenuating speech bins
            over_suppress = F.relu(target_mask - pred_mask)
            l_speech_pres = torch.mean(over_suppress ** 2)
            
            total_loss = 1.0 * l_mask + 2.0 * l_spec + 1.5 * l_speech_pres
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
                speech_mag = batch["speech_mag"].to(device)
                target_mask = batch["target_mask"].to(device)
                
                pred_mask = model(noisy_mag)
                enhanced_mag = noisy_mag * pred_mask
                
                l_mask = mask_loss_fn(pred_mask, target_mask)
                l_spec = F.l1_loss(enhanced_mag, speech_mag)
                total_loss = l_mask + 2.0 * l_spec
                val_loss += total_loss.item()
                
        val_loss /= len(val_loader)
        epoch_dur = time.time() - start_time
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_dur:.2f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0]
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = "models/markusblue_esp32s3_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model checkpoint to '{save_path}' (Val Loss: {best_val_loss:.4f})")
            
    # Save final model & training history
    final_path = "models/markusblue_esp32s3_final.pt"
    torch.save(model.state_dict(), final_path)
    
    with open("models/markusblue_esp32s3_history.json", "w") as f:
        json.dump(history, f, indent=2)
        
    print("=" * 70)
    print(f"[+] Training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"[+] Model checkpoint: {best_val_loss:.4f} saved at '{save_path}'")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Train MARKUSBLUE ESP32-S3 Speech Enhancer")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--sr", type=int, default=16000, help="Audio sampling rate (Hz)")
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()
