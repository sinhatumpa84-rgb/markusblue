import os
import time
import json
import random
import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.training.esp82_student_model import MARKUSBLUE_ESP82_Student

class ESP82SpeechNoiseDataset(Dataset):
    """
    Dynamic Online Mixture Dataset for ESP8266 Speech Enhancement.
    """
    def __init__(
        self,
        speech_files: list,
        noise_files: list,
        sr: int = 8000,
        n_fft: int = 128,
        hop_length: int = 64,
        duration_samples: int = 8000, # 1.0 second @ 8kHz
        snr_levels: list = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    ):
        self.speech_files = speech_files
        self.noise_files = noise_files
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration_samples
        self.snr_levels = snr_levels
        self.window = np.hanning(n_fft).astype(np.float32)

    def __len__(self):
        return len(self.speech_files)

    def _load_audio(self, path: str) -> np.ndarray:
        data, file_sr = sf.read(path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1) # Mono
        # Simple decimation if original is 16kHz
        if file_sr == 16000 and self.sr == 8000:
            data = data[::2]
        if len(data) < self.duration:
            data = np.pad(data, (0, self.duration - len(data)))
        else:
            data = data[:self.duration]
        return data.astype(np.float32)

    def __getitem__(self, idx):
        # 1. Load clean speech
        clean_speech = self._load_audio(self.speech_files[idx])
        
        # 2. Load random noise
        noise_path = random.choice(self.noise_files)
        noise = self._load_audio(noise_path)
        
        # 3. Mix at random SNR level
        snr_db = random.choice(self.snr_levels)
        speech_pwr = np.mean(clean_speech ** 2) + 1e-10
        noise_pwr = np.mean(noise ** 2) + 1e-10
        
        target_noise_pwr = speech_pwr / (10.0 ** (snr_db / 10.0))
        noise_scaled = noise * np.sqrt(target_noise_pwr / noise_pwr)
        
        noisy_mix = clean_speech + noise_scaled
        
        # Normalize mix
        max_val = max(1e-4, np.max(np.abs(noisy_mix)))
        clean_norm = clean_speech / max_val
        noisy_norm = noisy_mix / max_val
        
        # 4. Compute Spectrograms using STFT
        # Shape: [num_bins, num_frames]
        noisy_frames = np.lib.stride_tricks.sliding_window_view(
            np.pad(noisy_norm, (0, self.hop_length)), self.n_fft)[::self.hop_length] * self.window
        clean_frames = np.lib.stride_tricks.sliding_window_view(
            np.pad(clean_norm, (0, self.hop_length)), self.n_fft)[::self.hop_length] * self.window
            
        noisy_stft = np.abs(np.fft.rfft(noisy_frames, n=self.n_fft, axis=-1)).T.astype(np.float32)
        clean_stft = np.abs(np.fft.rfft(clean_frames, n=self.n_fft, axis=-1)).T.astype(np.float32)
        
        # Ideal Ratio Mask target (Teacher Target)
        ideal_mask = np.clip((clean_stft ** 2) / (noisy_stft ** 2 + 1e-6), 0.0, 1.0).astype(np.float32)
        
        return (
            torch.from_numpy(noisy_stft),
            torch.from_numpy(clean_stft),
            torch.from_numpy(ideal_mask),
            torch.from_numpy(noisy_norm),
            torch.from_numpy(clean_norm),
            snr_db
        )

def train_esp82():
    parser = argparse.ArgumentParser(description="Train MARKUSBLUE ESP82 Student Model")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--smoke_test", action="store_true", help="Run 100-sample smoke test first")
    args = parser.parse_args()

    print("==================================================")
    print("MARKUSBLUE — ESP82 / ESP8266 Student Model Training")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")
    
    # Gather speech and noise dataset files
    speech_files = []
    noise_files = []
    
    for root, _, files in os.walk("datasets/speech"):
        for f in files:
            if f.endswith(".wav"):
                speech_files.append(os.path.join(root, f))
                
    for n_dir in ["datasets/background_noise", "datasets/gunshot", "datasets/other_impulse"]:
        for root, _, files in os.walk(n_dir):
            for f in files:
                if f.endswith(".wav"):
                    noise_files.append(os.path.join(root, f))
                    
    print(f"[*] Found {len(speech_files)} speech files and {len(noise_files)} noise files.")
    
    if args.smoke_test:
        print("[!] SMOKE TEST MODE: Limiting dataset to 100 samples.")
        speech_files = speech_files[:100]
        args.epochs = 1
        
    random.shuffle(speech_files)
    split_idx = int(0.85 * len(speech_files))
    train_speech = speech_files[:split_idx]
    val_speech = speech_files[split_idx:]
    
    train_dataset = ESP82SpeechNoiseDataset(train_speech, noise_files)
    val_dataset = ESP82SpeechNoiseDataset(val_speech, noise_files)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Instantiate ESP82 model
    model = MARKUSBLUE_ESP82_Student(num_bins=65, hidden_dim=16).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total Trainable Parameters: {param_count:,}")
    print(f"[*] Estimated FP32 Memory: {param_count * 4 / 1024:.2f} KB")
    print(f"[*] Estimated INT8 Memory: {param_count / 1024:.2f} KB")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss functions: L1 Mask Loss + Spectral Reconstruction Loss
    criterion_mask = nn.MSELoss()
    criterion_spec = nn.L1Loss()
    
    best_val_loss = float("inf")
    os.makedirs("models", exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_snr_gain": []}
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for noisy_stft, clean_stft, ideal_mask, _, _, _ in train_loader:
            noisy_stft = noisy_stft.to(device)
            clean_stft = clean_stft.to(device)
            ideal_mask = ideal_mask.to(device)
            
            optimizer.zero_grad()
            
            pred_mask = model(noisy_stft)
            pred_enhanced_stft = noisy_stft * pred_mask
            
            # Composite Loss = 0.6 * Mask Loss + 0.4 * Enhanced Spectrogram Loss
            loss_mask = criterion_mask(pred_mask, ideal_mask)
            loss_spec = criterion_spec(pred_enhanced_stft, clean_stft)
            loss = 0.6 * loss_mask + 0.4 * loss_spec
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= max(1, len(train_loader))
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_snr_gains = []
        
        with torch.no_grad():
            for noisy_stft, clean_stft, ideal_mask, noisy_audio, clean_audio, snr in val_loader:
                noisy_stft = noisy_stft.to(device)
                clean_stft = clean_stft.to(device)
                ideal_mask = ideal_mask.to(device)
                
                pred_mask = model(noisy_stft)
                pred_enhanced_stft = noisy_stft * pred_mask
                
                loss_mask = criterion_mask(pred_mask, ideal_mask)
                loss_spec = criterion_spec(pred_enhanced_stft, clean_stft)
                val_loss += (0.6 * loss_mask + 0.4 * loss_spec).item()
                
        val_loss /= max(1, len(val_loader))
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss and not args.smoke_test:
            best_val_loss = val_loss
            save_path = "models/markusblue_esp82_student_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  [+] Saved new best model to '{save_path}'")
            
    total_time = time.time() - start_time
    print(f"[*] Training finished in {total_time:.1f}s.")
    
    if not args.smoke_test:
        final_save_path = "models/markusblue_esp82_student_final.pt"
        torch.save(model.state_dict(), final_save_path)
        with open("reports/esp82_training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"[*] Saved final model to '{final_save_path}'")

if __name__ == "__main__":
    train_esp82()
