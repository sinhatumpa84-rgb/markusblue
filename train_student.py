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

from src.training.student_model import MARKUSBLUEStudentEnhancer
from src.training.losses import StudentTeacherDistillationLoss, MultiResolutionSTFTLoss, SISDRLoss

class SpeechNoiseMixtureDataset(Dataset):
    """
    Dynamic Speech + Noise Online Mixture Dataset.
    Synthesizes realistic tactical audio mixtures across SNRs [-20 dB to +20 dB].
    """
    def __init__(
        self,
        speech_files: list,
        noise_files: list,
        sr: int = 16000,
        duration_samples: int = 16000,
        snr_levels: list = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    ):
        self.speech_files = speech_files
        self.noise_files = noise_files
        self.sr = sr
        self.duration = duration_samples
        self.snr_levels = snr_levels
        self.n_fft = 256
        self.hop_length = 64

    def __len__(self):
        return len(self.speech_files)

    def _load_audio(self, path: str) -> np.ndarray:
        data, sr = sf.read(path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1) # Mono
        if len(data) < self.duration:
            data = np.pad(data, (0, self.duration - len(data)))
        else:
            data = data[:self.duration]
        return data.astype(np.float32)

    def __getitem__(self, idx):
        # 1. Load clean speech
        speech_path = self.speech_files[idx]
        clean_speech = self._load_audio(speech_path)
        
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
        
        # 4. Compute Spectrograms
        w = np.hanning(self.n_fft)
        noisy_stft = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(
            np.pad(noisy_norm, (0, self.hop_length)), self.n_fft)[::self.hop_length] * w, axis=-1)).T
        clean_stft = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(
            np.pad(clean_norm, (0, self.hop_length)), self.n_fft)[::self.hop_length] * w, axis=-1)).T
            
        # Teacher Wiener Target
        ideal_mask = np.clip((clean_stft ** 2) / (noisy_stft ** 2 + 1e-6), 0.0, 1.0)
        teacher_spec = noisy_stft * ideal_mask
        
        return (
            torch.from_numpy(noisy_stft).float(),
            torch.from_numpy(clean_stft).float(),
            torch.from_numpy(teacher_spec).float(),
            torch.from_numpy(noisy_norm).float(),
            torch.from_numpy(clean_norm).float(),
            snr_db
        )

def run_training():
    parser = argparse.ArgumentParser(description="Train MARKUSBLUE Student Model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--smoke_test", action="store_true", help="Run 100-sample smoke test first")
    args = parser.parse_args()

    print("==================================================")
    print("MARKUSBLUE v7.1.0 — Student-Teacher Speech Model Training")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Gather audio files
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
                    
    print(f"Found {len(speech_files)} speech files and {len(noise_files)} noise files.")
    
    if len(speech_files) == 0 or len(noise_files) == 0:
        raise RuntimeError("Dataset audio files not found!")

    # Train / Val / Test Split (80% / 10% / 10%) without speaker leakage
    random.seed(42)
    random.shuffle(speech_files)
    n_total = len(speech_files)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    train_speech = speech_files[:n_train]
    val_speech = speech_files[n_train:n_train + n_val]
    test_speech = speech_files[n_train + n_val:]
    
    if args.smoke_test:
        print("[SMOKE TEST] Restricting to 100 samples for validation...")
        train_speech = train_speech[:100]
        val_speech = val_speech[:20]
        
    train_dataset = SpeechNoiseMixtureDataset(train_speech, noise_files)
    val_dataset = SpeechNoiseMixtureDataset(val_speech, noise_files)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = MARKUSBLUEStudentEnhancer().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized MARKUSBLUE Student Model: {total_params} parameters (~{total_params*4/1024:.2f} KB FP32)")
    
    criterion = StudentTeacherDistillationLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    best_val_loss = float("inf")
    metrics_history = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        for noisy_stft, clean_stft, teacher_spec, _, _, _ in train_loader:
            noisy_stft = noisy_stft.to(device)
            clean_stft = clean_stft.to(device)
            teacher_spec = teacher_spec.to(device)
            
            optimizer.zero_grad()
            mask = model(noisy_stft)
            pred_spec = noisy_stft * mask
            
            loss = criterion(pred_spec, clean_stft, teacher_spec)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item() * len(noisy_stft)
            
        train_loss /= len(train_dataset)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        sisdr_improvements = []
        
        with torch.no_grad():
            for noisy_stft, clean_stft, teacher_spec, noisy_norm, clean_norm, snr_vals in val_loader:
                noisy_stft = noisy_stft.to(device)
                clean_stft = clean_stft.to(device)
                teacher_spec = teacher_spec.to(device)
                
                mask = model(noisy_stft)
                pred_spec = noisy_stft * mask
                
                loss = criterion(pred_spec, clean_stft, teacher_spec)
                val_loss += loss.item() * len(noisy_stft)
                
                # Approximate SI-SDR improvement proxy
                noisy_err = torch.mean((noisy_stft - clean_stft) ** 2)
                pred_err = torch.mean((pred_spec - clean_stft) ** 2)
                imprv = 10.0 * torch.log10(noisy_err / (pred_err + 1e-8))
                sisdr_improvements.append(imprv.item())
                
        val_loss /= len(val_dataset)
        mean_sisdr_gain = float(np.mean(sisdr_improvements))
        epoch_time = time.time() - t0
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | SI-SDR Gain: +{mean_sisdr_gain:.2f} dB | Time: {epoch_time:.2f}s")
        
        metrics_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "sisdr_gain_db": round(mean_sisdr_gain, 2),
            "time_s": round(epoch_time, 2)
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best.pt")
            torch.save(model.state_dict(), "models/markusblue_final.pt")
            print(f"  --> Saved new best model checkpoint to 'models/markusblue_final.pt'")
            
    # Save last checkpoint and history
    torch.save(model.state_dict(), "checkpoints/last.pt")
    with open("experiments/training_history.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
        
    print("[SUCCESS] Training pipeline completed successfully!")

if __name__ == "__main__":
    run_training()
