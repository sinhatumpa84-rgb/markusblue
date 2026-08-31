import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import soundfile as sf
from typing import Dict, Optional, Tuple

from src.features.feature_extractor import AudioFeatureExtractor
from src.features.augmentation import TacticalAudioAugmenter

class TacticalAudioDataset(Dataset):
    """
    PyTorch Dataset for Tactical Impulse Detection and Hearing Protection.
    Extracts on-the-fly Log-Mel Spectrogram representations with tactical augmentation.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_mode: str = "edge",  # "edge" (32 mels) or "baseline" (64 mels)
        augment: bool = False,
        sr: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        seed: int = 42
    ):
        self.df = df.reset_index(drop=True)
        self.feature_mode = feature_mode
        self.augment = augment
        self.sr = sr
        
        self.extractor = AudioFeatureExtractor(
            sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels_baseline=64, n_mels_edge=32
        )
        self.augmenter = TacticalAudioAugmenter(sr=sr, seed=seed) if augment else None
        
        # Class mapping
        self.class_to_idx = {
            "DANGEROUS_IMPULSE": 0,
            "NORMAL_SPEECH": 1,
            "BACKGROUND_NOISE": 2,
            "OTHER_IMPULSE": 3
        }
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        class_label = row["class_label"]
        label_idx = self.class_to_idx.get(class_label, 0)
        
        # Load audio
        try:
            audio, sr = sf.read(filepath, dtype='float32')
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        except Exception:
            audio = np.zeros(self.sr, dtype=np.float32)
            
        # Ensure exact 1-second length (16000 samples)
        target_len = self.sr
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        elif len(audio) > target_len:
            audio = audio[:target_len]
            
        # Apply tactical augmentation if training
        if self.augment and self.augmenter is not None:
            is_gunshot = (label_idx == 0)
            audio = self.augmenter.augment(audio, is_gunshot=is_gunshot)
            
        # Extract Log-Mel Spectrogram
        mel = self.extractor.extract_log_mel_spectrogram(audio, mode=self.feature_mode)
        
        # Shape: [1, n_mels, time_frames]
        mel_tensor = torch.from_numpy(mel).unsqueeze(0).float()
        
        meta = {
            "sample_id": row.get("sample_id", f"sample_{idx}"),
            "source_group": row.get("source_group", "unknown"),
            "class_label": class_label
        }
        
        return mel_tensor, label_idx, meta

def get_data_loaders(
    splits_dir: str = "data/splits",
    feature_mode: str = "edge",
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoader instances for train, validation, and test splits."""
    train_df = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "validation.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test.csv"))
    
    train_ds = TacticalAudioDataset(train_df, feature_mode=feature_mode, augment=True)
    val_ds = TacticalAudioDataset(val_df, feature_mode=feature_mode, augment=False)
    test_ds = TacticalAudioDataset(test_df, feature_mode=feature_mode, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
