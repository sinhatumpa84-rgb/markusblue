import math
import numpy as np
import scipy.signal as signal
import librosa
import torch
from typing import Dict, Tuple, Optional

class AudioFeatureExtractor:
    """
    Unified 9-representation research-grade acoustic feature extractor.
    Extracts time-frequency, spectral dynamics, and transient impulse metrics.
    """
    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels_baseline: int = 64,
        n_mels_edge: int = 32,
        n_mfcc: int = 13,
        f_min: float = 50.0,
        f_max: float = 8000.0
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels_baseline = n_mels_baseline
        self.n_mels_edge = n_mels_edge
        self.n_mfcc = n_mfcc
        self.f_min = f_min
        self.f_max = f_max
        
        # Precompute Mel Filterbank basis using librosa
        self.mel_basis_baseline = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels_baseline, fmin=f_min, fmax=f_max
        )
        self.mel_basis_edge = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels_edge, fmin=f_min, fmax=f_max
        )
        
    def extract_log_mel_spectrogram(
        self,
        audio: np.ndarray,
        mode: str = "edge"
    ) -> np.ndarray:
        """
        Extract Log-Mel Spectrogram (dB normalized).
        'edge' mode yields [32, T], 'baseline' yields [64, T].
        """
        mel_basis = self.mel_basis_edge if mode == "edge" else self.mel_basis_baseline
        
        # STFT
        stft = librosa.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window='hann', center=True
        )
        power_spec = np.abs(stft) ** 2
        mel_spec = np.dot(mel_basis, power_spec)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)
        
        # Normalize to [-1.0, 1.0] range
        log_mel = (log_mel + 40.0) / 40.0
        return log_mel.astype(np.float32)

    def extract_mfcc(self, audio: np.ndarray, n_mfcc: Optional[int] = None) -> np.ndarray:
        """Extract Mel-Frequency Cepstral Coefficients (MFCCs)."""
        n_mfcc = n_mfcc or self.n_mfcc
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length,
            fmin=self.f_min, fmax=self.f_max
        )
        return mfccs.astype(np.float32)

    def extract_rms_energy(self, audio: np.ndarray) -> np.ndarray:
        """Extract frame-level RMS energy envelope."""
        rms = librosa.feature.rms(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )
        return rms.squeeze(0).astype(np.float32)

    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """Extract frame-level Zero-Crossing Rate (ZCR)."""
        zcr = librosa.feature.zero_crossing_rate(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )
        return zcr.squeeze(0).astype(np.float32)

    def extract_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """Extract Spectral Centroid (brightness of impulsive explosions)."""
        cent = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return cent.squeeze(0).astype(np.float32)

    def extract_spectral_bandwidth(self, audio: np.ndarray) -> np.ndarray:
        """Extract Spectral Bandwidth (frequency spread)."""
        bw = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return bw.squeeze(0).astype(np.float32)

    def extract_spectral_rolloff(self, audio: np.ndarray, roll_percent: float = 0.85) -> np.ndarray:
        """Extract Spectral Rolloff frequency."""
        ro = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
            roll_percent=roll_percent
        )
        return ro.squeeze(0).astype(np.float32)

    def extract_crest_factor(self, audio: np.ndarray) -> float:
        """
        Compute Peak-to-RMS (Crest Factor) ratio.
        Impulsive gunshots exhibit Crest Factors > 15-25 dB, continuous noise < 10 dB.
        """
        peak = np.max(np.abs(audio))
        rms = math.sqrt(np.mean(audio ** 2)) + 1e-9
        return float(peak / rms)

    def extract_short_time_energy(self, audio: np.ndarray, frame_len: int = 256, hop_len: int = 128) -> np.ndarray:
        """Extract fast short-time windowed energy for transient attack detection."""
        num_frames = (len(audio) - frame_len) // hop_len + 1
        energy = np.zeros(num_frames, dtype=np.float32)
        for i in range(num_frames):
            start = i * hop_len
            frame = audio[start:start + frame_len]
            energy[i] = np.sum(frame ** 2)
        return energy

    def extract_all_9_features(self, audio: np.ndarray) -> Dict:
        """Extract complete 9-representation feature dictionary."""
        return {
            "log_mel_edge": self.extract_log_mel_spectrogram(audio, mode="edge"),
            "log_mel_baseline": self.extract_log_mel_spectrogram(audio, mode="baseline"),
            "mfcc": self.extract_mfcc(audio),
            "rms_energy": self.extract_rms_energy(audio),
            "zcr": self.extract_zero_crossing_rate(audio),
            "spectral_centroid": self.extract_spectral_centroid(audio),
            "spectral_bandwidth": self.extract_spectral_bandwidth(audio),
            "spectral_rolloff": self.extract_spectral_rolloff(audio),
            "crest_factor": self.extract_crest_factor(audio),
            "short_time_energy": self.extract_short_time_energy(audio)
        }

def extract_all_features(audio: np.ndarray, sr: int = 16000) -> Dict:
    extractor = AudioFeatureExtractor(sr=sr)
    return extractor.extract_all_9_features(audio)
