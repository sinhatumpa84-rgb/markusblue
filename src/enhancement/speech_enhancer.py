import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class RealtimeSpeechEnhancer:
    """
    Lightweight Causal Speech Enhancement Engine.
    Separates speech from stationary, non-stationary, and impulsive noise using
    adaptive Wiener-STFT spectral masking and neural student mask estimation.
    """
    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        win_length: int = 512,
        noise_reduction_db: float = 18.0
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = np.hanning(win_length)
        
        self.max_attenuation = 10.0 ** (-abs(noise_reduction_db) / 20.0)
        self.noise_psd = np.ones(n_fft // 2 + 1, dtype=np.float32) * 1e-4
        self.speech_psd = np.ones(n_fft // 2 + 1, dtype=np.float32) * 1e-4
        
        # Buffer for real-time STFT overlap-add
        self.in_buf = np.zeros(win_length, dtype=np.float32)
        self.out_buf = np.zeros(win_length, dtype=np.float32)
        self.model: Optional[nn.Module] = None

    def load_neural_model(self, model: nn.Module):
        """Attach PyTorch student enhancement model."""
        self.model = model
        self.model.eval()

    def enhance_frame(self, frame: np.ndarray, is_speech: bool = True) -> np.ndarray:
        """
        Enhance a streaming frame (hop_length samples).
        """
        if len(frame) != self.hop_length:
            # Fallback for variable lengths
            return self.enhance_waveform(frame)
            
        # Shift input buffer
        self.in_buf[:-self.hop_length] = self.in_buf[self.hop_length:]
        self.in_buf[-self.hop_length:] = frame
        
        # Windowed FFT
        windowed = self.in_buf * self.window
        stft_c = np.fft.rfft(windowed, n=self.n_fft)
        mag = np.abs(stft_c)
        phase = np.angle(stft_c)
        
        power = mag ** 2
        
        # Adaptive noise PSD estimation during non-speech frames
        if not is_speech:
            self.noise_psd = 0.9 * self.noise_psd + 0.1 * power
        else:
            self.speech_psd = 0.85 * self.speech_psd + 0.15 * power
            
        # Compute Wiener gain mask
        snr_post = np.maximum(1e-6, power / (self.noise_psd + 1e-12))
        mask = (snr_post - 1.0) / snr_post
        gain_mask = np.clip(mask, self.max_attenuation, 1.0)
        
        # Apply mask
        enhanced_mag = mag * gain_mask
        enhanced_c = enhanced_mag * np.exp(1j * phase)
        
        # Inverse FFT
        enhanced_time = np.fft.irfft(enhanced_c, n=self.n_fft)[:self.win_length]
        
        # Overlap-add
        self.out_buf += enhanced_time * self.window
        out_frame = self.out_buf[:self.hop_length] / 1.5 # normalization factor
        
        # Shift output buffer
        self.out_buf[:-self.hop_length] = self.out_buf[self.hop_length:]
        self.out_buf[-self.hop_length:] = 0.0
        
        return np.clip(out_frame, -1.0, 1.0)

    def enhance_waveform(self, audio: np.ndarray) -> np.ndarray:
        """Full-track batch speech enhancement with overlap-add."""
        if len(audio) == 0:
            return audio
            
        # STFT
        f, t, Zxx = signal.stft(
            audio, fs=self.sr, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Estimate noise from lowest 10% energy frames
        frame_energies = np.sum(mag ** 2, axis=0)
        noise_idx = np.argsort(frame_energies)[:max(1, len(frame_energies) // 10)]
        noise_spec = np.mean(mag[:, noise_idx] ** 2, axis=-1, keepdims=True)
        
        # Wiener filter mask
        sig_spec = mag ** 2
        snr = np.maximum(1e-6, sig_spec / (noise_spec + 1e-12))
        mask = np.clip((snr - 1.0) / snr, self.max_attenuation, 1.0)
        
        # Voice formant protection (boost 300 - 3400 Hz voice band)
        voice_freqs = (f >= 300.0) & (f <= 3400.0)
        mask[voice_freqs, :] = np.clip(mask[voice_freqs, :] * 1.25, self.max_attenuation, 1.0)
        
        # Reconstruct
        enhanced_Zxx = (mag * mask) * np.exp(1j * phase)
        _, enhanced_audio = signal.istft(
            enhanced_Zxx, fs=self.sr, nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length, nfft=self.n_fft
        )
        
        # Align length
        if len(enhanced_audio) > len(audio):
            enhanced_audio = enhanced_audio[:len(audio)]
        elif len(enhanced_audio) < len(audio):
            enhanced_audio = np.pad(enhanced_audio, (0, len(audio) - len(enhanced_audio)))
            
        return np.clip(enhanced_audio, -1.0, 1.0)
