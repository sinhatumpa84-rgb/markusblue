import math
import numpy as np
import scipy.signal as signal
from typing import Dict, Optional, Tuple

class TacticalAudioAugmenter:
    """
    Tactical acoustic augmentation engine designed for edge hearing protection models.
    Simulates field conditions without destroying critical transient characteristics of impulse sounds.
    """
    def __init__(
        self,
        sr: int = 16000,
        snr_min_db: float = -5.0,
        snr_max_db: float = 20.0,
        gain_min_db: float = -6.0,
        gain_max_db: float = 3.0,
        time_shift_max_ms: int = 100,
        reverb_prob: float = 0.35,
        mic_sim_prob: float = 0.40,
        clipping_prob: float = 0.20,
        wind_noise_prob: float = 0.30,
        seed: Optional[int] = 42
    ):
        self.sr = sr
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.gain_min_db = gain_min_db
        self.gain_max_db = gain_max_db
        self.time_shift_max_samples = int((time_shift_max_ms / 1000.0) * sr)
        self.reverb_prob = reverb_prob
        self.mic_sim_prob = mic_sim_prob
        self.clipping_prob = clipping_prob
        self.wind_noise_prob = wind_noise_prob
        self.rng = np.random.RandomState(seed)

    def apply_gain_variation(self, audio: np.ndarray) -> np.ndarray:
        """Apply moderate gain variation within controlled bounds."""
        gain_db = self.rng.uniform(self.gain_min_db, self.gain_max_db)
        gain_linear = 10.0 ** (gain_db / 20.0)
        return np.clip(audio * gain_linear, -1.0, 1.0)

    def apply_time_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply small random temporal translation."""
        shift = self.rng.randint(-self.time_shift_max_samples, self.time_shift_max_samples)
        return np.roll(audio, shift)

    def apply_background_noise(self, audio: np.ndarray, snr_db: Optional[float] = None) -> np.ndarray:
        """Mix tactical background noise at specified/random SNR."""
        if snr_db is None:
            snr_db = self.rng.uniform(self.snr_min_db, self.snr_max_db)
            
        signal_power = np.mean(audio ** 2) + 1e-10
        target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        
        noise_type = self.rng.choice(["pink", "vehicle", "white", "bandpass"])
        n_samples = len(audio)
        
        if noise_type == "pink":
            # 1/f ambient noise
            white = self.rng.randn(n_samples)
            b, a = signal.butter(1, 400 / (self.sr / 2), btype='low')
            noise = signal.lfilter(b, a, white)
        elif noise_type == "vehicle":
            # Engine harmonic rumble
            rpm = self.rng.uniform(30, 60)
            t = np.linspace(0, n_samples / self.sr, n_samples, endpoint=False)
            noise = np.sin(2 * np.pi * rpm * t) + 0.5 * np.sin(2 * np.pi * 2 * rpm * t) + 0.2 * self.rng.randn(n_samples)
        elif noise_type == "bandpass":
            # Tactical radio passband hiss
            white = self.rng.randn(n_samples)
            sos = signal.butter(2, [300, 3400], btype='bandpass', fs=self.sr, output='sos')
            noise = signal.sosfilt(sos, white)
        else:
            noise = self.rng.randn(n_samples)
            
        current_noise_power = np.mean(noise ** 2) + 1e-10
        scaled_noise = noise * math.sqrt(target_noise_power / current_noise_power)
        
        mixed = audio + scaled_noise
        return np.clip(mixed, -1.0, 1.0)

    def apply_mic_frequency_response(self, audio: np.ndarray) -> np.ndarray:
        """Simulate MEMS (INMP441) microphone frequency curve with high-pass cut and HF resonance."""
        # High-pass cut below 80 Hz
        sos_hp = signal.butter(2, 80, btype='highpass', fs=self.sr, output='sos')
        filtered = signal.sosfilt(sos_hp, audio)
        
        # Slight presence boost around 4.5 kHz
        sos_boost = signal.butter(1, [4000, 5500], btype='bandpass', fs=self.sr, output='sos')
        presence = signal.sosfilt(sos_boost, audio) * 0.15
        
        return np.clip(filtered + presence, -1.0, 1.0)

    def apply_synthetic_reverberation(self, audio: np.ndarray) -> np.ndarray:
        """Simulate tactical acoustic enclosure / indoor vs outdoor wall reflection."""
        n_taps = int(self.sr * self.rng.uniform(0.02, 0.08)) # 20ms to 80ms room response
        decay = np.exp(-np.linspace(0, 6.0, n_taps))
        impulse_response = self.rng.randn(n_taps) * decay
        impulse_response[0] = 1.0 # Direct path
        impulse_response = impulse_response / np.sum(np.abs(impulse_response))
        
        reverbed = signal.convolve(audio, impulse_response, mode='same')
        return np.clip(reverbed, -1.0, 1.0)

    def apply_clipping_distortion(self, audio: np.ndarray) -> np.ndarray:
        """Simulate acoustic transducer saturation from extreme sound pressure level."""
        threshold = self.rng.uniform(0.70, 0.90)
        clipped = np.clip(audio, -threshold, threshold) / threshold
        return clipped

    def apply_wind_sensor_noise(self, audio: np.ndarray) -> np.ndarray:
        """Simulate aerodynamic low-frequency turbulent wind buffeting."""
        n_samples = len(audio)
        t = np.linspace(0, n_samples / self.sr, n_samples, endpoint=False)
        wind_mod = 0.5 * (1 + np.sin(2 * np.pi * self.rng.uniform(0.5, 3.0) * t))
        white = self.rng.randn(n_samples)
        sos = signal.butter(2, 120, btype='lowpass', fs=self.sr, output='sos')
        wind = signal.sosfilt(sos, white) * wind_mod * 0.25
        return np.clip(audio + wind, -1.0, 1.0)

    def augment(self, audio: np.ndarray, is_gunshot: bool = False) -> np.ndarray:
        """Apply random combination of realistic tactical augmentations."""
        aug_audio = audio.copy()
        
        # 1. Gain
        aug_audio = self.apply_gain_variation(aug_audio)
        
        # 2. Time Shift
        aug_audio = self.apply_time_shift(aug_audio)
        
        # 3. Background Noise
        if self.rng.rand() < 0.60:
            aug_audio = self.apply_background_noise(aug_audio)
            
        # 4. Mic simulation
        if self.rng.rand() < self.mic_sim_prob:
            aug_audio = self.apply_mic_frequency_response(aug_audio)
            
        # 5. Reverb
        if self.rng.rand() < self.reverb_prob:
            aug_audio = self.apply_synthetic_reverberation(aug_audio)
            
        # 6. Wind noise
        if self.rng.rand() < self.wind_noise_prob:
            aug_audio = self.apply_wind_sensor_noise(aug_audio)
            
        # 7. Clipping / Saturation
        if is_gunshot and (self.rng.rand() < self.clipping_prob):
            aug_audio = self.apply_clipping_distortion(aug_audio)
            
        return aug_audio.astype(np.float32)
