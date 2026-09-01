import numpy as np

class ESP82VoiceActivityDetector:
    """
    Lightweight energy and spectral flux based VAD for ESP8266.
    Uses running noise floor tracking to distinguish speech from ambient noise.
    """
    def __init__(self, alpha_noise: float = 0.98, snr_threshold_db: float = 4.5):
        self.alpha_noise = alpha_noise
        self.snr_threshold_db = snr_threshold_db
        self.noise_energy = 1e-4
        self.speech_energy = 1e-3
        self.speech_state = False

    def update(self, frame_energy: float) -> bool:
        if frame_energy < self.noise_energy * 2.0:
            # Update background noise estimate
            self.noise_energy = self.alpha_noise * self.noise_energy + (1.0 - self.alpha_noise) * frame_energy
        else:
            self.speech_energy = 0.95 * self.speech_energy + 0.05 * frame_energy
            
        snr_db = 10.0 * np.log10(max(1e-6, frame_energy) / max(1e-6, self.noise_energy))
        self.speech_state = (snr_db > self.snr_threshold_db)
        return self.speech_state

class ESP82AutomaticGainControl:
    """
    Smart Audio AGC with noise gating.
    Compensates for speech attenuation during noise suppression without boosting the residual noise floor.
    """
    def __init__(
        self,
        target_level: float = 0.35, # Target RMS speech amplitude
        max_gain: float = 4.0,       # +12 dB max gain boost
        min_gain: float = 0.5,       # -6 dB max attenuation
        attack_rate: float = 0.05,
        decay_rate: float = 0.005
    ):
        self.target_level = target_level
        self.max_gain = max_gain
        self.min_gain = min_gain
        self.attack_rate = attack_rate
        self.decay_rate = decay_rate
        self.current_gain = 1.0

    def process_frame(self, frame: np.ndarray, is_speech: bool) -> np.ndarray:
        frame_rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        
        if is_speech and frame_rms > 1e-3:
            # Calculate desired gain to reach target level
            desired_gain = np.clip(self.target_level / frame_rms, self.min_gain, self.max_gain)
            
            # Smooth gain update
            if desired_gain < self.current_gain:
                self.current_gain += self.attack_rate * (desired_gain - self.current_gain)
            else:
                self.current_gain += self.decay_rate * (desired_gain - self.current_gain)
        else:
            # Slowly decay gain back to unity when no speech is present (prevent noise pumping)
            self.current_gain += self.decay_rate * (1.0 - self.current_gain)
            
        return frame * self.current_gain

class ESP82PeakLimiter:
    """
    Low-latency Peak Limiter / Soft Knee Clipper.
    Prevents digital clipping and harsh artifacts.
    """
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Fast tanh-like soft clipping near threshold
        clipped = np.copy(frame)
        mask = np.abs(clipped) > self.threshold
        if np.any(mask):
            excess = np.abs(clipped[mask]) - self.threshold
            compressed = self.threshold + (1.0 - self.threshold) * np.tanh(excess / (1.0 - self.threshold + 1e-6))
            clipped[mask] = np.sign(clipped[mask]) * compressed
        return np.clip(clipped, -1.0, 1.0)

class ESP82AudioSynthesizer:
    """
    Overlap-add IFFT synthesis buffer.
    """
    def __init__(self, n_fft: int = 128, hop_length: int = 64):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.window = np.hanning(self.win_length).astype(np.float32)
        self.out_buffer = np.zeros(self.win_length, dtype=np.float32)

    def synthesize_frame(self, mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
        # Complex spectrum
        c_spec = mag * np.exp(1j * phase)
        
        # Real inverse FFT
        time_frame = np.fft.irfft(c_spec, n=self.n_fft)[:self.win_length]
        
        # Overlap-add
        self.out_buffer += time_frame * self.window
        out_frame = self.out_buffer[:self.hop_length] / 1.5 # Normalization factor for 50% Hanning overlap
        
        # Shift out buffer
        self.out_buffer[:-self.hop_length] = self.out_buffer[self.hop_length:]
        self.out_buffer[-self.hop_length:] = 0.0
        
        return np.clip(out_frame, -1.0, 1.0)
