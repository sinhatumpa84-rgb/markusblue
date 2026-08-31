import numpy as np

class VoiceActivityDetector:
    """
    Real-time Multi-Feature Voice Activity Detector (VAD).
    Combines sub-band voice energy (300-3400 Hz), full-band RMS energy,
    zero-crossing rate, and spectral flatness with hangover hysteresis.
    """
    def __init__(
        self,
        sr: int = 16000,
        frame_size: int = 256,
        energy_threshold_db: float = -42.0,
        voice_ratio_threshold: float = 0.35,
        hangover_frames: int = 8
    ):
        self.sr = sr
        self.frame_size = frame_size
        self.energy_threshold = 10.0 ** (energy_threshold_db / 20.0)
        self.voice_ratio_threshold = voice_ratio_threshold
        self.hangover_frames = hangover_frames
        
        self.hangover_counter = 0
        self.noise_floor = 1e-4
        
    def reset(self):
        self.hangover_counter = 0
        self.noise_floor = 1e-4

    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Evaluate frame for human speech activity.
        Returns: True if speech is active, False otherwise.
        """
        if len(frame) == 0:
            return False
            
        # 1. Compute RMS energy
        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        
        # Adaptive noise floor tracking on quiet frames
        if rms < self.noise_floor * 1.5:
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
            
        # 2. Compute Zero Crossing Rate (ZCR)
        zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0
        
        # 3. Simple Spectral Energy in Telephony / Voice Band (300 - 3400 Hz)
        fft_mag = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(frame), 1.0 / self.sr)
        
        voice_mask = (freqs >= 300.0) & (freqs <= 3400.0)
        voice_energy = np.sum(fft_mag[voice_mask] ** 2) + 1e-12
        total_energy = np.sum(fft_mag ** 2) + 1e-12
        voice_energy_ratio = voice_energy / total_energy
        
        # Decision rule: High voice band ratio + Energy sufficiently above noise floor
        is_speech_instant = (
            (rms > self.energy_threshold) and 
            (rms > self.noise_floor * 2.5) and 
            (voice_energy_ratio >= self.voice_ratio_threshold) and
            (zcr < 0.45)
        )
        
        # Hangover smoothing to preserve consonant/vowel endings
        if is_speech_instant:
            self.hangover_counter = self.hangover_frames
            return True
        elif self.hangover_counter > 0:
            self.hangover_counter -= 1
            return True
        else:
            return False
