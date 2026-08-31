import math
import numpy as np

class AutomaticGainControl:
    """
    VAD-Aware Dual-Rate Automatic Gain Control (AGC).
    Restores weak/damped speech to a standard conversational listening level
    while freezing gain adaptation during pauses to prevent background noise explosion.
    """
    def __init__(
        self,
        sr: int = 16000,
        target_rms_dbfs: float = -16.0,
        max_gain_db: float = 24.0,
        min_gain_db: float = -12.0,
        attack_ms: float = 10.0,
        release_ms: float = 250.0,
        noise_gate_dbfs: float = -48.0
    ):
        self.sr = sr
        self.target_rms = 10.0 ** (target_rms_dbfs / 20.0) # e.g. ~0.1585 (-16 dBFS)
        self.max_gain = 10.0 ** (max_gain_db / 20.0)       # e.g. ~15.84 (+24 dB)
        self.min_gain = 10.0 ** (min_gain_db / 20.0)       # e.g. ~0.251 (-12 dB)
        self.noise_gate_rms = 10.0 ** (noise_gate_dbfs / 20.0)
        
        # Time constants
        self.alpha_attack = math.exp(-1.0 / (max(0.001, attack_ms * 1e-3) * sr))
        self.alpha_release = math.exp(-1.0 / (max(0.001, release_ms * 1e-3) * sr))
        
        self.current_gain = 1.0
        self.smoothed_rms = self.target_rms
        
    def reset(self):
        self.current_gain = 1.0
        self.smoothed_rms = self.target_rms

    def process_frame(self, audio_frame: np.ndarray, is_speech: bool = True) -> np.ndarray:
        """
        Process an audio block through VAD-aware gain adaptation.
        """
        if len(audio_frame) == 0:
            return audio_frame
            
        frame_rms = np.sqrt(np.mean(audio_frame ** 2) + 1e-12)
        
        # Only update target gain when speech is actively detected and above noise gate
        if is_speech and (frame_rms > self.noise_gate_rms):
            self.smoothed_rms = 0.9 * self.smoothed_rms + 0.1 * frame_rms
            # Desired gain to bring smoothed RMS to target RMS
            desired_gain = self.target_rms / (self.smoothed_rms + 1e-6)
            desired_gain = float(np.clip(desired_gain, self.min_gain, self.max_gain))
        else:
            # During silence / non-speech, slowly decay gain toward nominal unity (1.0)
            desired_gain = 0.995 * self.current_gain + 0.005 * 1.0
            
        # Asymmetric smoothing across samples in the frame
        out = np.zeros_like(audio_frame)
        for i, s in enumerate(audio_frame):
            if desired_gain < self.current_gain:
                # Fast attack (reduce gain quickly when too loud)
                self.current_gain = self.alpha_attack * self.current_gain + (1.0 - self.alpha_attack) * desired_gain
            else:
                # Slow release (gradual increase to avoid breathing/pumping)
                self.current_gain = self.alpha_release * self.current_gain + (1.0 - self.alpha_release) * desired_gain
                
            out[i] = s * self.current_gain
            
        return out
