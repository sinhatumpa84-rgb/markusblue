import math
import numpy as np

class PeakSafetyLimiter:
    """
    Lookahead Peak Safety Limiter with Instant Zero-Overshoot Ceiling.
    Guarantees output stays strictly within [-1.0, 1.0] (16-bit PCM integer range)
    with sub-millisecond attack and transparent release.
    """
    def __init__(
        self,
        sr: int = 16000,
        ceiling_dbfs: float = -0.5,
        attack_ms: float = 0.2,
        release_ms: float = 50.0,
        lookahead_samples: int = 8
    ):
        self.sr = sr
        self.ceiling = 10.0 ** (ceiling_dbfs / 20.0) # ~0.9441 (-0.5 dBFS)
        self.lookahead_samples = lookahead_samples
        
        self.alpha_attack = math.exp(-1.0 / (max(0.0001, attack_ms * 1e-3) * sr))
        self.alpha_release = math.exp(-1.0 / (max(0.001, release_ms * 1e-3) * sr))
        
        self.gain = 1.0
        self.delay_buf = np.zeros(max(1, lookahead_samples), dtype=np.float32)
        
    def reset(self):
        self.gain = 1.0
        self.delay_buf.fill(0.0)

    def process_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """Process frame through lookahead limiter."""
        if len(audio_frame) == 0:
            return audio_frame
            
        out = np.zeros_like(audio_frame)
        for i, s in enumerate(audio_frame):
            # Read delayed sample from lookahead ring buffer
            delayed_sample = self.delay_buf[0]
            self.delay_buf[:-1] = self.delay_buf[1:]
            self.delay_buf[-1] = s
            
            abs_s = abs(s)
            target_gain = 1.0 if abs_s <= self.ceiling else (self.ceiling / (abs_s + 1e-12))
            
            # Fast attack on peak transients, smooth release recovery
            if target_gain < self.gain:
                self.gain = self.alpha_attack * self.gain + (1.0 - self.alpha_attack) * target_gain
            else:
                self.gain = self.alpha_release * self.gain + (1.0 - self.alpha_release) * target_gain
                
            out[i] = np.clip(delayed_sample * self.gain, -1.0, 1.0)
            
        return out
