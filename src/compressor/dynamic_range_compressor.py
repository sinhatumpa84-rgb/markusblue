import math
import numpy as np

class DynamicRangeCompressor:
    """
    Feed-Forward Soft-Knee Dynamic Range Compressor (DRC).
    Controls high dynamic variance in soldier speech, boosting whisper/quiet consonants
    while smoothly compressing loud vocal peaks.
    """
    def __init__(
        self,
        sr: int = 16000,
        threshold_db: float = -18.0,
        ratio: float = 3.0,
        knee_width_db: float = 4.0,
        attack_ms: float = 5.0,
        release_ms: float = 60.0,
        makeup_gain_db: float = 3.0
    ):
        self.sr = sr
        self.threshold_db = threshold_db
        self.ratio = max(1.0, ratio)
        self.knee_width_db = max(0.0, knee_width_db)
        self.makeup_gain = 10.0 ** (makeup_gain_db / 20.0)
        
        self.alpha_attack = math.exp(-1.0 / (max(0.001, attack_ms * 1e-3) * sr))
        self.alpha_release = math.exp(-1.0 / (max(0.001, release_ms * 1e-3) * sr))
        
        self.envelope = 0.0
        
    def reset(self):
        self.envelope = 0.0

    def _compute_gain_reduction_db(self, in_db: float) -> float:
        """Calculate soft-knee gain reduction characteristic."""
        t = self.threshold_db
        w = self.knee_width_db
        r = self.ratio
        
        if 2.0 * (in_db - t) < -w:
            # Below knee: 1:1 unity slope
            out_db = in_db
        elif 2.0 * abs(in_db - t) <= w:
            # Inside soft knee region (quadratic smoothing)
            out_db = in_db + ((1.0 / r - 1.0) * ((in_db - t + w / 2.0) ** 2)) / (2.0 * w)
        else:
            # Above knee: 1:R compression slope
            out_db = t + (in_db - t) / r
            
        return out_db - in_db # negative dB gain

    def process_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression to audio frame."""
        if len(audio_frame) == 0:
            return audio_frame
            
        out = np.zeros_like(audio_frame)
        for i, s in enumerate(audio_frame):
            abs_s = abs(s)
            
            # Envelope detector
            if abs_s > self.envelope:
                self.envelope = self.alpha_attack * self.envelope + (1.0 - self.alpha_attack) * abs_s
            else:
                self.envelope = self.alpha_release * self.envelope + (1.0 - self.alpha_release) * abs_s
                
            env_db = 20.0 * math.log10(max(1e-6, self.envelope))
            gr_db = self._compute_gain_reduction_db(env_db)
            gr_linear = 10.0 ** (gr_db / 20.0)
            
            out[i] = s * gr_linear * self.makeup_gain
            
        return out
