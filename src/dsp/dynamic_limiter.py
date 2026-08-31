import math
import numpy as np

class DynamicTransientLimiter:
    """
    Sub-millisecond Deterministic Transient Limiter and Dynamic Range Compressor.
    Provides immediate zero-delay peak attenuation on dangerous acoustic spikes (<0.5 ms)
    and smooth exponential recovery (80 ms) for hearing safety.
    """
    def __init__(
        self,
        sr: int = 16000,
        attack_ms: float = 0.5,
        release_ms: float = 80.0,
        max_attenuation_db: float = -28.0,
        threshold_db: float = -12.0
    ):
        self.sr = sr
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.max_attenuation_db = max_attenuation_db
        self.threshold_linear = 10.0 ** (threshold_db / 20.0) # ~0.251 (-12 dB)
        self.max_attenuation_linear = 10.0 ** (max_attenuation_db / 20.0) # ~0.0398 (-28 dB)
        
        # Release time constant (smooth exponential recovery)
        self.alpha_release = math.exp(-1.0 / (max(0.001, release_ms * 1e-3) * sr))
        # Attack time constant (for peak smoothing when not instantly hard-clamped)
        self.alpha_attack = math.exp(-1.0 / (max(0.0001, attack_ms * 1e-3) * sr))
        
        self.envelope = 0.0
        self.current_gain = 1.0
        
    def reset(self):
        self.envelope = 0.0
        self.current_gain = 1.0

    def process_sample(self, sample: float, force_protect: bool = False) -> float:
        """
        Process a single audio sample in real time.
        Deterministic sub-millisecond peak limiting with instant attack and smooth release.
        """
        abs_val = abs(sample)
        
        # 1. Instant peak detection on rising edge, smooth release on falling edge
        if abs_val > self.envelope:
            # Instantaneous peak tracking ensures zero-delay clamping on blast onset
            self.envelope = abs_val
        else:
            self.envelope = self.alpha_release * self.envelope + (1.0 - self.alpha_release) * abs_val
            
        # 2. Target gain computation
        if force_protect or (self.envelope > self.threshold_linear):
            # Proportional attenuation or maximum safe clamping
            target_gain = self.max_attenuation_linear
        else:
            target_gain = 1.0
            
        # 3. Fast attack clamp, smooth release recovery
        if target_gain < self.current_gain:
            # Instantaneous attack transition (< 0.1 ms)
            self.current_gain = target_gain
        else:
            # Smooth exponential recovery back to unity gain
            self.current_gain = self.alpha_release * self.current_gain + (1.0 - self.alpha_release) * target_gain
            
        output = sample * self.current_gain
        return float(np.clip(output, -1.0, 1.0))

    def process_block(self, audio_block: np.ndarray, force_protect: bool = False) -> np.ndarray:
        """Process an audio block/buffer in real time."""
        out = np.zeros_like(audio_block)
        for i in range(len(audio_block)):
            out[i] = self.process_sample(float(audio_block[i]), force_protect=force_protect)
        return out
