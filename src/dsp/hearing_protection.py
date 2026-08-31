import enum
import time
import numpy as np
from typing import Dict, Tuple

from .dynamic_limiter import DynamicTransientLimiter
from .speech_preservation import SpeechPreservationFilter

class ProtectionState(enum.Enum):
    NORMAL = "NORMAL"
    PROTECTION_TRIGGERED = "PROTECTION_TRIGGERED"
    RECOVERY = "RECOVERY"

class HearingProtectionController:
    """
    Tactical Hearing Protection State Machine and DSP Orchestrator.
    Bridges ML detection confidence with deterministic acoustic safety limiting.
    """
    def __init__(
        self,
        sr: int = 16000,
        detection_threshold: float = 0.65,
        recovery_threshold: float = 0.30,
        hold_time_ms: float = 60.0,
        attack_ms: float = 0.5,
        release_ms: float = 80.0,
        max_attenuation_db: float = -28.0
    ):
        self.sr = sr
        self.detection_threshold = detection_threshold
        self.recovery_threshold = recovery_threshold
        self.hold_samples = int((hold_time_ms / 1000.0) * sr)
        
        self.limiter = DynamicTransientLimiter(
            sr=sr, attack_ms=attack_ms, release_ms=release_ms,
            max_attenuation_db=max_attenuation_db
        )
        self.speech_filter = SpeechPreservationFilter(sr=sr)
        
        self.state = ProtectionState.NORMAL
        self.hold_counter = 0
        self.trigger_count = 0
        self.last_probability = 0.0

    def update_state(self, impulse_prob: float) -> ProtectionState:
        """Update protection state machine based on streaming ML impulse probability."""
        self.last_probability = impulse_prob
        
        if self.state == ProtectionState.NORMAL:
            if impulse_prob >= self.detection_threshold:
                self.state = ProtectionState.PROTECTION_TRIGGERED
                self.hold_counter = self.hold_samples
                self.trigger_count += 1
        elif self.state == ProtectionState.PROTECTION_TRIGGERED:
            if impulse_prob >= self.detection_threshold:
                self.hold_counter = self.hold_samples
            else:
                self.hold_counter -= 1
                if self.hold_counter <= 0:
                    self.state = ProtectionState.RECOVERY
        elif self.state == ProtectionState.RECOVERY:
            if impulse_prob >= self.detection_threshold:
                self.state = ProtectionState.PROTECTION_TRIGGERED
                self.hold_counter = self.hold_samples
            elif impulse_prob < self.recovery_threshold:
                self.state = ProtectionState.NORMAL
                
        return self.state

    def process_frame(
        self,
        audio_frame: np.ndarray,
        impulse_prob: float
    ) -> Tuple[np.ndarray, ProtectionState, float]:
        """
        Process incoming audio frame through detection-driven hearing protection DSP.
        Returns: (processed_audio, state, attenuation_gain)
        """
        current_state = self.update_state(impulse_prob)
        is_protecting = (current_state == ProtectionState.PROTECTION_TRIGGERED)
        
        # 1. Deterministic Limiting
        limited_audio = self.limiter.process_block(audio_frame, force_protect=is_protecting)
        
        # 2. Speech Preservation Filtering
        protected_audio = self.speech_filter.process(limited_audio, protection_active=is_protecting)
        
        return protected_audio, current_state, self.limiter.current_gain
