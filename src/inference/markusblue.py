import os
import numpy as np
import torch
from typing import Optional, Union

from src.enhancement.speech_enhancer import RealtimeSpeechEnhancer
from src.vad.voice_activity_detector import VoiceActivityDetector
from src.agc.automatic_gain_control import AutomaticGainControl
from src.compressor.dynamic_range_compressor import DynamicRangeCompressor
from src.limiter.peak_limiter import PeakSafetyLimiter

class MARKUSBLUE:
    """
    High-Level MARKUSBLUE Audio Intelligence & Speech Enhancement API.
    Combines AI speech separation with VAD-aware AGC, dynamic range compression,
    and lookahead peak safety limiting.
    """
    def __init__(
        self,
        model_path: Optional[str] = "models/markusblue_final.pt",
        sr: int = 16000,
        target_rms_dbfs: float = -16.0,
        enable_agc: bool = True,
        enable_compressor: bool = True,
        enable_limiter: bool = True
    ):
        self.sr = sr
        self.enable_agc = enable_agc
        self.enable_compressor = enable_compressor
        self.enable_limiter = enable_limiter
        
        self.enhancer = RealtimeSpeechEnhancer(sr=sr)
        self.vad = VoiceActivityDetector(sr=sr)
        self.agc = AutomaticGainControl(sr=sr, target_rms_dbfs=target_rms_dbfs)
        self.compressor = DynamicRangeCompressor(sr=sr)
        self.limiter = PeakSafetyLimiter(sr=sr)
        
        if model_path and os.path.exists(model_path):
            try:
                from src.training.student_model import MARKUSBLUEStudentEnhancer
                model = MARKUSBLUEStudentEnhancer()
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                self.enhancer.load_neural_model(model)
            except Exception as e:
                print(f"[WARN] Failed to load neural weights ({e}), using adaptive Wiener DSP fallback.")

    def enhance(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance a 1D audio waveform.
        Pipeline: Input -> AI Enhancement -> VAD -> AGC -> DRC -> Limiter.
        """
        if len(audio) == 0:
            return audio
            
        # 1. AI Speech Separation / Enhancement
        enhanced = self.enhancer.enhance_waveform(audio)
        
        # 2. VAD Detection
        is_speech = self.vad.process_frame(enhanced[:256])
        
        # 3. Automatic Gain Control (Loudness Restoration)
        if self.enable_agc:
            enhanced = self.agc.process_frame(enhanced, is_speech=is_speech)
            
        # 4. Dynamic Range Compressor
        if self.enable_compressor:
            enhanced = self.compressor.process_frame(enhanced)
            
        # 5. Lookahead Peak Limiter
        if self.enable_limiter:
            enhanced = self.limiter.process_frame(enhanced)
            
        return enhanced
