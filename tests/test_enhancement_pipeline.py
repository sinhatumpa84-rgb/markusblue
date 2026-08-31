import pytest
import numpy as np
import torch

from src.vad.voice_activity_detector import VoiceActivityDetector
from src.agc.automatic_gain_control import AutomaticGainControl
from src.compressor.dynamic_range_compressor import DynamicRangeCompressor
from src.limiter.peak_limiter import PeakSafetyLimiter
from src.enhancement.speech_enhancer import RealtimeSpeechEnhancer
from src.inference.markusblue import MARKUSBLUE
from src.training.student_model import MARKUSBLUEStudentEnhancer

class TestEnhancementPipeline:

    def test_vad_active_and_silence_detection(self):
        sr = 16000
        vad = VoiceActivityDetector(sr=sr)
        
        # 1. Pure silence
        silence = np.zeros(256, dtype=np.float32)
        assert not vad.process_frame(silence)
        
        # 2. Simulated speech formant tone (1000 Hz in voice band 300-3400 Hz)
        t = np.arange(256) / sr
        speech_tone = 0.4 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        assert vad.process_frame(speech_tone)
        
    def test_agc_gain_adaptation_and_noise_freeze(self):
        sr = 16000
        agc = AutomaticGainControl(sr=sr, target_rms_dbfs=-16.0, max_gain_db=20.0)
        
        # Quiet speech (-30 dBFS)
        quiet_speech = np.ones(256, dtype=np.float32) * 0.0316
        out_boosted = agc.process_frame(quiet_speech, is_speech=True)
        # Should apply upward gain
        assert np.mean(out_boosted) > np.mean(quiet_speech)
        
        # Low energy noise with is_speech=False -> should not rapidly boost
        noise = np.ones(256, dtype=np.float32) * 1e-4
        out_noise = agc.process_frame(noise, is_speech=False)
        assert np.max(np.abs(out_noise)) < 0.05

    def test_dynamic_range_compressor(self):
        sr = 16000
        drc = DynamicRangeCompressor(sr=sr, threshold_db=-18.0, ratio=4.0, makeup_gain_db=0.0)
        
        # Continuous loud signal to reach steady state
        loud_signal = np.ones(1024, dtype=np.float32) * 0.95
        compressed = drc.process_frame(loud_signal)
        # Steady-state compressed signal should be significantly attenuated below input
        assert np.mean(compressed[256:]) < np.mean(loud_signal[256:])

    def test_peak_safety_limiter_hard_ceiling(self):
        sr = 16000
        limiter = PeakSafetyLimiter(sr=sr, ceiling_dbfs=-0.5)
        
        # Overshooting signal (+6 dBFS / 2.0 amplitude)
        extreme_signal = np.sin(np.linspace(0, 20 * np.pi, 512)).astype(np.float32) * 2.5
        limited = limiter.process_frame(extreme_signal)
        
        # Guaranteed strict bound <= 1.0
        assert np.max(limited) <= 1.0
        assert np.min(limited) >= -1.0

    def test_student_neural_model_forward_shape(self):
        model = MARKUSBLUEStudentEnhancer()
        model.eval()
        dummy_spec = torch.randn(2, 129, 32)
        mask = model(dummy_spec)
        assert mask.shape == (2, 129, 32)
        assert torch.all(mask >= 0.0) and torch.all(mask <= 1.0)

    def test_high_level_markusblue_api(self):
        pipeline = MARKUSBLUE(model_path=None, sr=16000)
        audio = np.random.randn(1024).astype(np.float32) * 0.1
        enhanced = pipeline.enhance(audio)
        assert len(enhanced) == len(audio)
        assert np.max(np.abs(enhanced)) <= 1.0
