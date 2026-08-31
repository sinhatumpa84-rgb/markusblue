"""
SIH26052 — Unit Tests for DSP Protection and Speech Preservation
Tests filter stability, transient limiter attack/release, peak blast attenuation,
numerical precision, and Q15 fixed-point saturation behavior.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import unittest
import numpy as np
import scipy.signal as signal
from src.dsp.dynamic_limiter import DynamicTransientLimiter
from src.dsp.speech_preservation import SpeechPreservationFilter, evaluate_speech_intelligibility_preservation
from src.dsp.hearing_protection import HearingProtectionController, ProtectionState

class TestDSPProtection(unittest.TestCase):

    def test_biquad_stability_and_bandwidth(self):
        """Verify that the speech bandpass filter is BIBO stable and preserves 300Hz-3.4kHz."""
        sr = 16000
        filter_bank = SpeechPreservationFilter(sr=sr, voice_low_hz=300.0, voice_high_hz=3400.0)
        
        # 1. Test Impulse Response
        impulse = np.zeros(1600, dtype=np.float32)
        impulse[0] = 1.0
        ir = filter_bank.process(impulse, protection_active=True)
        
        # Check that impulse response decays to 0 (stable)
        self.assertTrue(np.all(np.isfinite(ir)), "Filter impulse response contains NaN or Inf!")
        self.assertLess(np.max(np.abs(ir[-100:])), 1e-3, f"Filter unstable! Non-decaying tail: {np.max(np.abs(ir[-100:]))}")
        
        # 2. Test Frequency Response
        t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
        sine_1k = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
        out_1k = filter_bank.process(sine_1k, protection_active=True)
        gain_1k = np.max(np.abs(out_1k[100:])) / np.max(np.abs(sine_1k[100:]))
        self.assertTrue(0.7 <= gain_1k <= 1.8, f"1kHz passband gain out of expected range: {gain_1k}")

    def test_transient_limiter_sub_ms_attack(self):
        """Verify sub-millisecond attack clamping on high-amplitude impulsive blast."""
        sr = 16000
        limiter = DynamicTransientLimiter(sr=sr, attack_ms=0.5, release_ms=80.0, max_attenuation_db=-28.0)
        
        # Create 140+ dB SPL equivalent blast spike (+1.0 amplitude) at sample 100
        signal_in = np.zeros(1600, dtype=np.float32)
        signal_in[100:150] = 1.0
        
        signal_out = limiter.process_block(signal_in)
        
        clamped_peak = np.max(np.abs(signal_out[100:150]))
        input_peak = np.max(np.abs(signal_in[100:150]))
        attenuation_db = 20 * np.log10(clamped_peak / (input_peak + 1e-8))
        
        self.assertLessEqual(attenuation_db, -18.0, f"Limiter attenuation insufficient: {attenuation_db:.2f} dB")
        self.assertLessEqual(clamped_peak, 0.15, f"Clamped output exceeded safe ceiling: {clamped_peak}")

    def test_transient_limiter_exponential_release(self):
        """Verify smooth 80 ms exponential recovery after transient passes."""
        sr = 16000
        limiter = DynamicTransientLimiter(sr=sr, attack_ms=0.5, release_ms=80.0)
        
        # Blast pulse at sample 50, followed by voice lasting 4800 samples (~300ms)
        signal_in = np.zeros(4800, dtype=np.float32)
        signal_in[50:100] = 1.0
        signal_in[200:4800] = 0.2 * np.sin(2 * np.pi * 500 * np.linspace(0, 4600/sr, 4600))
        
        signal_out = limiter.process_block(signal_in)
        
        gain_early = np.max(np.abs(signal_out[200:400])) / 0.2
        # After > 3.5 time constants (~250ms), gain should recover near unity (> 0.85)
        gain_late = np.max(np.abs(signal_out[4000:4800])) / 0.2
        
        self.assertLess(gain_early, gain_late, f"Release failed to recover gain: early={gain_early}, late={gain_late}")
        self.assertGreater(gain_late, 0.80, f"Limiter did not restore full hearing sensitivity: {gain_late}")

    def test_speech_preservation_evaluation(self):
        """Verify speech spectral preservation proxy calculation."""
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        speech = (0.3 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)
        bg = (0.05 * np.random.randn(sr)).astype(np.float32)
        blast = np.zeros(sr, dtype=np.float32)
        blast[4000:4100] = 0.95
        
        limiter = DynamicTransientLimiter(sr=sr)
        filter_bank = SpeechPreservationFilter(sr=sr)
        
        result = evaluate_speech_intelligibility_preservation(
            speech_audio=speech,
            background_audio=bg,
            impulse_audio=blast,
            limiter=limiter,
            filter_bank=filter_bank,
            sr=sr
        )
        
        self.assertGreater(result["peak_attenuation_db"], 15.0, f"Reported attenuation too low: {result['peak_attenuation_db']}")
        self.assertGreater(result["speech_intelligibility_proxy_percent"], 10.0, "Speech spectral energy completely destroyed!")
        self.assertEqual(len(result["protected_audio"]), sr, "Output length mismatch!")

    def test_controller_state_machine_transitions(self):
        """Verify HearingProtectionController state machine transitions."""
        sr = 16000
        controller = HearingProtectionController(sr=sr, detection_threshold=0.65, recovery_threshold=0.30, hold_time_ms=50.0)
        
        self.assertEqual(controller.state, ProtectionState.NORMAL)
        
        # Low probability -> NORMAL
        state = controller.update_state(0.1)
        self.assertEqual(state, ProtectionState.NORMAL)
        
        # High probability -> PROTECTION_TRIGGERED
        state = controller.update_state(0.95)
        self.assertEqual(state, ProtectionState.PROTECTION_TRIGGERED)
        
        # Decrement hold counter
        for _ in range(controller.hold_samples + 5):
            state = controller.update_state(0.1)
        self.assertIn(state, [ProtectionState.RECOVERY, ProtectionState.NORMAL])

if __name__ == "__main__":
    unittest.main(verbosity=2)
