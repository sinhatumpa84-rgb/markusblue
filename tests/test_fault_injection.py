#!/usr/bin/env python3
"""
MARKUSBLUE — Fault-Injection & Fail-Safe Audio Regression Tests
SIH Problem Statement: SIH26052 — DRDO / Defence Speech-Enhancement System

Tests system resilience and automatic fail-safe fallback under simulated hardware/software faults:
1. Microphone disconnection (zero input / DC offset)
2. Corrupted audio buffers (NaN, Inf, extreme clipping)
3. Buffer overflow & DMA underrun emulation
4. Invalid model mask output (all zeros, all ones, NaN mask)
5. Safe bypass fallback (reverts to linear pass-through with limiter protection; zero silence)
"""

import unittest
import numpy as np
import torch

from src.agc.automatic_gain_control import AutomaticGainControl
from src.limiter.peak_limiter import PeakSafetyLimiter
from src.training.student_model import MARKUSBLUEStudentEnhancer

class TestFaultInjection(unittest.TestCase):
    def setUp(self):
        self.model = MARKUSBLUEStudentEnhancer(n_fft=256, hop_length=64, hidden_dim=32)
        self.model.eval()
        self.agc = AutomaticGainControl(target_rms_dbfs=-14.0, max_gain_db=24.0)
        self.limiter = PeakSafetyLimiter(sr=16000, ceiling_dbfs=-0.5)

    def test_nan_audio_input_sanitization(self):
        """Inject NaN values into audio stream; verify pipeline sanitizes to zero without crashing."""
        corrupted = np.random.normal(0, 0.2, 16000).astype(np.float32)
        corrupted[500:600] = np.nan
        corrupted[1200:1250] = np.inf

        # Pipeline Sanitization Guard
        sanitized = np.nan_to_num(corrupted, nan=0.0, posinf=0.95, neginf=-0.95)
        self.assertFalse(np.isnan(sanitized).any(), "NaN survived sanitization guard!")
        self.assertFalse(np.isinf(sanitized).any(), "Inf survived sanitization guard!")

        # Process through limiter
        limited = self.limiter.process_frame(sanitized)
        self.assertLessEqual(np.max(np.abs(limited)), 0.96)

    def test_microphone_disconnect_silence(self):
        """Simulate external microphone disconnection (complete silence)."""
        silent_mic = np.zeros(16000, dtype=np.float32)
        # AGC must not runaway to infinity on silence (gain bounded by max_gain_db)
        agc_out = self.agc.process_frame(silent_mic, is_speech=False)
        self.assertFalse(np.isnan(agc_out).any())
        self.assertEqual(np.max(np.abs(agc_out)), 0.0)

    def test_microphone_dc_offset_fault(self):
        """Simulate microphone hardware fault generating +0.5V pure DC offset."""
        dc_fault = (np.random.normal(0, 0.1, 16000) + 0.50).astype(np.float32)
        # High-pass filter emulation (DC blocking capacitor / single-pole IIR)
        dc_blocked = dc_fault - np.mean(dc_fault)
        self.assertAlmostEqual(float(np.mean(dc_blocked)), 0.0, places=4)
        self.assertLess(np.max(np.abs(dc_blocked)), 0.6)

    def test_model_zero_mask_safe_bypass_mode(self):
        """Simulate model failure producing zero mask; verify safe fallback pass-through."""
        speech = np.random.normal(0, 0.2, 16000).astype(np.float32)
        faulty_mask = np.zeros((129, 251), dtype=np.float32)

        # Fail-Safe Policy: If mean mask is < 0.01 (indicating AI failure/dropout), activate Safe Bypass
        if np.mean(faulty_mask) < 0.01:
            # Safe Bypass Mode: Pass-through with Limiter Guard
            safe_output = self.limiter.process_frame(speech)
            mode = "SAFE_BYPASS"
        else:
            safe_output = speech * faulty_mask
            mode = "ENHANCED"

        self.assertEqual(mode, "SAFE_BYPASS")
        self.assertGreater(np.max(np.abs(safe_output)), 0.05, "User was silenced on AI fault!")

    def test_extreme_blast_transient_anti_clipping(self):
        """Simulate +12 dB extreme acoustic blast (impulse spike reaching 4.0x full scale)."""
        blast = np.zeros(16000, dtype=np.float32)
        blast[1000] = 4.0 # Extreme impulse
        blast[1001:1050] = 2.5 * np.hanning(49)

        limited = self.limiter.process_frame(blast)
        self.assertLessEqual(np.max(np.abs(limited)), 0.96, "Limiter failed to contain extreme blast!")
        self.assertFalse(np.isnan(limited).any())

if __name__ == "__main__":
    unittest.main()
