#!/usr/bin/env python3
"""
Unit Tests for SNR Calculation and Mathematical Correctness
"""

import unittest
import numpy as np

class TestSNRMathematics(unittest.TestCase):
    def test_snr_power_formula(self):
        """Test SNR power formula: SNR = 10 * log10(P_speech / P_noise)"""
        # 1.0s sinusoidal tone
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        speech = np.sin(2 * np.pi * 400.0 * t).astype(np.float32)
        noise = np.sin(2 * np.pi * 1200.0 * t).astype(np.float32)

        # Equal power -> 0 dB SNR
        sp_pwr = np.mean(speech ** 2)
        no_pwr = np.mean(noise ** 2)
        snr_0db = 10.0 * np.log10(sp_pwr / no_pwr)
        self.assertAlmostEqual(snr_0db, 0.0, places=4)

        # 10x power -> 10 dB SNR
        speech_10x = speech * np.sqrt(10.0)
        sp_pwr_10x = np.mean(speech_10x ** 2)
        snr_10db = 10.0 * np.log10(sp_pwr_10x / no_pwr)
        self.assertAlmostEqual(snr_10db, 10.0, places=4)

        # 0.1x power -> -10 dB SNR
        speech_01x = speech * np.sqrt(0.10)
        sp_pwr_01x = np.mean(speech_01x ** 2)
        snr_neg10db = 10.0 * np.log10(sp_pwr_01x / no_pwr)
        self.assertAlmostEqual(snr_neg10db, -10.0, places=4)

if __name__ == "__main__":
    unittest.main()
