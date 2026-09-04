#!/usr/bin/env python3
"""
Unit Tests for MARKUSBLUE Dynamic Multi-Noise Mixer
"""

import unittest
import numpy as np
import glob
from src.dataset.multi_noise_mixer import MultiNoiseMixer

class TestMultiNoiseMixer(unittest.TestCase):
    def setUp(self):
        # Generate dummy noise pool
        self.noise_files = glob.glob("datasets/external_noise/suppressible/*/*.wav")[:20]
        if not self.noise_files:
            self.noise_files = glob.glob("datasets/external_noise/*/*.wav")[:20]
        if not self.noise_files:
            self.noise_files = glob.glob("datasets/background_noise/*.wav")[:20]
        self.mixer = MultiNoiseMixer(self.noise_files, sr=16000, duration_samples=16000)
        
        # Synthetic 1.0s speech tone (440Hz + 880Hz)
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        self.clean_speech = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

    def test_multi_source_mixing(self):
        """Test mixing with 1, 2, 3, and 4 concurrent noise sources."""
        for num_sources in [1, 2, 3, 4]:
            noisy, speech, noise, meta = self.mixer.mix(
                self.clean_speech, target_snr_db=0.0, num_sources=num_sources, return_metadata=True
            )
            self.assertEqual(len(noisy), 16000)
            self.assertEqual(meta["num_sources"], num_sources)
            self.assertEqual(len(meta["sources"]), num_sources)

    def test_snr_scaling(self):
        """Test that achieved SNR matches target within +/- 1.5 dB tolerance."""
        for target_snr in [-10.0, 0.0, 10.0, 15.0]:
            noisy, speech, noise, meta = self.mixer.mix(
                self.clean_speech, target_snr_db=target_snr, num_sources=2, return_metadata=True
            )
            sp_pwr = np.mean(speech ** 2)
            no_pwr = np.mean(noise ** 2)
            calc_snr = 10.0 * np.log10(sp_pwr / (no_pwr + 1e-10))
            self.assertAlmostEqual(calc_snr, target_snr, delta=1.5)

    def test_temporal_envelopes(self):
        """Test that all temporal profiles execute without NaNs or infinities."""
        dummy_noise = np.random.normal(0, 0.2, 16000).astype(np.float32)
        for profile in ["stationary", "approaching", "receding", "flyby", "burst"]:
            enveloped = self.mixer.apply_temporal_envelope(dummy_noise, profile)
            self.assertEqual(len(enveloped), 16000)
            self.assertFalse(np.isnan(enveloped).any())
            self.assertFalse(np.isinf(enveloped).any())

    def test_anti_clipping_protection(self):
        """Test that noisy output never exceeds ceiling limit."""
        loud_speech = (self.clean_speech * 2.0).astype(np.float32)
        noisy, _, _ = self.mixer.mix(loud_speech, target_snr_db=-15.0, num_sources=4)
        self.assertLessEqual(np.max(np.abs(noisy)), 0.96)

if __name__ == "__main__":
    unittest.main()
