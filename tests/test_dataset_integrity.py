#!/usr/bin/env python3
"""
Unit Tests for MARKUSBLUE Dataset Integrity & Invariant Compliance
"""

import unittest
import os
import glob
import soundfile as sf

class TestDatasetIntegrity(unittest.TestCase):
    def test_original_dataset_counts(self):
        """Assert that baseline original dataset files are 100% intact."""
        speech_count = len(glob.glob("datasets/speech/*.wav"))
        gunshot_count = len(glob.glob("datasets/gunshot/*.wav"))
        bg_count = len(glob.glob("datasets/background_noise/*.wav"))
        other_count = len(glob.glob("datasets/other_impulse/*.wav"))

        self.assertEqual(speech_count, 2400, "Speech asset count changed!")
        self.assertEqual(gunshot_count, 6000, "Gunshot asset count changed!")
        self.assertEqual(bg_count, 2400, "Background noise asset count changed!")
        self.assertEqual(other_count, 2400, "Other impulse asset count changed!")

    def test_external_noise_format(self):
        """Assert that external noise files are valid 16kHz mono WAVs."""
        noise_files = glob.glob("datasets/external_noise/suppressible/*/*.wav")
        if not noise_files:
            noise_files = glob.glob("datasets/external_noise/*/*.wav")
        self.assertGreaterEqual(len(noise_files), 100, "Insufficient external noise samples!")
        
        # Test sample of files
        for fpath in noise_files[:15]:
            info = sf.info(fpath)
            self.assertEqual(info.samplerate, 16000, f"Sample rate mismatch in {fpath}")
            self.assertEqual(info.channels, 1, f"Channels mismatch in {fpath}")
            self.assertGreater(info.duration, 0.5, f"Audio too short in {fpath}")

    def test_critical_audio_format(self):
        """Assert that critical audio cue files (speech, radio, alarms, footsteps) exist."""
        crit_files = glob.glob("datasets/critical_audio/*/*.wav")
        self.assertGreaterEqual(len(crit_files), 100, "Insufficient critical audio samples!")
        self.assertTrue(os.path.exists("datasets/metadata/critical_audio_manifest.csv"))

    def test_manifest_existence(self):
        """Assert that manifest and source registries exist."""
        self.assertTrue(os.path.exists("datasets/metadata/external_noise_manifest.csv"))
        self.assertTrue(os.path.exists("datasets/metadata/critical_audio_manifest.csv"))
        self.assertTrue(os.path.exists("datasets/metadata/source_registry.json"))
        self.assertTrue(os.path.exists("metadata/dataset_catalog.json"))

if __name__ == "__main__":
    unittest.main()
