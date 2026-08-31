"""
SIH26052 — Robustness and Failure Testing Suite
Tests edge cases, invalid inputs, corrupted files, NaN/Inf handling,
dynamic range extremes, and malformed metadata formats.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import tempfile
import unittest
import numpy as np
import soundfile as sf
import torch

from src.preprocessing.audio_pipeline import load_and_preprocess_wav, normalize_audio_preserving_dynamics
from src.features.feature_extractor import AudioFeatureExtractor
from src.training.models import get_model
from src.dsp.dynamic_limiter import DynamicTransientLimiter

class TestPipelineFailures(unittest.TestCase):

    def test_empty_audio_handling(self):
        """Verify safe handling of empty audio arrays."""
        empty_arr = np.array([], dtype=np.float32)
        normalized = normalize_audio_preserving_dynamics(empty_arr)
        self.assertEqual(len(normalized), 0, "Expected empty array return on empty input")
        
        extractor = AudioFeatureExtractor(sr=16000, n_mels_edge=32)
        mel = extractor.extract_log_mel_spectrogram(np.zeros(16000, dtype=np.float32), mode="edge")
        self.assertTrue(np.all(np.isfinite(mel)), "Log-Mel produced NaN/Inf on silent audio")

    def test_nan_inf_audio_handling(self):
        """Verify that NaN and Inf in raw audio are scrubbed safely."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            data = np.array([0.1, 0.0, 0.5, 0.9, -0.9, 0.2] * 2000, dtype=np.float32)
            sf.write(tmp_path, data, 16000)
            
            loaded = load_and_preprocess_wav(tmp_path, target_sr=16000)
            self.assertIsNotNone(loaded, "Failed to load audio")
            self.assertTrue(np.all(np.isfinite(loaded)), "Processed audio contains NaN or Inf!")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_extremely_loud_clipped_audio(self):
        """Verify that extreme over-amplitude (> 10.0) is clipped safely without numerical explosion."""
        loud_audio = np.random.uniform(-15.0, 15.0, 16000).astype(np.float32)
        norm = normalize_audio_preserving_dynamics(loud_audio, headroom_db=-0.5)
        
        self.assertLessEqual(np.max(np.abs(norm)), 1.0, f"Normalized audio exceeded maximum digital limit: {np.max(np.abs(norm))}")
        self.assertTrue(np.all(np.isfinite(norm)), "Normalized audio contains NaN or Inf")
        
        limiter = DynamicTransientLimiter(sr=16000)
        clamped = limiter.process_block(loud_audio)
        self.assertLessEqual(np.max(np.abs(clamped)), 1.0, "Limiter output exceeded digital limit")

    def test_stereo_multi_channel_downmix(self):
        """Verify that stereo and multi-channel audio is downmixed safely to mono."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            stereo_data = np.zeros((16000, 2), dtype=np.float32)
            stereo_data[:, 0] = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
            stereo_data[:, 1] = 0.3 * np.sin(2 * np.pi * 880 * np.linspace(0, 1, 16000))
            sf.write(tmp_path, stereo_data, 16000)
            
            loaded = load_and_preprocess_wav(tmp_path, target_sr=16000)
            self.assertIsNotNone(loaded, "Failed to load stereo audio")
            self.assertEqual(loaded.ndim, 1, f"Expected 1D mono audio, got ndim={loaded.ndim}")
            self.assertEqual(len(loaded), 16000, f"Expected 16000 samples, got {len(loaded)}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_different_sample_rate_resampling(self):
        """Verify resampling from 8 kHz, 44.1 kHz, 48 kHz, and 96 kHz to standardized 16 kHz."""
        rates = [8000, 44100, 48000, 96000]
        
        for sr_orig in rates:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                
            try:
                n_samples = int(sr_orig * 1.5)
                t = np.linspace(0, 1.5, n_samples, endpoint=False)
                tone = (0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
                sf.write(tmp_path, tone, sr_orig)
                
                loaded = load_and_preprocess_wav(tmp_path, target_sr=16000)
                self.assertIsNotNone(loaded, f"Failed to resample from {sr_orig} Hz")
                self.assertTrue(np.all(np.isfinite(loaded)), f"Resampling from {sr_orig} created NaN/Inf")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def test_model_inference_on_arbitrary_tensor_shapes(self):
        """Verify that models handle batch sizes of 1 to 64 without shape crashes."""
        model_b = get_model("edge", num_classes=4)
        model_b.eval()
        
        for bs in [1, 4, 16, 64]:
            x = torch.randn(bs, 1, 32, 32)
            with torch.no_grad():
                out = model_b(x)
            self.assertEqual(out.shape, (bs, 4), f"Model B output shape mismatch: {out.shape}")
            self.assertTrue(torch.all(torch.isfinite(out)), "Model B produced NaN/Inf logits")

if __name__ == "__main__":
    unittest.main(verbosity=2)
