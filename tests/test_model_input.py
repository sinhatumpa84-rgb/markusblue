#!/usr/bin/env python3
"""
Unit Tests for MARKUSBLUE Model Input/Output Dimensions & ESP32-S3 Compatibility
"""

import unittest
import torch
from src.training.student_model import MARKUSBLUEStudentEnhancer

class TestModelInput(unittest.TestCase):
    def setUp(self):
        self.model = MARKUSBLUEStudentEnhancer(n_fft=256, hop_length=64, hidden_dim=32)
        self.model.eval()

    def test_forward_dimensions(self):
        """Test forward pass output shape matches [B, 129, T] for 256-pt STFT."""
        batch_size = 4
        num_bins = 129 # (256 // 2 + 1)
        num_frames = 251 # 16000 samples // 64 hop + 1
        
        dummy_input = torch.randn(batch_size, num_bins, num_frames)
        with torch.no_grad():
            output_mask = self.model(dummy_input)

        self.assertEqual(output_mask.shape, (batch_size, num_bins, num_frames))
        # Mask must be in range [0.0, 1.0] due to Sigmoid
        self.assertGreaterEqual(output_mask.min().item(), 0.0)
        self.assertLessEqual(output_mask.max().item(), 1.0)

    def test_single_frame_streaming_latency(self):
        """Test single frame streaming forward pass (ESP32-S3 frame inference)."""
        single_frame = torch.randn(1, 129, 1)
        with torch.no_grad():
            frame_mask = self.model(single_frame)
        self.assertEqual(frame_mask.shape, (1, 129, 1))

    def test_parameter_budget(self):
        """Ensure parameter count remains within ESP32-S3 budget (< 25,000 params)."""
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertLess(total_params, 25000, f"Model parameters ({total_params}) exceed ESP32-S3 budget!")

if __name__ == "__main__":
    unittest.main()
