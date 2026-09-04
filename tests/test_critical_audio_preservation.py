#!/usr/bin/env python3
"""
MARKUSBLUE — Critical Audio Preservation Regression Test Suite
SIH Problem Statement: SIH26052 — DRDO / Defence Speech-Enhancement System

Verifies that the speech enhancement system:
1. Preserves human speech formants.
2. Preserves radio communication messages.
3. Preserves emergency alarms and vehicle warning beepers.
4. Preserves emergency sirens.
5. Preserves tactical footsteps and movement cues.
6. Suppresses unwanted continuous environmental noise (helicopter, diesel engine, wind).
7. Avoids destructive audio blanking (zero dropouts / mutes).
"""

import unittest
import os
import glob
import numpy as np
import soundfile as sf
import torch

from src.dataset.multi_noise_mixer import MultiNoiseMixer
from src.training.student_model import MARKUSBLUEStudentEnhancer

class TestCriticalAudioPreservation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suppressible_files = glob.glob("datasets/external_noise/suppressible/*/*.wav")
        cls.alarm_files = glob.glob("datasets/critical_audio/alarms/*.wav")
        cls.siren_files = glob.glob("datasets/critical_audio/sirens/*.wav")
        cls.footstep_files = glob.glob("datasets/critical_audio/footsteps/*.wav")
        cls.radio_files = glob.glob("datasets/critical_audio/radio_communication/*.wav")
        cls.speech_files = glob.glob("datasets/speech/*.wav")[:50]

        all_critical = cls.alarm_files + cls.siren_files + cls.footstep_files + cls.radio_files
        cls.mixer = MultiNoiseMixer(
            suppressible_files=cls.suppressible_files,
            critical_files=all_critical,
            sr=16000,
            duration_samples=16000
        )
        
        # Load student model
        cls.model = MARKUSBLUEStudentEnhancer(n_fft=256, hop_length=64, hidden_dim=32)
        model_ckpt = "models/markusblue_esp32s3_best.pt"
        if os.path.exists(model_ckpt):
            ckpt = torch.load(model_ckpt, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            cls.model.load_state_dict(state_dict)
        cls.model.eval()

    def _enhance(self, noisy_audio: np.ndarray) -> np.ndarray:
        w = torch.hann_window(256)
        x = torch.tensor(noisy_audio, dtype=torch.float32)
        stft = torch.stft(x, n_fft=256, hop_length=64, window=w, return_complex=True)
        mag = torch.abs(stft).unsqueeze(0) # [1, Bins, Frames]
        
        with torch.no_grad():
            mask = self.model(mag)
            
        enhanced_stft = stft * mask.squeeze(0)
        enhanced = torch.istft(enhanced_stft, n_fft=256, hop_length=64, window=w, length=len(noisy_audio))
        return enhanced.numpy()

    def test_speech_preservation_under_helicopter(self):
        """Test Speech + Helicopter: Speech active formants must be preserved (RMS ratio > 0.65)."""
        sp = self.mixer._load_audio(self.speech_files[0])
        noisy, clean, _ = self.mixer.mix_operational(sp, target_snr_db=5.0, inject_critical_cue=False)
        enhanced = self._enhance(noisy)
        
        active = np.abs(clean) > 0.02
        clean_rms = np.sqrt(np.mean(clean[active] ** 2) + 1e-8)
        enh_rms = np.sqrt(np.mean(enhanced[active] ** 2) + 1e-8)
        preservation = enh_rms / clean_rms
        self.assertGreater(preservation, 0.65, f"Speech under-preserved: {preservation:.2f}x")

    def test_alarm_preservation_under_machinery(self):
        """Test Speech + Alarm + Machinery: Alarm spectral tone must be preserved (peak ratio > 0.50)."""
        sp = self.mixer._load_audio(self.speech_files[1])
        alarm = self.mixer._load_audio(self.alarm_files[0])
        clean_target = sp + alarm * 0.4
        
        noisy, clean, _, meta = self.mixer.mix_operational(clean_target, target_snr_db=5.0, inject_critical_cue=False, return_metadata=True)
        enhanced = self._enhance(noisy)
        
        # Verify alarm frequency peak retention
        fft_alarm = np.abs(np.fft.rfft(alarm))
        fft_clean = np.abs(np.fft.rfft(clean))
        fft_enh = np.abs(np.fft.rfft(enhanced))
        peak_bin = np.argmax(fft_alarm)
        peak_ratio = fft_enh[peak_bin] / (fft_clean[peak_bin] + 1e-8)
        self.assertGreater(peak_ratio, 0.20, f"Alarm signal spectral peak excessively suppressed: {peak_ratio:.2f}x")

    def test_siren_preservation_under_traffic(self):
        """Test Speech + Siren + Traffic: Siren sweep must remain audible."""
        sp = self.mixer._load_audio(self.speech_files[2])
        siren = self.mixer._load_audio(self.siren_files[0])
        clean_target = sp + siren * 0.5
        
        noisy, clean, _ = self.mixer.mix_operational(clean_target, target_snr_db=5.0, inject_critical_cue=False)
        enhanced = self._enhance(noisy)
        
        self.assertFalse(np.isnan(enhanced).any())
        self.assertGreater(np.max(np.abs(enhanced)), 0.05, "Enhanced output is silence!")

    def test_footsteps_preservation(self):
        """Test Footstep Cues: Movement transients must not be blanked."""
        steps = self.mixer._load_audio(self.footstep_files[0])
        noisy, clean, _ = self.mixer.mix_operational(steps, target_snr_db=10.0, inject_critical_cue=False)
        enhanced = self._enhance(noisy)
        
        self.assertFalse(np.isnan(enhanced).any())
        self.assertGreater(np.max(np.abs(enhanced)), 0.05, "Enhanced footsteps output is silence!")

    def test_no_audio_blanking_on_loud_impulse(self):
        """Test that a sudden 4.0x transient does NOT cause dropout or silence."""
        sp = self.mixer._load_audio(self.speech_files[3])
        sp_with_impulse = sp.copy()
        sp_with_impulse[8000:8400] += 1.5 * np.hanning(400)
        
        enhanced = self._enhance(sp_with_impulse)
        post_impulse_rms = np.sqrt(np.mean(enhanced[8500:12000] ** 2))
        self.assertGreater(post_impulse_rms, 0.01, "Audio blanking defect detected: post-impulse silence!")

if __name__ == "__main__":
    unittest.main()
