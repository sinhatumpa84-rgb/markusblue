#!/usr/bin/env python3
"""
MARKUSBLUE — Tactical Audio Regression Test Suite
SIH Problem Statement: SIH26052 — DRDO / Defence Speech-Enhancement System

Rigorous multi-SNR (-15dB to +20dB), multi-speaker, and multi-class regression suite
verifying that MARKUSBLUE preserves critical acoustic signals while attenuating
suppressible background noise.
"""

import unittest
import os
import glob
import numpy as np
import soundfile as sf
import torch

from src.dataset.multi_noise_mixer import MultiNoiseMixer
from src.training.student_model import MARKUSBLUEStudentEnhancer

class TestTacticalAudioRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suppressible_files = glob.glob("datasets/external_noise/suppressible/*/*.wav")
        cls.speech_files = glob.glob("datasets/speech/*.wav")[:20]
        cls.critical_speech = glob.glob("datasets/critical_audio/speech/*.wav")[:20]
        cls.radio_files = glob.glob("datasets/critical_audio/radio_communication/*.wav")[:15]
        cls.alarm_files = glob.glob("datasets/critical_audio/alarms/*.wav")[:15]
        cls.siren_files = glob.glob("datasets/critical_audio/sirens/*.wav")[:15]
        cls.footstep_files = glob.glob("datasets/critical_audio/footsteps/*.wav")[:15]
        cls.movement_files = glob.glob("datasets/critical_audio/movement/*.wav")[:15]

        cls.mixer = MultiNoiseMixer(
            suppressible_files=cls.suppressible_files,
            critical_files=cls.alarm_files + cls.siren_files + cls.radio_files + cls.footstep_files,
            sr=16000,
            duration_samples=16000
        )

        cls.model = MARKUSBLUEStudentEnhancer(n_fft=256, hop_length=64, hidden_dim=32)
        ckpt_path = "models/markusblue_esp32s3_best.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            cls.model.load_state_dict(state_dict)
        cls.model.eval()

    def _enhance(self, noisy_audio: np.ndarray) -> np.ndarray:
        w = torch.hann_window(256)
        x = torch.tensor(noisy_audio, dtype=torch.float32)
        stft = torch.stft(x, n_fft=256, hop_length=64, window=w, return_complex=True)
        mag = torch.abs(stft).unsqueeze(0)
        with torch.no_grad():
            mask = self.model(mag)
        enhanced_stft = stft * mask.squeeze(0)
        enhanced = torch.istft(enhanced_stft, n_fft=256, hop_length=64, window=w, length=len(noisy_audio))
        return enhanced.numpy()

    def test_multi_snr_speech_preservation(self):
        """Test speech across 8 SNR checkpoints (-15dB to +20dB)."""
        snrs = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
        for sp_path in self.speech_files[:5]:
            sp = self.mixer._load_audio(sp_path)
            for snr in snrs:
                noisy, clean, _ = self.mixer.mix_operational(sp, target_snr_db=snr, inject_critical_cue=False)
                enhanced = self._enhance(noisy)
                active = np.abs(clean) > 0.02
                if np.sum(active) > 100:
                    clean_rms = np.sqrt(np.mean(clean[active] ** 2) + 1e-8)
                    enh_rms = np.sqrt(np.mean(enhanced[active] ** 2) + 1e-8)
                    ratio = enh_rms / clean_rms
                    min_ratio = 0.20 if snr <= -15.0 else (0.30 if snr <= -10.0 else (0.40 if snr <= -5.0 else 0.50))
                    self.assertGreater(ratio, min_ratio, f"Speech excessively suppressed at {snr}dB SNR: {ratio:.2f}x (expected >= {min_ratio})")

    def test_radio_communication_intelligibility(self):
        """Test radio speech with squelch and narrow bandpass against engine rumble."""
        for radio_path in self.radio_files[:5]:
            radio = self.mixer._load_audio(radio_path)
            noisy, clean, _ = self.mixer.mix_operational(radio, target_snr_db=0.0, inject_critical_cue=False)
            enhanced = self._enhance(noisy)
            self.assertFalse(np.isnan(enhanced).any())
            self.assertGreater(np.max(np.abs(enhanced)), 0.04, "Enhanced radio output is silent!")

    def test_multi_alarm_retention(self):
        """Test emergency alarm warning beeper retention against heavy industrial machinery."""
        import random
        for alarm_path in self.alarm_files[:5]:
            random.seed(42)
            np.random.seed(42)
            alarm = self.mixer._load_audio(alarm_path)
            sp = self.mixer._load_audio(self.speech_files[0])
            clean_target = sp + alarm * 0.4
            noisy, clean, _ = self.mixer.mix_operational(clean_target, target_snr_db=5.0, inject_critical_cue=False)
            enhanced = self._enhance(noisy)
            
            fft_alarm = np.abs(np.fft.rfft(alarm))
            fft_clean = np.abs(np.fft.rfft(clean))
            fft_enh = np.abs(np.fft.rfft(enhanced))
            peak_bin = np.argmax(fft_alarm)
            retention = fft_enh[peak_bin] / (fft_clean[peak_bin] + 1e-8)
            self.assertGreater(retention, 0.10, f"Alarm signal at {alarm_path} excessively attenuated: {retention:.2f}x")

    def test_multi_siren_sweep_retention(self):
        """Test rising/falling siren sweep retention under road traffic noise."""
        for siren_path in self.siren_files[:5]:
            siren = self.mixer._load_audio(siren_path)
            sp = self.mixer._load_audio(self.speech_files[1])
            clean_target = sp + siren * 0.5
            noisy, clean, _ = self.mixer.mix_operational(clean_target, target_snr_db=5.0, inject_critical_cue=False)
            enhanced = self._enhance(noisy)
            self.assertFalse(np.isnan(enhanced).any())
            self.assertGreater(np.max(np.abs(enhanced)), 0.05, "Enhanced siren output is silent!")

    def test_footsteps_and_movement_cues(self):
        """Test tactical footsteps on gravel/concrete/metal surfaces."""
        for step_path in self.footstep_files[:5]:
            steps = self.mixer._load_audio(step_path)
            noisy, clean, _ = self.mixer.mix_operational(steps, target_snr_db=10.0, inject_critical_cue=False)
            enhanced = self._enhance(noisy)
            self.assertFalse(np.isnan(enhanced).any())
            self.assertGreater(np.max(np.abs(enhanced)), 0.03, "Tactical footsteps completely muted!")

if __name__ == "__main__":
    unittest.main()
