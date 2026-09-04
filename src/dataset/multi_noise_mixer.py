#!/usr/bin/env python3
"""
MARKUSBLUE — Dynamic Multi-Noise Mixer & Critical Signal Preservation Engine
SIH Problem Statement: SIH26052 — DRDO / Defence Speech-Enhancement System

Supports:
1. Dynamic online mixing of:
     Clean Speech (Primary) + Critical Audio Cues (Alarms, Sirens, Footsteps, Radio) + Suppressible Noise
2. Critical Preservation Mask Formulation:
     M_target(f, t) = |S_critical(f, t)| / (|S_critical(f, t)| + |N_suppress(f, t)| + eps)
   Ensuring that speech, radio communications, alarms, and tactical footsteps are preserved
   while helicopter, vehicle, wind, and industrial noise are suppressed.
3. Configurable wideband SNR (-15 dB to +20 dB).
4. Realistic temporal movement envelopes (approaching, receding, flyby, bursts, stationary).
"""

import os
import random
import numpy as np
import soundfile as sf
import torch

class MultiNoiseMixer:
    """
    Operational multi-source acoustic mixer with critical signal preservation.
    """
    def __init__(
        self,
        suppressible_files: list,
        critical_files: list = None,
        sr: int = 16000,
        duration_samples: int = 16000,
        snr_range: tuple = (-15.0, 20.0),
        max_noise_sources: int = 3,
        temporal_profiles: list = None
    ):
        self.suppressible_files = suppressible_files
        self.critical_files = critical_files or []
        self.sr = sr
        self.duration = duration_samples
        self.snr_min, self.snr_max = snr_range
        self.max_sources = max_noise_sources
        self.profiles = temporal_profiles or ["stationary", "approaching", "receding", "flyby", "burst"]

    def _load_audio(self, path: str) -> np.ndarray:
        try:
            data, _ = sf.read(path)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1) # Mono
            data = data.astype(np.float32)
            if len(data) < self.duration:
                repeats = int(np.ceil(self.duration / len(data)))
                data = np.tile(data, repeats)[:self.duration]
            else:
                max_start = len(data) - self.duration
                start = random.randint(0, max_start) if max_start > 0 else 0
                data = data[start:start + self.duration]
            return data
        except Exception:
            return np.zeros(self.duration, dtype=np.float32)

    def apply_temporal_envelope(self, noise: np.ndarray, profile: str) -> np.ndarray:
        N = len(noise)
        t = np.linspace(0, 1, N, endpoint=False)

        if profile == "stationary":
            envelope = np.ones(N, dtype=np.float32)
        elif profile == "approaching":
            envelope = 0.15 + 0.85 * (t ** 1.5)
        elif profile == "receding":
            envelope = 1.0 - 0.85 * (t ** 1.5)
        elif profile == "flyby":
            envelope = np.exp(-((t - 0.5) ** 2) / (2 * (0.18 ** 2)))
            envelope = 0.10 + 0.90 * (envelope / (np.max(envelope) + 1e-8))
        elif profile == "burst":
            burst_len = random.randint(int(0.2 * self.sr), int(0.4 * self.sr))
            start = random.randint(0, max(0, N - burst_len))
            envelope = np.ones(N, dtype=np.float32) * 0.20
            window = np.hanning(burst_len)
            envelope[start:start + burst_len] = 0.20 + 0.80 * window
        else:
            envelope = np.ones(N, dtype=np.float32)

        return (noise * envelope).astype(np.float32)

    def mix_operational(
        self,
        clean_speech: np.ndarray,
        target_snr_db: float = None,
        num_sources: int = None,
        inject_critical_cue: bool = True,
        return_metadata: bool = False
    ):
        """
        Mixes clean speech + optional critical cue (alarm/siren/footsteps/radio)
        against 1 to 3 suppressible environmental noises.
        Returns:
          noisy_mixture: S_critical + N_suppress
          target_critical: S_speech + S_cue (what the model MUST preserve)
          suppress_noise: N_suppress (what the model MUST attenuate)
        """
        if target_snr_db is None:
            target_snr_db = random.uniform(self.snr_min, self.snr_max)

        # 1. Prepare Critical Signal (Speech + Optional Cue)
        target_critical = clean_speech.copy()
        cue_meta = None

        if inject_critical_cue and self.critical_files and random.random() < 0.40:
            cue_path = random.choice(self.critical_files)
            cue_audio = self._load_audio(cue_path)
            # Scale cue to -6 dB to 0 dB relative to speech
            sp_rms = np.sqrt(np.mean(clean_speech ** 2) + 1e-8)
            cue_rms = np.sqrt(np.mean(cue_audio ** 2) + 1e-8)
            cue_gain = random.uniform(0.5, 1.0) * (sp_rms / (cue_rms + 1e-8))
            scaled_cue = cue_audio * cue_gain
            target_critical += scaled_cue
            cue_meta = {"path": cue_path, "gain": cue_gain}

        # 2. Combine 1 to N Suppressible Noise Sources
        if num_sources is not None:
            num_noises = max(1, min(num_sources, len(self.suppressible_files)))
        else:
            num_noises = random.randint(1, min(self.max_sources, len(self.suppressible_files)))
        selected_noises = random.sample(self.suppressible_files, num_noises)
        combined_noise = np.zeros(self.duration, dtype=np.float32)
        noise_meta = []

        for p in selected_noises:
            raw_no = self._load_audio(p)
            profile = random.choice(self.profiles)
            enveloped = self.apply_temporal_envelope(raw_no, profile)
            w = random.uniform(0.6, 1.0)
            combined_noise += w * enveloped
            noise_meta.append({"path": p, "profile": profile, "weight": w})

        # 3. Scale Suppressible Noise to achieve Target SNR relative to Critical Target
        crit_pwr = np.mean(target_critical ** 2) + 1e-10
        no_pwr = np.mean(combined_noise ** 2) + 1e-10
        target_no_pwr = crit_pwr * (10.0 ** (-target_snr_db / 10.0))
        scale = np.sqrt(target_no_pwr / no_pwr)
        scaled_noise = combined_noise * scale

        # 4. Synthesize Final Noisy Mixture
        noisy = target_critical + scaled_noise

        # 5. Anti-clipping safety normalization
        peak = np.max(np.abs(noisy))
        if peak > 0.95:
            factor = 0.95 / peak
            noisy = noisy * factor
            target_critical = target_critical * factor
            scaled_noise = scaled_noise * factor

        if return_metadata:
            return noisy, target_critical, scaled_noise, {
                "snr_db": target_snr_db,
                "num_sources": num_noises,
                "num_noise_sources": num_noises,
                "sources": noise_meta,
                "noises": noise_meta,
                "critical_cue": cue_meta
            }
        return noisy, target_critical, scaled_noise

    def mix(self, clean_speech: np.ndarray, target_snr_db: float = None, num_sources: int = None, return_metadata: bool = False):
        """Backward-compatible mixing call."""
        return self.mix_operational(clean_speech, target_snr_db=target_snr_db, num_sources=num_sources, inject_critical_cue=False, return_metadata=return_metadata)
