import os
import sys
from typing import Optional
import numpy as np
import soundfile as sf
import torch

try:
    from src.training.esp82_student_model import MARKUSBLUE_ESP82_Student
    from src.preprocessing.esp82_features import ESP82FeatureExtractor
    from src.postprocessing.esp82_dsp import (
        ESP82VoiceActivityDetector,
        ESP82AutomaticGainControl,
        ESP82PeakLimiter,
        ESP82AudioSynthesizer
    )
except ImportError:
    from training.esp82_student_model import MARKUSBLUE_ESP82_Student
    from preprocessing.esp82_features import ESP82FeatureExtractor
    from postprocessing.esp82_dsp import (
        ESP82VoiceActivityDetector,
        ESP82AutomaticGainControl,
        ESP82PeakLimiter,
        ESP82AudioSynthesizer
    )

class ESP82ReferencePipeline:
    """
    Python Reference Pipeline for MARKUSBLUE on ESP8266.
    Faithfully implements the exact fixed-memory streaming pipeline running on embedded hardware.
    """
    def __init__(
        self,
        model_path: str = "models/markusblue_esp82_student_best.pt",
        sr: int = 8000,
        n_fft: int = 128,
        hop_length: int = 64
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_bins = n_fft // 2 + 1
        
        # Modules
        self.feature_extractor = ESP82FeatureExtractor(sr=sr, n_fft=n_fft, hop_length=hop_length)
        self.synthesizer = ESP82AudioSynthesizer(n_fft=n_fft, hop_length=hop_length)
        self.vad = ESP82VoiceActivityDetector(alpha_noise=0.98, snr_threshold_db=4.0)
        self.agc = ESP82AutomaticGainControl(target_level=0.32, max_gain=4.0, min_gain=0.5)
        self.limiter = ESP82PeakLimiter(threshold=0.95)
        
        # Neural Model
        self.model = MARKUSBLUE_ESP82_Student(num_bins=self.num_bins, hidden_dim=16)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            print(f"[*] Loaded ESP82 weights from '{model_path}'")
        self.model.eval()

    def process_frame(self, in_frame: np.ndarray) -> np.ndarray:
        """
        Streaming single-frame inference loop (hop_length samples = 8.0 ms).
        """
        # 1. Feature Extraction (STFT Magnitude & Phase)
        mag, phase = self.feature_extractor.process_frame(in_frame)
        
        # 2. Neural Mask Inference
        with torch.no_grad():
            mag_tensor = torch.from_numpy(mag).unsqueeze(0).unsqueeze(-1).float() # [1, num_bins, 1]
            mask_tensor = self.model(mag_tensor)
            mask = mask_tensor.squeeze().numpy()
            
        # 3. Apply Speech Mask
        enhanced_mag = mag * mask
        
        # 4. Overlap-Add IFFT Synthesis
        synth_frame = self.synthesizer.synthesize_frame(enhanced_mag, phase)
        
        # 5. VAD & Speech Level Estimation
        frame_energy = float(np.mean(synth_frame ** 2) + 1e-10)
        is_speech = self.vad.update(frame_energy)
        
        # 6. Automatic Gain Control (Compensates for speech attenuation without noise pumping)
        agc_frame = self.agc.process_frame(synth_frame, is_speech)
        
        # 7. Peak Limiter / Soft Clipper
        out_frame = self.limiter.process_frame(agc_frame)
        
        return out_frame

    def process_wav(self, input_wav_path: str, output_wav_path: Optional[str] = None) -> np.ndarray:
        """
        End-to-end WAV processing using continuous streaming frames.
        """
        audio, in_sr = sf.read(input_wav_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1) # Mono
            
        if in_sr == 16000 and self.sr == 8000:
            audio = audio[::2]
            
        audio = audio.astype(np.float32)
        n_samples = len(audio)
        enhanced_chunks = []
        
        # Stream frame by frame
        for i in range(0, n_samples - self.hop_length + 1, self.hop_length):
            chunk = audio[i:i + self.hop_length]
            out_chunk = self.process_frame(chunk)
            enhanced_chunks.append(out_chunk)
            
        enhanced_audio = np.concatenate(enhanced_chunks)
        
        if output_wav_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
            sf.write(output_wav_path, enhanced_audio, self.sr)
            print(f"[OK] Saved enhanced audio to '{output_wav_path}'")
            
        return enhanced_audio

if __name__ == "__main__":
    pipeline = ESP82ReferencePipeline()
    print("[*] ESP82 Reference Pipeline initialized successfully.")
