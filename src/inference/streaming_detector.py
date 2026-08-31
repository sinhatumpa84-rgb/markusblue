import time
import math
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

from src.features.feature_extractor import AudioFeatureExtractor
from src.dsp.hearing_protection import HearingProtectionController, ProtectionState

class StreamingImpulseDetector:
    """
    Streaming real-time rolling-window acoustic impulse detector and protection runner.
    Processes live/simulated audio blocks with sub-frame sliding window inference.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        sr: int = 16000,
        window_size_samples: int = 16000,   # 1.0 second context
        hop_size_samples: int = 400,         # 25 ms streaming step
        feature_mode: str = "edge",          # "edge" (32 mels) or "baseline" (64 mels)
        detection_threshold: float = 0.65
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.sr = sr
        self.window_size_samples = window_size_samples
        self.hop_size_samples = hop_size_samples
        self.feature_mode = feature_mode
        self.detection_threshold = detection_threshold
        
        self.extractor = AudioFeatureExtractor(sr=sr)
        self.protection_controller = HearingProtectionController(
            sr=sr, detection_threshold=detection_threshold
        )
        
        # Internal circular audio buffer
        self.buffer = np.zeros(window_size_samples, dtype=np.float32)
        
        # Telemetry stats
        self.inference_latencies_ms = []
        self.dsp_latencies_ms = []
        self.total_blocks_processed = 0

    def reset(self):
        self.buffer.fill(0)
        self.protection_controller.limiter.reset()
        self.protection_controller.state = ProtectionState.NORMAL
        self.inference_latencies_ms.clear()
        self.dsp_latencies_ms.clear()

    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[np.ndarray, ProtectionState, float, float]:
        """
        Process a single incoming chunk (e.g., 25ms / 400 samples).
        Returns: (protected_output_chunk, state, impulse_probability, inference_latency_ms)
        """
        n_samples = len(audio_chunk)
        
        # Roll buffer and insert new chunk
        self.buffer = np.roll(self.buffer, -n_samples)
        self.buffer[-n_samples:] = audio_chunk
        
        # 1. Feature Extraction & Inference
        t0 = time.perf_counter()
        mel = self.extractor.extract_log_mel_spectrogram(self.buffer, mode=self.feature_mode)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            logits = self.model(mel_tensor)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        t1 = time.perf_counter()
        inf_lat_ms = (t1 - t0) * 1000.0
        self.inference_latencies_ms.append(inf_lat_ms)
        
        impulse_prob = float(probs[0])
        
        # 2. Deterministic DSP Protection
        t_dsp0 = time.perf_counter()
        protected_chunk, state, gain = self.protection_controller.process_frame(audio_chunk, impulse_prob)
        t_dsp1 = time.perf_counter()
        self.dsp_latencies_ms.append((t_dsp1 - t_dsp0) * 1000.0)
        
        self.total_blocks_processed += 1
        return protected_chunk, state, impulse_prob, inf_lat_ms

    def run_stream_simulation(
        self,
        full_audio: np.ndarray,
        chunk_size: Optional[int] = None
    ) -> Dict:
        """Simulate continuous streaming playback through the system."""
        chunk_size = chunk_size or self.hop_size_samples
        self.reset()
        
        output_chunks = []
        state_history = []
        probability_history = []
        latencies = []
        
        num_chunks = len(full_audio) // chunk_size
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = full_audio[start:end]
            
            out_chunk, state, prob, lat = self.process_chunk(chunk)
            output_chunks.append(out_chunk)
            state_history.append(state.value)
            probability_history.append(prob)
            latencies.append(lat)
            
        full_output = np.concatenate(output_chunks) if output_chunks else np.array([])
        
        return {
            "processed_audio": full_output,
            "state_history": state_history,
            "probability_history": probability_history,
            "mean_inference_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p95_inference_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "total_blocks": num_chunks
        }
