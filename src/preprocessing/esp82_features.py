import numpy as np

class ESP82FeatureExtractor:
    """
    Fixed-size streaming feature extractor for ESP8266.
    Extracts magnitude spectrum using windowed Hanning FFT.
    """
    def __init__(self, sr: int = 8000, n_fft: int = 128, hop_length: int = 64):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.num_bins = n_fft // 2 + 1 # 65 bins for n_fft=128
        self.window = np.hanning(self.win_length).astype(np.float32)
        
        # Circular / streaming input buffer
        self.in_buffer = np.zeros(self.win_length, dtype=np.float32)

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process incoming PCM audio frame (hop_length samples).
        Returns: (magnitude_spectrum, phase_spectrum)
        """
        # Shift buffer and insert new samples
        self.in_buffer[:-self.hop_length] = self.in_buffer[self.hop_length:]
        self.in_buffer[-self.hop_length:] = frame
        
        # Windowing
        windowed = self.in_buffer * self.window
        
        # Real FFT
        fft_c = np.fft.rfft(windowed, n=self.n_fft)
        mag = np.abs(fft_c).astype(np.float32)
        phase = np.angle(fft_c).astype(np.float32)
        
        return mag, phase

    def compute_spectrogram(self, audio: np.ndarray) -> tuple:
        """
        Compute full spectrogram for offline evaluation / training.
        """
        n_samples = len(audio)
        n_frames = max(1, (n_samples - self.win_length) // self.hop_length + 1)
        
        mags = np.zeros((self.num_bins, n_frames), dtype=np.float32)
        phases = np.zeros((self.num_bins, n_frames), dtype=np.float32)
        
        for t in range(n_frames):
            start = t * self.hop_length
            chunk = audio[start:start + self.win_length]
            if len(chunk) < self.win_length:
                chunk = np.pad(chunk, (0, self.win_length - len(chunk)))
            windowed = chunk * self.window
            fft_c = np.fft.rfft(windowed, n=self.n_fft)
            mags[:, t] = np.abs(fft_c)
            phases[:, t] = np.angle(fft_c)
            
        return mags, phases
