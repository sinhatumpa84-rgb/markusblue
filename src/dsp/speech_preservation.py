import math
import numpy as np
import scipy.signal as signal
from typing import Dict, Tuple

class SpeechPreservationFilter:
    """
    Adaptive Speech-Preserving Tactical Filter Bank.
    Separates tactical voice communication formant bands (300 Hz - 3.4 kHz)
    from wideband shockwave blast frequencies (<200 Hz, >4 kHz).
    """
    def __init__(
        self,
        sr: int = 16000,
        voice_low_hz: float = 300.0,
        voice_high_hz: float = 3400.0
    ):
        self.sr = sr
        self.voice_low_hz = voice_low_hz
        self.voice_high_hz = voice_high_hz
        
        # Second-order sections (SOS) bandpass filter for voice
        self.sos_voice = signal.butter(
            4, [voice_low_hz, voice_high_hz], btype='bandpass', fs=sr, output='sos'
        )
        # Notch / Bandstop filter for complementary blast noise
        self.sos_blast_cut = signal.butter(
            2, [voice_low_hz, voice_high_hz], btype='bandstop', fs=sr, output='sos'
        )

    def process(
        self,
        audio_block: np.ndarray,
        protection_active: bool = False,
        voice_boost_db: float = 3.0
    ) -> np.ndarray:
        """
        Process audio block: If protection is active, heavily attenuate
        wideband blast energy while passing and slightly boosting speech formants.
        """
        if not protection_active:
            return audio_block
            
        # Isolate voice band
        voice_band = signal.sosfilt(self.sos_voice, audio_block)
        # Isolate extreme sub-bass shockwave & high-freq blast resonance
        blast_band = signal.sosfilt(self.sos_blast_cut, audio_block)
        
        # Attenuate blast band by -32 dB (linear ~0.025)
        blast_attenuation = 0.025
        # Maintain/boost voice band
        voice_gain = 10.0 ** (voice_boost_db / 20.0)
        
        protected_output = (voice_band * voice_gain) + (blast_band * blast_attenuation)
        return np.clip(protected_output, -1.0, 1.0)

def compute_spectral_snr(clean_ref: np.ndarray, test_signal: np.ndarray) -> float:
    """Compute Spectral SNR proxy between clean speech reference and processed audio."""
    min_len = min(len(clean_ref), len(test_signal))
    clean_ref = clean_ref[:min_len]
    test_signal = test_signal[:min_len]
    
    error = test_signal - clean_ref
    clean_pow = np.mean(clean_ref ** 2) + 1e-10
    err_pow = np.mean(error ** 2) + 1e-10
    
    snr_db = 10.0 * np.log10(clean_pow / err_pow)
    return float(snr_db)

def evaluate_speech_intelligibility_preservation(
    speech_audio: np.ndarray,
    background_audio: np.ndarray,
    impulse_audio: np.ndarray,
    limiter,
    filter_bank: SpeechPreservationFilter,
    sr: int = 16000
) -> Dict:
    """
    Test scenario: Background noise + Speech + Sudden impulse event.
    Evaluates hearing protection attenuation and speech intelligibility preservation.
    """
    target_len = sr # 1.0 second
    # Truncate / pad all components
    speech = np.pad(speech_audio, (0, max(0, target_len - len(speech_audio))))[:target_len]
    bg = np.pad(background_audio, (0, max(0, target_len - len(background_audio))))[:target_len]
    impulse = np.pad(impulse_audio, (0, max(0, target_len - len(impulse_audio))))[:target_len]
    
    # Scale components
    clean_speech_ref = speech * 0.5
    bg_scaled = bg * 0.15
    impulse_scaled = impulse * 0.95 # Dangerous loud impulse
    
    # Raw dangerous acoustic mix entering the microphone
    raw_acoustic_mix = clean_speech_ref + bg_scaled + impulse_scaled
    
    # System processing
    protected_signal = limiter.process_block(raw_acoustic_mix, force_protect=True)
    preserved_signal = filter_bank.process(protected_signal, protection_active=True)
    
    # Metrics
    raw_peak = np.max(np.abs(raw_acoustic_mix))
    raw_peak_db = 20.0 * np.log10(raw_peak + 1e-6)
    
    protected_peak = np.max(np.abs(preserved_signal))
    protected_peak_db = 20.0 * np.log10(protected_peak + 1e-6)
    
    peak_attenuation_db = raw_peak_db - protected_peak_db
    
    # Measure voice intelligibility preservation during non-impulse frames
    speech_snr_raw = compute_spectral_snr(clean_speech_ref, raw_acoustic_mix)
    speech_snr_protected = compute_spectral_snr(clean_speech_ref, preserved_signal)
    
    # Speech intelligibility proxy metric (Normalized Correlation)
    corr = np.corrcoef(clean_speech_ref, preserved_signal)[0, 1]
    intelligibility_index = float(np.clip(corr, 0.0, 1.0)) * 100.0
    
    return {
        "raw_peak_amplitude": float(raw_peak),
        "raw_peak_db": float(round(raw_peak_db, 2)),
        "protected_peak_amplitude": float(protected_peak),
        "protected_peak_db": float(round(protected_peak_db, 2)),
        "peak_attenuation_db": float(round(peak_attenuation_db, 2)),
        "speech_snr_raw_db": float(round(speech_snr_raw, 2)),
        "speech_snr_protected_db": float(round(speech_snr_protected, 2)),
        "speech_intelligibility_proxy_percent": float(round(intelligibility_index, 2)),
        "hearing_safety_clamped": bool(protected_peak <= 0.35),
        "speech_preserved": bool(intelligibility_index > 60.0),
        "raw_mix_audio": raw_acoustic_mix,
        "protected_audio": preserved_signal,
        "clean_speech_ref": clean_speech_ref
    }
