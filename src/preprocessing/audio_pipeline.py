import os
import io
import math
import wave
import numpy as np
import soundfile as sf
import scipy.signal as signal
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

def load_and_preprocess_wav(
    filepath: str,
    target_sr: int = 16000,
    target_duration_sec: float = 1.0,
    preserve_dynamics: bool = True
) -> Optional[np.ndarray]:
    """
    Load an audio file, convert to mono, resample to target_sr, 
    and apply controlled dynamic-preserving normalization.
    """
    try:
        data, sr = sf.read(filepath, dtype='float32')
    except Exception as e:
        return None
        
    if data is None or len(data) == 0:
        return None
        
    # Convert multi-channel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
        
    # Resample to target_sr if needed
    if sr != target_sr:
        num_target_samples = int(len(data) * (target_sr / sr))
        data = signal.resample(data, num_target_samples)
        
    # Check for NaN / Inf
    if not np.all(np.isfinite(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
        
    # Controlled normalization preserving transient dynamics
    data = normalize_audio_preserving_dynamics(data, preserve_dynamics=preserve_dynamics)
    return data

def normalize_audio_preserving_dynamics(
    audio: np.ndarray,
    headroom_db: float = -0.5,
    preserve_dynamics: bool = True
) -> np.ndarray:
    """
    Carefully normalize waveform without flattening transient crest factor.
    Ensures peaks do not clip (> 1.0 or < -1.0) while maintaining relative amplitude.
    """
    if audio is None or len(audio) == 0:
        return audio
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio # Silent or near-silent
        
    max_allowed = 10 ** (headroom_db / 20.0) # ~0.944 for -0.5 dB
    
    if peak > max_allowed:
        audio = audio * (max_allowed / peak)
    elif not preserve_dynamics:
        audio = audio * (max_allowed / peak)
    else:
        pass
        
    return np.clip(audio, -1.0, 1.0)

def extract_windows_around_peaks(
    audio: np.ndarray,
    target_len: int = 16000,
    peak_offset_ratio: float = 0.2
) -> List[np.ndarray]:
    """
    Extract fixed-length audio windows centered/aligned to impulsive peaks.
    Aligns the peak at roughly 20% into the window for optimal temporal context.
    """
    if len(audio) == 0:
        return []
        
    if len(audio) <= target_len:
        pad_width = target_len - len(audio)
        peak_idx = int(np.argmax(np.abs(audio)))
        desired_peak_pos = int(target_len * peak_offset_ratio)
        lead_pad = max(0, min(pad_width, desired_peak_pos - peak_idx))
        trail_pad = pad_width - lead_pad
        padded = np.pad(audio, (lead_pad, trail_pad), mode='constant')
        return [padded]
        
    # Find prominent energy peaks
    peak_idx = int(np.argmax(np.abs(audio)))
    start = max(0, peak_idx - int(target_len * peak_offset_ratio))
    end = start + target_len
    if end > len(audio):
        end = len(audio)
        start = max(0, end - target_len)
        
    win = audio[start:end]
    if len(win) < target_len:
        win = np.pad(win, (0, target_len - len(win)), mode='constant')
    return [win]

def process_raw_audio(
    extracted_dir: str = "data/extracted",
    processed_dir: str = "data/processed",
    target_sr: int = 16000,
    target_duration_sec: float = 1.0,
    max_samples_per_category: int = 3000
) -> pd.DataFrame:
    """
    Process all extracted raw gunshot audio and curate diverse negative acoustic classes.
    Standardizes to 16 kHz mono 16-bit PCM WAV.
    """
    target_samples = int(target_sr * target_duration_sec)
    os.makedirs(processed_dir, exist_ok=True)
    
    categories = ["gunshot", "speech", "background", "other_impulse"]
    for cat in categories:
        os.makedirs(os.path.join(processed_dir, cat), exist_ok=True)
        
    processed_records = []
    
    # 1. Process Gunshot Audio from extracted directories
    print("[*] Processing extracted real gunshot recordings...")
    all_extracted_wavs = []
    for root, _, files in os.walk(extracted_dir):
        for f in files:
            if f.lower().endswith('.wav') and not f.startswith('._'):
                all_extracted_wavs.append(os.path.join(root, f))
                
    print(f"[*] Found {len(all_extracted_wavs)} extracted WAV files.")
    
    # Track source sessions to prevent intra-recording correlation
    count_saved = 0
    for wav_path in tqdm(all_extracted_wavs, desc="Processing Gunshots"):
        if count_saved >= max_samples_per_category:
            break
            
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        rel_path = os.path.relpath(wav_path, extracted_dir)
        source_group = rel_path.split(os.sep)[0]
        
        # Sub-session grouping
        if "C3GD-Dataset" in source_group:
            # e.g., 7-oh_farm-benelli_m4-DAVE-yyjjHTJQoHkP-0
            parts = base_name.split('-')
            session_id = f"c3gd_{parts[4]}" if len(parts) >= 5 else f"c3gd_{source_group}"
        elif "edge" in source_group.lower():
            # e.g., 880b3ce5-9c19-4c12-a813-b223bb4f2897_v1
            parts = base_name.split('_')
            session_id = f"edge_{parts[0]}" if len(parts) >= 2 else f"edge_{source_group}"
        else:
            session_id = f"zoom_{source_group}"
            
        audio = load_and_preprocess_wav(
            wav_path,
            target_sr=target_sr,
            target_duration_sec=target_duration_sec,
            preserve_dynamics=True
        )
        
        if audio is None or len(audio) == 0:
            continue
            
        rms = math.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        if peak < 0.02:
            continue # Skip near-silent
            
        windows = extract_windows_around_peaks(audio, target_len=target_samples)
        for w_idx, win in enumerate(windows):
            sample_id = f"gun_{session_id}_{base_name}_w{w_idx}"
            out_filename = f"{sample_id}.wav"
            out_path = os.path.join(processed_dir, "gunshot", out_filename)
            
            sf.write(out_path, win, target_sr, subtype='PCM_16')
            
            processed_records.append({
                "sample_id": sample_id,
                "source_group": session_id,
                "category": "gunshot",
                "class_label": "DANGEROUS_IMPULSE",
                "filepath": out_path,
                "duration": target_duration_sec,
                "sample_rate": target_sr,
                "channels": 1,
                "peak_amplitude": float(np.max(np.abs(win))),
                "rms_energy": float(math.sqrt(np.mean(win ** 2))),
                "crest_factor": float(np.max(np.abs(win)) / (math.sqrt(np.mean(win ** 2)) + 1e-8))
            })
            count_saved += 1
            
    print(f"[OK] Processed {count_saved} gunshot samples.")
    
    # 2. Synthesize High-Fidelity Tactical Speech, Background, and Hard Negative Impulse Sets
    print("[*] Generating realistic tactical acoustic sets (Speech, Ambient Noise, Hard Negative Transients)...")
    records_extra = generate_realistic_tactical_acoustic_sets(
        processed_dir=processed_dir,
        target_sr=target_sr,
        target_samples=target_samples,
        n_samples_per_class=1200
    )
    processed_records.extend(records_extra)
    
    df = pd.DataFrame(processed_records)
    csv_out = os.path.join(processed_dir, "processed_dataset_catalog.csv")
    df.to_csv(csv_out, index=False)
    print(f"[OK] Total processed samples in catalog: {len(df)}. Saved to '{csv_out}'.")
    return df

def generate_realistic_tactical_acoustic_sets(
    processed_dir: str,
    target_sr: int = 16000,
    target_samples: int = 16000,
    n_samples_per_class: int = 1200
) -> List[Dict]:
    """
    Generates diverse tactical speech, realistic military ambient noise, and hard negative non-hazardous impulses.
    Includes male/female speech, distance filters, radio bandpass, engine acoustics, wind turbulence,
    and mechanical transients (door slams, metal clatter, tool drops, weapon handling clicks).
    """
    rng = np.random.RandomState(42)
    records = []
    
    # -------------------------------------------------------------
    # CLASS 1: NORMAL_SPEECH (Varied F0, Formants, Radio, Whispers, Shouts)
    # -------------------------------------------------------------
    print("  -> Curating NORMAL_SPEECH...")
    for i in range(n_samples_per_class):
        session_idx = i // 40 # 30 distinct speaker session groups
        session_id = f"speaker_session_{session_idx:02d}"
        sample_id = f"speech_{session_id}_{i:04d}"
        t = np.linspace(0, 1.0, target_samples, endpoint=False)
        
        # Gender & pitch variation: Male (85-160 Hz), Female (165-280 Hz), Child/Radio (200-350 Hz)
        speaker_type = session_idx % 3
        if speaker_type == 0:
            f0 = rng.uniform(90, 150)   # Male
        elif speaker_type == 1:
            f0 = rng.uniform(170, 260)  # Female
        else:
            f0 = rng.uniform(130, 220)  # Tactical radio operator
            
        # Intonation & pitch contour
        pitch_contour = f0 * (1.0 + 0.08 * np.sin(2 * np.pi * rng.uniform(2, 5) * t) + 0.03 * rng.randn(target_samples))
        phase = np.cumsum(2 * np.pi * pitch_contour / target_sr)
        
        # Glottal pulse synthesis (Liljencrants-Fant glottal model approximation)
        voice = np.zeros(target_samples)
        num_harmonics = int(rng.uniform(12, 28))
        for h in range(1, num_harmonics):
            # Glottal spectral tilt: -6 to -12 dB/octave
            tilt = 1.0 / (h ** rng.uniform(0.9, 1.4))
            voice += tilt * np.sin(h * phase + rng.uniform(0, 2*np.pi))
            
        # Dynamic vocal tract formant filtering (F1: 300-800Hz, F2: 900-2400Hz, F3: 2500-3500Hz)
        f1 = rng.uniform(350, 750)
        f2 = rng.uniform(1100, 2200)
        f3 = rng.uniform(2500, 3200)
        
        # Multi-stage biquad resonance
        sos_formant1 = signal.butter(2, [max(100, f1 - 120), min(target_sr//2 - 100, f1 + 120)], btype='bandpass', fs=target_sr, output='sos')
        sos_formant2 = signal.butter(2, [max(100, f2 - 150), min(target_sr//2 - 100, f2 + 150)], btype='bandpass', fs=target_sr, output='sos')
        sos_formant3 = signal.butter(2, [max(100, f3 - 200), min(target_sr//2 - 100, f3 + 200)], btype='bandpass', fs=target_sr, output='sos')
        
        filtered_speech = (0.5 * signal.sosfilt(sos_formant1, voice) + 
                           0.35 * signal.sosfilt(sos_formant2, voice) + 
                           0.15 * signal.sosfilt(sos_formant3, voice))
                           
        # Syllabic amplitude envelope modulation (conversational cadence: 3.5 - 6.5 Hz)
        cadence_freq = rng.uniform(3.5, 6.5)
        syllables = (0.5 * (1 + np.sin(2 * np.pi * cadence_freq * t + rng.uniform(0, 2*np.pi)))) ** 2
        
        # Add consonant noise bursts (fricatives /s/, /t/, /k/)
        consonant_mask = (rng.rand(target_samples) > 0.85).astype(float)
        consonant_noise = signal.sosfilt(
            signal.butter(2, [3000, 7000], btype='bandpass', fs=target_sr, output='sos'),
            rng.randn(target_samples)
        ) * 0.15 * consonant_mask
        
        speech = (filtered_speech * syllables) + consonant_noise
        
        # Add realistic tactical acoustic variations:
        # Distance attenuation, low-pass room reflection, or military radio bandpass
        proc_mode = i % 4
        if proc_mode == 0:
            # Tactical VHF/UHF Radio Bandpass (300 Hz - 3.4 kHz) with slight channel hiss
            sos_radio = signal.butter(3, [300, 3400], btype='bandpass', fs=target_sr, output='sos')
            speech = signal.sosfilt(sos_radio, speech) + 0.02 * rng.randn(target_samples)
        elif proc_mode == 1:
            # Distant speaker (air absorption high-shelf rolloff)
            sos_dist = signal.butter(2, 2200 / (target_sr/2), btype='low', output='sos')
            speech = signal.sosfilt(sos_dist, speech) * 0.6 + 0.03 * rng.randn(target_samples)
        elif proc_mode == 2:
            # Close tactical whisper / headset microphone
            speech = speech * 0.9 + 0.01 * rng.randn(target_samples)
        else:
            # Shouted tactical command (higher vocal effort, higher harmonics)
            speech = np.tanh(speech * 1.8) * 0.6
            
        speech = normalize_audio_preserving_dynamics(speech * 0.45)
        out_path = os.path.join(processed_dir, "speech", f"{sample_id}.wav")
        sf.write(out_path, speech.astype(np.float32), target_sr, subtype='PCM_16')
        
        records.append({
            "sample_id": sample_id,
            "source_group": session_id,
            "category": "speech",
            "class_label": "NORMAL_SPEECH",
            "filepath": out_path,
            "duration": 1.0,
            "sample_rate": target_sr,
            "channels": 1,
            "peak_amplitude": float(np.max(np.abs(speech))),
            "rms_energy": float(math.sqrt(np.mean(speech ** 2))),
            "crest_factor": float(np.max(np.abs(speech)) / (math.sqrt(np.mean(speech ** 2)) + 1e-8))
        })

    # -------------------------------------------------------------
    # CLASS 2: BACKGROUND_NOISE (Engines, Wind, Generators, Rain, Radio)
    # -------------------------------------------------------------
    print("  -> Curating BACKGROUND_NOISE...")
    for i in range(n_samples_per_class):
        session_idx = i // 40 # 30 distinct background session groups
        session_id = f"bg_session_{session_idx:02d}"
        sample_id = f"bg_{session_id}_{i:04d}"
        t = np.linspace(0, 1.0, target_samples, endpoint=False)
        noise_type = session_idx % 5
        
        if noise_type == 0:
            # Heavy Military Armored Vehicle / Diesel Engine (BMP/T-90)
            rpm = rng.uniform(25, 48)
            engine = np.zeros(target_samples)
            for h in range(1, 10):
                engine += (1.0 / (h ** 0.9)) * np.sin(2 * np.pi * h * rpm * t + rng.uniform(0, 2*np.pi))
            # Exhaust rumble
            exhaust_noise = signal.sosfilt(signal.butter(2, [20, 450], btype='bandpass', fs=target_sr, output='sos'), rng.randn(target_samples))
            bg = engine * 0.5 + exhaust_noise * 0.4
        elif noise_type == 1:
            # High-Altitude Wind Turbulence & Sensor Flutter
            white = rng.randn(target_samples)
            sos_wind = signal.butter(1, rng.uniform(150, 450) / (target_sr / 2), btype='low', output='sos')
            wind_base = signal.sosfilt(sos_wind, white)
            # Gust envelope
            gust = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.3, 1.8) * t) + 0.2 * np.sin(2 * np.pi * rng.uniform(3, 7) * t)
            bg = wind_base * np.clip(gust, 0.2, 1.5)
        elif noise_type == 2:
            # Tactical Diesel Field Generator (60 Hz / 50 Hz Hum + Turbocharger Whine)
            grid_hz = rng.choice([50.0, 60.0])
            gen = np.sin(2 * np.pi * grid_hz * t) + 0.5 * np.sin(2 * np.pi * 2 * grid_hz * t) + 0.25 * np.sin(2 * np.pi * 3 * grid_hz * t)
            turbo = 0.1 * np.sin(2 * np.pi * rng.uniform(1800, 3200) * t)
            gen_noise = signal.sosfilt(signal.butter(2, [40, 900], btype='bandpass', fs=target_sr, output='sos'), rng.randn(target_samples))
            bg = gen * 0.4 + turbo + gen_noise * 0.35
        elif noise_type == 3:
            # Tactical Radio Channel Squelch & Static Hiss
            white = rng.randn(target_samples)
            sos_squelch = signal.butter(3, [350, 3200], btype='bandpass', fs=target_sr, output='sos')
            bg = signal.sosfilt(sos_squelch, white) * 0.45
        else:
            # Open Field Ambient (Rain, Foliage Rustle, Distant Noise)
            rain_noise = signal.sosfilt(signal.butter(2, [800, 6000], btype='bandpass', fs=target_sr, output='sos'), rng.randn(target_samples))
            low_rumble = signal.sosfilt(signal.butter(2, 200 / (target_sr/2), btype='low', output='sos'), rng.randn(target_samples))
            bg = rain_noise * 0.4 + low_rumble * 0.3
            
        bg = normalize_audio_preserving_dynamics(bg * 0.35)
        out_path = os.path.join(processed_dir, "background", f"{sample_id}.wav")
        sf.write(out_path, bg.astype(np.float32), target_sr, subtype='PCM_16')
        
        records.append({
            "sample_id": sample_id,
            "source_group": session_id,
            "category": "background",
            "class_label": "BACKGROUND_NOISE",
            "filepath": out_path,
            "duration": 1.0,
            "sample_rate": target_sr,
            "channels": 1,
            "peak_amplitude": float(np.max(np.abs(bg))),
            "rms_energy": float(math.sqrt(np.mean(bg ** 2))),
            "crest_factor": float(np.max(np.abs(bg)) / (math.sqrt(np.mean(bg ** 2)) + 1e-8))
        })

    # -------------------------------------------------------------
    # CLASS 3: OTHER_IMPULSE (Hard Negative Non-Gunfire Transients)
    # -------------------------------------------------------------
    print("  -> Curating OTHER_IMPULSE (Door slams, metal drops, gear clicks, weapon racks)...")
    for i in range(n_samples_per_class):
        session_idx = i // 40 # 30 distinct transient session groups
        session_id = f"other_imp_session_{session_idx:02d}"
        sample_id = f"other_imp_{session_id}_{i:04d}"
        t = np.linspace(0, 1.0, target_samples, endpoint=False)
        imp_type = session_idx % 6
        
        impulse_pos = int(rng.uniform(0.12, 0.45) * target_samples)
        imp = np.zeros(target_samples)
        
        if imp_type == 0:
            # Heavy Vehicle Door / Armored Hatch Slam (low resonance 60-140 Hz, ~60ms decay)
            decay_len = int(target_sr * rng.uniform(0.05, 0.12))
            decay = np.exp(-np.linspace(0, 6.0, decay_len))
            res_freq = rng.uniform(60, 130)
            slam_tone = np.sin(2 * np.pi * res_freq * np.linspace(0, decay_len/target_sr, decay_len)) * decay
            thud_noise = signal.sosfilt(signal.butter(2, [40, 300], btype='bandpass', fs=target_sr, output='sos'), rng.randn(decay_len)) * decay
            imp[impulse_pos:impulse_pos+decay_len] = slam_tone * 0.6 + thud_noise * 0.4
        elif imp_type == 1:
            # Dropped Metal Tool / Empty Magazine on Concrete (multi-modal metal ring: 1.5k, 2.8k, 4.2k Hz)
            decay_len = int(target_sr * rng.uniform(0.04, 0.10))
            decay = np.exp(-np.linspace(0, 7.5, decay_len))
            time_vec = np.linspace(0, decay_len/target_sr, decay_len)
            metal_ring = (
                0.4 * np.sin(2 * np.pi * rng.uniform(1400, 1900) * time_vec) +
                0.35 * np.sin(2 * np.pi * rng.uniform(2600, 3200) * time_vec) +
                0.25 * np.sin(2 * np.pi * rng.uniform(4000, 4800) * time_vec)
            ) * decay
            imp[impulse_pos:impulse_pos+decay_len] = metal_ring
        elif imp_type == 2:
            # Tactical Rifle Charging Handle / Bolt Rack / Mechanical Click (sharp 1-3 kHz transient, ~15ms decay)
            decay_len = int(target_sr * rng.uniform(0.010, 0.025))
            decay = np.exp(-np.linspace(0, 9.0, decay_len))
            click_noise = signal.sosfilt(signal.butter(3, [1200, 5500], btype='bandpass', fs=target_sr, output='sos'), rng.randn(decay_len)) * decay
            imp[impulse_pos:impulse_pos+decay_len] = click_noise
            # Double click (secondary sear release)
            sec_pos = impulse_pos + int(target_sr * rng.uniform(0.03, 0.06))
            if sec_pos + decay_len < target_samples:
                imp[sec_pos:sec_pos+decay_len] += click_noise * 0.6
        elif imp_type == 3:
            # Construction Hammer / Shovel Impact on Soil / Rock
            decay_len = int(target_sr * rng.uniform(0.02, 0.05))
            decay = np.exp(-np.linspace(0, 8.0, decay_len))
            freq = rng.uniform(300, 800)
            thump = np.sin(2 * np.pi * freq * np.linspace(0, decay_len/target_sr, decay_len)) * decay
            thump += 0.3 * rng.randn(decay_len) * decay
            imp[impulse_pos:impulse_pos+decay_len] = thump
        elif imp_type == 4:
            # Hand Clap / Finger Snap / Gear Buckle Snap
            decay_len = int(target_sr * rng.uniform(0.015, 0.035))
            decay = np.exp(-np.linspace(0, 8.0, decay_len))
            snap_noise = signal.sosfilt(signal.butter(2, [700, 4200], btype='bandpass', fs=target_sr, output='sos'), rng.randn(decay_len)) * decay
            imp[impulse_pos:impulse_pos+decay_len] = snap_noise
        else:
            # Boot on Gravel / Heavy Footstep
            decay_len = int(target_sr * rng.uniform(0.03, 0.07))
            decay = np.exp(-np.linspace(0, 6.0, decay_len))
            step_noise = signal.sosfilt(signal.butter(2, [150, 2500], btype='bandpass', fs=target_sr, output='sos'), rng.randn(decay_len)) * decay
            imp[impulse_pos:impulse_pos+decay_len] = step_noise
            
        # Add ambient floor
        imp = imp + 0.015 * rng.randn(target_samples)
        imp = normalize_audio_preserving_dynamics(imp * 0.75)
        out_path = os.path.join(processed_dir, "other_impulse", f"{sample_id}.wav")
        sf.write(out_path, imp.astype(np.float32), target_sr, subtype='PCM_16')
        
        records.append({
            "sample_id": sample_id,
            "source_group": session_id,
            "category": "other_impulse",
            "class_label": "OTHER_IMPULSE",
            "filepath": out_path,
            "duration": 1.0,
            "sample_rate": target_sr,
            "channels": 1,
            "peak_amplitude": float(np.max(np.abs(imp))),
            "rms_energy": float(math.sqrt(np.mean(imp ** 2))),
            "crest_factor": float(np.max(np.abs(imp)) / (math.sqrt(np.mean(imp ** 2)) + 1e-8))
        })
        
    return records
