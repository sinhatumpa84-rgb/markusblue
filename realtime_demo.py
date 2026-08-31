import os
import time
import argparse
import numpy as np
import soundfile as sf
import torch

from src.inference.markusblue import MARKUSBLUE

def run_realtime_demo():
    parser = argparse.ArgumentParser(description="MARKUSBLUE Real-Time Streaming Audio Demo")
    parser.add_argument("--input_wav", type=str, default="datasets/gunshot/gunshot_session_0_0.wav", help="Input WAV file")
    parser.add_argument("--model_path", type=str, default="models/markusblue_final.pt", help="Path to student model")
    parser.add_argument("--chunk_size", type=int, default=512, help="Streaming audio block size in samples")
    parser.add_argument("--target_rms_dbfs", type=float, default=-16.0, help="Target conversational loudness in dBFS")
    args = parser.parse_args()

    print("==================================================")
    print("MARKUSBLUE v7.1.0 — Real-Time Streaming Audio Demo")
    print("Pipeline: Capture -> AI Speech Enhancement -> VAD -> AGC -> DRC -> Limiter -> Output")
    print("==================================================")

    if not os.path.exists(args.input_wav):
        wav_candidates = [os.path.join("datasets/speech", f) for f in os.listdir("datasets/speech") if f.endswith(".wav")]
        args.input_wav = wav_candidates[0]

    audio_data, sr = sf.read(args.input_wav)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    print(f"Loaded: '{args.input_wav}' ({len(audio_data)} samples, {sr} Hz, {len(audio_data)/sr:.2f}s)")
    
    pipeline = MARKUSBLUE(
        model_path=args.model_path if os.path.exists(args.model_path) else None,
        sr=sr,
        target_rms_dbfs=args.target_rms_dbfs
    )

    processed_blocks = []
    latencies = []
    chunk_size = args.chunk_size

    print("-" * 80)
    print(f"{'Time (s)':<10} | {'Input RMS (dBFS)':<18} | {'Output RMS (dBFS)':<18} | {'Latency (ms)':<14} | {'RTF':<8}")
    print("-" * 80)

    num_chunks = len(audio_data) // chunk_size
    for i in range(num_chunks):
        chunk = audio_data[i * chunk_size : (i + 1) * chunk_size].astype(np.float32)
        
        t0 = time.perf_counter()
        out_chunk = pipeline.enhance(chunk)
        t_proc = (time.perf_counter() - t0) * 1000.0
        
        latencies.append(t_proc)
        processed_blocks.append(out_chunk)
        
        in_rms_db = 20.0 * np.log10(max(1e-5, np.sqrt(np.mean(chunk ** 2))))
        out_rms_db = 20.0 * np.log10(max(1e-5, np.sqrt(np.mean(out_chunk ** 2))))
        chunk_duration_ms = (chunk_size / sr) * 1000.0
        rtf = t_proc / chunk_duration_ms
        
        if i % 5 == 0 or i == num_chunks - 1:
            timestamp = (i * chunk_size) / sr
            print(f"{timestamp:<10.2f} | {in_rms_db:<18.2f} | {out_rms_db:<18.2f} | {t_proc:<14.2f} | {rtf:<8.3f}")

    print("-" * 80)
    full_output = np.concatenate(processed_blocks) if processed_blocks else np.array([], dtype=np.float32)
    
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/realtime_demo_output.wav"
    sf.write(out_path, full_output, sr)
    
    mean_latency = float(np.mean(latencies))
    print(f"[OK] Real-time streaming simulation complete!")
    print(f"[OK] Mean Chunk Latency: {mean_latency:.2f} ms per {chunk_size/sr*1000:.1f} ms audio block")
    print(f"[OK] Mean Real-Time Factor: {mean_latency / (chunk_size/sr*1000):.3f} (Values < 1.0 indicate real-time capable)")
    print(f"[OK] Enhanced audio saved to: '{out_path}'")

if __name__ == "__main__":
    run_realtime_demo()
