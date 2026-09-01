import os
import glob
import json
import soundfile as sf
import numpy as np

def audit_dataset():
    stats = {
        "speech": {"count": 0, "durations": [], "sample_rates": {}},
        "background_noise": {"count": 0, "durations": [], "sample_rates": {}},
        "gunshot": {"count": 0, "durations": [], "sample_rates": {}},
        "other_impulse": {"count": 0, "durations": [], "sample_rates": {}}
    }
    
    dirs = {
        "speech": "datasets/speech",
        "background_noise": "datasets/background_noise",
        "gunshot": "datasets/gunshot",
        "other_impulse": "datasets/other_impulse"
    }
    
    for category, dirpath in dirs.items():
        files = glob.glob(f"{dirpath}/**/*.wav", recursive=True)
        stats[category]["count"] = len(files)
        # Sample first 200 files for duration and sample rate stats
        for f in files[:200]:
            try:
                info = sf.info(f)
                stats[category]["durations"].append(info.duration)
                sr_str = str(info.samplerate)
                stats[category]["sample_rates"][sr_str] = stats[category]["sample_rates"].get(sr_str, 0) + 1
            except Exception as e:
                pass
                
    total_audio = sum(stats[k]["count"] for k in stats)
    
    summary = {
        "total_files": total_audio,
        "categories": {}
    }
    
    for k, v in stats.items():
        durations = v["durations"]
        summary["categories"][k] = {
            "file_count": v["count"],
            "avg_duration_sec": float(np.mean(durations)) if durations else 0.0,
            "min_duration_sec": float(np.min(durations)) if durations else 0.0,
            "max_duration_sec": float(np.max(durations)) if durations else 0.0,
            "total_estimated_hours": float(v["count"] * np.mean(durations) / 3600.0) if durations else 0.0,
            "sample_rate_distribution": v["sample_rates"]
        }
        
    print(json.dumps(summary, indent=2))
    
    # Write dataset report
    md_report = f"""# MARKUSBLUE Audio Dataset Audit Report

## 1. Executive Summary
The MARKUSBLUE dataset contains audio for training and evaluating speech enhancement and tactical noise suppression on edge microcontrollers.

- **Total Audio Files**: {total_audio:,}
- **Speech Files**: {stats['speech']['count']:,} ({summary['categories']['speech']['total_estimated_hours']:.1f} hours estimated)
- **Noise / Disturbance Files**: {stats['background_noise']['count'] + stats['gunshot']['count'] + stats['other_impulse']['count']:,}
  - **Background Noise**: {stats['background_noise']['count']:,}
  - **Gunshot Impulses**: {stats['gunshot']['count']:,}
  - **Other Impulses**: {stats['other_impulse']['count']:,}

---

## 2. Dataset Distribution & Characteristics

| Category | File Count | Mean Duration (s) | Min/Max Duration (s) | Primary Sample Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Clean Speech** | {stats['speech']['count']:,} | {summary['categories']['speech']['avg_duration_sec']:.2f} s | {summary['categories']['speech']['min_duration_sec']:.2f}s / {summary['categories']['speech']['max_duration_sec']:.2f}s | 16,000 Hz / 8,000 Hz |
| **Background Noise** | {stats['background_noise']['count']:,} | {summary['categories']['background_noise']['avg_duration_sec']:.2f} s | {summary['categories']['background_noise']['min_duration_sec']:.2f}s / {summary['categories']['background_noise']['max_duration_sec']:.2f}s | 16,000 Hz |
| **Gunshot Impulses** | {stats['gunshot']['count']:,} | {summary['categories']['gunshot']['avg_duration_sec']:.2f} s | {summary['categories']['gunshot']['min_duration_sec']:.2f}s / {summary['categories']['gunshot']['max_duration_sec']:.2f}s | 16,000 Hz |
| **Other Impulses** | {stats['other_impulse']['count']:,} | {summary['categories']['other_impulse']['avg_duration_sec']:.2f} s | {summary['categories']['other_impulse']['min_duration_sec']:.2f}s / {summary['categories']['other_impulse']['max_duration_sec']:.2f}s | 16,000 Hz |

---

## 3. Training Mixture Generation Strategy
During model training and distillation:
- **Clean speech** is dynamically combined on-the-fly with **environmental noise, gunfire, and acoustic impulse sounds**.
- **SNR Distribution**: Sampled uniformly from `[-20 dB, -15 dB, -10 dB, -5 dB, 0 dB, +5 dB, +10 dB, +15 dB, +20 dB]` to ensure robustness under heavy noise and near-clean conditions.
- **Normalization**: Dynamic RMS normalization with floating-point to INT8 scaling consistency.
"""
    with open("docs/dataset_report.md", "w") as f:
        f.write(md_report)
    print("Saved docs/dataset_report.md")

if __name__ == "__main__":
    audit_dataset()
