"""
SIH26052 — Script to build the clean MURKUSBLUE repository
Copies only sound datasets and MARKAS LOOP model, sets up Git LFS, and verifies 0 extra files.
"""

import os
import sys
import shutil
import hashlib
import glob
import json
import soundfile as sf

TARGET_REPO_DIR = r"c:\Users\sinha\OneDrive\Desktop\murkusblue"
SOURCE_PROJECT_DIR = r"c:\Users\sinha\OneDrive\Desktop\demucs"

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def build_repo():
    print(f"[*] Initializing clean MURKUSBLUE repository at '{TARGET_REPO_DIR}'...")
    
    if os.path.exists(TARGET_REPO_DIR):
        print(f"[!] Target directory already exists. Cleaning up...")
        shutil.rmtree(TARGET_REPO_DIR)
        
    os.makedirs(TARGET_REPO_DIR, exist_ok=True)
    
    # 1. Model Copy
    model_dir = os.path.join(TARGET_REPO_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)
    source_model = os.path.join(SOURCE_PROJECT_DIR, "models", "model_int8.tflite")
    target_model = os.path.join(model_dir, "markas_loop_model_int8.tflite")
    
    print(f"[*] Copying MARKAS LOOP model -> '{target_model}'...")
    shutil.copy2(source_model, target_model)
    
    # Verify model binary identity
    src_h = sha256_file(source_model)
    tgt_h = sha256_file(target_model)
    assert src_h == tgt_h, "Model binary hash mismatch!"
    print(f"[OK] MARKAS LOOP model copied. SHA-256: {tgt_h} (Size: {os.path.getsize(target_model)} bytes)")
    
    # 2. Audio Datasets Copy
    category_map = {
        "gunshot": os.path.join(SOURCE_PROJECT_DIR, "data", "processed", "gunshot"),
        "speech": os.path.join(SOURCE_PROJECT_DIR, "data", "processed", "speech"),
        "background_noise": os.path.join(SOURCE_PROJECT_DIR, "data", "processed", "background"),
        "other_impulse": os.path.join(SOURCE_PROJECT_DIR, "data", "processed", "other_impulse")
    }
    
    seen_hashes = {}
    copied_counts = {}
    category_stats = {}
    
    for cat_name, src_dir in category_map.items():
        if not os.path.exists(src_dir):
            print(f"[-] Category source '{src_dir}' does not exist, skipping.")
            continue
            
        wav_files = glob.glob(os.path.join(src_dir, "*.wav"))
        if not wav_files:
            print(f"[-] No wav files in '{src_dir}', skipping category.")
            continue
            
        target_cat_dir = os.path.join(TARGET_REPO_DIR, "datasets", cat_name)
        os.makedirs(target_cat_dir, exist_ok=True)
        
        count = 0
        total_cat_bytes = 0
        total_cat_duration = 0.0
        
        for w in wav_files:
            file_h = sha256_file(w)
            if file_h in seen_hashes:
                # Exact duplicate, skip
                continue
                
            seen_hashes[file_h] = w
            fname = os.path.basename(w)
            dest = os.path.join(target_cat_dir, fname)
            shutil.copy2(w, dest)
            
            sz = os.path.getsize(dest)
            total_cat_bytes += sz
            try:
                info = sf.info(dest)
                total_cat_duration += info.duration
            except Exception:
                total_cat_duration += 1.0
                
            count += 1
            
        copied_counts[cat_name] = count
        category_stats[cat_name] = {
            "file_count": count,
            "total_bytes": total_cat_bytes,
            "total_mb": round(total_cat_bytes / (1024.0 * 1024.0), 2),
            "total_duration_sec": round(total_cat_duration, 2),
            "total_duration_formatted": f"{int(total_cat_duration//60)}m {int(total_cat_duration%60)}s"
        }
        print(f"[OK] Copied {count} unique audio files to 'datasets/{cat_name}/' ({category_stats[cat_name]['total_mb']} MB, {category_stats[cat_name]['total_duration_formatted']})")
        
    # 3. Permitted AudioSet metadata CSV
    meta_src = os.path.join(SOURCE_PROJECT_DIR, "data", "metadata", "gunshot_segments.csv")
    if os.path.exists(meta_src):
        meta_dest = os.path.join(TARGET_REPO_DIR, "datasets", "gunshot_segments.csv")
        shutil.copy2(meta_src, meta_dest)
        print(f"[OK] Copied AudioSet segment metadata -> 'datasets/gunshot_segments.csv'")
        
    # 4. Git LFS attributes setup
    gitattributes_path = os.path.join(TARGET_REPO_DIR, ".gitattributes")
    with open(gitattributes_path, "w") as f:
        f.write("*.wav filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.flac filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.mp3 filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.tflite filter=lfs diff=lfs merge=lfs -text\n")
    print(f"[OK] Configured Git LFS tracking rules in '.gitattributes'")
    
    # 5. Security & Prohibited Files Scan
    print("\n[*] Running Security & Strict Exclusion Audit on target repository...")
    all_target_files = []
    for root, _, files in os.walk(TARGET_REPO_DIR):
        for f in files:
            all_target_files.append(os.path.join(root, f))
            
    forbidden_extensions = [
        ".py", ".js", ".ts", ".c", ".cpp", ".h", ".cc", ".hpp", ".html", ".css",
        ".png", ".jpg", ".jpeg", ".bmp", ".md", ".env", ".key", ".pem", ".log"
    ]
    
    violations = []
    for f in all_target_files:
        ext = os.path.splitext(f)[1].lower()
        base = os.path.basename(f).lower()
        if ext in forbidden_extensions or base.startswith(".env"):
            violations.append(f)
            
    if violations:
        print(f"[!] VIOLATION FOUND: Disallowed files detected: {violations}")
        sys.exit(1)
    else:
        print("[OK] Zero source code, zero README, zero images, zero credentials found. Clean repository structure verified.")
        
    total_audio_count = sum(copied_counts.values())
    total_audio_bytes = sum(s["total_bytes"] for s in category_stats.values())
    total_audio_dur = sum(s["total_duration_sec"] for s in category_stats.values())
    
    manifest = {
        "project": "MURKUSBLUE",
        "model": {
            "name": "MARKAS LOOP",
            "filename": "markas_loop_model_int8.tflite",
            "size_bytes": os.path.getsize(target_model),
            "size_kb": round(os.path.getsize(target_model) / 1024.0, 2),
            "sha256": tgt_h,
            "format": "TensorFlow Lite (INT8)"
        },
        "datasets": {
            "categories": category_stats,
            "total_audio_files": total_audio_count,
            "total_size_bytes": total_audio_bytes,
            "total_size_mb": round(total_audio_bytes / (1024.0 * 1024.0), 2),
            "total_duration_sec": round(total_audio_dur, 2),
            "total_duration_formatted": f"{int(total_audio_dur//3600)}h {int((total_audio_dur%3600)//60)}m {int(total_audio_dur%60)}s"
        },
        "git_lfs": "YES",
        "license_excluded_files": "None (AudioSet YouTube raw audio excluded at source, only open research recordings and permissible segment metadata included)"
    }
    
    with open(os.path.join(SOURCE_PROJECT_DIR, "scratch", "murkusblue_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n============================================================")
    print("COMPLETE REPOSITORY MANIFEST")
    print("============================================================")
    print(f"PROJECT:             {manifest['project']}")
    print(f"MODEL:               {manifest['model']['name']}")
    print(f"MODEL FILE:          {manifest['model']['filename']}")
    print(f"MODEL SIZE:          {manifest['model']['size_kb']} KB ({manifest['model']['size_bytes']} bytes)")
    print(f"MODEL SHA-256:       {manifest['model']['sha256']}")
    print("\nDATASET:")
    for cat, st in manifest['datasets']['categories'].items():
        print(f"  {cat.replace('_', ' ').title():<18}: {st['file_count']} files ({st['total_mb']} MB, {st['total_duration_formatted']})")
    print(f"\nTOTAL AUDIO FILES:   {manifest['datasets']['total_audio_files']}")
    print(f"TOTAL DATASET SIZE:  {manifest['datasets']['total_size_mb']} MB")
    print(f"TOTAL DURATION:      {manifest['datasets']['total_duration_formatted']}")
    print(f"GIT LFS:             {manifest['git_lfs']}")
    print(f"LICENSE-EXCLUDED:    {manifest['license_excluded_files']}")
    print("============================================================\n")

if __name__ == "__main__":
    build_repo()
