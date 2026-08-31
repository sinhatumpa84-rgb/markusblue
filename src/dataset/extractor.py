import os
import zipfile
import shutil
import hashlib
from typing import Dict, List, Tuple
from tqdm import tqdm

def get_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def extract_all_datasets(
    gunsound_dir: str = "gunsound",
    raw_dir: str = "data/raw",
    extracted_dir: str = "data/extracted"
) -> Dict:
    """
    Safely extract all ZIP archives from gunsound/ into data/raw/ and data/extracted/.
    Deduplicates repeated archives and removes OS artifacts (__MACOSX, ._*).
    """
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(extracted_dir, exist_ok=True)
    
    zip_files = sorted([f for f in os.listdir(gunsound_dir) if f.endswith('.zip')])
    print(f"[*] Found {len(zip_files)} ZIP archives in '{gunsound_dir}'.")
    
    # Track unique archives by content hash
    processed_hashes = {}
    duplicate_archives = []
    extraction_summary = {
        "extracted_archives": [],
        "duplicate_archives_skipped": [],
        "total_extracted_wavs": 0,
        "archive_details": {}
    }
    
    for zf in zip_files:
        zpath = os.path.join(gunsound_dir, zf)
        file_hash = get_file_hash(zpath)
        
        if file_hash in processed_hashes:
            print(f"[!] Skipping duplicate archive: '{zf}' (matches '{processed_hashes[file_hash]}')")
            duplicate_archives.append((zf, processed_hashes[file_hash]))
            extraction_summary["duplicate_archives_skipped"].append({
                "file": zf,
                "original": processed_hashes[file_hash]
            })
            continue
            
        processed_hashes[file_hash] = zf
        archive_name = os.path.splitext(zf)[0].replace(" (1)", "")
        target_dir = os.path.join(extracted_dir, archive_name)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"[+] Extracting '{zf}' -> '{target_dir}'...")
        wav_count = 0
        with zipfile.ZipFile(zpath, 'r') as zip_ref:
            members = zip_ref.namelist()
            for member in members:
                # Filter macOS metadata artifacts
                if member.startswith('__MACOSX') or os.path.basename(member).startswith('._'):
                    continue
                
                zip_ref.extract(member, target_dir)
                if member.lower().endswith('.wav'):
                    wav_count += 1
                    
        extraction_summary["extracted_archives"].append(zf)
        extraction_summary["total_extracted_wavs"] += wav_count
        extraction_summary["archive_details"][zf] = {
            "target_dir": target_dir,
            "wav_count": wav_count
        }
        print(f"    -> Extracted {wav_count} WAV files from {zf}")
        
    print(f"\n[OK] Extraction Complete! Total WAVs extracted: {extraction_summary['total_extracted_wavs']}")
    return extraction_summary
