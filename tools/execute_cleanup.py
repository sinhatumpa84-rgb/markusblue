#!/usr/bin/env python3
"""
MARKUSBLUE — Safe Automated Repository Cleanup Script
SIH Problem Statement: SIH26052

Deletes verified obsolete components (Demucs, LoRa, ESP82, and scratch clones)
with rigorous invariant assertion for protected datasets (datasets, data, gunsound).
"""

import os
import sys
import shutil
import json

PROTECTED_DIRS = ["datasets", "data", "gunsound"]

def count_files_in_dir(path):
    if not os.path.exists(path):
        return 0
    return sum(len(files) for _, _, files in os.walk(path))

def main():
    print("=" * 75)
    print("MARKUSBLUE (SIH26052) — SAFE REPOSITORY CLEANUP EXECUTION")
    print("=" * 75)

    # 1. Capture Before Baseline
    counts_before = {d: count_files_in_dir(d) for d in PROTECTED_DIRS}
    print("[*] Baseline Protected Dataset Verification:")
    for d, count in counts_before.items():
        print(f"    • {d}/: {count:,} files")
    assert counts_before["datasets"] >= 13200, "Error: datasets baseline mismatch!"
    assert counts_before["data"] >= 27000, "Error: data baseline mismatch!"

    # 2. Targeted Directories for Removal
    dirs_to_remove = [
        "demucs",
        "demucs.egg-info",
        "conf",
        "embedded/esp82",
        "scratch/fresh_clone_test",
        "models/retrained_model_a",
        "models/retrained_model_b",
        ".pytest_cache",
        "checkpoints"
    ]

    # 3. Targeted Files for Removal
    files_to_remove = [
        # Upstream Demucs root files
        "Demucs.ipynb",
        "demucs.png",
        "hubconf.py",
        "Makefile",
        "setup.py",
        "setup.cfg",
        "MANIFEST.in",
        "environment-cpu.yml",
        "environment-cuda.yml",
        "outputs.tar.gz",
        "test.mp3",
        "train.py",
        "benchmark.py",
        "evaluate.py",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",

        # Demucs docs
        "docs/api.md",
        "docs/linux.md",
        "docs/mac.md",
        "docs/windows.md",
        "docs/release.md",
        "docs/sdx23.md",
        "docs/mdx.md",
        "docs/training.md",
        "docs/training_pipeline.md",

        # ESP8266/ESP82 obsolete code
        "train_esp82_student.py",
        "evaluate_esp82.py",
        "export_esp82_tflite.py",
        "src/postprocessing/esp82_dsp.py",
        "src/training/esp82_student_model.py",
        "src/preprocessing/esp82_features.py",
        "src/inference/esp82_reference.py",

        # ESP82 docs & reports
        "docs/esp82_deployment.md",
        "docs/esp82_feasibility.md",
        "docs/esp82_hardware_constraints.md",
        "docs/esp82_model_budget.md",
        "reports/esp82_evaluation.md",
        "reports/esp82_training_history.json",

        # Obsolete model weights
        "models/markusblue_esp82_fp32.tflite",
        "models/markusblue_esp82_int8.tflite",
        "models/markusblue_esp82_student_best.pt",
        "models/markusblue_esp82_student_final.pt",
        "models/tactical_baseline_model_best.pt",
        "models/tactical_baseline_model_final.pt",
        "models/tactical_baseline_model_history.json",
        "models/tactical_edge_model_best.pt",
        "models/tactical_edge_model_final.pt",
        "models/tactical_edge_model_history.json",
        "models/model_float32.tflite",
        "models/model_int8.tflite",
        "models/markusblue_final.pt",
        "models/markusblue_final.tflite",
        "models/markusblue_final_int8.tflite",
        "models/markusblue_final_traced.pt",

        # Old scratch scripts
        "scratch/build_complete_murkusblue_project.py",
        "scratch/build_murkusblue_repo.py",
        "export_edge_model.py",
        "export_tflite.py",
        "train_student.py",
        "evaluate_speech_enhancement.py",
        "realtime_demo.py",
        "prepare_dataset.py"
    ]

    deleted_dirs = []
    deleted_files = []

    # Execute Directory Deletion with Protection Guards
    for d in dirs_to_remove:
        norm_d = os.path.normpath(d)
        assert not any(norm_d.startswith(p) for p in PROTECTED_DIRS), f"FATAL: Attempted to delete protected directory '{norm_d}'!"
        if os.path.exists(norm_d):
            print(f"[-] Deleting directory: '{norm_d}' ...")
            shutil.rmtree(norm_d, ignore_errors=True)
            deleted_dirs.append(norm_d)

    # Execute File Deletion with Protection Guards
    for f in files_to_remove:
        norm_f = os.path.normpath(f)
        assert not any(norm_f.startswith(p) for p in PROTECTED_DIRS), f"FATAL: Attempted to delete protected file '{norm_f}'!"
        if os.path.exists(norm_f):
            print(f"[-] Deleting obsolete file: '{norm_f}'")
            try:
                os.remove(norm_f)
                deleted_files.append(norm_f)
            except Exception as e:
                print(f"[!] Error deleting {norm_f}: {e}")

    # Clean __pycache__ in src/, tests/, tools/
    for root, dirs, _ in os.walk(".", topdown=False):
        if any(p in root.split(os.sep) for p in PROTECTED_DIRS):
            continue
        for d in dirs:
            if d == "__pycache__":
                pycache_path = os.path.join(root, d)
                shutil.rmtree(pycache_path, ignore_errors=True)
                deleted_dirs.append(pycache_path)

    # 4. Invariant Verification After Deletion
    counts_after = {d: count_files_in_dir(d) for d in PROTECTED_DIRS}
    print("\n" + "=" * 75)
    print("[*] Verifying Dataset Protection Invariant After Cleanup:")
    print("=" * 75)
    all_intact = True
    for d in PROTECTED_DIRS:
        diff = counts_after[d] - counts_before[d]
        status = "PASSED (100% INTACT)" if diff == 0 else f"FAILED (Diff: {diff})"
        print(f"    • {d:<12}: Before={counts_before[d]:<6} After={counts_after[d]:<6} | {status}")
        if diff != 0:
            all_intact = False

    assert all_intact, "CRITICAL ERROR: Dataset integrity invariant violated!"

    log_data = {
        "deleted_directories_count": len(deleted_dirs),
        "deleted_files_count": len(deleted_files),
        "deleted_directories": deleted_dirs,
        "deleted_files": deleted_files,
        "dataset_integrity_verified": True,
        "counts_before": counts_before,
        "counts_after": counts_after
    }

    os.makedirs("audit_results", exist_ok=True)
    with open("audit_results/deletion_log.json", "w") as fp:
        json.dump(log_data, fp, indent=2)

    print(f"\n[+] Cleanup successful! Deleted {len(deleted_dirs)} directories and {len(deleted_files)} files.")
    print(f"[+] Deletion manifest logged to 'audit_results/deletion_log.json'")
    print("=" * 75)

if __name__ == "__main__":
    main()
