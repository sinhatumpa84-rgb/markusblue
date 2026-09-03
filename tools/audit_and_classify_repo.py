#!/usr/bin/env python3
"""
MARKUSBLUE — Comprehensive Repository Audit & File Classifier
SIH Problem Statement: SIH26052

Performs recursive inventory and dependency tracing across all files,
categorizing them into Classes A through J, auditing Demucs/LoRa/ESP82 references,
and verifying dataset counts.
"""

import os
import sys
import json
import re

# Protected dataset folders that must NEVER be modified or deleted
PROTECTED_DATASET_DIRS = {"datasets", "data", "gunsound"}

def audit_repository():
    print("=" * 75)
    print("MARKUSBLUE (SIH26052) — COMPLETE REPOSITORY INVENTORY & CLASSIFIER")
    print("=" * 75)

    root_items = os.listdir(".")
    all_files = []
    dataset_counts = {k: 0 for k in PROTECTED_DATASET_DIRS}

    # 1. Baseline dataset count
    for d in PROTECTED_DATASET_DIRS:
        if os.path.exists(d):
            count = sum(len(files) for _, _, files in os.walk(d))
            dataset_counts[d] = count
    print(f"[*] Baseline Protected Dataset File Counts:")
    for d, cnt in dataset_counts.items():
        print(f"    • {d}/: {cnt:,} files")

    # 2. Search patterns for legacy / forbidden technologies
    demucs_regex = re.compile(r'\bdemucs\b|\bfacebookresearch\b|\bhtshift\b', re.IGNORECASE)
    lora_regex = re.compile(r'\blora\b|\bsx1262\b|\bsx1278\b|\blorawan\b', re.IGNORECASE)
    esp82_regex = re.compile(r'\besp8266\b|\besp82\b|\besp-12e\b', re.IGNORECASE)

    demucs_hits = []
    lora_hits = []
    esp82_hits = []

    inventory = []

    # Demucs-exclusive legacy root files and folders
    demucs_legacy_items = {
        "demucs", "demucs.egg-info", "conf", "Demucs.ipynb", "demucs.png",
        "hubconf.py", "Makefile", "setup.py", "environment-cpu.yml",
        "environment-cuda.yml", "outputs.tar.gz", "test.mp3"
    }

    # ESP8266/ESP82 legacy items
    esp82_legacy_items = {
        "evaluate_esp82.py", "export_esp82_tflite.py", "train_esp82_student.py",
        "embedded/esp82", "src/postprocessing/esp82_dsp.py",
        "src/training/esp82_student_model.py", "src/preprocessing/esp82_features.py",
        "src/inference/esp82_reference.py"
    }

    for root, dirs, files in os.walk('.'):
        if '.git' in root.split(os.sep):
            continue

        rel_root = os.path.relpath(root, '.').replace(os.sep, '/')
        top_dir = rel_root.split('/')[0]

        for f in files:
            file_rel = os.path.normpath(os.path.join(root, f)).replace(os.sep, '/')
            size = os.path.getsize(file_rel)
            ext = os.path.splitext(f)[1].lower()

            is_dataset = (top_dir in PROTECTED_DATASET_DIRS)

            classification = "UNKNOWN"
            reason = ""

            # Check contents for legacy strings if text file and not dataset
            if not is_dataset and size < 5000000 and ext in ['.py', '.cpp', '.h', '.cc', '.md', '.txt', '.json', '.yaml', '.yml', '.toml', '.ini', '.rst']:
                try:
                    with open(file_rel, 'r', encoding='utf-8', errors='ignore') as fp:
                        content = fp.read()
                        if demucs_regex.search(content):
                            demucs_hits.append(file_rel)
                        if lora_regex.search(content):
                            lora_hits.append(file_rel)
                        if esp82_regex.search(content):
                            esp82_hits.append(file_rel)
                except Exception:
                    pass

            if is_dataset:
                classification = "PROTECTED_DATASET"
                reason = "Original raw acoustic dataset asset (strictly preserved)"
            elif file_rel.startswith("firmware/esp32s3/"):
                classification = "A" # REQUIRED FOR ESP32-S3 HARDWARE
                reason = "Core ESP32-S3 firmware code"
            elif file_rel.startswith("hardware/"):
                classification = "A" # REQUIRED FOR ESP32-S3 HARDWARE
                reason = "Hardware pinout, power tree, and wiring specification"
            elif file_rel in ["models/markusblue_esp32s3_best.pt", "models/markusblue_esp32s3_int8.tflite", "models/esp32s3_model_metadata.json"]:
                classification = "B" # REQUIRED FOR MODEL DEPLOYMENT
                reason = "Active trained and quantized MARKUSBLUE student model"
            elif file_rel in ["firmware/esp32s3/platformio.ini", "requirements.txt", "pyproject.toml"]:
                classification = "C" # REQUIRED FOR BUILD
                reason = "Primary build & dependency configuration"
            elif file_rel in ["evaluate_esp32s3.py", "tools/sih_demo_suite.py"] or file_rel.startswith("tests/"):
                classification = "D" # REQUIRED FOR HARDWARE TESTING
                reason = "Active DSP/AI hardware test or evaluation suite"
            elif file_rel in ["README.md", "docs/architecture.md", "docs/sih26052_mapping.md", "docs/validation_report.md"]:
                classification = "E" # REQUIRED FOR DOCUMENTATION
                reason = "Active primary project documentation"
            elif file_rel in ["tools/verify_datasets.py", "export_esp32s3_model.py", "train_esp32s3_student.py"]:
                classification = "F" # OPTIONAL DEVELOPMENT TOOL
                reason = "Active model training / export tool"
            elif any(file_rel.startswith(item) or file_rel == item for item in demucs_legacy_items):
                classification = "G" # OLD / OBSOLETE (Demucs)
                reason = "Legacy upstream Demucs file (unrelated to ESP32-S3 hardware)"
            elif any(file_rel.startswith(item) or file_rel == item for item in esp82_legacy_items) or "esp82" in file_rel.lower():
                classification = "G" # OLD / OBSOLETE (ESP8266/ESP82)
                reason = "Obsolete ESP8266/ESP82 legacy file (superseded by ESP32-S3)"
            elif "demucs" in file_rel.lower() and not is_dataset:
                classification = "G" # OLD / OBSOLETE (Demucs)
                reason = "Demucs-specific legacy component"
            elif any(x in file_rel for x in ["__pycache__", ".pytest_cache", ".vscode", "audit_results"]):
                classification = "I" # GENERATED / CACHE
                reason = "IDE cache or local build artifact"
            elif file_rel.startswith("docs/"):
                if any(x in file_rel for x in ["esp82", "linux.md", "mac.md", "windows.md", "release.md", "sdx23.md", "mdx.md"]):
                    classification = "G" # OLD / OBSOLETE
                    reason = "Outdated platform or Demucs documentation"
                else:
                    classification = "J" # UNSAFE TO DELETE WITHOUT DEPENDENCY CHECK
                    reason = "Supporting documentation"
            elif file_rel.startswith("models/"):
                if "esp82" in file_rel or "tactical_baseline" in file_rel or "model_float32" in file_rel:
                    classification = "G"
                    reason = "Obsolete model checkpoint"
                else:
                    classification = "J"
                    reason = "Model asset"
            elif file_rel.startswith("src/"):
                if "esp82" in file_rel:
                    classification = "G"
                    reason = "Legacy ESP82 DSP/features/inference"
                else:
                    classification = "J" # Core Python library
                    reason = "Supporting DSP/training module"
            else:
                classification = "J"
                reason = "Unclassified file requiring manual dependency review"

            inventory.append({
                "path": file_rel,
                "size": size,
                "classification": classification,
                "reason": reason
            })

    # Summary by classification
    summary = {}
    for item in inventory:
        c = item["classification"]
        summary[c] = summary.get(c, 0) + 1

    print("\n" + "=" * 75)
    print("CLASSIFICATION SUMMARY (Non-Dataset Items):")
    print("=" * 75)
    class_names = {
        "A": "REQUIRED FOR ESP32-S3 HARDWARE",
        "B": "REQUIRED FOR MODEL DEPLOYMENT",
        "C": "REQUIRED FOR BUILD",
        "D": "REQUIRED FOR HARDWARE TESTING",
        "E": "REQUIRED FOR DOCUMENTATION",
        "F": "OPTIONAL DEVELOPMENT TOOL",
        "G": "OLD / OBSOLETE (Demucs / ESP8266 / Unrelated)",
        "H": "DUPLICATE",
        "I": "GENERATED / CACHE",
        "J": "UNSAFE TO DELETE WITHOUT DEPENDENCY CHECK",
        "PROTECTED_DATASET": "PROTECTED RAW DATASETS (DO NOT TOUCH)"
    }
    for c, desc in class_names.items():
        print(f"  Class {c:<18}: {summary.get(c, 0):>6} files ({desc})")

    print("\n" + "=" * 75)
    print("LEGACY STRING AUDIT (Files containing matches):")
    print("=" * 75)
    print(f"  • Files containing 'Demucs' references: {len(set(demucs_hits)):>4}")
    print(f"  • Files containing 'LoRa' references:   {len(set(lora_hits)):>4}")
    print(f"  • Files containing 'ESP82' references:  {len(set(esp82_hits)):>4}")

    os.makedirs("audit_results", exist_ok=True)
    report_file = "audit_results/cleanup_classification_manifest.json"
    with open(report_file, "w") as fp:
        json.dump({
            "dataset_baseline": dataset_counts,
            "classification_summary": summary,
            "demucs_files": sorted(list(set(demucs_hits))),
            "lora_files": sorted(list(set(lora_hits))),
            "esp82_files": sorted(list(set(lora_hits))),
            "inventory": inventory
        }, fp, indent=2)

    print(f"\n[+] Detailed manifest written to: '{report_file}'")
    print("=" * 75)

if __name__ == "__main__":
    audit_repository()
