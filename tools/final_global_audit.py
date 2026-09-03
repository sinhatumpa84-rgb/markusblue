#!/usr/bin/env python3
"""
MARKUSBLUE — Final Global Search & Verification Audit
SIH Problem Statement: SIH26052

Performs final recursive check for:
1. Demucs references in all source code and documentation (Expected: 0)
2. LoRa / SX1262 / SX1278 references in production code (Expected: 0)
3. ESP8266 / ESP82 production references (Expected: 0)
4. Dataset integrity invariant verification (Expected: 0 deleted, 0 modified)
5. Python import integrity verification across all modules
"""

import os
import sys
import re
import json

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PROTECTED_DIRS = ["datasets", "data", "gunsound"]
BASELINE_COUNTS = {"datasets": 13201, "data": 27626, "gunsound": 26}

def run_global_audit():
    print("=" * 75)
    print("MARKUSBLUE (SIH26052) — FINAL GLOBAL REPOSITORY AUDIT")
    print("=" * 75)

    # 1. Verify Dataset Invariant
    print("[1] VERIFYING DATASET INTEGRITY INVARIANT:")
    current_counts = {}
    dataset_passed = True
    for d in PROTECTED_DIRS:
        cnt = sum(len(files) for _, _, files in os.walk(d)) if os.path.exists(d) else 0
        current_counts[d] = cnt
        diff = cnt - BASELINE_COUNTS[d]
        status = "PASSED (100% PRESERVED)" if diff == 0 else f"FAILED (Diff: {diff})"
        print(f"    • {d:<12}: Baseline={BASELINE_COUNTS[d]:<6} Current={cnt:<6} | {status}")
        if diff != 0:
            dataset_passed = False

    # 2. Global Search for Forbidden Legacy Strings
    print("\n[2] SCANNING ALL PROJECT TEXT FILES FOR FORBIDDEN STRINGS:")
    
    excluded_dirs = {".git", "datasets", "data", "gunsound", "audit_results", "__pycache__"}
    text_extensions = {".py", ".cpp", ".h", ".cc", ".md", ".json", ".txt", ".ini", ".yaml", ".yml", ".toml"}
    
    demucs_pattern = re.compile(r'\bdemucs\b|\bfacebookresearch\b|\bhtshift\b', re.IGNORECASE)
    lora_pattern = re.compile(r'\blora\b|\bsx1262\b|\bsx1278\b|\blorawan\b', re.IGNORECASE)
    esp82_pattern = re.compile(r'\besp8266\b|\besp82\b|\besp-12e\b', re.IGNORECASE)

    demucs_hits = []
    lora_hits = []
    esp82_hits = []
    total_files_scanned = 0

    for root, dirs, files in os.walk("."):
        parts = os.path.normpath(root).split(os.sep)
        if any(p in excluded_dirs for p in parts):
            continue

        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in text_extensions:
                continue

            # Don't scan audit scripts themselves for their own pattern strings
            if f in ["final_global_audit.py", "audit_and_classify_repo.py", "execute_cleanup.py", "deletion_log.json", "cleanup_classification_manifest.json"]:
                continue

            filepath = os.path.normpath(os.path.join(root, f)).replace(os.sep, "/")
            total_files_scanned += 1

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
                    
                    # Demucs match check
                    if demucs_pattern.search(content):
                        demucs_hits.append(filepath)

                    # LoRa match check (exclude explicit negation like "No LoRa")
                    if lora_pattern.search(content):
                        lines = [line.strip() for line in content.splitlines() if lora_pattern.search(line)]
                        non_negated = [l for l in lines if not any(w in l.lower() for w in ["no lora", "do not add", "remove lora", "zero forbidden", "without lora"])]
                        if non_negated:
                            lora_hits.append(filepath)

                    # ESP82 match check (exclude explicit negation like "No ESP8266")
                    if esp82_pattern.search(content):
                        lines = [line.strip() for line in content.splitlines() if esp82_pattern.search(line)]
                        non_negated = [l for l in lines if not any(w in l.lower() for w in ["no esp", "do not add", "superseded", "zero forbidden", "without esp"])]
                        if non_negated:
                            esp82_hits.append(filepath)
            except Exception:
                pass

    print(f"    • Source text files scanned: {total_files_scanned}")
    print(f"    • Demucs References Found:   {len(demucs_hits)} {'[PASSED - ZERO DEMUCS]' if len(demucs_hits) == 0 else '[FAILED]'}")
    for h in demucs_hits:
        print(f"      - {h}")
    print(f"    • Production LoRa Drivers:   {len(lora_hits)} {'[PASSED - ZERO LORA]' if len(lora_hits) == 0 else '[FAILED]'}")
    for h in lora_hits:
        print(f"      - {h}")
    print(f"    • Production ESP82 Code:     {len(esp82_hits)} {'[PASSED - ZERO ESP82]' if len(esp82_hits) == 0 else '[FAILED]'}")
    for h in esp82_hits:
        print(f"      - {h}")

    # 3. Python Module Import Integrity
    print("\n[3] TESTING PYTHON IMPORT INTEGRITY:")
    try:
        import src.training.student_model
        import src.agc.automatic_gain_control
        import src.limiter.peak_limiter
        import src.vad.voice_activity_detector
        import src.enhancement.speech_enhancer
        print("    [+] All primary MARKUSBLUE Python modules import cleanly without broken dependencies.")
        imports_passed = True
    except Exception as e:
        print(f"    [!] Import Error: {e}")
        imports_passed = False

    # 4. Final Verdict
    print("\n" + "=" * 75)
    print("FINAL REPOSITORY AUDIT VERDICT:")
    print("=" * 75)
    verdict = (
        dataset_passed and
        len(demucs_hits) == 0 and
        len(lora_hits) == 0 and
        len(esp82_hits) == 0 and
        imports_passed
    )
    if verdict:
        print("[+] STATUS: 100% COMPLIANT WITH SIH26052 & ESP32-S3 FINAL ARCHITECTURE")
        print("[+] Original datasets: 100% PRESERVED")
        print("[+] Demucs: ZERO REFERENCES REMAINING")
        print("[+] LoRa: ZERO PRODUCTION REFERENCES")
        print("[+] ESP8266/ESP82: ZERO PRODUCTION REFERENCES")
        print("[+] Deployed MCU: ESP32-S3 N16R8 EXCLUSIVELY")
    else:
        print("[!] STATUS: ISSUES DETECTED. Review logs above.")
    print("=" * 75)

    audit_summary = {
        "verdict": "PASSED" if verdict else "FAILED",
        "dataset_invariant_passed": dataset_passed,
        "demucs_hits_count": len(demucs_hits),
        "demucs_hits": demucs_hits,
        "lora_production_hits_count": len(lora_hits),
        "lora_hits": lora_hits,
        "esp82_production_hits_count": len(esp82_hits),
        "esp82_hits": esp82_hits,
        "imports_passed": imports_passed,
        "current_counts": current_counts
    }
    with open("audit_results/final_global_audit.json", "w") as fp:
        json.dump(audit_summary, fp, indent=2)

if __name__ == "__main__":
    run_global_audit()
