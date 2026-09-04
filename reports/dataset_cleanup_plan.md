# MARKUSBLUE (SIH26052) — Operational Dataset Cleanup & Restructuring Plan

## 1. Inventory & Classification of Existing Datasets

| Dataset Path | Current File Count | Classification | Action | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `datasets/speech/` | 2,400 files | `KEEP_REQUIRED` | **PRESERVE READ-ONLY** | Primary clean speech baseline. Zero deletions. |
| `datasets/gunshot/` | 6,000 files | `KEEP_REQUIRED` | **PRESERVE READ-ONLY** | Primary gunfire impulse recordings. Zero deletions. |
| `datasets/background_noise/` | 2,400 files | `KEEP_REQUIRED` | **PRESERVE READ-ONLY** | Primary ambient baseline. Zero deletions. |
| `datasets/other_impulse/` | 2,400 files | `KEEP_REQUIRED` | **PRESERVE READ-ONLY** | Primary mechanical impact baseline. Zero deletions. |
| `data/processed/` | 15,200 files | `KEEP_REQUIRED` | **PRESERVE READ-ONLY** | Extended processed dataset. Zero deletions. |
| `data/extracted/` | 12,406 files | `KEEP_REQUIRED` | **PRESERVE READ-ONLY** | Raw extracted recordings. Zero deletions. |
| `gunsound/` | 26 files | `KEEP_REQUIRED` | **PRESERVE READ-ONLY** | Archive zips. Zero deletions. |
| `datasets/external_noise/` | 250 files | `REVIEW` | **RESTRUCTURE** | Restructure into `suppressible/` and scale to 100+ files per category. |
| `datasets/derived/` | 250 files | `REVIEW` | **RESTRUCTURE** | Regenerate standardized splits for the expanded operational dataset. |

---

## 2. Restructuring & Classification Rules

1. **Original Dataset Invariant**:
   - `datasets/speech/`, `datasets/gunshot/`, `datasets/background_noise/`, `datasets/other_impulse/`, `data/`, and `gunsound/` are strictly `KEEP_REQUIRED` and will **never be deleted, renamed, or modified in place**.
2. **Critical vs. Suppressible Partitioning**:
   - Previously, alarms, sirens, footsteps, and radio static were placed in generic noise folders.
   - Under the new architecture, alarms, sirens, footsteps, and radio communications are classified as `CRITICAL_AUDIO_TO_PRESERVE` and migrated to `datasets/critical_audio/`.
   - Continuous industrial machinery, engines, helicopters, jets, wind, and electrical hum are classified as `SUPPRESSIBLE_NOISE` in `datasets/external_noise/suppressible/`.
3. **Scale Expansion**:
   - Every primary class will be populated with **100+ distinct recordings** to provide genuine acoustic diversity.
