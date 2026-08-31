import os
import csv
import pandas as pd
from typing import Dict, List

def parse_audioset_annotations(
    gunsound_dir: str = "gunsound",
    metadata_out_dir: str = "data/metadata"
) -> pd.DataFrame:
    """
    Parse AudioSet balanced train segments and class labels indices to extract
    ground-truth gunshot annotations and generate gunshot_segments.csv.
    """
    os.makedirs(metadata_out_dir, exist_ok=True)
    class_labels_file = os.path.join(gunsound_dir, "class_labels_indices.csv")
    balanced_segments_file = os.path.join(gunsound_dir, "balanced_train_segments.csv")
    output_csv = os.path.join(metadata_out_dir, "gunshot_segments.csv")
    
    # 1. Build MID -> Display Name mapping
    mid_to_name = {}
    if os.path.exists(class_labels_file):
        with open(class_labels_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    mid_to_name[row[1].strip()] = row[2].strip()
                    
    gunshot_target_mids = {
        "/m/032s66": "Gunshot, gunfire",
        "/m/04zjc": "Machine gun",
        "/m/0_1c": "Artillery fire",
        "/m/073cg4": "Cap gun"
    }
    
    records = []
    if os.path.exists(balanced_segments_file):
        with open(balanced_segments_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip().replace('"', '') for p in line.split(',')]
                if len(parts) >= 4:
                    ytid = parts[0]
                    start_sec = float(parts[1])
                    end_sec = float(parts[2])
                    labels = parts[3:]
                    
                    # Check if any label is gunshot
                    matched_gun_labels = [gunshot_target_mids[lbl] for lbl in labels if lbl in gunshot_target_mids]
                    if matched_gun_labels:
                        sample_id = f"audioset_{ytid}_{int(start_sec)}_{int(end_sec)}"
                        records.append({
                            "sample_id": sample_id,
                            "source_id": ytid,
                            "start_time": start_sec,
                            "end_time": end_sec,
                            "label": matched_gun_labels[0],
                            "source_file": f"https://www.youtube.com/watch?v={ytid}",
                            "duration": round(end_sec - start_sec, 3),
                            "sample_rate": 16000,
                            "channels": 1
                        })
                        
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"[OK] Generated AudioSet gunshot metadata: '{output_csv}' with {len(df)} annotated segments.")
    return df
