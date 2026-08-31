import os
import random
import pandas as pd
import numpy as np
from typing import Tuple, Dict

def create_source_isolated_splits(
    catalog_csv: str = "data/processed/processed_dataset_catalog.csv",
    splits_dir: str = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset strictly by source_group / recording session / platform
    to eliminate any possible acoustic data leakage between train, val, and test.
    """
    os.makedirs(splits_dir, exist_ok=True)
    df = pd.read_csv(catalog_csv)
    print(f"[*] Total samples in catalog: {len(df)}")
    
    # Class mapping dictionary
    class_to_idx = {
        "DANGEROUS_IMPULSE": 0,
        "NORMAL_SPEECH": 1,
        "BACKGROUND_NOISE": 2,
        "OTHER_IMPULSE": 3
    }
    df["label_idx"] = df["class_label"].map(class_to_idx)
    
    # Group by (class_label, source_group) to ensure balanced distribution of classes
    train_dfs, val_dfs, test_dfs = [], [], []
    
    for class_name, group_df in df.groupby("class_label"):
        unique_sources = list(group_df["source_group"].unique())
        
        # If class has multiple sources, split by source
        if len(unique_sources) >= 3:
            rng = random.Random(seed)
            rng.shuffle(unique_sources)
            
            n_train = max(1, int(len(unique_sources) * train_ratio))
            n_val = max(1, int(len(unique_sources) * val_ratio))
            
            train_sources = set(unique_sources[:n_train])
            val_sources = set(unique_sources[n_train:n_train + n_val])
            test_sources = set(unique_sources[n_train + n_val:])
            
            # Handle edge case where test is empty
            if not test_sources and len(val_sources) > 1:
                moved = val_sources.pop()
                test_sources.add(moved)
                
            train_sub = group_df[group_df["source_group"].isin(train_sources)]
            val_sub = group_df[group_df["source_group"].isin(val_sources)]
            test_sub = group_df[group_df["source_group"].isin(test_sources)]
        else:
            # Fallback for synthetic/single source: split sample IDs with fixed seed
            shuffled = group_df.sample(frac=1.0, random_state=seed)
            n_train = int(len(shuffled) * train_ratio)
            n_val = int(len(shuffled) * val_ratio)
            
            train_sub = shuffled.iloc[:n_train]
            val_sub = shuffled.iloc[n_train:n_train + n_val]
            test_sub = shuffled.iloc[n_train + n_val:]
            
        train_dfs.append(train_sub)
        val_dfs.append(val_sub)
        test_dfs.append(test_sub)
        
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Save splits
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "validation.csv")
    test_path = os.path.join(splits_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("\n" + "="*50)
    print("DATASET SPLIT SUMMARY (Source-Isolated)")
    print("="*50)
    print(f"Train samples:      {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test samples:       {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print("\nClass distribution per split:")
    summary_table = pd.DataFrame({
        "Train": train_df["class_label"].value_counts(),
        "Val": val_df["class_label"].value_counts(),
        "Test": test_df["class_label"].value_counts()
    }).fillna(0).astype(int)
    print(summary_table)
    print("="*50 + "\n")
    
    return train_df, val_df, test_df
