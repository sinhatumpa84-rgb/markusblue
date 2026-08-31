"""
SIH26052 — Training Pipeline Entry Point
Trains Model A (Baseline CNN) and Model B (ESP32-S3 Depthwise-Separable Edge CNN)
with Focal Loss, early stopping, and high-recall optimization for dangerous acoustic impulses.
"""

import os
import json
import argparse
import torch

from src.training.trainer import TacticalTrainer, set_seed
from src.training.models import get_model, get_model_summary
from src.dataset.dataset_loader import get_data_loaders

def train_pipeline():
    parser = argparse.ArgumentParser(description="Train Tactical Impulse Detection Models for SIH26052.")
    parser.add_argument("--model_type", type=str, default="both", choices=["edge", "baseline", "both"],
                        help="Model architecture: 'edge' (ESP32 Model B), 'baseline' (Model A), or 'both'")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on compute device: {device}")
    
    models_to_train = ["edge", "baseline"] if args.model_type == "both" else [args.model_type]
    
    for m_type in models_to_train:
        feature_mode = "edge" if m_type == "edge" else "baseline"
        model_name = f"tactical_{m_type}_model"
        retrain_dir = f"models/retrained_model_{'b' if m_type == 'edge' else 'a'}"
        os.makedirs(retrain_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print(f"TRAINING {model_name.upper()} (Feature Mode: {feature_mode})")
        print("="*60)
        
        # 1. Load Data
        train_loader, val_loader, _ = get_data_loaders(
            splits_dir="data/splits",
            feature_mode=feature_mode,
            batch_size=args.batch_size
        )
        print(f"[*] Loaded Data: Train batches={len(train_loader)}, Val batches={len(val_loader)}")
        
        # 2. Instantiate Model
        model = get_model(m_type, num_classes=4)
        summary = get_model_summary(model, (1, 1, 32 if feature_mode == "edge" else 64, 32))
        print(f"[*] Architecture Summary: {summary['total_parameters']:,} params | Est. INT8 Size: {summary['int8_estimated_size_kb']} KB")
        
        # 3. Train with TacticalTrainer (saves to models/ and retrain_dir)
        trainer = TacticalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=args.lr,
            epochs=args.epochs,
            patience=args.patience,
            save_dir=retrain_dir,
            model_name=model_name
        )
        history = trainer.fit()
        
        # Also copy best weights to models/
        best_pt = os.path.join(retrain_dir, f"{model_name}_best.pt")
        target_pt = os.path.join("models", f"{model_name}_best.pt")
        if os.path.exists(best_pt):
            import shutil
            shutil.copy2(best_pt, target_pt)
            shutil.copy2(os.path.join(retrain_dir, f"{model_name}_history.json"), 
                         os.path.join("models", f"{model_name}_history.json"))
            
        # Save training metadata
        meta = {
            "model_type": m_type,
            "feature_mode": feature_mode,
            "total_parameters": summary["total_parameters"],
            "hyperparameters": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "patience": args.patience,
                "seed": args.seed,
                "loss": "Multi-Class Focal Loss (gamma=2.0)"
            },
            "dataset_version": "v2_hard_negatives_zero_leakage",
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "best_val_accuracy": max(history["val_acc"]) if "val_acc" in history else None,
            "best_val_loss": min(history["val_loss"]) if "val_loss" in history else None
        }
        with open(os.path.join(retrain_dir, "training_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

if __name__ == "__main__":
    train_pipeline()
