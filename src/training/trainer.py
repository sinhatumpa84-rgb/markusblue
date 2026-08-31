import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from .losses import FocalLoss

def set_seed(seed: int = 42):
    """Set global random seed for complete reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TacticalTrainer:
    """
    Tactical Audio ML Trainer with early stopping, mixed precision, and metric logging.
    Prioritizes high recall for dangerous acoustic impulses.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 35,
        patience: int = 7,
        save_dir: str = "models",
        model_name: str = "tactical_model"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.save_dir = save_dir
        self.model_name = model_name
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss function: Alpha weights to prioritize DANGEROUS_IMPULSE (class 0)
        # Class 0 (Impulse): 2.5x weight, Class 1 (Speech): 1.0x, Class 2 (Noise): 1.0x, Class 3 (Other): 1.5x
        alpha_weights = torch.tensor([2.5, 1.0, 1.0, 1.5], dtype=torch.float32).to(device)
        self.criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_impulse_recall": [],
            "val_impulse_precision": [],
            "val_macro_f1": [],
            "learning_rates": []
        }

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            
            # Gradient clipping for stable training
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_targets, all_preds)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict:
        loader = loader or self.val_loader
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        for inputs, targets, _ in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
        val_loss = total_loss / len(loader)
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        # Binary recall for DANGEROUS_IMPULSE (class 0)
        binary_targets = [1 if t == 0 else 0 for t in all_targets]
        binary_preds = [1 if p == 0 else 0 for p in all_preds]
        impulse_recall = recall_score(binary_targets, binary_preds, zero_division=0)
        impulse_prec = precision_score(binary_targets, binary_preds, zero_division=0)
        
        return {
            "loss": float(val_loss),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "impulse_recall": float(impulse_recall),
            "impulse_precision": float(impulse_prec),
            "predictions": all_preds,
            "probabilities": all_probs,
            "targets": all_targets
        }

    def fit(self) -> Dict:
        print(f"[*] Starting Training: {self.model_name} on {self.device}")
        best_val_loss = float('inf')
        best_recall = 0.0
        patience_counter = 0
        best_model_path = os.path.join(self.save_dir, f"{self.model_name}_best.pt")
        final_model_path = os.path.join(self.save_dir, f"{self.model_name}_final.pt")
        
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics["loss"])
            
            # Record history
            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_metrics["loss"]))
            self.history["train_acc"].append(float(train_acc))
            self.history["val_acc"].append(float(val_metrics["accuracy"]))
            self.history["val_impulse_recall"].append(float(val_metrics["impulse_recall"]))
            self.history["val_impulse_precision"].append(float(val_metrics["impulse_precision"]))
            self.history["val_macro_f1"].append(float(val_metrics["macro_f1"]))
            self.history["learning_rates"].append(float(current_lr))
            
            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']*100:.1f}% | "
                f"Impulse Recall: {val_metrics['impulse_recall']*100:.1f}% Prec: {val_metrics['impulse_precision']*100:.1f}% | "
                f"F1: {val_metrics['macro_f1']:.4f} LR: {current_lr:.6f}"
            )
            
            # Save best model (prioritizing low validation loss + high impulse recall)
            combined_metric = val_metrics["loss"] - 0.2 * val_metrics["impulse_recall"]
            if combined_metric < best_val_loss:
                best_val_loss = combined_metric
                best_recall = val_metrics["impulse_recall"]
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  --> Saved new best model checkpoint to '{best_model_path}'")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"[!] Early stopping triggered at epoch {epoch} (patience={self.patience}).")
                    break
                    
        # Save final model state
        torch.save(self.model.state_dict(), final_model_path)
        
        # Load best model weights for subsequent evaluation
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            
        history_path = os.path.join(self.save_dir, f"{self.model_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
            
        print(f"[OK] Training complete. Saved best model to '{best_model_path}' and history to '{history_path}'.")
        return self.history
