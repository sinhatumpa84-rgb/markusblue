import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
from typing import Dict, List, Tuple, Optional

CLASS_NAMES = ["DANGEROUS_IMPULSE", "NORMAL_SPEECH", "BACKGROUND_NOISE", "OTHER_IMPULSE"]

def compute_tactical_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    latencies_ms: Optional[List[float]] = None
) -> Dict:
    """
    Compute comprehensive tactical hearing protection evaluation metrics.
    Focuses specifically on impulse recall and safety constraints.
    """
    acc = accuracy_score(targets, predictions)
    prec_macro = precision_score(targets, predictions, average='macro', zero_division=0)
    rec_macro = recall_score(targets, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    
    # Class-wise metrics
    class_prec = precision_score(targets, predictions, average=None, zero_division=0)
    class_rec = recall_score(targets, predictions, average=None, zero_division=0)
    class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(targets, predictions, labels=[0, 1, 2, 3])
    
    # Binary metrics for DANGEROUS_IMPULSE (Class 0)
    bin_targets = np.array([1 if t == 0 else 0 for t in targets])
    bin_preds = np.array([1 if p == 0 else 0 for p in predictions])
    impulse_probs = probabilities[:, 0]
    
    tp = np.sum((bin_targets == 1) & (bin_preds == 1))
    fn = np.sum((bin_targets == 1) & (bin_preds == 0))
    fp = np.sum((bin_targets == 0) & (bin_preds == 1))
    tn = np.sum((bin_targets == 0) & (bin_preds == 0))
    
    impulse_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    impulse_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    impulse_fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    impulse_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    impulse_specificity = float(tn / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    # ROC-AUC
    try:
        roc_auc_ovr = roc_auc_score(targets, probabilities, multi_class='ovr')
        roc_auc_impulse = roc_auc_score(bin_targets, impulse_probs)
    except Exception:
        roc_auc_ovr = 0.0
        roc_auc_impulse = 0.0
        
    # Latency statistics
    lat_stats = {}
    if latencies_ms and len(latencies_ms) > 0:
        lat_arr = np.array(latencies_ms)
        lat_stats = {
            "mean_ms": float(np.mean(lat_arr)),
            "median_ms": float(np.median(lat_arr)),
            "p95_ms": float(np.percentile(lat_arr, 95)),
            "p99_ms": float(np.percentile(lat_arr, 99)),
            "min_ms": float(np.min(lat_arr)),
            "max_ms": float(np.max(lat_arr))
        }
        
    return {
        "overall": {
            "accuracy": float(round(acc, 4)),
            "macro_precision": float(round(prec_macro, 4)),
            "macro_recall": float(round(rec_macro, 4)),
            "macro_f1": float(round(f1_macro, 4)),
            "roc_auc_ovr": float(round(roc_auc_ovr, 4))
        },
        "dangerous_impulse": {
            "recall": float(round(impulse_recall, 4)),
            "precision": float(round(impulse_precision, 4)),
            "f1_score": float(round(class_f1[0], 4)),
            "false_negative_rate": float(round(impulse_fnr, 4)),
            "false_positive_rate": float(round(impulse_fpr, 4)),
            "specificity": float(round(impulse_specificity, 4)),
            "roc_auc": float(round(roc_auc_impulse, 4)),
            "true_positives": int(tp),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_negatives": int(tn)
        },
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(round(class_prec[i], 4)),
                "recall": float(round(class_rec[i], 4)),
                "f1_score": float(round(class_f1[i], 4))
            } for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": cm.tolist(),
        "latency_stats": lat_stats
    }

def evaluate_model_on_split(
    model: torch.nn.Module,
    loader,
    device: torch.device
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """Run thorough inference and timing on a DataLoader."""
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []
    latencies = []
    
    import time
    
    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            
            t0 = time.perf_counter()
            logits = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            # Latency per sample in ms
            batch_lat = ((t1 - t0) / inputs.size(0)) * 1000.0
            latencies.extend([batch_lat] * inputs.size(0))
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            
    targets_np = np.array(all_targets)
    preds_np = np.array(all_preds)
    probs_np = np.array(all_probs)
    
    metrics = compute_tactical_metrics(targets_np, preds_np, probs_np, latencies)
    return metrics, targets_np, preds_np, probs_np, latencies
