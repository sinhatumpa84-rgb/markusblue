import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from typing import Dict, List, Optional

CLASS_NAMES = ["DANGEROUS_IMPULSE", "NORMAL_SPEECH", "BACKGROUND_NOISE", "OTHER_IMPULSE"]

class TacticalVisualizer:
    """High-resolution visualization suite for tactical hearing protection AI."""
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        # Apply clean publication theme
        sns.set_theme(style="darkgrid", palette="muted")
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.autolayout': True
        })

    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str = "Tactical Model", out_path: Optional[str] = None):
        """Generate high-resolution annotated confusion matrix."""
        out_path = out_path or os.path.join(self.reports_dir, "confusion_matrix.png")
        fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
        
        # Normalize for display
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
                
        sns.heatmap(
            cm, annot=annot, fmt='', cmap="Blues", cbar=True,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
        )
        ax.set_title(f"Confusion Matrix — {model_name}", fontweight='bold', pad=15)
        ax.set_xlabel("Predicted Class", fontweight='bold')
        ax.set_ylabel("True Class", fontweight='bold')
        plt.xticks(rotation=20, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved confusion matrix: '{out_path}'")

    def plot_training_history(self, history: Dict, model_name: str = "Tactical Model", out_path: Optional[str] = None):
        """Generate multi-panel training loss, accuracy, and impulse recall curves."""
        out_path = out_path or os.path.join(self.reports_dir, "training_history.png")
        epochs = range(1, len(history["train_loss"]) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
        
        # 1. Loss
        axes[0, 0].plot(epochs, history["train_loss"], 'o-', label='Train Loss', color='#2b5c8f', linewidth=2)
        axes[0, 0].plot(epochs, history["val_loss"], 's--', label='Val Loss', color='#d95f02', linewidth=2)
        axes[0, 0].set_title("Training & Validation Loss (Focal Loss)", fontweight='bold')
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        
        # 2. Accuracy
        axes[0, 1].plot(epochs, [a*100 for a in history["train_acc"]], 'o-', label='Train Acc', color='#2b5c8f', linewidth=2)
        axes[0, 1].plot(epochs, [a*100 for a in history["val_acc"]], 's--', label='Val Acc', color='#2ca02c', linewidth=2)
        axes[0, 1].set_title("Overall Accuracy (%)", fontweight='bold')
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].legend()
        
        # 3. Dangerous Impulse Recall
        axes[1, 0].plot(epochs, [r*100 for r in history["val_impulse_recall"]], '^-', label='Impulse Recall', color='#d62728', linewidth=2.5)
        axes[1, 0].plot(epochs, [p*100 for p in history["val_impulse_precision"]], 'v--', label='Impulse Precision', color='#9467bd', linewidth=2)
        axes[1, 0].set_title("Dangerous Impulse Detection (Recall & Precision %)", fontweight='bold')
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Percentage (%)")
        axes[1, 0].legend()
        
        # 4. Macro F1 Score
        axes[1, 1].plot(epochs, history["val_macro_f1"], 'D-', label='Macro F1', color='#17becf', linewidth=2)
        axes[1, 1].set_title("Validation Macro F1 Score", fontweight='bold')
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("F1 Score")
        axes[1, 1].legend()
        
        plt.suptitle(f"Tactical Training Progression — {model_name}", fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved training history plot: '{out_path}'")

    def plot_roc_and_pr_curves(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        out_roc: Optional[str] = None,
        out_pr: Optional[str] = None
    ):
        """Generate ROC and Precision-Recall curves for all 4 classes."""
        out_roc = out_roc or os.path.join(self.reports_dir, "roc_curve.png")
        out_pr = out_pr or os.path.join(self.reports_dir, "pr_curve.png")
        
        colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
        
        # 1. ROC Curve
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        for i, class_name in enumerate(CLASS_NAMES):
            bin_targets = (targets == i).astype(int)
            fpr, tpr, _ = roc_curve(bin_targets, probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2.5 if i==0 else 1.8,
                    label=f"{class_name} (AUC = {roc_auc:.4f})")
            
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)", fontweight='bold')
        ax.set_ylabel("True Positive Rate (Recall)", fontweight='bold')
        ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight='bold')
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_roc, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved ROC curves: '{out_roc}'")
        
        # 2. Precision-Recall Curve
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        for i, class_name in enumerate(CLASS_NAMES):
            bin_targets = (targets == i).astype(int)
            prec, rec, _ = precision_recall_curve(bin_targets, probabilities[:, i])
            pr_auc = auc(rec, prec)
            ax.plot(rec, prec, color=colors[i], lw=2.5 if i==0 else 1.8,
                    label=f"{class_name} (PR-AUC = {pr_auc:.4f})")
            
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall (Sensitivity)", fontweight='bold')
        ax.set_ylabel("Precision (Positive Predictive Value)", fontweight='bold')
        ax.set_title("Precision-Recall (PR) Curves", fontweight='bold')
        ax.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(out_pr, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved PR curves: '{out_pr}'")

    def plot_speech_preservation_comparison(
        self,
        eval_result: Dict,
        sr: int = 16000,
        out_path: Optional[str] = None
    ):
        """Generate waveform & spectrogram comparison of Dangerous Impulse vs Protected Speech."""
        out_path = out_path or os.path.join(self.reports_dir, "speech_preservation_demo.png")
        
        raw_mix = eval_result["raw_mix_audio"]
        protected = eval_result["protected_audio"]
        clean_ref = eval_result["clean_speech_ref"]
        
        t = np.linspace(0, len(raw_mix)/sr, len(raw_mix))
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 9), dpi=300)
        
        # Waveforms
        axes[0, 0].plot(t, clean_ref, color='#2ca02c', alpha=0.9)
        axes[0, 0].set_title("Clean Tactical Voice Reference", fontweight='bold')
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].set_ylim([-1.1, 1.1])
        
        axes[1, 0].plot(t, raw_mix, color='#d62728', alpha=0.9)
        axes[1, 0].set_title("Raw Microphone Input (Speech + Background + Dangerous Impulse)", fontweight='bold')
        axes[1, 0].set_ylabel("Amplitude")
        axes[1, 0].set_ylim([-1.1, 1.1])
        
        axes[2, 0].plot(t, protected, color='#1f77b4', alpha=0.9)
        axes[2, 0].set_title(f"DSP Protected Output (Peak Attenuation: {eval_result['peak_attenuation_db']:.1f} dB)", fontweight='bold')
        axes[2, 0].set_xlabel("Time (seconds)")
        axes[2, 0].set_ylabel("Amplitude")
        axes[2, 0].set_ylim([-1.1, 1.1])
        
        # Spectrograms
        axes[0, 1].specgram(clean_ref, Fs=sr, NFFT=512, noverlap=256, cmap='inferno')
        axes[0, 1].set_title("Clean Speech Spectrogram", fontweight='bold')
        axes[0, 1].set_ylabel("Frequency (Hz)")
        
        axes[1, 1].specgram(raw_mix, Fs=sr, NFFT=512, noverlap=256, cmap='inferno')
        axes[1, 1].set_title("Unprotected Blast Spectrogram", fontweight='bold')
        axes[1, 1].set_ylabel("Frequency (Hz)")
        
        axes[2, 1].specgram(protected, Fs=sr, NFFT=512, noverlap=256, cmap='inferno')
        axes[2, 1].set_title("Protected Speech-Preserved Spectrogram", fontweight='bold')
        axes[2, 1].set_xlabel("Time (seconds)")
        axes[2, 1].set_ylabel("Frequency (Hz)")
        
        plt.suptitle("SIH26052 Adaptive Hearing Protection & Speech Preservation Demonstration", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved speech preservation demo plot: '{out_path}'")
