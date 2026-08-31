import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for Tactical Audio AI.
    Down-weights well-classified ambient sounds and concentrates gradients on hard/impulsive events.
    Includes custom alpha weights to enforce maximum Recall on DANGEROUS_IMPULSE.
    """
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute log-softmax and cross-entropy per sample
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)
        
        # Gather probabilities for target classes
        target_log_p = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_p = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal modulation factor: (1 - p_t)^gamma
        focal_weight = (1.0 - target_p) ** self.gamma
        
        loss = -focal_weight * target_log_p
        
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_weight = self.alpha.gather(0, targets)
            loss = alpha_weight * loss
            
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class WeightedCrossEntropyLoss(nn.Module):
    """Standard Cross-Entropy with custom class weighting for high impulse sensitivity."""
    def __init__(self, weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.weights is not None and self.weights.device != logits.device:
            self.weights = self.weights.to(logits.device)
        return F.cross_entropy(logits, targets, weight=self.weights)
