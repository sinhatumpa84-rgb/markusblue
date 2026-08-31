"""Model architectures, loss functions, and training pipeline."""
from .models import BaselineCNN, ESP32EdgeCNN, get_model
from .losses import FocalLoss, WeightedCrossEntropyLoss
from .trainer import TacticalTrainer

__all__ = [
    "BaselineCNN",
    "ESP32EdgeCNN",
    "get_model",
    "FocalLoss",
    "WeightedCrossEntropyLoss",
    "TacticalTrainer"
]
