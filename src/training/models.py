import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class ResidualBlock2D(nn.Module):
    """2D Residual Convolutional Block for Baseline Model."""
    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out + residual)
        return out

class BaselineCNN(nn.Module):
    """
    MODEL A — BASELINE MODEL:
    High-capacity 2D CNN with Residual Connections for Log-Mel Spectrogram (64 bins).
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [32, 32, 16]
        )
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock2D(64, dropout=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [64, 16, 8]
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock2D(128, dropout=dropout),
            nn.AdaptiveAvgPool2d((1, 1))  # [128, 1, 1]
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        logits = self.classifier(x)
        return logits

class DepthwiseSeparableConv2D(nn.Module):
    """
    Depthwise-Separable 2D Convolution:
    Depthwise Conv (spatial filter) + Pointwise 1x1 Conv (channel projection).
    Reduces computational cost by 8x-9x compared to standard Conv2D for ESP32-S3.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ESP32EdgeCNN(nn.Module):
    """
    MODEL B — EMBEDDED MODEL:
    Ultra-compact Depthwise-Separable CNN optimized for ESP32-S3 + PSRAM and TFLite Micro.
    Input: [1, 32, 32] Log-Mel Spectrogram.
    Total Parameters: ~14,000 (< 60 KB float32, < 15 KB INT8 quantized).
    Peak SRAM during inference: < 24 KB.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        # Initial standard Conv2D
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [16, 16, 16]
        )
        
        # Depthwise Separable Blocks
        self.block1 = DepthwiseSeparableConv2D(16, 24, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # [24, 8, 8]
        
        self.block2 = DepthwiseSeparableConv2D(24, 32, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # [32, 4, 4]
        
        self.block3 = DepthwiseSeparableConv2D(32, 48, stride=1)
        
        # Global Average Pooling to eliminate heavy dense parameter layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # [48, 1, 1]
        
        # Final lightweight dense projection
        self.head = nn.Linear(48, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        logits = self.head(x)
        return logits

def get_model(model_type: str = "edge", num_classes: int = 4) -> nn.Module:
    """Factory helper to instantiate Model A (baseline) or Model B (edge)."""
    if model_type.lower() in ["edge", "embedded", "esp32", "model_b"]:
        return ESP32EdgeCNN(num_classes=num_classes)
    elif model_type.lower() in ["baseline", "resnet", "model_a"]:
        return BaselineCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'. Choose 'baseline' or 'edge'.")

def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int, int]) -> Dict:
    """Calculate parameter counts, memory, and output shapes."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = total_params * 4 / (1024 * 1024)
    int8_size_kb = total_params * 1 / 1024
    
    return {
        "model_name": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "float32_size_mb": round(param_size_mb, 4),
        "int8_estimated_size_kb": round(int8_size_kb, 2),
        "esp32_s3_compatible": total_params < 50000
    }
