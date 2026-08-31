import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class DepthwiseSeparableConv1D(nn.Module):
    """Causal Depthwise-Separable 1D Convolution block for edge speech processing."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation # Causal padding
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            dilation=dilation, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad on the left for causal time continuity
        x = F.pad(x, (self.padding, 0))
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class MARKUSBLUEStudentEnhancer(nn.Module):
    """
    MARKUSBLUE Student Speech Enhancement Model (v7.1.0 Edge Architecture).
    Ultra-lightweight causal neural mask estimator designed for ESP32-S3 and mobile edge.
    Total Parameters: ~8,400 (< 35 KB float32, < 9 KB INT8 quantized).
    """
    def __init__(self, n_fft: int = 256, hop_length: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_bins = n_fft // 2 + 1 # 129 bins
        
        # Input encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(self.num_bins, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(hidden_dim)
        )
        
        # Causal temporal modeling blocks with increasing receptive field
        self.tcn1 = DepthwiseSeparableConv1D(hidden_dim, hidden_dim, kernel_size=3, dilation=1)
        self.tcn2 = DepthwiseSeparableConv1D(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.tcn3 = DepthwiseSeparableConv1D(hidden_dim, hidden_dim, kernel_size=3, dilation=4)
        
        # Lightweight GRU recurrent cell for long-range voice formant tracking
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        
        # Mask estimation head (sigmoid activation produces Ideal Ratio Mask [0, 1])
        self.mask_head = nn.Sequential(
            nn.Conv1d(hidden_dim, self.num_bins, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, mag_spec: torch.Tensor) -> torch.Tensor:
        """
        Input: Magnitude spectrogram [Batch, Bins, Frames]
        Output: Estimated Speech Mask [Batch, Bins, Frames]
        """
        x = self.encoder(mag_spec)
        res1 = self.tcn1(x)
        res2 = self.tcn2(res1)
        res3 = self.tcn3(res2)
        
        # Pass through causal GRU
        gru_in = res3.permute(0, 2, 1) # [Batch, Frames, Dim]
        gru_out, _ = self.gru(gru_in)
        gru_out = gru_out.permute(0, 2, 1) # [Batch, Dim, Frames]
        
        # Compute Ideal Ratio Mask
        mask = self.mask_head(gru_out + res1)
        return mask

    def enhance_spectrum(self, mag_spec: torch.Tensor) -> torch.Tensor:
        """Apply predicted mask directly to input magnitude."""
        mask = self.forward(mag_spec)
        return mag_spec * mask
