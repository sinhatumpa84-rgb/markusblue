import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalDepthwiseSeparableConv1D(nn.Module):
    """
    Causal Depthwise-Separable 1D Convolution block.
    Optimized for microcontroller cache and minimal MAC footprint.
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation # Left causal padding
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            dilation=dilation, groups=channels, bias=False
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal padding on left
        x = F.pad(x, (self.padding, 0))
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class MARKUSBLUE_ESP82_Student(nn.Module):
    """
    MARKUSBLUE Student Speech Enhancement Model (ESP82 / ESP8266 Version).
    Ultra-lightweight streaming causal neural mask estimator.
    
    Target Specifications:
    - Platform: Tensilica Xtensa L106 @ 160 MHz
    - Parameters: ~2,800
    - Model INT8 Flash: < 4 KB
    - Tensor Arena RAM: < 3.5 KB
    - Frame Latency: < 0.15 ms on L106
    """
    def __init__(self, num_bins: int = 65, hidden_dim: int = 16):
        super().__init__()
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        
        # 1. Input Linear Projection
        self.encoder = nn.Sequential(
            nn.Conv1d(num_bins, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 2. Causal Temporal Blocks (Receptive Field = 7 frames)
        self.tcn1 = CausalDepthwiseSeparableConv1D(hidden_dim, kernel_size=3, dilation=1)
        self.tcn2 = CausalDepthwiseSeparableConv1D(hidden_dim, kernel_size=3, dilation=2)
        
        # 3. Mask Head (Sigmoid bounded [0, 1])
        self.mask_head = nn.Sequential(
            nn.Conv1d(hidden_dim, num_bins, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [Batch, Bins, Frames]
        Output: [Batch, Bins, Frames] Speech Mask in range [0, 1]
        """
        enc = self.encoder(x)
        h1 = self.tcn1(enc)
        h2 = self.tcn2(h1)
        mask = self.mask_head(h2 + enc) # Residual skip connection
        return mask

    def enhance_spectrum(self, mag_spec: torch.Tensor) -> torch.Tensor:
        """Apply predicted mask to spectral magnitude."""
        mask = self.forward(mag_spec)
        return mag_spec * mask
