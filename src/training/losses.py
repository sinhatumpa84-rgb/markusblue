import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Multi-class Focal Loss for mitigating extreme class imbalance."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class WeightedCrossEntropyLoss(nn.Module):
    """Standard weighted cross entropy loss with label smoothing."""
    def __init__(self, weight: torch.Tensor = None, label_smoothing: float = 0.05):
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(inputs, targets, weight=self.weight, label_smoothing=self.label_smoothing)

class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss for high-fidelity speech synthesis and enhancement.
    Computes spectral convergence and log magnitude differences over multiple FFT resolutions.
    """
    def __init__(self, fft_sizes=(512, 256, 128), hop_sizes=(128, 64, 32), win_lengths=(512, 256, 128)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(win, device=x.device)
            x_stft = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            y_stft = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            
            x_mag = torch.abs(x_stft) + 1e-7
            y_mag = torch.abs(y_stft) + 1e-7
            
            # Spectral Convergence
            sc_loss = torch.norm(y_mag - x_mag, p='fro') / (torch.norm(y_mag, p='fro') + 1e-7)
            # Log Magnitude Loss
            log_loss = F.l1_loss(torch.log(x_mag), torch.log(y_mag))
            
            total_loss += sc_loss + log_loss
            
        return total_loss / len(self.fft_sizes)

class SISDRLoss(nn.Module):
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) Loss."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Zero-mean normalization
        target = target - torch.mean(target, dim=-1, keepdim=True)
        estimated = estimated - torch.mean(estimated, dim=-1, keepdim=True)
        
        # Optimal scaling factor
        dot = torch.sum(target * estimated, dim=-1, keepdim=True)
        target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + self.eps
        s_target = (dot / target_energy) * target
        
        e_noise = estimated - s_target
        
        s_target_energy = torch.sum(s_target ** 2, dim=-1) + self.eps
        e_noise_energy = torch.sum(e_noise ** 2, dim=-1) + self.eps
        
        sisdr = 10.0 * torch.log10(s_target_energy / e_noise_energy)
        return -torch.mean(sisdr) # Negative for minimization

class StudentTeacherDistillationLoss(nn.Module):
    """
    Combined Loss for MARKUSBLUE Student Training:
    L = λ_clean * L_speech + λ_spec * L_spectral + λ_distill * L_teacher
    """
    def __init__(self, lambda_clean: float = 1.0, lambda_spec: float = 0.5, lambda_distill: float = 0.3):
        super().__init__()
        self.lambda_clean = lambda_clean
        self.lambda_spec = lambda_spec
        self.lambda_distill = lambda_distill
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        pred_spec: torch.Tensor,
        clean_spec: torch.Tensor,
        teacher_spec: torch.Tensor
    ) -> torch.Tensor:
        loss_clean = self.l1_loss(pred_spec, clean_spec)
        loss_spec = self.mse_loss(pred_spec, clean_spec)
        loss_teacher = self.mse_loss(pred_spec, teacher_spec)
        
        total_loss = (
            self.lambda_clean * loss_clean +
            self.lambda_spec * loss_spec +
            self.lambda_distill * loss_teacher
        )
        return total_loss
