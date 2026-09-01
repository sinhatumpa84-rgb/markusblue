# MARKUSBLUE Training Pipeline & Distillation Strategy

## 1. Teacher-Student Distillation Architecture

```
                    Noisy Speech Mixture
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
        Demucs / Ideal                MARKUSBLUE Tiny
        Wiener Teacher                 Student Model
               │                             │
               ▼                             ▼
        Target Spectral Mask         Estimated Mask
               │                             │
               └──────────────┬──────────────┘
                              ▼
        Composite Loss = 0.6 * L_mask + 0.4 * L_spec
```

---

## 2. Loss Formulations
1. **Mask Distillation Loss ($\mathcal{L}_{\text{mask}}$)**:
   $$\mathcal{L}_{\text{mask}} = \frac{1}{B \cdot K} \sum_{b=1}^{B} \sum_{k=1}^{K} \left( \hat{M}_{b,k} - M_{b,k}^{\text{teacher}} \right)^2$$
2. **Spectral Reconstruction Loss ($\mathcal{L}_{\text{spec}}$)**:
   $$\mathcal{L}_{\text{spec}} = \frac{1}{B \cdot K} \sum_{b=1}^{B} \sum_{k=1}^{K} \left| \hat{X}_{b,k}^{\text{enhanced}} - X_{b,k}^{\text{clean}} \right|$$
3. **Total Optimization Objective**:
   $$\mathcal{L}_{\text{total}} = 0.6 \cdot \mathcal{L}_{\text{mask}} + 0.4 \cdot \mathcal{L}_{\text{spec}}$$

---

## 3. Training Hyperparameters
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, Weight Decay = $10^{-4}$)
- **Learning Rate**: $2 \times 10^{-3}$ with Cosine Annealing scheduler
- **Batch Size**: 32
- **Dynamic SNR Sampling**: Uniform across `[-20 dB, -15 dB, -10 dB, -5 dB, 0 dB, +5 dB, +10 dB, +15 dB, +20 dB]`
- **Augmentation**: Random noise gain scaling and time-shift offsets.
