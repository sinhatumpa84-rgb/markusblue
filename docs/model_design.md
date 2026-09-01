# MARKUSBLUE Model Design & TinyML Mask Estimator

## 1. Network Topology (MARKUSBLUE_ESP82_Student)

The MARKUSBLUE student model is designed specifically for extreme microcontrollers without hardware floating-point units.

```
Input: Spectral Magnitude [1, 65, 1]
  │
  ├─► Linear Projection (Conv1D: 65 -> 16, ReLU) [1,056 params]
  │     │
  │     ├── (Residual Skip Connection) ──────────────────────────┐
  │     ▼                                                        │
  ├─► Causal DW-Conv1D (k=3, dilation=1, Pointwise: 16 -> 16)    │
  │     ▼                                                        │
  ├─► Causal DW-Conv1D (k=3, dilation=2, Pointwise: 16 -> 16)    │
  │     ▼                                                        │
  ├─► Sum with Skip: [16 channels] ◄─────────────────────────────┘
  │     ▼
  └─► Mask Output Head (Conv1D: 16 -> 65, Sigmoid) [1,105 params]
        ▼
Output: Ideal Ratio Speech Mask [1, 65, 1] in range [0.0, 1.0]
```

### Parameter Breakdown:
- **Encoder Conv1x1 (65 $\to$ 16)**: $65 \times 16 + 16 = 1,056$ params
- **Causal DW-TCN 1 (16 ch, k=3, dil=1)**: $16 \times 3 + 16 \times 16 + 16 = 320$ params
- **Causal DW-TCN 2 (16 ch, k=3, dil=2)**: $16 \times 3 + 16 \times 16 + 16 = 320$ params
- **Mask Head Conv1x1 (16 $\to$ 65)**: $16 \times 65 + 65 = 1,105$ params
- **Batch Normalization & Biases**: ~147 params
- **Total Parameters**: **2,948 parameters**
- **INT8 Flatbuffer Size**: **2.88 KB**
- **MACs per Frame**: ~5,800 MACs
