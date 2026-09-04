# Rerun with the SATNAC 2025 configuration, and updated codebase.


| Section | Parameter | Value |
|---|---|---|
| Data | Batch size | $8$ |
| Data | No. frames | $16, 32$ |
| Data | Frame size | $224 \times 224$ |
| Optimizer: AdamW | Classifier LR | $1 \times 10^{-3}$ |
| Optimizer: AdamW | Classifier WD | $1 \times 10^{-7}$ |
| Optimizer: AdamW | Backbone LR | $1 \times 10^{-5}$ |
| Optimizer: AdamW | Backbone WD | $1 \times 10^{-4}$ |
| Optimizer: AdamW | Epsilon | $1 \times 10^{-3}$ |
| Scheduler: Cosine Annealing | T max | $100$ |
| Scheduler: Cosine Annealing | Eta min | $1 \times 10^{-5}$ |
| Early stopping: Min val loss | Min delta | $0.01$ |
| Early stopping: Min val loss | Patience | $50$ |
| Training | Dropout | $0.50$ |
| Training | Max epochs | $200$ |

- LR: Learning rate.
- WD: Weight decay.

Only `3D CNNs` are run with 32 frames. The models run are:
- ***3D CNN:*** `R3D_18`, `R(2+1)D_18`, `S3D`
- ***ViT:*** `Swin3D-T`, `Swin3D-S`, `Swin3D-B`, `MViTv1_B`, `MViTv2_S`