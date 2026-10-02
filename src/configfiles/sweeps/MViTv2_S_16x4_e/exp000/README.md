# Sweep 0 : imootel6

Wandb [link](https://wandb.ai/ljgoodall2001-rhodes-university/WLASL-100_cutoff_9/sweeps/imootel6)

This sweep was based on MViTv2_B_32x3 [Sweep 3](../../MViTv2_B_32x3/exp003/README.md), which was itself based on S3D [Sweep 4](../../S3D/exp004/README.md). 

## Result

Best val loss 0.590 (`2iehfua5`), the best of any model on `asl100_cutoff_9` so far. Hyperband
stopped 18 of the 24 trials at epoch 15, and the 6 it let finish are the 6 best. The analysis is
in [suggest_sweep.ipynb](../../../../results/sweeping/suggest_sweep.ipynb), and it feeds into
[Sweep 1](../exp001/README.md).
