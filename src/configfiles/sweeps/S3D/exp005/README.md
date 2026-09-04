# Sweep 5: i6g6do1w


Wandb [link](https://wandb.ai/ljgoodall2001-rhodes-university/Sweeps/sweeps/i6g6do1w)

This sweep is a fresh start after [MViTv2_S_16x4_e/exp000](../../MViTv2_S_16x4_e/exp000/README.md), and will probably delete those old ones if this sweep can run for long enough. The changes include: 
- `base_config` and `sweep_key_map` in same file, and they can be imported and modified from other [base.py](./base.py) files. 
- This sweep does the broadest non parameter, non scheduler sweep using both 16 and 32 frames. Ideally, the follow up will contain schedulers and augmentations in a more refined range of parameters. 