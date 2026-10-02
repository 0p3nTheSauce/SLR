# Sweep 5: n4brjkmn


Wandb [link](https://wandb.ai/ljgoodall2001-rhodes-university/Sweeps/sweeps/n4brjkmn)

This sweep is a fresh start after [MViTv2_S_16x4_e/exp000](../../MViTv2_S_16x4_e/exp000/README.md), and will probably delete those old ones if this sweep can run for long enough. The changes include: 
- `base_config` and `sweep_key_map` in same file, and they can be imported and modified from other [base.py](./base.py) files. 
- non augmentation, non scheduler sweep to set a basline.
- expanded range to consistent values for all optimizer params
- expanded ramnge of drop_p
- reduced early stopping patience to 15

