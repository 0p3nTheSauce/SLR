# Sweep 0 : xcj43xka

Wandb [link](https://wandb.ai/ljgoodall2001-rhodes-university/WLASL-100_cutoff_9/sweeps/xcj43xka)

This Sweep is the first S3D sweep, it is also the longest running sweep, and covers nearly all parameters in quite a broad range. It was the first sweep after the original two MViTv2_B_32x3 [Sweep 0](../../MViTv2_B_32x3/exp000/README.md), which showed the potential. 

This sweep also pioneered the base.py + config.yaml approach used currently, over the sweep_base.toml. Honestly, I think the toml files still look cooler, and could rely on existing loading methods, however I didn't like the defaults, but I see now that could be avoided, I just don't have time to rewrite the whole system again. 

This sweep needed a few further refinements, as seen in the later sweeps. 

**This sweep was affected by the move to the new split**