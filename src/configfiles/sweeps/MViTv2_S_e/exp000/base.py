from src.configfiles.sweeps.MViTv2_S_16x4_e.exp001.base import (
    base_config,
    sweep_key_map,
)

# MViTv2_S_16x4_e sweep 1's base config unchanged (S3D sweep 4's augmentation pipeline, warmup,
# one cosine cycle, LR held at eta_min afterwards); only the model and the yaml's values differ.
__all__ = ["base_config", "sweep_key_map"]
