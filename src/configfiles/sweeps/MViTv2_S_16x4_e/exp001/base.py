import copy

from src.configfiles.sweeps.S3D.exp007.base import base_config as _exp007_base_config
from src.configfiles.sweeps.S3D.exp007.base import sweep_key_map

# S3D sweep 7's base config (S3D sweep 4's augmentation pipeline, warmup + a single cosine
# annealing cycle), with the LR held at eta_min after the cycle instead of rising again.
base_config = copy.deepcopy(_exp007_base_config)
base_config["scheduler"]["hold_after_tmax"] = True

__all__ = ["base_config", "sweep_key_map"]
