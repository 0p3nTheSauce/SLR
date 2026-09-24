import copy

from src.configfiles.sweeps.S3D.exp004.base import base_config as _exp004_base_config
from src.configfiles.sweeps.S3D.exp004.base import (
    sweep_key_map as _exp004_sweep_key_map,
)

# exp004's augmentation pipeline (chunked temporal, centre crop, RandAugment), with its
# warm-restart scheduler swapped for warmup + a single cosine annealing cycle.
base_config = copy.deepcopy(_exp004_base_config)
base_config["scheduler"] = {
    "type": "CosineAnnealingLR",
    "tmax": None,
    "eta_min": None,
    "warm_up": {"start_factor": None, "end_factor": None, "warmup_epochs": None},
}

sweep_key_map = {
    k: v for k, v in _exp004_sweep_key_map.items() if k not in ("t0", "tmult")
} | {"tmax": "scheduler.tmax"}
