"""Minimal valid runs for Que tests (imported by the test modules, which pytest puts on sys.path)."""

import logging
from collections.abc import Callable
from typing import Any, TypeAlias

from src.que.core import Que
from src.run_types import CompExpInfo, ExpInfo, FailedExp

MakeQue: TypeAlias = Callable[[], Que]
"""Builds a Que on a fixed runs path; calling it again simulates a restart."""

_AUGS: dict[str, Any] = {
    "normalise": True,
    "norm_dict": {"mean": [0.43, 0.39, 0.37], "std": [0.22, 0.22, 0.21]},
    "temporal_aug": [{"target_length": 32, "max_wobble": 0, "type": "uniform"}],
    "spatial_aug": [{"frame_size": 224, "type": "Centre_crop"}],
    "strict_size": True,
    "target_length": 32,
    "frame_size": 224,
}
_SPLIT_RES: dict[str, Any] = {
    "top_k_average_per_class_acc": {"top1": 0.7, "top5": 0.9, "top10": 0.95},
    "top_k_per_instance_acc": {"top1": 0.7, "top5": 0.9, "top10": 0.95},
    "average_loss": 1.2,
}
RESULTS: dict[str, Any] = {
    "check_name": "best_val",
    "best_val_acc": 70.0,
    "best_val_loss": 1.2,
    "test": _SPLIT_RES,
    "val": _SPLIT_RES,
}


def run_dict(exp_no: str, sweep_id: str | None = None) -> dict[str, Any]:
    """A minimal valid ExpInfo payload; `exp_no` doubles as the wandb run id."""
    return {
        "admin": {
            "model": "S3D",
            "dataset": "WLASL",
            "split": "asl100_cutoff_9",
            "save_path": f"runs/{exp_no}/checkpoints",
            "seed": 42,
            "exp_no": exp_no,
            "recover": False,
            "config_path": "configfiles/base.py",
            "weight_path": None,
        },
        "training": {"batch_size": 8, "update_per_step": 1, "max_epoch": 200},
        "optimizer": {
            "eps": 1e-6,
            "backbone_init_lr": 1e-4,
            "backbone_weight_decay": 0.1,
            "classifier_init_lr": 1e-4,
            "classifier_weight_decay": 0.1,
        },
        "model_params": {"drop_p": 0.5},
        "data": {"train_augs": _AUGS, "test_augs": _AUGS, "strict_size": True,
                 "target_length": 32, "frame_size": 224},
        "scheduler": {"type": "CosineAnnealingLR", "tmax": 73, "eta_min": 0.0},
        "stopping": {"type": "early_stopper", "metric": "loss", "phase": "val",
                     "mode": "min", "patience": 20, "min_delta": 0.01},
        "wandb": {"entity": "e", "project": "p", "tags": [], "run_id": exp_no,
                  "sweep_id": sweep_id},
    }


def exp_run(exp_no: str, sweep_id: str | None = None) -> ExpInfo:
    return ExpInfo.model_validate(run_dict(exp_no, sweep_id))


def comp_run(exp_no: str, sweep_id: str | None = None) -> CompExpInfo:
    return CompExpInfo.model_validate({**run_dict(exp_no, sweep_id), "results": RESULTS})


def failed_run(exp_no: str, sweep_id: str | None = None) -> FailedExp:
    return FailedExp.model_validate({**run_dict(exp_no, sweep_id), "error": "boom"})


def silent_logger(name: str) -> logging.Logger:
    """A logger that doesn't propagate to the root logger (which writes the real Server.log)."""
    logger = logging.getLogger(name)
    logger.propagate = False
    return logger
