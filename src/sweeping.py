"""Standalone entrypoint for wandb sweep agents.

Each invocation merges this trial's sweep-selected hyperparameters into a
base TOML config, builds a fresh RunInfo, and runs training exactly as
training.py would -- but without the interactive confirmation prompt, and
with save_path/exp_no derived from the wandb run id rather than filesystem
enumeration, so concurrent agents never collide.
"""

from __future__ import annotations

import argparse
try:
    import tomllib  # type: ignore
except ImportError:
    import tomli as tomllib
from pathlib import Path
from typing import Any, Dict, List

import wandb

from src.run_types import RunInfo, AdminInfo, RUNS_PATH, strict_validate, AVAIL_SPLITS
from src.configs import print_config, set_seed, get_avail_splits, get_model_checkpoint_dir
from src.training import train_model  # adjust if train_model lives elsewhere



def _resolve_list_index(lst: list, selector: str) -> int:
    """selector like 'type:RANDAUGMENT' -> index of the first list item whose
    'type' field matches. Falls back to plain int index for backward compat."""
    if ":" not in selector:
        return int(selector)
    field, _, val = selector.partition(":")
    for i, item in enumerate(lst):
        if isinstance(item, dict) and item.get(field) == val:
            return i
    raise KeyError(f"No item in list matches selector {selector!r}")


def _set_nested(d: Any, keys: List[str], value: Any) -> None:
    """Set a value at a dotted/list-indexed path inside a nested dict/list.

    Path segments that are integers index into lists (TOML arrays of tables,
    e.g. `data.train_augs.spatial_aug.2.magnitude`); all other segments index
    into dicts, creating intermediate dicts as needed.
    """
    key = keys[0]

    if isinstance(d, list):
        idx = _resolve_list_index(d, key)
        if len(keys) == 1:
            d[idx] = value
        else:
            _set_nested(d[idx], keys[1:], value)
        return

    if len(keys) == 1:
        d[key] = value
        return

    if key not in d or not isinstance(d[key], (dict, list)):
        d[key] = {}
    _set_nested(d[key], keys[1:], value)



SWEEP_KEY_MAP = {
    # optimizer
    "backbone_init_lr":        "optimizer.backbone_init_lr",
    "classifier_init_lr":      "optimizer.classifier_init_lr",
    "backbone_weight_decay":   "optimizer.backbone_weight_decay",
    "classifier_weight_decay": "optimizer.classifier_weight_decay",
    "eps":                     "optimizer.eps",

    # model
    "drop_p":                  "model_params.drop_p",

    # scheduler (WarmRestartInfo)
    "t0":                      "scheduler.t0",
    "tmult":                   "scheduler.tmult",
    "eta_min":                 "scheduler.eta_min",

    # temporal aug — selector-based, safe against reordering
    "max_wobble":              "data.train_augs.temporal_aug.type:chunked.max_wobble",

    # spatial aug
    "magnitude":               "data.train_augs.spatial_aug.type:RANDAUGMENT.magnitude",
    "num_ops":                 "data.train_augs.spatial_aug.type:RANDAUGMENT.num_ops",
    "num_magnitude_bins":      "data.train_augs.spatial_aug.type:RANDAUGMENT.num_magnitude_bins",
    "hflip_p":                 "data.train_augs.spatial_aug.type:HORIZONTAL_FLIP.p",

    # early stopping
    "patience":                "stopping.patience",
    "min_delta":                "stopping.min_delta",
}

def validate_sweep_key_map(base_path: Path, key_map: dict = SWEEP_KEY_MAP) -> None:
    with open(base_path, "rb") as f:
        raw = tomllib.load(f)
    for name, dotted in key_map.items():
        d = raw
        for k in dotted.split("."):
            try:
                d = d[_resolve_list_index(d, k)] if isinstance(d, list) else d[k]
            except (KeyError, IndexError) as e:
                raise ValueError(
                    f"SWEEP_KEY_MAP[{name!r}] -> {dotted!r} does not resolve "
                    f"against {base_path}: {e}"
                ) from None

def apply_sweep_overrides(raw: Dict[str, Any], wandb_config: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in wandb_config.items():
        dotted_key = SWEEP_KEY_MAP.get(key, key)
        _set_nested(raw, dotted_key.split("."), value)
    return raw


def get_sweep_exp_dir(split: str, model: str, sweep_id: str, run_id: str,
                       runs_path: Path | str = RUNS_PATH) -> Path:
    """Per-trial output dir, namespaced by sweep so concurrent agents never collide."""
    return Path(runs_path) / split / model / f"sweep_{sweep_id}" / run_id


def get_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single trial of a wandb sweep")
    parser.add_argument("model", type=str)
    parser.add_argument("split", type=str, choices=get_avail_splits())
    parser.add_argument(
        "-bc", "--base_config", type=str, required=True,
        help="TOML template; sweep parameters override its values",
    )
    parser.add_argument("-ds", "--dataset", type=str, default="WLASL")
    parser.add_argument("-se", "--save_every", type=int, default=5)
    return parser

def create_sweep_run(base_path: Path, split: AVAIL_SPLITS, model: str, dataset: str):
    
    validate_sweep_key_map(base_path)
    
    
    with open(base_path, "rb") as f:
        raw = tomllib.load(f)
        
    # wandb agent sets WANDB_SWEEP_ID / WANDB_ENTITY / WANDB_PROJECT in the env;
    # wandb.init() attaches to the sweep and populates run.config with this
    # trial's chosen hyperparameters.
    run = wandb.init()
    
    sweep_overrides = dict(run.config)
    if sweep_overrides:
        raw = apply_sweep_overrides(raw, sweep_overrides)

    sweep_id = run.sweep_id or "manual"
    exp_dir = get_sweep_exp_dir(split, model, sweep_id, run.id)
    save_path = get_model_checkpoint_dir(exp_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    admin = AdminInfo(
        model=model,
        dataset=dataset,
        split=split,
        exp_no=run.id,
        recover=False,
        config_path=str(base_path),
        save_path=str(save_path),
        weight_path=None,
    )

    config = strict_validate(RunInfo, {"admin": admin.model_dump(), **raw})

    run.name = f"{admin.model}_{admin.split}_{run.id}"
    run.config.update(config.model_dump(), allow_val_change=True)  # log resolved config

    print_config(config)
    
    return config, run


def main():
    args = get_sweep_parser().parse_args()

    base_path = Path(args.base_config)
    if not base_path.exists():
        raise FileNotFoundError(f"{base_path} not found")

    config, run = create_sweep_run(
        base_path=base_path,
        split=args.split,
        model=args.model,
        dataset=args.dataset
    )

    train_model(args.model, config, run, save_every=args.save_every, recover=False)
    run.finish()


if __name__ == "__main__":
    main()