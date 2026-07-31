"""Helpers for running wandb sweep trials, plus a standalone CLI entrypoint.
 
This module serves two roles:
 
1. A library of commonly-used pieces (build_base_config, SWEEP_KEY_MAP,
   apply_sweep_overrides, create_sweep_run, ...) called in-process by the Que
   system: Worker._sweep_train hands these to wandb.agent(function=...), so a
   trial's RunInfo is built from an in-memory config skeleton plus that
   trial's sweep-selected hyperparameters, rather than merging onto a
   base_config.toml file on disk.
 
2. A standalone script (see `main`/`get_sweep_parser` below) that can itself
   be the target of a sweep yaml's `program`/`command` field, so the same
   sweep can be launched the plain way -- `wandb agent <entity>/<project>/
   <sweep_id>` -- for quick testing outside the Daemon/Worker stack. Keep
   `main()` a valid, self-sufficient entrypoint: the `command` block in any
   sweep yaml pointing at this file depends on it.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import wandb
from src.configs import get_avail_splits, get_model_checkpoint_dir
from src.run_types import AVAIL_SPLITS, RUNS_PATH, AdminInfo, RunInfo, strict_validate
from src.training import train_model
from src.utils import load_module_from_path


class SweepConfigError(ValueError):
    """Raised for any sweep-config problem that should fail loudly and early:
    a SWEEP_KEY_MAP entry that doesn't resolve against the skeleton, or a
    skeleton leaf that never got overridden by a sweep value."""


BASE_CONFIG_ATTR = "base_config"


def build_base_config(config_path: Path) -> dict:
    """The fixed shape every sweep trial's config takes, loaded from an
    arbitrary Python file at `config_path` rather than hardcoded here.

    `config_path` must define a top-level dict named `BASE_CONFIG`. `None`
    marks a leaf that MUST be supplied by SWEEP_KEY_MAP/run.config --
    validate_resolved() checks this after overrides are applied. The
    skeleton typically assumes exactly one sampler and one cropper per split
    (train: flip + crop + randaugment; test: crop only) -- this is what makes
    flat wandb params workable instead of needing list-index paths -- and
    hardcodes the sampler and crop method, so a different combination needs
    its own base_config.py and sweep.

    Returns a fresh deep copy each call, since apply_sweep_overrides mutates
    its input in place and the loaded module-level dict must stay pristine
    across repeated calls (e.g. multiple trials in the same sweep agent
    process).
    """
    module = load_module_from_path(config_path, module_prefix="_base_config")

    if not hasattr(module, BASE_CONFIG_ATTR):
        raise SweepConfigError(
            f"{config_path} has no top-level `{BASE_CONFIG_ATTR}` dict for build_base_config to load."
        )

    base_config = getattr(module, BASE_CONFIG_ATTR)
    if not isinstance(base_config, dict):
        raise SweepConfigError(
            f"{config_path}: `{BASE_CONFIG_ATTR}` must be a dict, got {type(base_config).__name__}."
        )

    return copy.deepcopy(base_config)


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


def _set_nested(d: Any, keys: list[str], value: Any) -> None:
    """Set a value at a dotted/selector path inside a nested dict/list."""
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


def _find_unresolved(d: Any, path: str = "") -> list[str]:
    """Recursively collect dotted paths of any leaf still set to None."""
    unresolved: list[str] = []
    if isinstance(d, dict):
        for k, v in d.items():
            unresolved.extend(_find_unresolved(v, f"{path}.{k}" if path else k))
    elif isinstance(d, list):
        for i, item in enumerate(d):
            unresolved.extend(_find_unresolved(item, f"{path}.{i}"))
    elif d is None:
        unresolved.append(path)
    return unresolved


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
    "start_factor":            "scheduler.warm_up.start_factor",
    "end_factor":              "scheduler.warm_up.end_factor",
    "warmup_epochs":           "scheduler.warm_upwarmup_epcochs",

    # training
    "batch_size":              "training.batch_size",
    "update_per_step":         "training.update_per_step",

    # temporal aug -- now maps to both train and test
    "max_wobble":              "data.train_augs.temporal_aug.type:chunked.max_wobble",
    "target_length": [
        "data.train_augs.temporal_aug.type:chunked.target_length",
        "data.test_augs.temporal_aug.type:uniform.target_length"
    ],

    # spatial aug -- now maps to both train and test
    "hflip_p":                 "data.train_augs.spatial_aug.type:HORIZONTAL_FLIP.p",
    "frame_size": [
        "data.train_augs.spatial_aug.type:Centre_crop.frame_size",
        "data.test_augs.spatial_aug.type:Centre_crop.frame_size"
    ],
    "magnitude":               "data.train_augs.spatial_aug.type:RANDAUGMENT.magnitude",
    "num_ops":                 "data.train_augs.spatial_aug.type:RANDAUGMENT.num_ops",
    "num_magnitude_bins":      "data.train_augs.spatial_aug.type:RANDAUGMENT.num_magnitude_bins",

    # early stopping
    "max_epoch":               "stopping.max_epoch",
    "patience":                "stopping.patience",
    "min_delta":               "stopping.min_delta",
}


def apply_sweep_overrides(raw: dict[str, Any], wandb_config: dict[str, Any]) -> dict[str, Any]:
    """Mutate `raw` in place, applying each wandb.config key via SWEEP_KEY_MAP
    (or, if absent from the map, as a literal dotted path)."""
    for key, value in wandb_config.items():
        dotted_or_list = SWEEP_KEY_MAP.get(key, key)
        dotted_keys = [dotted_or_list] if isinstance(dotted_or_list, str) else dotted_or_list
        for dotted_key in dotted_keys:
            _set_nested(raw, dotted_key.split("."), value)
    return raw


def validate_sweep_key_map(config_path: Path, key_map: dict = SWEEP_KEY_MAP) -> None:
    """Check every SWEEP_KEY_MAP target resolves against the skeleton shape.
    Structural check only -- doesn't require any particular run's values, so
    it can run once at sweep-launch time, independent of wandb.init().
    """
    raw = build_base_config(config_path)
    for name, dotted_or_list in key_map.items():
        dotted_list = [dotted_or_list] if isinstance(dotted_or_list, str) else dotted_or_list
        for dotted in dotted_list:
            d = raw
            for k in dotted.split("."):
                try:
                    d = d[_resolve_list_index(d, k)] if isinstance(d, list) else d[k]
                except (KeyError, IndexError) as e:
                    raise SweepConfigError(
                        f"SWEEP_KEY_MAP[{name!r}] -> {dotted!r} does not resolve "
                        f"against the base config skeleton: {e}"
                    ) from None

def validate_resolved(config: dict[str, Any]) -> None:
    """Raise if any skeleton placeholder was never overwritten by a sweep
    value -- catches a yaml `parameters` entry that got renamed/removed
    without updating SWEEP_KEY_MAP (or vice versa) before it silently reaches
    Pydantic as None."""
    unresolved = _find_unresolved(config)
    if unresolved:
        raise SweepConfigError(
            "Sweep config has unresolved placeholders (no value supplied "
            f"for): {unresolved}. Check that the sweep yaml's `parameters` "
            "block and SWEEP_KEY_MAP are in sync."
        )


def get_sweep_exp_dir(split: str, model: str, sweep_id: str, run_id: str,
                       runs_path: Path | str = RUNS_PATH) -> Path:
    """Per-trial output dir, namespaced by sweep so concurrent agents never collide."""
    return Path(runs_path) / split / model / f"sweep_{sweep_id}" / run_id


def create_sweep_run(model: str, split: AVAIL_SPLITS, config_path: Path, dataset: str = "WLASL"):
    """Build a fresh RunInfo + wandb Run for one sweep trial.

    Called from inside the callback passed to wandb.agent(function=...).
    wandb.agent sets sweep context via env vars before invoking that callback;
    wandb.init() attaches to the sweep and populates run.config with this
    trial's chosen hyperparameters -- do NOT pass `config=` or `id=` here,
    that would shadow the sweep-sampled values with defaults.

    `model`/`split`/`dataset` are NOT sweep parameters -- they're passed in by
    the caller (Worker, via SweepInfo from the Daemon), since they determine
    architecture/data plumbing rather than being tuned. `config_path` points
    at the base_config.py defining the BASE_CONFIG skeleton for this sweep
    (see build_base_config) and is likewise caller-supplied rather than tuned.
    """
    validate_sweep_key_map(config_path)  # structural check, fails before any wandb call

    run = wandb.init()

    raw = build_base_config(config_path)
    sweep_overrides = dict(run.config)
    if sweep_overrides:
        raw = apply_sweep_overrides(raw, sweep_overrides)

    validate_resolved(raw)

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
        config_path="<in-memory:sweep>",  # no file backs this config anymore
        save_path=str(save_path),
        weight_path=None,
    )

    config = strict_validate(RunInfo, {"admin": admin.model_dump(), **raw})

    run.name = f"{admin.model}_{admin.split}_{run.id}"
    run.config.update(config.model_dump(), allow_val_change=True)  # log resolved config

    return config, run


# --- CLI entrypoint, for running this as a subprocess under `wandb agent` ---
#
# Function mode (Worker._sweep_train calling create_sweep_run directly) is the
# path used by the Que system. This entrypoint exists so the same sweep can
# also be launched the plain way -- `wandb agent <entity>/<project>/<sweep_id>`
# -- for quick standalone testing outside the Daemon/Worker stack. To use this
# mode, the sweep yaml needs `program`/`command` pointing back at this script,
# e.g.:
#
#   command:
#     - ${env}
#     - python
#     - ${program}
#     - MViTv2_B_32x3
#     - asl100
#     - /path/to/base_config.py
#
# model/split/config_path/dataset/save_every are CLI args here, NOT wandb
# parameters -- they aren't tuned, so they don't belong in the sweep's
# `parameters` block (see create_sweep_run's docstring).

def get_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single trial of a wandb sweep")
    parser.add_argument("model", type=str)
    parser.add_argument("split", type=str, choices=get_avail_splits())
    parser.add_argument("config_path", type=Path, help="Path to a base_config.py defining BASE_CONFIG")
    parser.add_argument("-ds", "--dataset", type=str, default="WLASL")
    parser.add_argument("-se", "--save_every", type=int, default=5)
    return parser


def main():
    args = get_sweep_parser().parse_args()

    config, run = create_sweep_run(
        model=args.model, split=args.split, config_path=args.config_path, dataset=args.dataset
    )

    train_model(args.model, config, run, save_every=args.save_every, recover=False)
    run.finish()


if __name__ == "__main__":
    main()