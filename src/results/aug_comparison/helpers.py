from collections.abc import Callable, Iterable
from statistics import mean

# locals
import pandas as pd

from src.que.core import CompExpInfo
from src.run_types import BaseRes, CompRes

# ------------------------------
# Converting to Dataframe
# ------------------------------


def get_crop_name(run: CompExpInfo) -> str:
    assert run.data.train_augs is not None, "Spatial augmentations are None"
    assert run.data.train_augs.spatial_aug is not None, "Spatial augmentations are None"
    return run.data.train_augs.spatial_aug[0].type


def get_sampler_name(run: CompExpInfo) -> str:
    assert run.data.train_augs is not None, "Temporal augmentations are None"
    assert run.data.train_augs.temporal_aug is not None, (
        "Temporal augmentations are None"
    )
    return run.data.train_augs.temporal_aug[0].type


auto_aug_policies = ["SVHN", "CIFAR10", "IMAGENET"]
keys = [
    "num_ops",
    "magnitude",
    "mean",
    "std",
    "max_wobble",
    "speed_min",
    "speed_max",
]


def get_parameters_and_values(run: CompExpInfo) -> dict:
    assert run.data.train_augs is not None, "Train augmentations are None"
    assert run.data.train_augs.spatial_aug is not None, "Spatial augmentations are None"
    assert run.data.train_augs.temporal_aug is not None, (
        "Temporal augmentations are None"
    )

    key_value_pairs = {}

    for aug in run.data.train_augs.spatial_aug + run.data.train_augs.temporal_aug:
        aug_dict = aug.model_dump()
        match_keys = [k for k in keys if k in aug_dict]
        values = [aug_dict[k] for k in match_keys]

        key_value_pairs.update({k: v for k, v in zip(match_keys, values)})

        if "type" in aug_dict and aug_dict["type"] in auto_aug_policies:
            key_value_pairs.update({"policy": aug_dict["type"]})

    return key_value_pairs


def unpack_parameters_and_values(run: CompExpInfo) -> dict:
    # max three parameters, fill columns with blank -
    params_and_values = get_parameters_and_values(run)
    params = []
    keys = list(params_and_values.keys())
    for i in range(3):
        params.append(
            f"{keys[i]}: {params_and_values[keys[i]]}" if i < len(keys) else "-"
        )

    return {f"param_{i + 1}": params[i] for i in range(3)}


def to_df(
    runs: list[CompExpInfo], split_name: str, set_names: list[str], acc_type: str
) -> pd.DataFrame:
    df_format = [
        {
            "exp no": run.admin.exp_no,
            "run_id": run.wandb.run_id,
            "subset": run.admin.split,
            "type": "control" if len(run.wandb.tags) == 1 else run.wandb.tags[0],
            "strategy": run.wandb.tags[-1],
            # "crop": get_crop_name(run),
            # "sampler": get_sampler_name(run),
        }
        | unpack_parameters_and_values(run)
        | {
            f"{set_name} {k}": v
            for set_name in set_names
            for k, v in run.results.model_dump()[set_name][acc_type].items()
        }
        | {
            "best_val_acc": run.results.best_val_acc,
            "best_val_loss": run.results.best_val_loss,
            "test_loss": run.results.test.average_loss,
            "config path": run.admin.config_path,
        }
        for run in runs
    ]

    df = pd.DataFrame(df_format)

    # Rename columns and format values
    ns = [1, 5, 10]
    for set_name in set_names:
        for n in ns:
            old_name, new_name = (
                f"{set_name} top{n}",
                f"{set_name.capitalize()} Top-{n}",
            )

            df = df.rename(columns={old_name: new_name})
            df[new_name] = df[new_name].apply(lambda x: f"{x * 100:.2f}")

    subdf = df[df["subset"] == split_name]

    return subdf


# ------------------------------
# Averaged two runs
# ------------------------------


def _average_dict(ds: Iterable[dict]) -> dict:
    ds = list(ds)
    return {k: mean(d[k] for d in ds) for k in ds[0]}


def _average_set(base_models: Iterable[BaseRes]) -> BaseRes:
    ds = [b.model_dump() for b in base_models]
    k1 = "top_k_average_per_class_acc"
    k2 = "top_k_per_instance_acc"
    k3 = "average_loss"
    acc_keys = [k1, k2]
    return BaseRes.model_validate(
        {k: _average_dict(d[k] for d in ds) for k in acc_keys}
        | {k3: mean(d[k3] for d in ds)}
    )


def _average_results(reses: Iterable[CompRes]) -> CompRes:
    reses = list(reses)
    return CompRes(
        check_name="averaged",
        best_val_acc=mean(res.best_val_acc for res in reses),
        best_val_loss=mean(res.best_val_loss for res in reses),
        test=_average_set(res.test for res in reses),
        val=_average_set(res.val for res in reses),
    )


def to_average_rows_df(
    runs: Iterable[CompExpInfo], split_name: str, set_names: list[str] | None = None, acc_type: str = "top_k_per_instance_acc"
) -> pd.DataFrame:
    """Averages the `results` for all runs that have the same `config_path` and returns
    a summarised dataframe representation, where each row corresponds to a single config.

    Args:
        runs (Iterable[CompExpInfo]): Runs to average results across configs.
        split_name (str): Subdf to select. One of the valid WLASL splits. 
        set_names (list[str] | None, optional): List of valid set names, otherwise ['test', 'val']. Defaults to None.
        acc_type (str, optional): One of the valid accuracies. Defaults to "top_k_per_instance_acc".

    Returns:
        pd.DataFrame: Single `split` Dataframe containing averaged rows. 
    """
    if set_names is None:
        set_names = ['test', 'val']
    
    packed_runs: dict[str, list[CompExpInfo]] = {}
    averaged_runs: dict[str, CompExpInfo] = {}

    # collect runs under same config
    for run in runs:
        c_path = run.admin.config_path
        if c_path in packed_runs:
            packed_runs[c_path].append(run)
        else:
            packed_runs[c_path] = [run]
            averaged_runs[c_path] = run  # initialise with rest of config

    # average runs
    for k, v in packed_runs.items():
        averaged_runs[k].results = _average_results(run.results for run in v)

    # create df
    df_format = [
        {
            "subset": run.admin.split,
            "type": "control" if len(run.wandb.tags) == 1 else run.wandb.tags[0],
            "strategy": run.wandb.tags[-1],
            # "crop": get_crop_name(run),
            # "sampler": get_sampler_name(run),
        }
        | unpack_parameters_and_values(run)
        | {
            f"{set_name} {k}": v
            for set_name in set_names
            for k, v in run.results.model_dump()[set_name][acc_type].items()
        }
        | {
            "best_val_acc": run.results.best_val_acc,
            "best_val_loss": run.results.best_val_loss,
            "test_loss": run.results.test.average_loss,
            "val_loss": run.results.val.average_loss,
            "config path": run.admin.config_path,
        }
        for run in averaged_runs.values()
    ]

    df = pd.DataFrame(df_format)

    # Rename columns and format values
    ns = [1, 5, 10]
    for set_name in set_names:
        for n in ns:
            old_name, new_name = (
                f"{set_name} top{n}",
                f"{set_name.capitalize()} Top-{n}",
            )

            df = df.rename(columns={old_name: new_name})
            df[new_name] = df[new_name].apply(lambda x: f"{x * 100:.2f}")

    subdf = df[df["subset"] == split_name]

    return subdf


# ------------------------------
# Augmentations
# ------------------------------

name_maps: dict[str, Callable[[str], str]] = {
    "spatial": lambda x: _fix_part_spatial(x),
    "temporal": lambda x: _fix_part_temporal(x),
}


def _fix_part_spatial(part: str) -> str:
    if part == "resize":
        part += "d"
    if part.startswith("op") and len(part) == 3 and part[2].isdigit():
        return f"Ops: {part[2]},"
    elif part.startswith("mag") and 4 <= len(part) <= 5:
        return f"Mag: {part[3:]}"
    elif part in ["CIFAR10", "IMAGENET", "SVHN"]:
        return part
    else:
        return part.capitalize()


def _fix_part_temporal(part: str) -> str:
    if part.startswith("wobble") and part[6].isdigit():
        return f"Idx Disp: {part[6]}"
    elif part.startswith("std") and all(x.isdigit() for x in part[3:]):
        return f"Std: {float(part[3:]) / 100}"
    elif part.startswith("speed") and all(x.isdigit() for x in part[5:]):
        return f"Speed Dev: {float(part[5:]) / 10}"
    else:
        return part.capitalize()


def aug_name_mapper(aug_name: str, part_fixer: Callable[[str], str]) -> str:
    return " ".join(part_fixer(p) for p in aug_name.split("_"))

def get_y_max(subdf: pd.DataFrame,  loss_name: str, aug_types: list[str] | None = None) -> float:
    """Get shared y_max for consistent plots"""
    if aug_types is None:
        aug_types = ['spatial', 'temporal']
    vals = []
    for aug_type in aug_types:
        aug_df = subdf[subdf["type"].isin([aug_type, 'control'])]
        vals.append(max(aug_df[loss_name]))
        
        
    return max(vals)
    