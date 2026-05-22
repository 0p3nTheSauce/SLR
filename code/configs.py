import configparser
import argparse
import ast
from typing import Callable, Dict, Any, List, Optional, Union, Tuple, Literal

from pathlib import Path

import json
try:
    import tomllib #type: ignore
except ImportError:
    import tomli as tomllib
# locals
# from models import avail_models, norm_vals
from run_types import (
    WandbInfo,
    RunInfo,
    AdminInfo,
    AugInfo,
    OG_Sampler
)


# constants
ENTITY = "ljgoodall2001-rhodes-university"
PROJECT_BASE = "WLASL"
LABEL_SUFFIX = "instances_fixed_frange_bboxes_len.json"
NUM_INSTANCES_SUFFIX = "num_instances.json"
# LABEL_INSTANCES_SUFFIX = "instances_fixed_frange_bboxes_len.json"
CLASSES_PATH = "./info/wlasl_class_list.json"
WLASL_ROOT = "../data/WLASL"
LABELS_PATH = WLASL_ROOT + "/preprocessed/labels"
RAW_DIR = "WLASL2000"
SPLIT_DIR = "splits"
RUNS_PATH = "./runs"
CONFIGS_PATH = "./configfiles"
ZFILL = 3
CONFIG_FILETYPE = '.toml'
SEED = 42


def ask_nicely(
    message: str, requirment: Optional[Callable] = None, error: Optional[str] = None
) -> str:
    """Ask user for input, with optional requirement and error message."""
    while True:
        ans = input(message)
        if requirment is None or requirment(ans):
            return ans
        else:
            print(error if error else "Invalid input, please try again.")


####################### Utility functions ##############################


def set_seed(seed: int = SEED):
    """Set the random seed across multiple environments."""
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_avail_splits(pre_proc_dir: str = LABELS_PATH) -> List[str]:
    """Get the available splits from preprocessed labels directory."""
    ppd = Path(pre_proc_dir)
    if not ppd.exists() or not ppd.is_dir():
        raise ValueError(
            f"Invalid preprocessed directory: {pre_proc_dir}, must exist and be a directory"
        )
    return list(map(lambda x: x.name, ppd.iterdir()))


def print_config(config: RunInfo):
    """Print a RunInfo model in a readable format."""
    for section, data in config.model_dump().items():
        print(f"{section.upper()}:")
        if isinstance(data, dict):
            max_key_len = max((len(str(k)) for k in data), default=0)
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    print(json.dumps(value, indent=4))
                else:
                    print(f"    {key:<{max_key_len}} : {value}")
        else:
            print(f"  {data}")
        print()


def get_class_list(classes_path: Union[str, Path] = CLASSES_PATH) -> List[str]:
    """Retrieve the classes list from file."""
    with open(classes_path, "r") as f:
        class_list = json.load(f)
    return class_list


###################### Config generation ###############################


def _convert_type(value: str) -> Any:
    """Convert string values from .ini to appropriate Python types."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_ini_config(ini_file: Union[str, Path]) -> Dict[str, Any]:
    """Parse .ini file into a nested dict with correctly typed values."""
    config = configparser.ConfigParser()
    config.read(ini_file)
    return {
        section: {key: _convert_type(value) for key, value in config[section].items()}
        for section in config.sections()
    }


def _make_aug_info(
    aug_conf: Optional[Dict[str, Any]],
    model_name: str,
    mode: Literal["train", "test", "val"],
) -> AugInfo:
    from models import norm_vals
    """Build an AugInfo model, filling in defaults when aug_conf is None.

    Validation of strategy values is handled by AugInfo's Literal type annotations —
    pydantic will raise a ValidationError automatically on invalid values.
    """
    if aug_conf is None:
        aug_conf = {"norm_dict": norm_vals(model_name)}
        if mode == "train":
            aug_conf["spatial_aug"] = ["Horizontal_flip"]
            aug_conf["frame_size_strategy"] = "Random_crop"
            aug_conf['frame_sampler'] = OG_Sampler(target_length=aug_conf['target_length'])
        else:
            aug_conf["frame_size_strategy"] = "Centre_crop"
            aug_conf['frame_sampler'] = OG_Sampler(target_length=aug_conf['target_length'])

    # Resolve the "on" shorthand for norm_dict
    if aug_conf.get("norm_dict") == "on":
        aug_conf["norm_dict"] = norm_vals(model_name)

    return AugInfo.model_validate(aug_conf)


def load_config_retro(admin: AdminInfo) -> RunInfo:
    """ Old config loader for backward compatibility
    Load config from .ini file and merge with AdminInfo from CLI.

    Args:
        admin: Parsed admin info from command line arguments.

    Returns:
        RunInfo: Fully validated config model for the run.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        pydantic.ValidationError: If config values fail validation.
    """
    conf_path = Path(admin.config_path)
    if not conf_path.exists():
        raise FileNotFoundError(f"{conf_path} not found")

    raw = parse_ini_config(admin.config_path)

    # Resolve augmentations before handing off to pydantic
    data_conf = raw.get("data", {})
    data_conf["train_augs"] = _make_aug_info(raw.pop("train_augs", None), admin.model, "train")
    data_conf["test_augs"] = _make_aug_info(raw.pop("test_augs", None), admin.model, "test")
    raw["data"] = data_conf

    # Nest warmup into the scheduler subdict so pydantic sees the right shape
    if "scheduler" in raw and "warmup_epochs" in raw["scheduler"]:
        sched = raw["scheduler"]
        raw["scheduler"]["warm_up"] = {
            "start_factor": sched.pop("start_factor"),
            "end_factor": sched.pop("end_factor"),
            "warmup_epochs": sched.pop("warmup_epochs"),
        }

    # pydantic handles: type coercion, required field checks, discriminated union
    # dispatch, WarmUpSched factor validation, EarlyStopper metric checks, defaults
    return RunInfo.model_validate({"admin": admin.model_dump(), **raw})

def load_config(admin: AdminInfo) -> RunInfo:
    """Load config from .toml file and merge with AdminInfo from CLI.
    Supports backward compatibility with .ini files, using old loading mechanism.

    Args:
        admin: Parsed admin info from command line arguments.

    Returns:
        RunInfo: Fully validated config model for the run.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        pydantic.ValidationError: If config values fail validation.
    """
    conf_path = Path(admin.config_path)
    if not conf_path.exists():
        raise FileNotFoundError(f"{conf_path} not found")
    
    if conf_path.name.endswith('.ini'):
        return load_config_retro(admin)
    
    with open(conf_path, 'rb') as f:
        raw = tomllib.load(f)
        
    return RunInfo.model_validate({"admin": admin.model_dump(), **raw})
    


###################### Path utilities ###############################


def get_next_expno(split: str, model: str, runs_path: str = RUNS_PATH) -> int:
    model_dir = Path(runs_path) / split / model
    if not (model_dir.exists() and model_dir.is_dir()):
        return 0
    model_exps = sorted(model_dir.glob("exp*"))
    return int(model_exps[-1].name[-3:])


def get_model_exp_dir(
    split: str,
    model: str,
    exp_no: int = 0,
    runs_path: str = RUNS_PATH,
    zd: int = ZFILL,
) -> Path:
    return Path(f"{runs_path}/{split}/{model}/exp{str(exp_no).zfill(zd)}")


def get_model_checkpoint_dir(
    model_exp_dir: Path,
    checkpoint_num: Optional[int] = None,
    zd: int = ZFILL,
) -> Path:
    if checkpoint_num is None:
        return model_exp_dir / "checkpoints"
    return model_exp_dir / f"checkpoints{str(checkpoint_num).zfill(zd)}"


def get_model_results_dir(
    model_exp_dir: Path, checkpoint_num: Optional[int] = None, zd: int = ZFILL
) -> Path:
    if checkpoint_num is None:
        return model_exp_dir / "results"
    return model_exp_dir / f"results{str(checkpoint_num).zfill(zd)}"


def get_config_path(
    split: str,
    model: str,
    exp_no: int = 0,
    configs_dir: str = CONFIGS_PATH,
    zd: int = ZFILL,
    file_type: str = CONFIG_FILETYPE
) -> Path:
    return Path(configs_dir) / f"{split}/{model}/exp{str(exp_no).zfill(zd)}{file_type}"


###################### Argument parsing ###############################


def get_train_parser(
    prog: Optional[str] = None, desc: str = "Train a model", model_opts: Optional[List[str]] = None, split_opts: Optional[List[str]] = None
) -> argparse.ArgumentParser:
    """Get parser for a training configuration

    Args:
        prog (Optional[str], optional): Script name, (e.g. train.py). Defaults to None.
        desc (str, optional): Program desctiption. Defaults to "Train a model".

    Returns:
        argparse.ArgumentParser: Parser which takes training arguments
    """
    if model_opts is None:
        from models import avail_models
        models_available = avail_models()
    else:
        models_available = model_opts
        
    if split_opts is None:
        splits_available = get_avail_splits()
    else:
        splits_available = split_opts

    parser = argparse.ArgumentParser(description=desc, prog=prog)

    parser.add_argument("model", type=str, choices=models_available, help="Model name from one of the implemented model")
    parser.add_argument("split", type=str, choices=splits_available, help="The class split")

    # experiment_gen_type = parser.add_mutually_exclusive_group(required=True)
    # experiment_gen_type.add_argument("-en", "--exp_no", type=int, help="Experiment number (e.g. 10)")
    # experiment_gen_type.add_argument("-c", "--config_path", help="Path to config file")

    # experiment_gen_type = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("-en", "--exp_no", type=int, help="Experiment number (e.g. 10)")
    parser.add_argument("-c", "--config_path", help="Path to config file")


    parser.add_argument("-ds", "--dataset", type=str, choices=["WLASL"], default="WLASL", help="Not implemented yet")
    parser.add_argument("-r", "--recover", action="store_true", help="Recover from last checkpoint")
    parser.add_argument("-ri", "--run_id", type=str, default=None, help="The run id to use (especially when also using recover)")
    parser.add_argument("-p", "--project", type=str, help=f"wandb project name, if not {PROJECT_BASE}-num_classes (e.g. {PROJECT_BASE}-100)")
    parser.add_argument("-et", "--entity", type=str, default=ENTITY, help=f"Entity if not {ENTITY}")
    parser.add_argument("-t", "--tags", nargs="+", type=str, help="Additional wandb tags")
    parser.add_argument("-w", "--weights_path", type=str, help="Path to model pretrained weights")
    parser.add_argument("-b", "--force_weight", action="store_true", help="Use weights_path even if it does not exist")
    parser.add_argument("-na", "--no_ask", action="store_true", help="Don't ask for confirmation")
    parser.add_argument("-nec", "--no_enum_chck", action="store_true", help="Do not enumerate the checkpoint dir num (for output)")
    parser.add_argument("-f", "--config_filetype", type=str, default=CONFIG_FILETYPE, help=f'Config file type, defaults to: {CONFIG_FILETYPE}')
    return parser


def take_args(
    sup_args: Optional[List[str]] = None,
    parsed_args: Optional[argparse.Namespace] = None,
) -> Optional[Tuple[AdminInfo, WandbInfo]]:
    """Retrieve and validate arguments for a new training run."""
    from models import avail_models
    from utils import enum_dir
    models_available = avail_models()
    splits_available = get_avail_splits()

    parser = get_train_parser(model_opts=models_available, split_opts=splits_available)
    if sup_args:
        args = parser.parse_args(sup_args)
    elif parsed_args:
        args = parsed_args
    else:
        args = parser.parse_args()

    if args.split not in splits_available:
        raise ValueError(f"{args.split} not processed yet. Available: {splits_available}")
    if args.model not in models_available:
        raise ValueError(f"{args.model} not implemented yet. Available: {models_available}")

    if args.project is None:
        args.project = f"{PROJECT_BASE}-{args.split[3:]}"

    if args.exp_no is None and args.config_path is None:
        parser.error("Either --exp_no or --config_path must be provided to identify the experiment configuration.")

    if args.exp_no is None:
        exp_no = get_next_expno(args.split, args.model)
        enum_exp = True
    else:
        exp_no = args.exp_no
        enum_exp = False

    if args.config_path is None:
        assert not enum_exp, "Enumerating the experiment is not valid in this case"
        args.config_path = str(get_config_path(args.split, args.model, exp_no, file_type=args.config_filetype))

    output = get_model_exp_dir(split=args.split, model=args.model, exp_no=exp_no)

    if enum_exp:
        output = enum_dir(output, decimals=ZFILL)
        exp_no = int(output.name[-3:])

    save_path = get_model_checkpoint_dir(output)

    if not (args.recover or args.no_enum_chck):
        save_path = enum_dir(save_path, decimals=ZFILL)
    elif args.recover:
        if not save_path.exists() or not save_path.is_dir():
            raise ValueError(f"Cannot recover: {save_path} does not exist or is not a directory")
        if len([f for f in save_path.iterdir() if f.is_file()]) == 0:
            raise ValueError(f"Cannot recover: {save_path} is empty")

    tags = []
    if args.recover:
        tags.append("Recovered")
    if args.tags:
        tags.extend(args.tags)

    weights_path = None
    if args.weights_path:
        wp = Path(args.weights_path)
        if not wp.exists() and not args.force_weight:
            raise FileNotFoundError(f"Weights file not found: {wp}")
        weights_path = str(wp)

    # Pydantic models — construction is the same, but now validated on init
    wandb_info = WandbInfo(
        entity=args.entity,
        project=args.project,
        tags=tags,
        run_id=args.run_id,
    )
    admin_info = AdminInfo(
        model=args.model,
        dataset=args.dataset,
        split=args.split,
        exp_no=str(exp_no).zfill(ZFILL),
        recover=args.recover,
        config_path=args.config_path,
        save_path=str(save_path),
        weight_path=weights_path,
    )

    return admin_info, wandb_info


def main():
    maybe_args = take_args()
    if isinstance(maybe_args, tuple):
        admin_info, wandb_info = maybe_args
    else:
        return
    config = load_config(admin_info)
    print_config(config)


if __name__ == "__main__":
    main()
