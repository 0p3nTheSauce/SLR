import configparser
import argparse
import ast
from typing import Dict, Any, List, Optional
from utils import enum_dir, ask_nicely
from stopping import EarlyStopper
from pathlib import Path

# locals
from models import avail_models

# TODO: make configs the sole source of these constants

# constants
# - wandb
ENTITY = "ljgoodall2001-rhodes-university"
PROJECT = "WLASL-100"
PROJECT_BASE = "WLASL"
# - data
LABELS_PATH = "preprocessed/labels"
LABEL_SUFFIX = "_fixed_frange_bboxes_len.json"
CLASSES_PATH = "./info/wlasl_class_list.json"
WLASL_ROOT = "../data/WLASL"
RAW_DIR = "WLASL2000"
SPLIT_DIR = "splits"
# - training
RUNS_PATH = "./runs"


def get_avail_splits(pre_proc_dir: str = LABELS_PATH) -> List[str]:
    ppd = Path(pre_proc_dir)
    if not ppd.exists() or not ppd.is_dir():
        raise ValueError(
            f"Invalied preprocessed directory: {pre_proc_dir}, must exist and be directory"
        )
    return list(map(lambda x: x.name, ppd.iterdir()))


def load_config(arg_dict: Dict, verbose: bool = False) -> Dict:
    """Load config from flat file and merge with command line args"""
    conf_path = Path(arg_dict["config_path"])
    if not conf_path.exists():
        raise ValueError(f"{conf_path} not found")
    config = parse_ini_config(arg_dict["config_path"])

    finished_keys = []

    if verbose:
        print()
    # update existing sections
    for key, value in arg_dict.items():
        if value is None:
            finished_keys.append(key)
            continue
        # print(f'{key}: {value}')
        for section in config.keys():  # config is nested
            if key in config[section]:
                if verbose:
                    print(f"Overide:\n config[{section}][{key}] = {value}")
                config[section][key] = value
                finished_keys.append(key)
                if verbose:
                    print(f"Becomes: \n config[{section}][{key}] = {value}")

    # update unset section
    not_finished = [key for key in arg_dict.keys() if key not in finished_keys]
    admin = {}
    for key in not_finished:
        admin[key] = arg_dict[key]
        if verbose:
            print(f"Added: conf[admin][{key}] = {arg_dict[key]}")
    config["admin"] = admin
    if verbose:
        print()

    # want to add equivalent batch size
    try:
        config["training"]["batch_size_equivalent"] = (
            config["training"]["batch_size"] * config["training"]["update_per_step"]
        )
    except KeyError as e:
        print(f"Warning: issue with config: {e}")
        print("available keys: ")
        for k in config.keys():
            print(k)
        raise e
    config = EarlyStopper.config_precheck(config)

    # return post_process(config, verbose)
    return config


def _convert_type(value: str) -> Any:
    """Convert string values to appropriate types"""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_ini_config(ini_file: str | Path) -> Dict[str, Any]:
    """Parse .ini file for wandb config"""
    config = configparser.ConfigParser()
    config.read(ini_file)

    # Nested structure
    wandb_config = {}
    for section in config.sections():
        wandb_config[section] = {}
        for key, value in config[section].items():
            wandb_config[section][key] = _convert_type(value)

    return wandb_config


def print_config(config_dict):
    """
    Print configuration dictionary in a more readable format.

    Args:
                    config_dict (dict): Dictionary containing configuration sections
                    title: 'Testing' or 'Training'
    """

    for section in config_dict.keys():
        print(f"{section.upper()}:")
        section_data = config_dict[section]

        # Calculate max key length for alignment
        max_key_len = (
            max(len(str(k)) for k in section_data.keys()) if section_data else 0
        )

        for key, value in section_data.items():
            print(f"    {key:<{max_key_len}} : {value}")
        print()


def take_args(
    sup_args: Optional[List[str]] = None,
    return_parser_only: bool = False,
    make_dirs: bool = False,
    prog: Optional[str] = None,
    desc: str = "Train a model",
) -> Optional[tuple | argparse.ArgumentParser]:
    """Retrieve arguments for new training run

    Args:
        sup_args (Optional[List[str]], optional): Supply arguments instead of taking from command line. Defaults to None.
        return_parser_only (bool, optional): Give the parser instead of arguments. Defaults to False.
        make_dirs (bool, optional): Make output and checkpoint dirs. Defaults to False.
        prog (Optional[str], optional): Script name. Defaults to configs.py.
        desc (str, optional): What does the script do? Defaults to "Train a model".

    Raises:
        ValueError: If model or split supplied are not available, or if recovering and save path is invalid.

    Returns:
        Optional[tuple | argparse.ArgumentParser]: Arguments or parser, if successful.
    """
    
    models_available = avail_models()
    splits_available = get_avail_splits()

    parser = argparse.ArgumentParser(description=desc, prog=prog)

    # admin
    parser.add_argument("exp_no", type=int, help="Experiment number (e.g. 10)")
    parser.add_argument(
        "model_name",
        type=str,
        help=f"Model name from one of the implemented models: {models_available}",
    )
    parser.add_argument(
        "split",
        type=str,
        help=f"The class split, one of:  {', '.join(splits_available)}",
    )
    parser.add_argument(
        "-r", "--recover", action="store_true", help="Recover from last checkpoint"
    )
    parser.add_argument(
        "-ri",
        "--run_id",
        type=str,
        default=None,
        help="The run id to use (especially when also usign recover)",
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        default=PROJECT,
        help=f"wandb project name, if not {PROJECT}",
    )
    parser.add_argument(
        "-et", "--entity", type=str, default=ENTITY, help=f"Entity if not {ENTITY}"
    )
    parser.add_argument(
        "-ee",
        "--enum_exp",
        action="store_true",
        help="enumerate the experiment dir num (for output)",
    )
    parser.add_argument(
        "-ec",
        "--enum_chck",
        action="store_true",
        help="enumerate the checkpoint dir num (for output)",
    )
    parser.add_argument(
        "-t", "--tags", nargs="+", type=str, help="Additional wandb tags"
    )
    parser.add_argument("-c", "--config_path", help="path to config .ini file")

    if return_parser_only:
        return parser

    if sup_args:
        args = parser.parse_args(sup_args)
    else:
        args = parser.parse_args()

    if args.split not in splits_available:
        raise ValueError(
            f"Sorry {args.split} not processed yet.\n\
			Currently available: {splits_available}"
        )
    if args.model_name not in models_available:
        raise ValueError(
            f"Sorry {args.model_name} not implemented yet.\n\
			Currently available: {models_available}"
        )

    exp_no = str(int(args.exp_no)).zfill(3)

    if args.project is None:
        args.project = f"{PROJECT_BASE}-{args.split[3:]}"

    args.exp_no = exp_no
    args.root = WLASL_ROOT + "/" + RAW_DIR
    args.labels = f"{LABELS_PATH}/{args.split}"
    output = Path(f"{RUNS_PATH}/{args.split}/{args.model_name}_exp{exp_no}")

    # recovering
    if not args.recover and output.exists():  # fresh run
        if not args.enum_exp:
            ans = ask_nicely(
                message=f"{output} already exists, do you want to cancel, proceed, or enumerate (c, p, e): ",
                requirment=lambda x: x.lower() in ["c", "p", "e"],
                error=f"Must choose one of: {['c', 'p', 'e']}",
            )
        else:
            ans = "e"

        if ans.lower() == "e":
            output = enum_dir(output, make_dirs)

        if ans.lower() == "c":
            return
    
    # saving
    save_path = output / "checkpoints"
    # if not args.recover and args.enum_chck:
    if not args.recover and save_path.exists():
        if not args.enum_chck:
            ans = ask_nicely(
                message=f"{save_path} already exists, do you want to cancel, overwrite, or enumerate (c, o, e): ",
                requirment=lambda x: x.lower() in ["c", "o", "e"],
                error=f"Must choose one of: {['c', 'o', 'e']}",
            )
        else:
            ans = "e"

        if ans.lower() == "e":
            args.save_path = enum_dir(save_path, make_dirs)

        if ans.lower() == "c":
            return
    elif args.recover:
        if not save_path.exists() or not save_path.is_dir():
            raise ValueError(
                f"Cannot recover, {save_path} does not exist or is not a directory"
            )
        if len([f for f in save_path.iterdir() if f.is_file()]) == 0:
            raise ValueError(f"Cannot recover, {save_path} is empty")
    else:
        args.save_path = save_path

    args.save_path = str(save_path)

    # Set config path
    if args.config_path is None:
        args.config_path = f"./configfiles/{args.split}/{args.model_name}_{exp_no}.ini"

    # Load config
    arg_dict = vars(args)
    clean_dict = {}

    redundants = ["project", "tags", "enum_exp", "enum_chck", "entity"]

    for key, value in arg_dict.items():
        if key in redundants:
            continue
        if value is not None:
            clean_dict[key] = value

    # NOTE: these tags are redundant
    tags = [
        args.split,
        args.model_name,
        f"exp-{exp_no}",
    ]
    if args.recover:
        tags.append("Recovered")
    if args.tags is not None:
        tags.extend(args.tags)

    return clean_dict, tags, args.project, args.entity


# def print_wandb_config(config):


def main():
    try:
        # maybe_args = take_args(available_splits,available_model,
        #                  sup_args=['-x', '5', '-m', 'S3D', '-sp', 'asl100'])
        # maybe_args = take_args(sup_args=["-h"])
        maybe_args = take_args()
    except Exception as e:
        maybe_args = None
        print(f"Parsing failed with error: {e}")

    if isinstance(maybe_args, tuple):
        arg_dict, tags, project, entity = maybe_args
    else:
        return

    config = load_config(arg_dict, verbose=True)
    print_config(config)
    print("\n" * 2)

    # print_dict(config)


if __name__ == "__main__":
    main()
