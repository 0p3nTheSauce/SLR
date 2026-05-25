from typing import Dict, Any, List, Optional, cast, Union, Callable
import copy

try:
    import tomllib  # type: ignore
except ImportError:
    import tomli as tomllib

from pathlib import Path
from argparse import ArgumentParser
import json
import logging

# locals
from src.que.core import Que, ExpQue, GenExp
from src.run_types import GenInfo, RunRes, CompExpInfo, CleverDict
# from results.saicair.saicair import additional_modifications


RESULTS_DIR = Path("results/")
CONFIG_PATH = RESULTS_DIR / "config.toml"
basic_logger = logging.getLogger(__name__)
basic_logger.addHandler(logging.StreamHandler())  # add this

def drop_max_wobble(temporal_aug: list) -> list:
    for item in temporal_aug:
        item.pop('max_wobble', None)
        item.pop('wobble', None)
    return temporal_aug

ignore_keys = {
    'data': {
        'train_augs': {'temporal_aug': drop_max_wobble},
        'test_augs':  {'temporal_aug': drop_max_wobble},
    }
}


def load_config(config_path: str, validate: bool = True) -> GenInfo:
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
    conf_path = Path(config_path)
    if not conf_path.exists():
        raise FileNotFoundError(f"{conf_path} not found")

    with open(conf_path, "rb") as f:
        raw = tomllib.load(f)
    return raw

    # if validate:
    #     return GenInfo.model_validate(raw)
    # else:
    #     return GenInfo.model_validate(raw, strict=False)
    
    

def snap(search: Any, spec: Any, logger: logging.Logger) -> bool:
    
    if isinstance(spec, dict):
        if not isinstance(search, dict):
            logger.debug(f"type mismatch: search is {type(search)}, spec is dict")
            return False
        
        for key, value in spec.items():
            if key not in search:
                logger.debug(f"key '{key}' not found in search")
                return False
            if not snap(search[key], value, logger):
                return False

    elif isinstance(spec, list):
        if not isinstance(search, list):
            logger.debug(f"type mismatch: search is {type(search)}, spec is list")
            return False
        
        # every item in spec must match at least one item in search
        for spec_item in spec:
            if not any(snap(search_item, spec_item, logger) for search_item in search):
                logger.debug(f"no match found in search for spec item: {spec_item}")
                return False

    else:
        # leaf value — must match exactly
        if search != spec:
            logger.debug(f"value mismatch: search={search}, spec={spec}")
            return False

    return True



def modify(
    search: Dict, spec: Dict, logger: logging.Logger, 
) -> Dict[str, Any]:
    
    for key, value in spec.items():
        
        if isinstance(value, dict):
            
            if key in search:
                search[key] = modify(search[key], value, logger)
            else:
                logger.info(f'skipping key: {key} not in search: {search.keys()}')
                continue
            
        elif isinstance(search, dict) and key in search:
            nv = value(search[key]) 
            # logger.debug(f'Mapping search[key] : {search[key]} to {nv}')
            search[key] = nv
        else:
            logger.warning(f'unexpected mismatch between types: search: {type(search)} criterion: {type(value)}')


    return search
    
    


def find_runs(
    runs: ExpQue, spec: GenInfo, logger: logging.Logger, ignore: Dict[str, Any] = {}
) -> List[GenExp]:
    # return [run for run in runs if snap(run.model_dump(), spec.model_dump())]

    return [
        run
        for run in runs
        if snap(run.model_dump(), modify(spec, ignore, logger), logger)
    ]


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=4))


def output_results(res_set: GenInfo, out_path: Path) -> None:
    with open(out_path, "w") as f:
        # json.dump(res_set.model_dump(), f, indent=4)
        json.dump(res_set, f, indent=4)


def build_GenInfo(
    runs: List[CompExpInfo],
    spec: GenInfo,
    logger: logging.Logger,
    exclude: List[List[str]] = [],
    extra_mods: Dict[str, Any] = {},
) -> GenInfo:
    run_set = []
    excluded = 0
    for run in runs:
        # run = cast(CompExpInfo, run)

        run_res = CleverDict(RunRes(admin=run.admin, results=run.results).model_dump())
        for key_chain in exclude:
            run_res.pop(key_chain)

        mods = CleverDict(extra_mods)

        if any([not crit(run_res[key_chain]) for key_chain, crit in mods]):
            excluded += 1
            continue

        run_set.append(run_res.to_dict())
    logger.info(f"Excluded {excluded} runs based on additional modifications")
    res_set = {"spec": spec, "results": run_set}

    return res_set


def load_config_and_find_runs(
    conf_path: Path,
    exclude: List[List[str]] = [],
    extra_mods: Dict[str, Any] = {},
    ignore: Dict[str, Any] = {},
    logger: logging.Logger = basic_logger,
    logging_level=logging.INFO,
) -> Optional[GenInfo]:
    gen_info = load_config(str(conf_path), validate=False)
    
    # find_que_runs(args.out_path)
    logger.setLevel(logging_level)  # or WARNING, INFO, ERROR, CRITICAL
    logger.debug(json.dumps(gen_info, indent=4))
    
    q = Que(logger)
    runs = q.list_runs(loc="old_runs")

    found_runs = find_runs(runs, gen_info, logger, ignore)
    logger.info(f"Found {len(found_runs)}/{len(runs)} runs matching the spec")

    if len(found_runs) == 0:
        logger.warning("No runs found matching the spec")
        return

    return build_GenInfo(
        [cast(CompExpInfo, run) for run in found_runs],
        gen_info,
        logger,
        exclude,
        extra_mods,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--result_dir",
        "-d",
        help=f"Path to result directory, default is {RESULTS_DIR}",
        type=str,
        default=RESULTS_DIR,
    )
    parser.add_argument(
        "--config_path",
        "-c",
        help=f"Path to config file, if different from {CONFIG_PATH}",
        type=str,
        default=CONFIG_PATH,
    )
    parser.add_argument(
        "--exclude_keys",
        "-e",
        help="Exclude a list of keys from the output",
        action="append",
        nargs="+",
        metavar="KEY",
    )
    parser.add_argument(
        "--out_path",
        "-o",
        help="Path to output file, if different from config path with .json suffix",
        type=str,
        default=None,
    )
    # parser.add_argument("--num_frames", "-f", help="Number of frames to filter by, default is 16", type=int, default=frames)
    parser.add_argument(
        "--extra_mods",
        "-m",
        action="store_true",
        help=f"Use {RESULTS_DIR}/saicair.py extra modifications",
    )
    parser.add_argument("--debug", "-g", help="Enable debug mode", action="store_true")

    args = parser.parse_args()

    resdir = Path(args.result_dir)
    if not resdir.exists():
        raise FileNotFoundError(f"{resdir} not found")

    output = None
    out_path = None

    if args.out_path is not None:
        out_path = Path(args.out_path)

    if args.config_path is not None:
        config_path = Path(args.config_path)
        if out_path is None:
            out_path = config_path.with_suffix(".json")
        exclude = args.exclude_keys if args.exclude_keys is not None else []
        print(f"Excluding keys: {exclude}")
        # extra_mods = additional_modifications if args.extra_mods else {}
        extra_mods = {}
        logging_level = logging.DEBUG if args.debug else logging.INFO
        basic_logger.setLevel(logging_level)
        # ignore_keys = {}
        output = load_config_and_find_runs(
            config_path,
            exclude,
            extra_mods=extra_mods,
            logger=basic_logger,
            logging_level=logging_level,
            ignore=ignore_keys
        )

        basic_logger.info("logger working")
        basic_logger.debug("debug mode")

        if output is None:
            print("No runs found matching the spec, cannot output results")
            return
        output_results(output, out_path)
        assert out_path.exists(), f"Output file not found at {out_path}"
        print(f"Output path: {out_path}")


if __name__ == "__main__":
    main()
