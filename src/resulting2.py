from __future__ import annotations

import json
import logging
import sys
from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

# locals
from src.que.core import ExpQue, GenExp, Que, QueLocation
from src.run_types import SRC_ROOT, CleverDict, CompExpInfo, RunRes
from src.utils import load_module_from_path

# from results.saicair.saicair import additional_modifications


RESULTS_DIR = SRC_ROOT / "results"
FILTERS_NAME = "filters.py"
OUTPUT_SUFFIX = ".json"
# verbose logger for debugging - configured directly (not via basicConfig)
# so it isn't swallowed by ipykernel/other libraries already holding the
# root logger's handlers.
basic_logger = logging.getLogger("resulting")
basic_logger.setLevel(logging.DEBUG)
basic_logger.propagate = False
if not basic_logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    basic_logger.addHandler(_handler)


def load_filters_module(filters_path: Path) -> ModuleType:
    """Load an arbitrary filters.py file by path as a standalone module."""
    return load_module_from_path(filters_path, module_prefix="_filters")

def get_filters_drop_keys(filters_path: Path) -> tuple[dict, list[list[str]]]:
    """Load filters_path and pull out filters and keys to drop

    Args:
        filters_path (Path): Path to filters.py file.  

    Raises:
        AttributeError: If file does not contain 'filters' and 'drop_keys' attributes

    Returns:
        tuple[dict, list[str]]: filters, drop_keys
    """
    module = load_filters_module(filters_path)

    missing = [
        name
        for name in ("filters", "drop_keys")
        if not hasattr(module, name)
    ]
    if missing:
        raise AttributeError(
            f"{filters_path} is missing required attribute(s): {missing}"
        )

    return module.filters, module.drop_keys

def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=4))


def output_results(obj: Any, out_path: Path) -> None:
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=4)


def _unpack_filters(filters: dict) -> tuple[list[list[str]], list[Callable[[Any], bool]]]:
    """Recursively flatten a nested dict. The ouput is a list of key sets which directly index a value, and a 
    corresponding list of citeria to match the value against. Useful for converting compact
    nested dict specifications to the form expecte by the Que.find_runs method. 
    

    Args:
        filters (dict): Nested dictionary.

    Raises:
        TypeError: If filters does not have str keys, or Callable[[Any], bool] leaf values.

    Returns:
        tuple[list[list[str]], list[Callable[[Any], bool]]]: filter_key_sets, criterions
    """
    filter_key_sets: list[list[str]] = []
    criterions: list[Callable[[Any], bool]] = []
    
    for key, value in filters.items():
        key_set = [key]
        if isinstance(value, Callable):
            criterions.append(value)
            filter_key_sets.append(key_set) 
            continue
            
        elif isinstance(value, dict):
            sub_key_sets, crits =  _unpack_filters(value)
            for sublist in sub_key_sets:
                filter_key_sets.append(key_set + sublist)
            
            criterions.extend(crits)
        else:
            raise TypeError(f'value should be dict or Callable, instead got: {type(value)}')
        

    return filter_key_sets, criterions
        
def _drop_keys(d: dict, keys: list[Any]) -> dict:
    """Drop a nested value from a dict and return the dict. 

    Args:
        d (dict): A dictionary to modify in place
        keys (list[Any]): List of keys in order to index. 

    Returns:
        dict: The reference the original dictionary
    """
    if len(keys) == 0:
        return d
    
    parent = d
    for key in keys[:-1]:
        parent = parent[key]

    parent.pop(keys[-1])
    
    return d
    
    

def load_config_and_find_runs(
    specification_path: Path,
    que_location: QueLocation = "old_runs",
    logger: logging.Logger = basic_logger,
    logging_level=logging.INFO,
) -> list[dict]:
    
    filters, drop_key_sets = get_filters_drop_keys(specification_path)
    filter_key_sets, criterions = _unpack_filters(filters)

    # find_que_runs(args.out_path)
    logger.setLevel(logging_level)  # or WARNING, INFO, ERROR, CRITICAL
    logger.debug(json.dumps(str(filters), indent=4))
    logger.debug(json.dumps(str(drop_key_sets), indent=4))

    q = Que(logger)
    runs = q.list_runs(loc=que_location)

    found_runs = Que.list_manipulation(runs, filter_keys=filter_key_sets, criterions=criterions)
    logger.info(f"Found {len(found_runs)}/{len(runs)} runs matching the spec")

    return [_drop_keys(run.model_dump(), drop_keys) for
    run, drop_keys in zip(found_runs, drop_key_sets)]
        
        


def main():
    parser = ArgumentParser(
        description="Run find runs using filters loaded from an arbitrary "
        "filters.py file."
    )
    parser.add_argument("filters", type=Path, help="Path to filters.py")
    parser.add_argument(
        "--out_path",
        "-o",
        help="Path to output file, if different from config path with .json suffix",
        type=Path,
    )
    parser.add_argument("--debug", "-d", help="Enable debug mode", action="store_true")
    args = parser.parse_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO
    basic_logger.setLevel(logging_level)

    out_path = args.out_path if args.out_path else args.config.with_suffix(OUTPUT_SUFFIX)
    
    output = load_config_and_find_runs(
        specification_path=args.filters,
        logger=basic_logger,
        logging_level=logging_level,
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
