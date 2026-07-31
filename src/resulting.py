from __future__ import annotations

import sys
from typing import Any, cast

try:
    import tomllib  # type: ignore
except ImportError:
    import tomli as tomllib
import importlib.util
import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from types import ModuleType

# locals
from src.que.core import ExpQue, GenExp, Que
from src.run_types import SRC_ROOT, CleverDict, CompExpInfo, GenInfo, RunRes

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
    """Load an arbitrary filters.py file by path as a standalone module.

    This works regardless of where filters_path lives on disk - it doesn't
    need to be on sys.path or part of any package.
    """
    if not filters_path.exists():
        raise FileNotFoundError(f"No such filters file: {filters_path}")

    module_name = f"_filters_{filters_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, filters_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {filters_path}")

    module = importlib.util.module_from_spec(spec)
    # Registering in sys.modules first lets the module's own top-level code
    # (e.g. dataclasses, or anything doing `import module_name`) resolve
    # correctly, and avoids it being garbage-collected mid-exec.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_filters(filters_path: Path) -> tuple[dict, list]:
    """Load filters_path and pull out the two attributes load_find needs.

    Raises AttributeError with a clear message if either is missing, rather
    than failing later inside load_config_and_find_runs.
    """
    module = load_filters_module(filters_path)

    missing = [
        name
        for name in ("additional_modifications", "exclude_keys")
        if not hasattr(module, name)
    ]
    if missing:
        raise AttributeError(
            f"{filters_path} is missing required attribute(s): {missing}"
        )

    return module.additional_modifications, module.exclude_keys


def load_config(config_path: str) -> dict[str, Any]:
    """load config file as dictionary"""
    conf_path = Path(config_path)
    if not conf_path.exists():
        raise FileNotFoundError(f"{conf_path} not found")

    with open(conf_path, "rb") as f:
        raw = tomllib.load(f)
    return raw


def snap0(search: Any, spec: Any, logger: logging.Logger) -> bool:

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
        
        spec = sorted(spec, key=lambda x: str(x))  # sort spec for consistent comparison
        search = sorted(search, key=lambda x: str(x))  # sort search for consistent comparison

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

        def _key(item: Any) -> str:
            # Group by 'type' when present (aug configs, etc.) so we only
            # compare like-with-like. Falls back to a stable serialization
            # for items without a 'type' field (or non-dict items).
            if isinstance(item, dict) and "type" in item:
                return str(item["type"])
            return json.dumps(item, sort_keys=True, default=str)

        search_by_key: dict[str, list[Any]] = {}
        for item in search:
            search_by_key.setdefault(_key(item), []).append(item)

        for spec_item in sorted(spec, key=_key):
            key = _key(spec_item)
            candidates = search_by_key.get(key, [])
            if not candidates:
                logger.debug(f"no search item with key '{key}' for spec item: {spec_item}")
                return False
            if not any(snap(candidate, spec_item, logger) for candidate in candidates):
                logger.debug(f"no candidate with key '{key}' matched spec item: {spec_item}")
                return False

    else:
        # leaf value — must match exactly
        if search != spec:
            logger.debug(f"value mismatch: search={search}, spec={spec}")
            return False

    return True
    
    
    

def modify(
    search: dict,
    spec: dict,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Recursively apply modifications from spec to search, where spec values are callables."""
    for key, value in spec.items():
        if isinstance(value, dict):
            if key in search:
                search[key] = modify(search[key], value, logger)
            else:
                logger.info(f"skipping key: {key} not in search: {search.keys()}")
                continue

        elif isinstance(search, dict) and key in search:
            nv = value(search[key])
            # logger.debug(f'Mapping search[key] : {search[key]} to {nv}')
            search[key] = nv
        else:
            logger.warning(
                f"unexpected mismatch between types: search: {type(search)} criterion: {type(value)}"
            )

    return search


def find_runs(
    runs: ExpQue, spec: GenInfo, logger: logging.Logger, ignore: dict[str, Any] | None = None
) -> list[GenExp]:
    # return [run for run in runs if snap(run.model_dump(), spec.model_dump())]

    if ignore is None:
        ignore = {}
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
    runs: list[CompExpInfo],
    spec: GenInfo,
    logger: logging.Logger,
    exclude: list[list[str]] | None = None,
    extra_mods: dict[str, Any] | None = None,
) -> GenInfo:
    if extra_mods is None:
        extra_mods = {}
    if exclude is None:
        exclude = []
    run_set = []
    excluded = 0
    for run in runs:
        # run = cast(CompExpInfo, run)

        run_res = CleverDict(RunRes(admin=run.admin, results=run.results, wandb=run.wandb).model_dump())
        for key_chain in exclude:
            run_res.pop(key_chain)

        mods = CleverDict(extra_mods)

        if any(not crit(run_res[key_chain]) for key_chain, crit in mods):
            excluded += 1
            continue

        run_set.append(run_res.to_dict())
    logger.info(f"Excluded {excluded} runs based on additional modifications")
    res_set = {"spec": spec, "results": run_set}

    return res_set


def load_config_and_find_runs(
    conf_path: Path,
    exclude: list[list[str]] | None = None,
    extra_mods: dict[str, Any] | None = None,
    ignore: dict[str, Any] | None = None,
    logger: logging.Logger = basic_logger,
    logging_level=logging.INFO,
) -> GenInfo | None:
    if ignore is None:
        ignore = {}
    if extra_mods is None:
        extra_mods = {}
    if exclude is None:
        exclude = []
    gen_info = load_config(str(conf_path))

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
    parser = ArgumentParser(
        description="Run load_find using filters loaded from an arbitrary "
        "filters.py file."
    )
    parser.add_argument("config", type=Path, help="Path to config.toml")
    parser.add_argument("--filters", "-f", type=Path, help="Path to filters.py")
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
    filters_path = args.filters if args.filters else args.config.parent / FILTERS_NAME

    extra_mods, exclude_keys = get_filters(filters_path)
    output = load_config_and_find_runs(
        args.config,
        exclude=exclude_keys,
        extra_mods=extra_mods,
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
