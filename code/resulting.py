from typing import Dict, Any, List, Optional, Tuple, Callable, cast, TypeAlias

try:
    import tomllib  # type: ignore
except ImportError:
    import tomli as tomllib

from pathlib import Path
from argparse import ArgumentParser
import json
import inspect

# locals
from que.core import Que, _get_basic_logger, ExpQue, GenExp
from run_types import GenInfo, RunRes, CompExpInfo, CleverDict
from results.saicair.saicair import additional_modifications


RESULTS_DIR = Path("results/saicair")
CONFIG_PATH = RESULTS_DIR / "config.toml"
logger = _get_basic_logger()


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


def snap(search: Dict[str, Any], spec: Dict[str, Any], debug: bool = False) -> bool:
    for key, value in spec.items():
        if key not in search:
            if debug:
                print(f"key: {key} not found in search")
                if key == "wobble":
                    print(f"search keys: {search.keys()}")
            return False

        if isinstance(value, Dict):
            # we only want to check the specified values are present,
            # not that the dictionaries match exctly
            if not snap(search[key], value, debug):
                # if debug:
                #     print(f'key: {key} does not match')
                return False

        elif search[key] != value:
            if debug:
                print(f"key: {key} search value: {search[key]} spec value: {value}")
            # print(type(value))
            return False

    return True


def find_runs(runs: ExpQue, spec: GenInfo, debug: bool = False) -> List[GenExp]:
    # return [run for run in runs if snap(run.model_dump(), spec.model_dump())]
    return [run for run in runs if snap(run.model_dump(), spec, debug)]


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=4))


def output_results(res_set: GenInfo, out_path: Path) -> None:
    with open(out_path, "w") as f:
        # json.dump(res_set.model_dump(), f, indent=4)
        json.dump(res_set, f, indent=4)


def build_GenInfo(
    runs: List[CompExpInfo], spec: GenInfo, exclude: List[List[str]] = []
) -> GenInfo:
    run_set = []
    excluded = 0
    for run in runs:
        # run = cast(CompExpInfo, run)

        run_res = CleverDict(RunRes(admin=run.admin, results=run.results).model_dump())
        for key_chain in exclude:
            run_res.pop(key_chain)

        mods = CleverDict(additional_modifications)

        if any([not crit(run_res[key_chain]) for key_chain, crit in mods]):
            excluded += 1
            continue

        run_set.append(run_res.to_dict())
    print(f"Excluded {excluded} runs based on additional modifications")
    # res_set = GenInfo.model_validate({'spec': spec, 'results': run_set})
    res_set = {"spec": spec, "results": run_set}

    return res_set
    # for key_chain in exclude:
    #     res_set.pop(key_chain)

    # print(len(run_set))

    # return GenInfo.model_validate(res_set.to_dict())


def load_config_and_find_runs(
    conf_path: Path, exclude: List[List[str]] = [], debug: bool = False
) -> Optional[GenInfo]:
    gen_info = load_config(str(conf_path), validate=False)
    print_json(gen_info)
    # find_que_runs(args.out_path)
    q = Que(logger)
    runs = q.list_runs(loc="old_runs")

    found_runs = find_runs(runs, gen_info, debug)
    print(f"Found {len(found_runs)}/{len(runs)} runs matching the spec")

    if len(found_runs) == 0:
        print("No runs found matching the spec")
        return

    return build_GenInfo(
        [cast(CompExpInfo, run) for run in found_runs], gen_info, exclude
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
        "--debug", "-g", help="Enable debug mode for snap function", action="store_true"
    )

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
        output = load_config_and_find_runs(config_path, exclude, debug=args.debug)
        assert output is not None, (
            "No runs found matching the spec, cannot output results"
        )
        output_results(output, out_path)
        assert out_path.exists(), f"Output file not found at {out_path}"
        print(f"Output path: {out_path}")


if __name__ == "__main__":
    main()
