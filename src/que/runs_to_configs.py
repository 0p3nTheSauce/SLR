import argparse
import json
from pathlib import Path
from typing import Literal

import pydantic
import tomli_w

# from que.shell import QueShell
from src.que.core import (
    CUR_RUN,
    FAIL_RUNS,
    OLD_RUNS,
    RUN_PATH,
    TO_RUN,
    GenExp,
)


def _strip_none(obj):
    """Recursively remove None values from a nested dict/list, since TOML
    has no concept of null and tomli_w will error on None values.

    Args:
            obj: A (possibly nested) dict, list, or scalar value.

    Returns:
            The same structure with all None values and the keys pointing
            to them removed.
    """
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [_strip_none(item) for item in obj]
    else:
        return obj


def _get_old_comments(contents: str) -> list[str]:
    """Given the contents of a config file as a string, extracts the comment lines (starting with #) and returns them as a list of strings.

    Args:
            contents (str): The full string content of a config file
    """
    lines = contents.splitlines()
    comments = [
        line.replace("#", "").replace(";", "").strip()
        for line in lines
        if line.strip().startswith("#") or line.strip().startswith(";")
    ]
    return comments


def _get_save_name(
    save_path: str, mode: Literal["overwrite", "duplicate"] = "duplicate"
) -> str:
    """Generate a save name based on the save path and mode.

    Args:
        save_path (str): The path where the file will be saved.
        mode (Literal['overwrite', 'duplicate'], optional): The mode for saving. Defaults to 'duplicate'.

    Returns:
        str: The generated save name.
    """
    path = Path(save_path)

    if mode == "duplicate":
        return str(path.with_stem(path.stem + "_updated").with_suffix(".toml"))
    else:
        return str(path.with_suffix(".toml"))


def _run_to_config(
    run: GenExp | dict,
    comments: list[str] | None = None,
    ignore_sections: list[str] | None = None,
) -> str:
    """Turn a general run into its TOML string representation for the config
    file system. Skips ignored sections, strips None values (TOML has no
    null), and appends comment lines.

    Args:
        run (GenExp | dict): Experiment from que
        comments (list[str] | None, optional): Comment lines to append at the end. Defaults to None.
        ignore_sections (list[str] | None, optional): Sections to ignore. Defaults to None (uses ["admin", "wandb", "results"]).

    Returns:
        str: String representation of the config file content for this run
    """

    if comments is None:
        comments = []
    if ignore_sections is None:
        ignore_sections = ["admin", "wandb", "results"]
    if isinstance(run, GenExp):
        run_info = run.model_dump()
    else:
        run_info = run

    filtered = {
        section_name: _strip_none(section_content)
        for section_name, section_content in run_info.items()
        if section_name not in ignore_sections and section_content
    }

    config_str = tomli_w.dumps(filtered)

    if len(comments) > 0:
        config_str += "\n"

    for comment in comments:
        config_str += f"\n# {comment}"

    return config_str


def update_config_file(
    run: GenExp | dict,
    default_mode: Literal["overwrite", "duplicate"] = "overwrite",
    dry_run: bool = True,
    retro_support: bool = False,
    output: Path | None = None,
):
    from src.configs import load_config
    from src.run_types import AdminInfo

    if isinstance(run, GenExp):
        run_info = run.model_dump()
    else:
        run_info = run

    conf_path = Path(run_info["admin"]["config_path"])
    print(f"Updating config file: {conf_path}")

    if conf_path.exists():
        # get old comments
        with open(conf_path, "r") as f:
            old_contents = f.read()
        old_comments = _get_old_comments(old_contents)

        # skip file if it is already valid
        try:
            _ = load_config(
                AdminInfo.model_validate(run_info["admin"]), retro_support=retro_support
            )
            print(f"Valid config found at {conf_path}, skipping overwrite mode.")
            return
        except (FileNotFoundError, pydantic.ValidationError, ValueError) as e:
            print(f"Validation failed for existing config: {e}")

            mode: Literal["overwrite", "duplicate"] = default_mode
            print(f"Proceeding with {mode} mode.")

    else:
        old_comments = []
        mode: Literal["overwrite", "duplicate"] = "overwrite"

    # generate new config file
    config_str = _run_to_config(run_info, comments=old_comments + ["updated by script"])
    # get save path
    save_name = _get_save_name(conf_path.as_posix(), mode=mode)

    if dry_run:
        print(config_str)
        print(f"Would save to: {save_name}")
    else:
        parent_dir = Path(save_name).parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        # write file
        with open(save_name, "w") as f:
            f.write(config_str)
        print(f"Saved config to: {save_name}")

    if output:
        with open(output, "w") as f:
            f.write(config_str)
        print(f"Debug config saved to: {output}")


def update_all_files(
    default_mode: Literal["overwrite", "duplicate"] = "overwrite",
    dry_run: bool = True,
    retro_support: bool = False,
    output: Path | None = None,
):

    KEYS = [TO_RUN, CUR_RUN, OLD_RUNS, FAIL_RUNS]
    with open(RUN_PATH, "r") as f:
        all_runs = json.load(f)

    flat_all_runs = []
    for key in KEYS:
        flat_all_runs.extend(all_runs[key])

    for run_info in flat_all_runs:
        update_config_file(
            run_info, default_mode, dry_run, retro_support, output=output
        )

    print(len(flat_all_runs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility script to update TOML configuration files from experiment run data."
    )

    # Global arguments
    parser.add_argument(
        "--mode",
        choices=["overwrite", "duplicate"],
        default="overwrite",
        help="Default saving mode if file exists (default: %(default)s).",
    )
    parser.add_argument(
        "--run",
        action="store_false",
        dest="dry_run",
        help="Actually write changes to files. If not specified, defaults to a dry run.",
    )
    parser.add_argument(
        "--retro-support",
        action="store_true",
        help="Enable legacy/retro support configuration parsing.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write a debug copy of the generated config.",
    )

    # Sub-commands (making dest="command" optional by not enforcing required=True)
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Sub-command: single
    single_parser = subparsers.add_parser(
        "single", help="Update a single configuration file."
    )
    single_parser.add_argument(
        "--run-data",
        type=str,
        default=str(RUN_PATH),
        help="JSON string or path to a JSON file. Defaults to RUN_PATH (default: %(default)s).",
    )
    single_parser.add_argument(
        "--key",
        type=str,
        default=OLD_RUNS,
        help="The dictionary key to extract from the run database (default: %(default)s).",
    )
    single_parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="The list index to extract from the specified key section (default: %(default)s).",
    )

    # Sub-command: all
    all_parser = subparsers.add_parser(
        "all", help="Update all configuration files found in the runs database."
    )

    args = parser.parse_args()

    # Fallback: If the user didn't specify 'single' or 'all', default to 'all'
    chosen_command = args.command if args.command else "all"

    if chosen_command == "single":
        p = Path(args.run_data)
        assert p.exists()

        with open(args.run_data, "r") as f:
            all_runs = json.load(f)

        update_config_file(
            run=all_runs[args.key][args.index],
            default_mode=args.mode,
            dry_run=args.dry_run,
            retro_support=args.retro_support,
            output=args.output,
        )

    elif chosen_command == "all":
        update_all_files(
            default_mode=args.mode,
            dry_run=args.dry_run,
            retro_support=args.retro_support,
            output=args.output,
        )
