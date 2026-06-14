from typing import Optional, Literal
from pathlib import Path
import tomli_w

# from que.shell import QueShell
from src.que.core import (
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
    comments: list[str] = [],
    ignore_sections: list[str] = ["admin", "wandb", "results",],
) -> str:
    """Turn a general run into its TOML string representation for the config
    file system. Skips ignored sections, strips None values (TOML has no
    null), and appends comment lines.

    Args:
        run (GenExp | dict): run (GenExp | dict): Experiment from que
        comments (list[str], optional): Comment lines to append at the end. Defaults to [].
        ignore_sections (list[str], optional): Sections to ignore. Defaults to ["admin", "wandb", "results"].

    Returns:
        str: String representation of the config file content for this run
    """

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
    output: Optional[Path] = None,
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
        with open(conf_path, "r") as f:
            old_contents = f.read()
        old_comments = _get_old_comments(old_contents)

        try:
            _ = load_config(
                AdminInfo.model_validate(run_info["admin"]), retro_support=retro_support
            )
            print(f"Valid config found at {conf_path}, skipping overwrite mode.")
            return
        except Exception as e:
            print(f"Validation failed for existing config: {e}")

            mode: Literal["overwrite", "duplicate"] = default_mode
            print(f"Proceeding with {mode} mode.")

    else:
        old_comments = []
        mode: Literal["overwrite", "duplicate"] = "overwrite"

    config_str = _run_to_config(run_info, comments=old_comments + ["updated by script"])

    save_name = _get_save_name(conf_path.as_posix(), mode=mode)

    if dry_run:
        print(config_str)
        print(f"Would save to: {save_name}")
    else:
        parent_dir = Path(save_name).parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        with open(save_name, "w") as f:
            f.write(config_str)
        print(f"Saved config to: {save_name}")

    if output:
        with open(output, "w") as f:
            f.write(config_str)
        print(f"Debug config saved to: {output}")
