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





def _make_list_section(base_list: list[dict], name: str) -> str:
    """Generates a piece of the toml file, a section which is a list of dicts, e.g. the data/train_augs section.
    No newline before or after

    Args:
            base_list (list[dict]): List of named parameters in dict format (i.e. from model_dump)
            name (str): The name of the section, will become [[name]] in the config file

    Returns:
            str: The string content of the section ready for concatenation into the full config file
    """
    section = f"[[{name}]]" 
    for subsec in base_list:
        for k, v in subsec.items():
            if v is None:
                continue
            section += f"\n{k} = {v!r}"
    return section


def _handle_dict(d: Optional[dict], name: str = "") -> str:
    """Given a nested dictionary structure (specific to config setup), returns the str representation
    to be used in the config file. Recurses into nested dicts and lists.

    Args:
            d (dict): Possibly nested dict to be converted into a config section
            name (str, optional): The name of the section. Defaults to ''.

    Returns:
            str: The string content of the section ready for concatenation into the full config file
    """

    if d is None or len(d) == 0:
        return ""

    section = f"[{name}]"

    subsecs = []

    for k, v in d.items():
        if v is None:
            continue
        elif isinstance(v, dict):
            subsecs.append(_handle_dict(v, f"{name}.{k}"))
        elif isinstance(v, list):
            subsecs.append(_handle_list(v, f"{name}.{k}"))
        else:
            section += f"\n{k} = {v!r}"

    return section + "\n" + "\n".join(subsecs)


def _handle_list(l: list, name: str = "") -> str:
    """Specifically handles lists within nested dictionary structure

    Args:
            l (list): List of elements or list of flat dicts (e.g. list of aug configs)
            name (str, optional): The name of the section. Defaults to ''.

    Returns:
            str: The string content of the section ready for concatenation into the full config file
    """
    if len(l) == 0:
        return ""

    item_0 = l[0]

    if isinstance(item_0, dict):
        return _make_list_section(l, name)
    else:
        return f"{name.split('.')[-1]} = {l!r}"


def run_to_config(run: GenExp | dict, comments: list[str] = []) -> str:
    """Turn a general run into its TOML string representation for the config
    file system. Skips ignored sections, strips None values (TOML has no
    null), and appends comment lines.

    Args:
            run (GenExp | dict): Experiment from que
            comments (list[str], optional): Comment lines to append at the end

    Returns:
            str: String representation of the config file content for this
            run, ready to be written to a .toml file
    """
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



def get_old_comments(contents: str) -> list[str]:
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


def get_save_name(
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


def update_config_file(
    run: GenExp | dict,
    default_mode: Literal["overwrite", "duplicate"] = "overwrite",
    dry_run: bool = True,
    retro_support: bool = False,
    output: Optional[Path] = None
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
        old_comments = get_old_comments(old_contents)

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

    config_str = run_to_config(run_info, comments=old_comments + ["updated by script"])

    save_name = get_save_name(conf_path.as_posix(), mode=mode)

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
        with open(output, 'w') as f:
            f.write(config_str)
        print(f"Debug config saved to: {output}")

