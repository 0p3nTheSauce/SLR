#!/usr/bin/env python3
"""Convert a wandb run's config.yaml export into a Que-style TOML config.

Background
----------
A wandb sweep run's `config.yaml` (once your `build_base_config()` /
`SWEEP_KEY_MAP` machinery has expanded the flat sweep params into a full
config) contains two overlapping things:

  1. The raw flat params wandb actually swept over, e.g.
     `batch_size`, `drop_p`, `backbone_init_lr`, `eps`, `max_epoch`, ...
  2. The fully composed nested sections those params were folded into, e.g.
     `admin`, `data`, `optimizer`, `scheduler`, `stopping`, `training`,
     `model_params`.

(1) is redundant - every value in it already lives somewhere inside (2).
So this script:

  - loads the yaml
  - unwraps wandb's `{value: ..., desc: ...}` wrapper on every top-level key
  - skips wandb bookkeeping keys (`_wandb`, `wandb_version`, ...)
  - keeps only the dict-shaped (composed) sections, dropping the flat
    scalar duplicates
  - recursively strips `None` values (TOML has no null)
  - applies any `--set section.key=value` overrides (useful for
    `admin.config_path` / `admin.save_path`, which are placeholders like
    `<in-memory:sweep>` for a run that never had a real config file)
  - writes the result out as TOML

Deriving admin.config_path automatically
-----------------------------------------
`admin.save_path` looks like:

    .../runs/{split}/{model}/sweep_{sweep_id}/{run_id}/checkpoints

With `--derive-config-path`, the script mirrors that structure under your
project's `CONFIGS_PATH` (imported from `src.configs`), dropping the
trailing `checkpoints` segment and turning the run id into a filename:

    CONFIGS_PATH/{split}/{model}/sweep_{sweep_id}/{run_id}.toml

This also sets `admin.config_path` in the written config, and - unless -o
is given explicitly - is used as the output path too.

Usage
-----
    python wandb_yaml_to_toml.py run_config.yaml -o run_config.toml
    python wandb_yaml_to_toml.py run_config.yaml --derive-config-path
    python wandb_yaml_to_toml.py run_config.yaml \\
        --set admin.config_path=/home/luke/Code/SLR/src/runs/asl100/S3D/sweep_xcj43xka/1wtvl0ke/config.toml \\
        --set admin.save_path=/home/luke/Code/SLR/src/runs/asl100/S3D/sweep_xcj43xka/1wtvl0ke/checkpoints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import tomli_w
import yaml

# wandb bookkeeping keys that never belong in the run config
_SKIP_KEYS = {"wandb_version", "_wandb"}


def _strip_none(obj: Any) -> Any:
    """Recursively drop None values (TOML has no concept of null)."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(item) for item in obj]
    return obj


def _unwrap(raw: dict) -> dict:
    """Strip wandb's per-key `{value: ..., desc: ...}` wrapper.

    Handles both wrapped (`key: {value: ...}`) and already-plain
    (`key: ...`) entries, and skips wandb's own bookkeeping keys.
    """
    unwrapped = {}
    for key, entry in raw.items():
        if key in _SKIP_KEYS or key.startswith("_"):
            continue
        if isinstance(entry, dict) and "value" in entry:
            unwrapped[key] = entry["value"]
        else:
            unwrapped[key] = entry
    return unwrapped


def _keep_sections(unwrapped: dict) -> dict:
    """Keep only dict-shaped (composed) sections; drop flat scalar params.

    The flat sweep params (batch_size, drop_p, eps, ...) are scalars/lists
    at the top level and are already duplicated inside the nested sections,
    so anything that isn't itself a dict gets dropped here.
    """
    return {k: v for k, v in unwrapped.items() if isinstance(v, dict)}


def _derive_config_path(save_path: str, configs_path: Path) -> Path:
    """Mirror a run's save_path structure under CONFIGS_PATH.

    save_path:    .../runs/{split}/{model}/sweep_{sweep_id}/{run_id}/checkpoints
    config_path:  CONFIGS_PATH/{split}/{model}/sweep_{sweep_id}/{run_id}.toml

    Anchors on the last literal "runs" path component if present (robust to
    save_path being an absolute path with an arbitrary prefix); otherwise
    falls back to just taking the last 4 path components.
    """
    parts = list(Path(save_path).parts)

    if parts and parts[-1] == "checkpoints":
        parts = parts[:-1]

    if "runs" in parts:
        anchor = len(parts) - 1 - parts[::-1].index("runs")
        relative = parts[anchor + 1:]
    else:
        relative = parts[-4:]

    if len(relative) < 2:
        raise ValueError(
            f"Could not derive a config path from save_path: {save_path!r}"
        )

    *dirs, run_id = relative
    return configs_path.joinpath(*dirs, f"{run_id}.toml")


def _apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply `section.key=value` overrides, converting value to
    int/float/bool where possible, else leaving it as a string.
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"--set expects section.key=value, got: {override!r}")
        path, _, value_str = override.partition("=")
        parts = path.split(".")

        value: Any = value_str
        if value_str.lower() in ("true", "false"):
            value = value_str.lower() == "true"
        else:
            for caster in (int, float):
                try:
                    value = caster(value_str)
                    break
                except ValueError:
                    continue

        node = config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    return config


def convert(
    yaml_path: Path,
    overrides: list[str] | None = None,
    derive_config_path: bool = False,
) -> tuple[str, Optional[Path]]:
    """Load a wandb config.yaml and return (toml_str, derived_config_path).

    derived_config_path is the CONFIGS_PATH-relative path computed from
    admin.save_path when derive_config_path=True, else None.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    unwrapped = _unwrap(raw)
    sections = _keep_sections(unwrapped)
    sections = _strip_none(sections)

    derived_path: Optional[Path] = None
    if derive_config_path:
        from src.configs import CONFIGS_PATH

        save_path = sections.get("admin", {}).get("save_path")
        if not save_path:
            raise ValueError(
                "--derive-config-path needs admin.save_path in the config, "
                "but none was found."
            )
        derived_path = _derive_config_path(save_path, Path(CONFIGS_PATH))
        sections.setdefault("admin", {})["config_path"] = str(derived_path)

    if overrides:
        sections = _apply_overrides(sections, overrides)

    return tomli_w.dumps(sections), derived_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a wandb run config.yaml into a Que-style TOML config."
    )
    parser.add_argument("yaml_path", type=Path, help="Path to the wandb config.yaml")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Path to write the TOML file. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="section.key=value",
        help="Override a value in the composed config, e.g. "
             "--set admin.config_path=/path/to/config.toml. Repeatable.",
    )
    parser.add_argument(
        "--derive-config-path", action="store_true",
        help="Derive admin.config_path from admin.save_path, mirroring its "
             "{split}/{model}/sweep_{id}/{run_id} structure under "
             "CONFIGS_PATH (imported from src.configs). Also used as the "
             "output path if -o/--output isn't given.",
    )
    args = parser.parse_args()

    if not args.yaml_path.exists():
        print(f"No such file: {args.yaml_path}", file=sys.stderr)
        sys.exit(1)

    toml_str, derived_path = convert(
        args.yaml_path,
        overrides=args.overrides,
        derive_config_path=args.derive_config_path,
    )

    output = args.output or derived_path

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(toml_str)
        print(f"Saved config to: {output}")
    else:
        print(toml_str)


if __name__ == "__main__":
    main()