import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# locals
from src.que.core import CompExpInfo, Que
from src.que.shell import get_filters_crits_dropkeys, output_filtered_runs


def _safe_get(d: dict | None, k: str) -> Any:
    """Safely get a value from a dictionary, returning None if the dictionary is None.
    """
    return d.get(k) if d is not None else None

def match(obj, target):
    d = obj.model_dump() if isinstance(obj, BaseModel) else obj
    return all(_safe_get(d, k) == v for k, v in target.items())


def fetch_runs(
    filters_path: Path,
    sort_keys: list[list[str]] | None = None,
    reverse: bool = False,
    top_n: int | None = None,
    output_path: str | None = None,
) -> list[CompExpInfo]:
    """
    Use the filters.py file to load filters, then search for runs from the `Que`.

    Args:
        filters_path (Path): Path to the filters.py file
        sort_keys (list[list[str]] | None, optional): Keys to sort the runs by. Defaults to None.
        reverse (bool, optional): Whether to sort in reverse order. Defaults to False.
        top_n (int | None, optional): Number of top runs to return. Defaults to None.
        output_path (str | None, optional): Path to output the filtered runs. Defaults to None.

    Returns:
        list[CompExpInfo]: List of CompExpInfo objects representing the filtered runs.
    """
    file_filter_keys, file_criterions, file_drop_key_sets = get_filters_crits_dropkeys(
        filters_path
    )
    que = Que()
    runs = list(
        Que.list_manipulation(
            que.list_runs("old_runs"),
            sort_keys=sort_keys,
            reverse=reverse,
            filter_keys=file_filter_keys,
            criterions=file_criterions,
        )
    )

    # retrieve top n if specified
    if top_n is not None:
        runs = runs[:top_n]

    if output_path:
        output_filtered_runs(
            runs=runs,
            output_path=output_path,
            file_drop_key_sets=file_drop_key_sets,
        )

    return [CompExpInfo.model_validate(run) for run in runs]


def load_runs(runs_path: Path) -> list[CompExpInfo]:
    """Load a presaved set of finished runs.

    Args:
        runs_path (Path): Path to the JSON file containing the runs.

    Returns:
        list[CompExpInfo]: List of CompExpInfo objects representing the runs.
    """
    with open(runs_path, "r") as f:
        return [CompExpInfo.model_validate(r) for r in json.load(f)]
