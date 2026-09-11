import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

# locals
from src.que.core import CompExpInfo, Que
from src.que.shell import (
    get_filters_drop_keys,
    output_filtered_runs,
    unpack_filters,
)
from src.run_types import TypeAlias


def _safe_get(d: dict | None, k: str) -> Any:
    """Safely get a value from a dictionary, returning None if the dictionary is None.
    """
    return d.get(k) if d is not None else None

RunInst : TypeAlias = BaseModel | dict | None


def match(obj : RunInst, target : RunInst) -> bool:
    o = obj.model_dump() if isinstance(obj, BaseModel) else obj
    t = target.model_dump() if isinstance(target, BaseModel) else target
    
    if o is None:
        return t is None
    elif t is None:
        return o is None
    
    return all(_safe_get(o, k) == v for k, v in t.items())


T = TypeVar("T")


def same_augs(augs1: list[T], augs2: list[T]) -> bool:
    """Checks that augs1 transform is the same as augs2.

    Args:
        augs1 (list[T]): Augmentations from first source
        augs2 (list[T]): Augmentations from second source

    Returns:
        bool: Wether the first and second source contain the same augmentations in the same order.
    """
    if len(augs1) != len(augs2):
        return False

    def _norm(a: T) -> Any:
        return a.model_dump() if isinstance(a, BaseModel) else a

    return all(_norm(a) == _norm(b) for a, b in zip(augs1, augs2))

def search_old_runs(
    filter_key_sets : list[list[str]],
    criterions: list[Callable[[Any], bool]],
    *,
    drop_key_sets: list[list[str]] | None = None,
    sort_keys: list[list[str]] | None = None,
    reverse: bool = False,
    top_n: int | None = None,
    output_path: str | None = None,
) -> list[CompExpInfo]:
    """Pass arguments directly to Que.list_manipulation. Optionally take top_n and output_path. 

    Args:
        filter_key_sets (list[list[str]]): Each list[str] directly indexes a leaf. 
        criterions (list[Callable[[Any], bool]]): Each criterion corresponds to a list[str] filter key set.
        drop_key_sets (list[list[str]] | None, optional): Sets of keys to drop, each list directly indexes a leaf node. Only applies to Json output. Defaults to None.
        sort_keys (list[list[str]] | None, optional): Keys to sort the runs by. Defaults to None.
        reverse (bool, optional): Whether to sort in reverse order. Defaults to False.
        top_n (int | None, optional): Number of top runs to return. Defaults to None.
        output_path (str | None, optional): Path to output the filtered runs. Defaults to None.

    Returns:
        list[CompExpInfo]: List of CompExpInfo objects representing the filtered runs.
    """
    drop_key_sets = drop_key_sets if drop_key_sets is not None else []
    que = Que()
    runs = list(
        Que.list_manipulation(
            que.list_runs("old_runs"),
            sort_keys=sort_keys,
            reverse=reverse,
            filter_keys=filter_key_sets,
            criterions=criterions,
        )
    )

    # retrieve top n if specified
    if top_n is not None:
        runs = runs[:top_n]

    if output_path:
        output_filtered_runs(
            runs=runs,
            output_path=output_path,
            file_drop_key_sets=drop_key_sets,
        )

    return [CompExpInfo.model_validate(run) for run in runs]

def find_runs(
    filters: dict[str, Any],
    *,
    drop_key_sets: list[list[str]] | None = None,
    sort_keys: list[list[str]] | None = None,
    reverse: bool = False,
    top_n: int | None = None,
    output_path: str | None = None,
) -> list[CompExpInfo]:
    """
    Find runs using fiters and apply list manipulation
        
    Args:
        filters (dict[str, Any]): Nested dictionary where the leaf is a callable boolean function. 
        drop_key_sets (list[list[str]] | None, optional): Sets of keys to drop, each list directly indexes a leaf node. Only applies to Json output. Defaults to None.
        sort_keys (list[list[str]] | None, optional): Keys to sort the runs by. Defaults to None.
        reverse (bool, optional): Whether to sort in reverse order. Defaults to False.
        top_n (int | None, optional): Number of top runs to return. Defaults to None.
        output_path (str | None, optional): Path to output the filtered runs. Defaults to None.

    Returns:
        list[CompExpInfo]: List of CompExpInfo objects representing the filtered runs.
    """
    
    drop_key_sets = drop_key_sets if drop_key_sets is not None else []
    file_filter_keys, file_criterions = unpack_filters(filters)
    return search_old_runs(
        file_filter_keys,
        file_criterions,
        drop_key_sets=drop_key_sets,
        sort_keys=sort_keys,
        reverse=reverse,
        top_n=top_n,
        output_path=output_path,
    )

def fetch_runs(
    filters_path: Path,
    *,
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
    file_filters, file_drop_key_sets = get_filters_drop_keys(filters_path)
    return find_runs(
        file_filters,
        drop_key_sets=file_drop_key_sets,
        sort_keys=sort_keys,
        reverse=reverse,
        top_n=top_n,
        output_path=output_path,
    )


def load_runs(runs_path: Path) -> list[CompExpInfo]:
    """Load a presaved set of finished runs.

    Args:
        runs_path (Path): Path to the JSON file containing the runs.

    Returns:
        list[CompExpInfo]: List of CompExpInfo objects representing the runs.
    """
    with open(runs_path, "r") as f:
        return [CompExpInfo.model_validate(r) for r in json.load(f)]
