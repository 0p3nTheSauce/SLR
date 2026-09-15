"""Shared helpers for the `src/results` analysis notebooks/scripts.

Wraps the `Que` run-history system (`src.que.core`, `src.que.shell`) with the
query/load/stash conveniences used across `src/results/*`:

- `find_runs`/`fetch_runs`/`search_old_runs` query finished runs out of the `Que`'s
  `old_runs` history, filtering/sorting/limiting them (`fetch_runs` reads its filters
  from a `filters.py` file; `find_runs` takes them directly).
- `load_runs` reads back a previously-saved JSON list of runs (e.g. via
  `output_path`/`output_filtered_runs`).
- `get_out_stub`/`get_stash_path`/`get_asset_path`/`stash_json`/`load_json`/
  `stash_from_saved` implement the local "stash" (raw JSON) vs "asset" (figures, LaTeX)
  saving convention for notebooks — see below.
- `match`/`same_augs` are equality helpers for comparing `RunInst`/augmentation configs.

`CompExpInfo`, `get_filters_drop_keys`, `output_filtered_runs`, `Que` and
`unpack_filters` are re-exported here so callers only need `src.results`, not the
underlying `src.que` submodules.

---

### When it comes to saving files in Notebooks:

1. By convention, when a notebook is run once, heavy running code should stash it's results
then the call sight must be commented out. A call site which loads the stashed results
is left uncommented, for future notebook runs

2. Additionally, certain items should be dumped into the results/outputs directory and thus
not tracked by git. These are generally large files that can be generated quickly by
the notebook, such as Figures. This serves a secondary purpose of localising all diagrams
so they can be extracted for the thesis. 
---
**Naming Convention:**

Stash
- Results (JSON) in a local subdirectory
- Allows for offline running of notebooks after they are run once
- Used by the notebook

Asset
- Figures, LaTeX etc. 
- Produced by the notebook, but has no impact on running the notebook
- Used in the thesis

"""

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
from src.run_types import RESULTS_OUTPUTS, TypeAlias

__all__ = [
    "STASH_DIR_NAME",
    "CompExpInfo",
    "Que",
    "RunInst",
    "fetch_runs",
    "find_runs",
    "format_exp_label",
    "get_asset_path",
    "get_filters_drop_keys",
    "get_out_stub",
    "get_stash_path",
    "load_json",
    "load_runs",
    "match",
    "output_filtered_runs",
    "same_augs",
    "search_old_runs",
    "stash_from_saved",
    "stash_json",
    "unpack_filters",
]


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

# ----------------------------------------------------------------------
# Stash and Assest saving and loading
# ----------------------------------------------------------------------

STASH_DIR_NAME : str = 'stashed_results'

def format_exp_label(exp_no: str) -> str:
    """Format an `exp_no` for use in stash/asset filenames.

    Regular experiments are zero-padded sequential numbers (e.g. `"000"`) and
    get an `"exp"` prefix to match the `exp{NNN}` directory convention. Sweep
    trials store a wandb run id (not numeric) in `exp_no` instead and are
    left as-is, since they don't live under that convention at all (see
    `sweeping.get_sweep_exp_dir`).
    """
    return f"exp{exp_no}" if exp_no.isdigit() else exp_no


def get_out_stub(split : str, model : str, exp : str, checkpoint_num: int | str | None = None) -> str:
    checknum = str(checkpoint_num) + '_' if checkpoint_num is not None else ''
    return f"{split}_{model}_{exp}_{checknum}"

def get_asset_path(metric_descriptor: str, stub: str, file_suffix: str, asset_dir: Path = RESULTS_OUTPUTS) -> Path:
    """Get the path for the asset to be saved to"""
    return (asset_dir / f'{metric_descriptor}_{stub}').with_suffix(file_suffix)

def get_stash_path(stub : str, base_name: str = "results", local_stash_dir_name: str = STASH_DIR_NAME) -> Path:
    """Get the path to the results json file"""
    return Path(local_stash_dir_name) / f"{stub}{base_name}.json"

def stash_json(results : Any, stash_path: Path, make: bool = True, indent: int | str | None  = None) -> Path:
    """Stash the results, make if necessary"""
    if make:
        stash_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stash_path, 'w') as f:
        json.dump(results, f, indent=indent)
        
    return stash_path 
        
def load_json(stash_path: Path) -> Any:
    """Load stashed results"""
    with open(stash_path, 'r') as f:
        return json.load(f)

def stash_from_saved(original_save_path: Path) -> Path:
    """Load pre-run results from the runs directory to the stashed directory"""
    results_dir = original_save_path.parent

    exp_dir = results_dir.parent
    model_dir = exp_dir.parent
    split_dir = model_dir.parent
    
    return stash_json(
        load_json(original_save_path),
        get_stash_path(split_dir.name, model_dir.name, exp_dir.name),
        )

