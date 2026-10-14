import json
import logging
import sys
from collections.abc import Callable
from typing import Any, get_args

# from .server import connect_manager
# from que.shell import QueShell
from src.que.core import QUE_LOCATIONS, RUN_PATH, GenExp, Que
from src.run_types import (
    CUTOFF_SPLITS,
    ORIGINAL_SPLITS,
    CompExpInfo,
    ExpInfo,
    FailedExp,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.propagate = False  # don't also send to root's (possibly broken) handlers


def update_runs_que_template():
    """Use the Que.update_runs method to apply a function to runs"""

    q = Que(auto_save=False)  # writes to Runs_updated.json below, not back to Runs.json

    def ident(x):
        # replace with custom function
        return x

    key_set = []
    q.update_runs(key_set, ident)
    q.save_state(RUN_PATH.with_name("Runs_updated.json"))


def update_runs_json_template():
    """Update Json directly"""
    with open(RUN_PATH, "r") as f:
        all_runs = json.load(f)

    for loc in QUE_LOCATIONS:
        que_list = all_runs[loc]
        new_quelist = []
        for run in que_list:
            # ---
            # edit run here
            # ---
            if loc in QUE_LOCATIONS[:2]:
                run = ExpInfo.model_validate(run).model_dump()
            elif loc == QUE_LOCATIONS[2]:
                run = CompExpInfo.model_validate(run).model_dump()
            else:
                run = FailedExp.model_validate(run).model_dump()

            new_quelist.append(run)

        all_runs[loc] = new_quelist

    with open(RUN_PATH.with_name("Runs_fixed.json"), "w") as f:
        json.dump(all_runs, f, indent=4)


def update_runs_json():
    """Update Json directly"""
    with open(RUN_PATH, "r") as f:
        all_runs = json.load(f)

    for loc in QUE_LOCATIONS:
        que_list = all_runs[loc]
        new_quelist = []
        for run in que_list:
            # ---
            del run["model_params"]["type"]
            # print(run['model_params'].keys())
            # break

            # ---
            if loc in QUE_LOCATIONS[:2]:
                run = ExpInfo.model_validate(run).model_dump()
            elif loc == QUE_LOCATIONS[2]:
                run = CompExpInfo.model_validate(run).model_dump()
            else:
                run = FailedExp.model_validate(run).model_dump()

            new_quelist.append(run)

        all_runs[loc] = new_quelist

    with open(RUN_PATH.with_name("Runs_fixed.json"), "w") as f:
        json.dump(all_runs, f, indent=4)


def _to_one_indexed_dir(path: str, splits: tuple[str, ...]) -> str:
    """Rename the `runs/<split>/` component of `path` for each of `splits` to `<split>_1_indexed`.

    Matches the component rather than a prefix, since paths may be relative (`runs/...`) and a
    run's `weight_path` can point into another split's run dir (transfer from a smaller split).
    """
    for split in splits:
        path = path.replace(f"runs/{split}/", f"runs/{split}_1_indexed/")
    return path


def to_one_indexed_split(
    admin: dict[str, Any], splits: tuple[str, ...] = get_args(ORIGINAL_SPLITS)
) -> dict[str, Any]:
    """Move a run on one of `splits` to its `*_1_indexed` split and run dir; others unchanged.

    For renaming the 1-based labels and their `src/runs` dirs ahead of a 0-based preprocessing
    rerun (see src/info/WLASL_info.md, "The 0-based rerun"). `splits` defaults to the plain
    splits (done 2026-10-05); the `*_cutoff_9` splits use `to_one_indexed_cutoff_split`. A run's
    `weight_path` is rewritten for any of `splits`, even when it isn't the run's own split, so a
    run on an unmoved split that loads a moved checkpoint still finds it.
    """
    moved = dict(admin)
    if admin["split"] in splits:
        moved["split"] = f"{admin['split']}_1_indexed"
    moved["save_path"] = _to_one_indexed_dir(admin["save_path"], splits)
    if admin.get("weight_path"):
        moved["weight_path"] = _to_one_indexed_dir(admin["weight_path"], splits)
    return moved


def to_one_indexed_cutoff_split(admin: dict[str, Any]) -> dict[str, Any]:
    """`to_one_indexed_split` for the `*_cutoff_9` splits."""
    return to_one_indexed_split(admin, get_args(CUTOFF_SPLITS))


def move_to_one_indexed_splits(
    move: Callable[[dict[str, Any]], dict[str, Any]] = to_one_indexed_cutoff_split,
) -> None:
    """Apply `move` to every run's admin, writing Runs_updated.json (not Runs.json)."""
    q = Que(auto_save=False)
    q.update_runs(["admin"], move)
    q.save_state(RUN_PATH.with_name("Runs_updated.json"))


def get_all_runs(q: Que) -> list[GenExp]:
    runs = []
    for loc in QUE_LOCATIONS:
        runs.extend(q.list_runs(loc))  # type: ignore
    return runs


def any_dups() -> bool:
    """Return True if any two runs across all Que locations serialise identically."""
    seen: set[str] = set()
    for run in get_all_runs(Que()):
        str_run = run.model_dump_json()
        if str_run in seen:
            return True
        seen.add(str_run)
    return False


if __name__ == "__main__":
    # update_runs_json()
    # print(any_dups())
    move_to_one_indexed_splits()
