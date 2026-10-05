import json
import logging
import sys
from pathlib import Path
from typing import Any, get_args

# from .server import connect_manager
# from que.shell import QueShell
from src.que.core import QUE_LOCATIONS, RUN_PATH, GenExp, Que
from src.run_types import (
    ORIGINAL_SPLITS,
    RUNS_PATH,
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


def to_one_indexed_split(admin: dict[str, Any]) -> dict[str, Any]:
    """Move a run on a plain asl split to its `*_1_indexed` split and run dir; others unchanged.

    For the 2026-10-05 rename of the 1-based labels and `src/runs/asl100` ahead of the 0-based
    preprocessing rerun (see src/info/WLASL_info.md, "What the rerun will change").
    """
    split = admin["split"]
    if split not in get_args(ORIGINAL_SPLITS):
        return admin
    new_split = f"{split}_1_indexed"
    run_dir = Path(admin["save_path"]).relative_to(RUNS_PATH / split)
    return {**admin, "split": new_split, "save_path": str(RUNS_PATH / new_split / run_dir)}


def move_to_one_indexed_splits() -> None:
    """Apply `to_one_indexed_split` to every run, writing Runs_updated.json (not Runs.json)."""
    q = Que(auto_save=False)
    q.update_runs(["admin"], to_one_indexed_split)
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
