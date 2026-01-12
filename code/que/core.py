#!/usr/bin/env python
"""que
---
A lightweight in-memory queue manager for experiment configurations with
simple JSON-backed persistence. The que class is responsible for:
- holding experiment entries in four named lists: to_run, cur_run, old_runs, fail_runs
- moving experiments between these lists safely (with runtime type checks)
- persisting and restoring queue state to/from a JSON file
- summarising and displaying runs for CLI interaction
- creating, adding, editing, finding, and removing runs
Key concepts / data model
- RunInfo / ExpInfo / GenExp: generic experiment dictionaries expected to follow
        the project's standard schema. Typical keys used by que:
                - admin: { model, exp_no, dataset, split, config_path, save_path, ... }
                - data, training, optimizer, model_params, scheduler, early_stopping
                - wandb: optional runtime info, e.g. { "run_id": ... }
                - results: present for completed runs (CompExpInfo), contains at least
                        best_val_acc and best_val_loss
                - error: present for failed runs (FailedExp) and contains an error message
- QueLocation constants (TO_RUN, CUR_RUN, OLD_RUNS, FAIL_RUNS) select which
        internal list is operated on.
- Persistence: queue state is saved/loaded as JSON with top-level keys matching
        the four QueLocation constants.
Public attributes
- runs_path (Path): path to JSON file used for persistence
- to_run (List[ExpInfo]): queued experiments to be executed
- cur_run (List[ExpInfo]): currently executing experiment (zero or one element)
- old_runs (List[CompExpInfo]): completed and fully tested experiments
- fail_runs (List[FailedExp]): experiments that failed during execution
- auto_save (bool): whether to trigger auto-save semantics (honoured externally)
- logger: logging.Logger instance used for informational/error messages
Important methods (summary)
- load_state() -> None
                Load queue lists from runs_path JSON. Initializes empty lists if file missing.
- save_state() -> None
                Write current queue lists to runs_path as JSON.
- create_run(arg_dict: AdminInfo, wandb_dict: WandbInfo) -> None
                Load a run configuration (via load_config), check duplicates, add to to_run.
                Raises QueDupExp if duplicate is found. Uses log_and_raise for logging.
- add_run(arg_dict: AdminInfo, wandb_dict: WandbInfo) -> None
                Construct a CompExpInfo for an already-completed run and insert into old_runs.
                If results cannot be loaded from disk, runs full_test to compute them.
- pop_cur_run() -> ExpInfo
                Remove and return the current run (from cur_run index 0). Raises QueEmpty if none.
- set_cur_run(run: ExpInfo) -> None
                Set cur_run to a provided run. Raises QueBusy if cur_run already contains an item.
- peak_run(loc: QueLocation, idx: int) -> GenExp
                Return (without removing) the run at index idx of the chosen location.
                Raises QueEmpty or QueIdxOOR for invalid access.
- stash_next_run() -> str
                Move the next run from to_run -> cur_run and return a human-readable summary.
                If cur_run is busy it restores the popped run back to to_run and re-raises QueBusy.
- store_fin_run() -> None
                Produce full_test results for the run in cur_run, convert to a CompExpInfo,
                move it into old_runs and remove it from cur_run. Raises QueEmpty if cur_run is empty.
- stash_failed_run(error: str) -> None
                Move the current run from cur_run to fail_runs and annotate with error.
- recover_run(move_to: QueLocation = TO_RUN) -> None
                Mark the current run for recovery (admin.recover = True) and move it to the
                specified location.
- clear_runs(loc: QueLocation) -> None
                Clear the list at the chosen location. Raises QueEmpty if already empty.
- remove_run(loc: QueLocation, idx: int) -> None
                Safely remove a single run at idx from the chosen location.
- shuffle(loc: QueLocation, o_idx: int, n_idx: int) -> None
                Reposition a run within the same list by moving the item at o_idx to n_idx.
- move(o_loc: QueLocation, n_loc: QueLocation, oi_idx: int, of_idx: Optional[int] = None) -> None
                Move a single run or a contiguous range from o_loc to n_loc. For ranges,
                preserves order. Raises QueIdxOORR on invalid index ranges.
- edit_run(loc: QueLocation, idx: int, key1: str, value: Any, key2: Optional[str] = None) -> None
                Edit a run dictionary in-place by popping it, updating the specified key
                (or nested key), and re-inserting it.
- find_runs(loc: QueLocation, key_set: List[List[str]], criterions: List[Callable[[Any], bool]]) -> Tuple[List[int], List[GenExp]]
                Search for runs that satisfy all provided criteria. Each key_set is a list of
                nested keys used to extract a value from the run and passed to the corresponding
                criterion predicate. Returns matching indexes and the matching run objects.
- list_runs(loc: QueLocation) -> List[Sumarised]
                Return a list of summarised runs (model, exp_no, dataset, split, config_path,
                optional run_id, best_val_acc, best_val_loss, error).
- disp_runs(loc: QueLocation, exc: Optional[List[str]] = None) -> None
                Pretty-print the runs at the chosen location to stdout with aligned columns.
                Columns can be excluded by name using exc.
- disp_run(loc: QueLocation, idx: int) -> None
                Print a full run configuration to stdout (via print_config).
Helper / internal methods
- _fetch_state(loc: QueLocation) -> ExpQue
                Return a reference to the corresponding internal list.
- _pop_run(loc: QueLocation, idx: int) -> GenExp
                Pop and return the run at idx from the chosen list with bounds checks.
- _set_run(loc: QueLocation, idx: int, run: GenExp) -> None
                Insert a run at idx into the chosen list with runtime type checks:
                        - FAIL_RUNS expects a FailedExp (contains "error")
                        - OLD_RUNS expects a CompExpInfo (contains "results")
                        - CUR_RUN disallows insertion if cur_run is non-empty
                Raises QueIdxOOR, TypeError, or QueBusy as appropriate.
- _is_failed_exp(run: RunInfo) -> TypeGuard[FailedExp]
- _is_comp_exp_info(run: GenExp) -> TypeGuard[CompExpInfo]
                Lightweight runtime guards used by _set_run.
- _run_sum(run: RunInfo) -> Sumarised
                Create a compact summary dict for display/comparison including optionally
                best_val_acc, best_val_loss, run_id and error.
- _is_dup_exp(new_run: RunInfo) -> bool
                Detect duplicates by comparing model, exp_no, dataset and split across
                to_run, cur_run and old_runs.
- _get_print_stats(runs: List[Sumarised]) -> Dict[str,int]
                Compute column widths for pretty-printing.
- _get_val(run: GenExp, keys: List[str]) -> Any
                Helper to walk a nested dictionary by a list of keys.
- _find_runs(to_search: ExpQue, keys: List[str], criterion: Callable[[Any], bool]) -> Tuple[List[int], List[GenExp]]
                Find elements in a single list matching a predicate on a nested value.
- log_and_raise(task: str = 'Operation') -> contextmanager
                Context manager used throughout to log success or log and re-raise exceptions.
Exceptions
- QueEmpty: raised when an operation expects an element but the chosen list is empty.
- QueIdxOOR: raised when an index is out of range for a single-item operation.
- QueIdxOORR: raised when a range of indexes is invalid for a range-move operation.
- QueBusy: raised when trying to set cur_run while another run occupies it.
- QueDupExp: raised when attempting to create/add a duplicate experiment.
Concurrency and persistence notes
- The class itself does not implement locks. Some methods' docstrings/comments
        mention "saves state with lock", but explicit locking must be implemented by
        the caller if multiple processes/threads will mutate the same runs_path.
- save_state/load_state perform JSON serialization of the internal lists.
        Consumers should ensure save_state is called as needed (auto_save is present
        but not enforced inside every mutating method).
Example usage
- Create and persist a new run to the queue:
                q = que(logger, runs_path="runs.json")
                q.create_run(admin_args, wandb_info)
                q.save_state()
- Worker stashing next job and later storing result:
                summary = q.stash_next_run()
                # worker executes, then:
                q.store_fin_run()
                q.save_state()
This docstring describes the intended behaviour and the main public API surface.
"""

from typing import (
    TYPE_CHECKING,
    Protocol,
    Optional,
    Callable,
    List,
    Literal,
    TypeAlias,
    Tuple,
    Dict,
    Any,
    TypedDict,
    cast,
    Union,
    TypeGuard,
)
from pathlib import Path
import json
from logging import Logger
import logging
from multiprocessing.managers import BaseManager
import time
from datetime import datetime
# locals
from run_types import (
    ExpInfo,
    CompExpInfo,
    AdminInfo,
    WandbInfo,
    RunInfo,
    Sumarised,
)
from testing import full_test, load_comp_res
from configs import print_config, load_config
from contextlib import contextmanager

# constants
# MR_NAME = "monitor"
QUE_DIR = Path(__file__).parent

QUE_NAME = "Que"
DN_NAME = "Daemon"
WORKER_NAME = "Worker"
SERVER_NAME = "Server"
RUN_PATH = QUE_DIR / "Runs.json"
DAEMON_STATE_PATH = QUE_DIR / "Daemon.json"
WR_LOG_PATH = QUE_DIR / "Worker.log"
DN_LOG_PATH = QUE_DIR / "Daemon.log"


SR_LOG_PATH = QUE_DIR / "Server.log"
WR_PATH = QUE_DIR / "worker.py"
WR_MODULE_PATH = f"{QUE_DIR.name}.worker"


TO_RUN = "to_run"  # havent run yet
CUR_RUN = "cur_run"  # busy running
OLD_RUNS = "old_runs"  # run already
FAIL_RUNS = "fail_runs"  # runs that crashed
# List for argparse choices
QUE_LOCATIONS = [TO_RUN, CUR_RUN, OLD_RUNS]
SYNONYMS = {
    "new": "to_run",
    "tr": "to_run",
    "cur": "cur_run",
    "busy": "cur_run",
    "old": "old_runs",
    "or": "old_runs",
    "fail": "fail_runs",
    "fr": "fail_runs",
}
QueLocation: TypeAlias = Literal["to_run", "cur_run", "old_runs", "fail_runs"]

#tmux
SESH_NAME = "train"


class FailedExp(ExpInfo):
    error: str


GenExp: TypeAlias = Union[ExpInfo, FailedExp, CompExpInfo]
ExpQue: TypeAlias = Union[List[ExpInfo], List[FailedExp], List[CompExpInfo]]


class AllRuns(TypedDict):
    old_runs: List[CompExpInfo]
    cur_run: List[ExpInfo]
    to_run: List[ExpInfo]
    fail_runs: List[FailedExp]


class QueException(Exception):
    """Base exception for Que-related errors"""

    pass


class QueDupExp(QueException):
    """Duplicate run detected"""

    def __init__(self, message: str = "Duplicate run detected"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def __reduce__(self):
        return (self.__class__, (self.message,))


class QueEmpty(QueException):
    """Raised when no runs are available in the queue"""

    def __init__(self, loc: QueLocation):
        self.loc = loc
        self.message = f"{loc} is empty"
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def __reduce__(self):
        return (self.__class__, (self.loc,))


class QueIdxOOR(QueException):
    """Raised when index is out of range for a given location"""

    def __init__(self, loc: QueLocation, idx: int, leng: int):
        self.loc = loc
        self.idx = idx
        self.length = leng
        self.message = (
            f"Index {idx} is out of range for que location {loc} (length: {leng})"
        )
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def __reduce__(self):
        return (self.__class__, (self.loc, self.idx, self.length))


class QueIdxOORR(QueException):
    """Raised when a range of values is out of range for a given location"""

    def __init__(self, loc: QueLocation, oi_idx: int, of_idx: int, leng: int):
        self.loc = loc
        self.oi_idx = oi_idx
        self.of_idx = of_idx
        self.length = leng
        self.message = f"Range: {oi_idx} - {of_idx} is an invalid range. Length of {loc} is: {leng}"
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def __reduce__(self):
        return (self.__class__, (self.loc, self.oi_idx, self.of_idx, self.length))


class QueBusy(QueException):
    """Raised when attempting to add a run when one already exists"""

    def __init__(self, message: str = "Run already exists in cur_run"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def __reduce__(self):
        return (self.__class__, (self.message,))

@contextmanager
def log_and_raise(logger: Logger, task: str = "Operation"):
    """
    Context manager that logs success or logs error and re-raises exception

    Args:
            success_msg: Message to log on success
            error_msg: Message to log on error (before re-raising)
    """
    try:
        yield
        logger.info(f"{task} completed successfully")
    except Exception as e:
        logger.error(f"{task} failed: {e}")
        raise e

def timestamp_path(path: Union[str, Path]) -> str:
    formatted = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return str(path).replace(".json", f"_{formatted}.json")


class Que:
    def __init__(
        self,
        logger: Logger,
        runs_path: str | Path = RUN_PATH,
        auto_save: bool = False,
    ) -> None:
        self.runs_path: Path = Path(runs_path)
        self.old_runs: List[CompExpInfo] = []
        self.cur_run: List[ExpInfo] = []
        self.to_run: List[ExpInfo] = []
        self.fail_runs: List[FailedExp] = []
        self.auto_save: bool = auto_save
        self.logger = logger
        self.load_state()

    def _fetch_state(self, loc: QueLocation) -> ExpQue:
        """Return reference to the specified list"""
        if loc == TO_RUN:
            return self.to_run
        elif loc == CUR_RUN:
            return self.cur_run
        elif loc == FAIL_RUNS:
            return self.fail_runs
        else:
            return self.old_runs

    def _pop_run(self, loc: QueLocation, idx: int) -> GenExp:
        """Pop the run at the given location with the provided index

        Args:
                loc (QueLocation): to_run, cur_run or old_runs
                idx (int): Index of the run

        Raises:
                QueEmpty: len(loc) == 0
                QueIdxOOR: abs(idx) >= len(loc)

        Returns:
                ExpInfo: The specified run
        """
        to_get = self._fetch_state(loc)
        if len(to_get) == 0:
            raise QueEmpty(loc)
        elif abs(idx) >= len(to_get):
            raise QueIdxOOR(loc, idx, len(to_get))
        return to_get.pop(idx)

    def _set_run(self, loc: QueLocation, idx: int, run: GenExp) -> None:
        """Set a run at a specified location and index

        Args:
                loc (QueLocation): to_run, cur_run or old_runs
                idx (int): New index, must be within [-len(loc), len(loc)]
                run (ExpInfo): Experiment info to add to loc

        Raises:
                QueIdxOOR: The provided index is out of range: [-len(loc), len(loc)]
                TypeError: The run type doesn't match the queue location
        """
        to_set = self._fetch_state(loc)
        if not (-len(to_set) <= idx <= len(to_set)):
            raise QueIdxOOR(loc, idx, len(to_set))

        # Runtime type checking with type narrowing
        if loc == FAIL_RUNS:
            if not self._is_failed_exp(run):
                raise TypeError("fail_runs requires FailedExp with 'error' field")
            self.fail_runs.insert(idx, run)
        elif loc == OLD_RUNS:
            if not self._is_comp_exp_info(run):
                raise TypeError("old_runs requires CompExpInfo with 'results' field")
            self.old_runs.insert(idx, run)
        elif loc == TO_RUN:
            self.to_run.insert(idx, run)
        else:  # CUR_RUN
            if len(self.cur_run) != 0:
                raise QueBusy()
            self.cur_run.insert(idx, run)

    def _is_failed_exp(self, run: RunInfo) -> TypeGuard[FailedExp]:
        """Check if run is a FailedExp"""
        return isinstance(run, dict) and "error" in run

    def _is_comp_exp_info(self, run: GenExp) -> TypeGuard[CompExpInfo]:
        """Check if run is a CompExpInfo"""
        return isinstance(run, dict) and "results" in run

    def _run_sum(self, run: RunInfo) -> Sumarised:
        """Extract key details from a run configuration.

        Args:
                                        run: Dictionary containing run configuration with admin details
                                        exc: Optional list of keys to exclude from the summary
        Returns:
                                        Sumarised: Dictionary with model, exp_no, split, and config_path
        """
        return Sumarised(
            model=run["admin"]["model"],
            exp_no=run["admin"]["exp_no"],
            dataset=run["admin"]["dataset"],
            split=run["admin"]["split"],
            config_path=run["admin"]["config_path"],
            run_id=run.get("wandb", {}).get("run_id") if "wandb" in run else None,
            best_val_acc=(run["results"]["best_val_acc"] if "results" in run else None),
            best_val_loss=(
                run["results"]["best_val_loss"] if "results" in run else None
            ),
            error=run.get("error") if self._is_failed_exp(run) else None,
        )

    def _run_to_str(self, run_sum: Sumarised) -> str:
        """Convert a summarised run to a string for display

        Args:
                                        run_sum (Sumarised): Summarised run information

        Returns:
                                        str: Formatted string
        """
        return (
            f"Model: {run_sum['model']}, Exp No: {run_sum['exp_no']}, "
            f"Dataset: {run_sum['dataset']}, Split: {run_sum['split']}, "
            f"Config Path: {run_sum['config_path']}"
        )

    def _is_dup_exp(self, new_run: RunInfo) -> bool:
        """Check if run is a duplicate experiment"""

        new_sum = self._run_sum(new_run)

        for run in self.to_run + self.old_runs + self.cur_run:
            run_sum = self._run_sum(run)
            if (
                new_sum["model"] == run_sum["model"]
                and new_sum["exp_no"] == run_sum["exp_no"]
                and new_sum["dataset"] == run_sum["dataset"]
                and new_sum["split"] == run_sum["split"]
            ):
                return True
        return False

    @classmethod
    def _get_print_stats(cls, runs: List[Sumarised]) -> Dict[str, int]:
        """Get statistics for string formatting"""
        stats = {
            "max_model_len": 0,
            "max_exp_no_len": 0,
            "max_run_id_len": 0,
            "max_dataset_len": 0,
            "max_split_len": 0,
            "max_config_path_len": 0,
        }

        for run in runs:
            stats["max_model_len"] = max(stats["max_model_len"], len(run["model"]))
            stats["max_exp_no_len"] = max(stats["max_exp_no_len"], len(run["exp_no"]))
            if run["run_id"] is not None:
                stats["max_run_id_len"] = max(
                    stats["max_run_id_len"], len(run["run_id"])
                )
            stats["max_dataset_len"] = max(
                stats["max_dataset_len"], len(run["dataset"])
            )
            stats["max_split_len"] = max(stats["max_split_len"], len(run["split"]))
            stats["max_config_path_len"] = max(
                stats["max_config_path_len"], len(run["config_path"])
            )

        if runs[0]["best_val_acc"] is not None:
            stats["max_best_val_acc_len"] = len("Best Val Acc")
            stats["max_best_val_loss_len"] = len("Best Val Loss")

        return stats

    def _get_val(self, run: GenExp, keys: List[str]) -> Any:
        """Unpack the value in a run using a list of keys

        Args:
                                        run (GenExp): Provided general run
                                        keys (List[str]): Keys to unpack dictionary

        Returns:
                                        Any: The value
        """
        unpack = cast(Dict[str, Any], run)
        for k in keys:
            unpack = unpack[k]
        return unpack

    def _find_runs(
        self, to_search: ExpQue, keys: List[str], criterion: Callable[[Any], bool]
    ) -> Tuple[List[int], List[GenExp]]:
        """Find runs with matching keys, if any

        Args:
                                        to_search (List[GenExp]): A run list
                                        keys (List[str]): Run keys
                                        value (Any): The desired value

        Returns:
                                        List[Tuple[int, GenExp]]: A List of runs
        """
        idxs = []
        runs = []
        for i, run in enumerate(to_search):
            if criterion(self._get_val(run, keys)):
                idxs.append(i)
                runs.append(run)
        return idxs, runs

    def load_state(self, in_path: Optional[Union[str, Path]]= None):
        """
        Load Que from file. Default load from RUN_PATH, unless in_path is provided
        
        :param in_path: Overide default load path. 
        :type in_path: Optional[Union[str, Path]]
        """
        if in_path is None:
            in_path = self.runs_path
        elif not Path(in_path).exists():
            self.logger.warning(f"No existing state found at {in_path}. Load unsuccessful.")
            return
            
        try:
            with open(in_path, "r") as f:
                data = json.load(f)
            self.to_run = data.get(TO_RUN, [])
            self.cur_run = data.get(CUR_RUN, [])
            self.old_runs = data.get(OLD_RUNS, [])
            self.fail_runs = data.get(FAIL_RUNS, [])
            self.logger.info(f"Loaded que state from {in_path}")
        except FileNotFoundError:
            self.logger.warning(
                f"No existing state found at {in_path}. Starting fresh."
            )
            self.to_run = []
            self.cur_run = []
            self.old_runs = []
            self.fail_runs = []

    def save_state(self, out_path: Optional[Union[str, Path]] = None, timestamp: bool = False):
        if out_path is None:
            out_path = self.runs_path
        elif Path(out_path).exists() and not timestamp:
            self.logger.warning(f"Overwriting existing state file: {out_path}")        

        if timestamp:
            out_path = timestamp_path(out_path)
            
        with open(out_path, "w") as f:
            all_runs = {
                TO_RUN: self.to_run,
                CUR_RUN: self.cur_run,
                OLD_RUNS: self.old_runs,
                FAIL_RUNS: self.fail_runs,
            }
            json.dump(all_runs, f, indent=4)
        self.logger.info(f"Saved que to {out_path}")

    # for worker

    def peak_run(self, loc: QueLocation, idx: int) -> GenExp:
        """Get the run at the given location with the provided index, but don't remove

        Args:
                loc (QueLocation): to_run, cur_run or old_runs
                idx (int): Index of the run

        Raises:
                QueEmpty: len(loc) == 0
                QueIdxOOR: abs(idx) >= len(loc)

        Returns:
                ExpInfo: The specified run
        """
        to_get = self._fetch_state(loc)
        if len(to_get) == 0:
            raise QueEmpty(loc)
        elif abs(idx) >= len(to_get):
            raise QueIdxOOR(loc, idx, len(to_get))
        return to_get[idx]
    
    def peak_cur_run(self) -> ExpInfo:
        """Peaks the current run
        
        Returns:
            ExpInfo: Dictionary of experiment info"""
        return self.peak_run(CUR_RUN, 0)
    
    def pop_cur_run(self) -> ExpInfo:
        """Pops the current run

        Returns:
                ExpInfo: Dictionary of experiment info
        """
        return self._pop_run(CUR_RUN, 0)

    def set_cur_run(self, run: ExpInfo) -> None:
        """Sets the current run

        Args:
                run (ExpInfo): Dictionary of experiment info
        """
        self._set_run(CUR_RUN, 0, run)
    
    def stash_next_run(self) -> str:
        """Moves next run from to_run to cur_run. Saves state with lock over both read and write"""
        next_run = self._pop_run(TO_RUN, 0)
        sum_str = self._run_to_str(self._run_sum(next_run))
        try:
            self.set_cur_run(next_run)
            self.logger.info(f"Stashed new run: {sum_str}")
        except QueBusy as qb:
            # put back
            self.logger.error(f"Failed to stash new run: {sum_str}")
            self._set_run(TO_RUN, 0, next_run)
            raise qb
        return sum_str

    def store_fin_run(self):
        """Moves finished run from cur_run to old_runs. Saves state with lock over both read and write

        Raises:
                QueEmpty: If cur_run is empty
        """
        # fin_run = self._pop_run(CUR_RUN, 0)
        fin_run = self.peak_run(CUR_RUN, 0)  # safer incase crash during test
        results = full_test(admin=fin_run["admin"], data=fin_run["data"])
        comp_run = CompExpInfo(
            admin=fin_run["admin"],
            training=fin_run["training"],
            optimizer=fin_run["optimizer"],
            model_params=fin_run["model_params"],
            data=fin_run["data"],
            scheduler=fin_run["scheduler"],
            early_stopping=fin_run["early_stopping"],
            wandb=fin_run["wandb"],
            results=results,
        )
        self._set_run(OLD_RUNS, 0, comp_run)
        _ = self._pop_run(CUR_RUN, 0)  # still remove from cur
        self.logger.info("Stored finished run")

    def stash_failed_run(self, error: str) -> None:
        """Move a run to the failed que"""
        run = self.pop_cur_run()
        failed = cast(FailedExp, run | {"error": error})
        self._set_run(FAIL_RUNS, 0, failed)

    # summarisation

    @classmethod
    def get_config(cls, next_run: RunInfo) -> str:
        admin = next_run["admin"]
        return admin["config_path"]

    def run_str(self, loc: QueLocation, idx: int) -> str:
        """Method to"""
        return self._run_to_str(self._run_sum(self.peak_run(loc, idx)))

    def list_runs(self, loc: QueLocation) -> List[Sumarised]:
        """List runs at a given location in summarised format

        Args:
                                        loc (QueLocation): to_run, cur_run or old_runs

        Returns:
                                        List[List[str]]: Summarised runs
        """
        return [self._run_sum(run) for run in self._fetch_state(loc)]

    def disp_runs(self, loc: QueLocation, exc: Optional[List[str]] = None) -> None:
        print(f"{loc} runs".title())
        runs = self.list_runs(loc)

        if len(runs) == 0:
            print("  No runs available")
            return
        stats = self._get_print_stats(runs)
        header_parts = [
            "Idx".ljust(5),
            "Run ID".ljust(stats.get("max_run_id_len", len("Run Id")) + 2),
            "Model".ljust(stats["max_model_len"] + 2),
            "Exp No".ljust(stats["max_exp_no_len"] + 2),
            "Dataset".ljust(stats["max_dataset_len"] + 2),
            "Split".ljust(stats["max_split_len"] + 2),
        ]

        if "best_val_acc" in runs[0]:
            header_parts.append("Best Val Acc".ljust(stats["max_best_val_acc_len"] + 2))
            header_parts.append(
                "Best Val Loss".ljust(stats["max_best_val_loss_len"] + 2)
            )

        header_parts.append("Config Path".ljust(stats["max_config_path_len"] + 2))

        if "error" in runs[0]:
            header_parts.append("Error")

        if exc is not None:
            header_parts = [h for h in header_parts if h.strip().lower() not in exc]

        header = " | ".join(header_parts)
        print(header)
        print("-" * len(header))
        for i, run in enumerate(runs):
            row_parts = [
                str(i).ljust(5),
                (run["run_id"] if run["run_id"] is not None else "N/A").ljust(
                    stats.get("max_run_id_len", len("Run Id")) + 2
                ),
                run["model"].ljust(stats["max_model_len"] + 2),
                run["exp_no"].ljust(stats["max_exp_no_len"] + 2),
                run["dataset"].ljust(stats["max_dataset_len"] + 2),
                run["split"].ljust(stats["max_split_len"] + 2),
            ]

            if "best_val_acc" in run:
                row_parts.append(
                    (
                        f"{run['best_val_acc']:.4f}"
                        if run["best_val_acc"] is not None
                        else "N/A"
                    ).ljust(stats["max_best_val_acc_len"] + 2)
                )
                row_parts.append(
                    (
                        f"{run['best_val_loss']:.4f}"
                        if run["best_val_loss"] is not None
                        else "N/A"
                    ).ljust(stats["max_best_val_loss_len"] + 2)
                )

            row_parts.append(run["config_path"].ljust(stats["max_config_path_len"] + 2))

            if "error" in run:
                row_parts.append(run["error"] if run["error"] is not None else "N/A")

            if exc is not None:
                row_parts = [
                    r
                    for r, h in zip(row_parts, header_parts)
                    if h.strip().lower() not in exc
                ]

            row = " | ".join(row_parts)
            print(row)

    @classmethod
    def print_runs(cls, runs: List[Sumarised], exc: Optional[List[str]] = None) -> None:
        """If you are working through the proxy and have already got the runs list"""


        if len(runs) == 0:
            print("  No runs available")
            return
        stats = cls._get_print_stats(runs)
        header_parts = [
            "Idx".ljust(5),
            "Run ID".ljust(stats.get("max_run_id_len", len("Run Id")) + 2),
            "Model".ljust(stats["max_model_len"] + 2),
            "Exp No".ljust(stats["max_exp_no_len"] + 2),
            "Dataset".ljust(stats["max_dataset_len"] + 2),
            "Split".ljust(stats["max_split_len"] + 2),
        ]

        if "best_val_acc" in runs[0]:
            header_parts.append("Best Val Acc".ljust(stats.get("max_best_val_acc_len", 4) + 2))
            header_parts.append(
                "Best Val Loss".ljust(stats.get("max_best_val_loss_len", 4) + 2)
            )

        header_parts.append("Config Path".ljust(stats["max_config_path_len"] + 2))

        if "error" in runs[0]:
            header_parts.append("Error")

        if exc is not None:
            header_parts = [h for h in header_parts if h.strip().lower() not in exc]

        header = " | ".join(header_parts)
        print(header)
        print("-" * len(header))
        for i, run in enumerate(runs):
            row_parts = [
                str(i).ljust(5),
                (run["run_id"] if run["run_id"] is not None else "N/A").ljust(
                    stats.get("max_run_id_len", len("Run Id")) + 2
                ),
                run["model"].ljust(stats["max_model_len"] + 2),
                run["exp_no"].ljust(stats["max_exp_no_len"] + 2),
                run["dataset"].ljust(stats["max_dataset_len"] + 2),
                run["split"].ljust(stats["max_split_len"] + 2),
            ]

            if "best_val_acc" in run:
                row_parts.append(
                    (
                        f"{run['best_val_acc']:.4f}"
                        if run["best_val_acc"] is not None
                        else "N/A"
                    ).ljust(stats.get("max_best_val_acc_len", 4) + 2)
                )
                row_parts.append(
                    (
                        f"{run['best_val_loss']:.4f}"
                        if run["best_val_loss"] is not None
                        else "N/A"
                    ).ljust(stats.get("max_best_val_loss_len", 4) + 2)
                )

            row_parts.append(run["config_path"].ljust(stats["max_config_path_len"] + 2))

            if "error" in run:
                row_parts.append(run["error"] if run["error"] is not None else "N/A")

            if exc is not None:
                row_parts = [
                    r
                    for r, h in zip(row_parts, header_parts)
                    if h.strip().lower() not in exc
                ]

            row = " | ".join(row_parts)
            print(row)

    def disp_run(self, loc: QueLocation, idx: int) -> None:
        """Print a run config at a specific location}

        Args:
                loc (QueLocation): Location
                idx (int): Index
        """
        print_config(self.peak_run(loc, idx))

    # for QueShell interface

    def recover_run(self, move_to: QueLocation = TO_RUN) -> None:
        """Set the run in cur_run to recover and move to to_run or cur_run. Raises a value error if run_id is not present"""
        with log_and_raise(self.logger, "recover"):
            run = self.peak_cur_run()
            run["admin"]["recover"] = True
            if run["wandb"]["run_id"] is None: #NOTE: run id could become optional if required
                raise QueException("Run was set to recover, but no run id was provided")
            _ = self.pop_cur_run()
            self._set_run(move_to, 0, run)

    def clear_runs(self, loc: QueLocation) -> None:
        """reset the runs queue"""
        to_clear = self._fetch_state(loc)
        with log_and_raise(self.logger, f"clear {loc}"):
            if len(to_clear) > 0:
                to_clear.clear()
            else:
                raise QueEmpty(loc)

    def create_run(
        self,
        arg_dict: AdminInfo,
        wandb_dict: WandbInfo,
    ) -> None:
        """Create and add a new training run entry

        Args:
                                        arg_dict (AdminInfo): Arguments used by training function
                                        wandb_dict (WandbInfo): Wandb information not included in arg_dict.
                                        ask (bool, optional): Pre-check run before creation. Defaults to True.
        """
        with log_and_raise(self.logger, "create"):
            config = load_config(arg_dict)

            if self._is_dup_exp(config):
                raise QueDupExp

            # print_config(config)
            config = cast(ExpInfo, config | {"wandb": wandb_dict})
            self.to_run.append(config)

    def add_run(self, arg_dict: AdminInfo, wandb_dict: WandbInfo) -> None:
        """Add a completed (and full tested) run to the old_runs que for storage

        Args:
                                        arg_dict (AdminInfo): Basic information to load the config and find the results
                                        wandb_dict (WandbInfo): Wandb information not included in the run config
        """
        with log_and_raise(self.logger, "add"):
            config = load_config(arg_dict)

            if self._is_dup_exp(config):
                raise QueDupExp

            save_path = Path(arg_dict["save_path"])
            res_dir = save_path.parent / "results"
            res_path = res_dir / "best_val_loss.json"
            try:
                results = load_comp_res(res_path)
                self.logger.info("Successfully loaded results")
            except FileNotFoundError:
                results = full_test(
                    admin=config["admin"],
                    data=config["data"],
                )  # NOTE: this will print to the terminal
                self.logger.info("Could not find results, running full test")
            comp_run = CompExpInfo(
                admin=config["admin"],
                training=config["training"],
                optimizer=config["optimizer"],
                model_params=config["model_params"],
                data=config["data"],
                scheduler=config["scheduler"],
                early_stopping=config["early_stopping"],
                wandb=wandb_dict,
                results=results,
            )
            self.old_runs.insert(0, comp_run)

    def remove_run(self, loc: QueLocation, idx: int) -> None:
        """Removes a run from the given location safely

        Args:
                loc (QueLocation): to_run, cur_run or old_runs
                idx (int): Index of run
        """
        with log_and_raise(self.logger, "remove"):
            _ = self._pop_run(loc, idx)

    def shuffle(self, loc: QueLocation, o_idx: int, n_idx: int) -> None:
        """Repositions a run from the que
        Args:
                                                                                                                                        loc: TO_RUN, CUR_RUN or OLD_RUNS
                                                                                                                                        o_idx: original index of run
                                                                                                                                        n_idx: new index of run
        """
        with log_and_raise(self.logger, "shuffle"):
            self._set_run(loc, n_idx, self._pop_run(loc, o_idx))

    def move(
        self,
        o_loc: QueLocation,
        n_loc: QueLocation,
        oi_idx: int,
        of_idx: Optional[int] = None,
    ) -> None:
        """Moves a run between locations in que (at beginning)

        Args:
                                                                                                                                        o_loc (QueLocation): Old location
                                                                                                                                        n_loc (QueLocation): New location
                                                                                                                                        oi_idx (int): Old initial index
                                                                                                                                        of_idx (int): Old final index, if specifying a range.
        """
        with log_and_raise(self.logger, "move"):
            if of_idx is None:
                self._set_run(n_loc, 0, self._pop_run(o_loc, oi_idx))
            else:
                # Range move
                old_location = self._fetch_state(o_loc)
                new_location = self._fetch_state(n_loc)

                # Validate range
                if abs(oi_idx) >= len(old_location) or abs(of_idx) >= len(old_location):
                    raise QueIdxOORR(o_loc, oi_idx, of_idx, len(old_location))

                # Extract the runs to move
                tomv = []
                for _ in range(oi_idx, of_idx + 1):
                    tomv.append(old_location.pop(oi_idx))

                # Insert into new location (in reverse to maintain order when inserting at 0)
                for run in tomv:
                    new_location.insert(0, run)

    def edit_run(
        self,
        loc: QueLocation,
        idx: int,
        key1: str,
        value: Any,
        key2: Optional[str] = None,
    ) -> None:
        with log_and_raise(self.logger, "edit"):
            run = self._pop_run(loc, idx)
            if key2 is not None:
                run[key1][key2] = value
            else:
                run[key1] = value
            self._set_run(loc, idx, run)

    def find_runs(
        self,
        loc: QueLocation,
        key_set: List[List[str]],
        criterions: List[Callable[[Any], bool]],
    ) -> Tuple[List[int], List[GenExp]]:
        """Find the set of runs which match all of the key list value pairs

        Args:
                                        loc (QueLocation): Location to search
                                        key_set (List[List[str]]): A list of keys to unpack a dictionary to get to a particular value. Multiple values can be searched with a list of these sets of keys
                                        values (List[Any]): The corresponding values for each set of keys

        Returns:
                                        Tuple[List[int], List[GenExp]]: Indexes, and runs, if found
        """

        assert len(key_set) == len(criterions), (
            f"Length of key_set: {len(key_set)} does not match length of values: {len(criterions)}"
        )
        runs = [run for run in self._fetch_state(loc)]
        idxs = []
        for k_lst, crit in zip(key_set, criterions):
            idxs, runs = self._find_runs(runs, k_lst, crit)
        return idxs, runs


        

# --- Daemon State Management --- #

class DaemonState(TypedDict):
    pid: Optional[int]
    worker_pid: Optional[int]
    stop_on_fail: bool
    awake: bool


def is_daemon_state(val: Any) -> TypeGuard[DaemonState]:
    """
    Type guard to check if an arbitrary value is structurally
    compatible with the DaemonState TypedDict.
    """
    # 1. Check if the value is a dictionary
    if not isinstance(val, dict):
        return False

    # 2. Check for the presence of all required keys
    # Since all keys are technically *optional* in the Python dictionary sense
    # but *required* by TypedDict (unless explicitly marked NotRequired),
    # we check for all keys listed in the TypedDict.
    required_keys = DaemonState.__annotations__.keys()
    if not all(key in val for key in required_keys):
        return False

    # 3. Check the type of each value
    # We use .get() here defensively, although the previous check makes it safe
    # to use val[key].

    # Check 'pid' and 'worker_pid' (Optional[int])
    if not (val.get("pid") is None or isinstance(val["pid"], int)):
        return False

    if not (val.get("worker_pid") is None or isinstance(val["worker_pid"], int)):
        return False

    # Check 'stop_on_fail' and 'awake' (bool)
    if not isinstance(val.get("stop_on_fail"), bool):
        return False

    if not isinstance(val.get("awake"), bool):
        return False

    # If all checks pass, it is a DaemonState
    return True


def read_daemon_state(state_path: Union[Path, str] = DAEMON_STATE_PATH) -> DaemonState:
    with open(state_path, "r") as f:
        data = json.load(f)
    if is_daemon_state(data):
        return data
    else:
        raise ValueError(
            f"Data read from: {state_path} is not compatible with DaemonState"
        )

default_state: DaemonState = {
    "pid": None,
    "worker_pid": None,
    "stop_on_fail": False,
    "awake": False,
}

class DaemonStateHandler:
    def __init__(
        self,
        logger: Logger,
        pid: Optional[int] = None,
        worker_pid: Optional[int] = None,
        stop_on_fail: bool = True,
        awake: bool = False,
        state_path: Union[Path, str] = DAEMON_STATE_PATH,
    ) -> None:
        self.logger = logger
        self.pid: Optional[int] = pid
        self.worker_pid: Optional[int] = worker_pid
        self.stop_on_fail: bool = stop_on_fail
        self.awake: bool = awake
        self.state_path: Path = Path(state_path)
        self.from_disk()

    def from_disk(self) -> None:

        try:
            state = read_daemon_state(self.state_path)
            self.pid = state["pid"]
            self.worker_pid = state["worker_pid"]
            self.stop_on_fail = state["stop_on_fail"]
            self.awake = state["awake"]
            self.logger.info(f"Loaded state from: {self.state_path}")
        except Exception as e:
            self.logger.warning(
                f"Ran into an error when loading state: {e}\nloading from scratch"
            )
            self.pid = None
            self.worker_pid = None
            self.stop_on_fail = False
            self.awake = False

    def to_disk(self) -> None:
        state: DaemonState = {
            "pid": self.pid,
            "worker_pid": self.worker_pid,
            "stop_on_fail": self.stop_on_fail,
            "awake": self.awake,
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f)
        self.logger.info(f"Saved state to: {self.state_path}")

    def get_state(self) -> DaemonState:
        return {
            "pid": self.pid,
            "worker_pid": self.worker_pid,
            "stop_on_fail": self.stop_on_fail,
            "awake": self.awake,
        }

    def set_state(self, state: DaemonState) -> None:
        self.pid = state["pid"]
        self.worker_pid = state["worker_pid"]
        self.stop_on_fail = state["stop_on_fail"]
        self.awake = state["awake"]

    def get_pid(self) -> Optional[int]:
        return self.pid

    def set_pid(self, pid: Optional[int]) -> None:
        self.pid = pid

    def get_worker_pid(self) -> Optional[int]:
        return self.worker_pid

    def set_worker_pid(self, worker_pid: Optional[int]) -> None:
        self.worker_pid = worker_pid

    def get_stop_on_fail(self) -> bool:
        return self.stop_on_fail

    def set_stop_on_fail(self, stop_on_fail: bool) -> None:
        self.stop_on_fail = stop_on_fail

    def get_awake(self) -> bool:
        return self.awake

    def set_awake(self, awake: bool) -> None:
        self.awake = awake

#------- Basmanager connections -------#

if TYPE_CHECKING:
    class DaemonControllerProtocol(Protocol):
        def save_state(self) -> None: ...
        def load_state(self) -> None: ...
        def start(self) -> None: ...
        def stop_worker(self, timeout: Optional[float] = None, hard: bool = False) -> None: ...
        def stop_supervisor(self, timeout: Optional[float] = None, hard: bool = False, and_worker: bool = False) -> None: ...
        def get_state(self) -> DaemonState: ...
        def set_stop_on_fail(self, value: bool) -> None: ...
        def set_awake(self, value: bool) -> None: ...

    class QueManagerProtocol(Protocol):
        def get_que(self) -> Que: ...
        def get_daemon_state(self) -> DaemonStateHandler: ...
        def DaemonController(self) -> DaemonControllerProtocol: ...

class QueManager(BaseManager): 
    pass

def connect_manager(max_retries=5, retry_delay=2) -> "QueManagerProtocol":
    """
    Useful helper for clients to connect to the QueManager server.
    
    :param max_retries: Maximum number of connection attempts
    :param retry_delay: Delay between retries in seconds
    :return: Connected QueManager instance
    :rtype: QueManagerProtocol
    """
    QueManager.register('DaemonController')
    QueManager.register('get_que')
    QueManager.register('get_daemon_state')

    for _ in range(max_retries):
        try:
            m = QueManager(address=('localhost', 50000), authkey=b'abracadabra')
            m.connect()
            return m # type: ignore
        except ConnectionRefusedError:
            print(f"Queue server not ready, retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            
    raise RuntimeError("Cannot connect to Queue server.")

def _get_basic_logger() -> Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=SR_LOG_PATH,  # Optional: log to file
    )

    logger = logging.getLogger(__name__)
    return logger


def main():
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #     filename=SR_LOG_PATH,  # Optional: log to file
    # )

    logger = _get_basic_logger()

    q = Que(logger)
    q.disp_runs(OLD_RUNS)

#TODO:
#- add more options to logs, (e.g. clear)
#- add more options for que copies (e.g. load)
#- add auto experiment num
#- add remote shell connection




if __name__ == "__main__":
    main()
