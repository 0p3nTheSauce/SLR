#!/usr/bin/env python
"""que
---
A lightweight in-memory queue manager for experiment configurations with
simple JSON-backed persistence.  See the original module docstring for the
full public-API description.  This version uses pydantic BaseModel objects
throughout instead of TypedDicts / plain dicts.
"""

from typing import (
    Protocol,
    Optional,
    Callable,
    List,
    Literal,
    Sequence,
    TypeAlias,
    Tuple,
    Dict,
    Any,
    Union,
    TypeGuard,
)
from typing_extensions import TypedDict, Unpack
import ast
import traceback
from pydantic import BaseModel
from pathlib import Path
import json
from logging import Logger
import logging
from multiprocessing.managers import BaseManager, DictProxy
import time
from datetime import datetime
from contextlib import contextmanager

# locals
from run_types import (
    ExpInfo,
    CompExpInfo,
    AdminInfo,
    WandbInfo,
    RunInfo,
    Sumarised,
    SummarisedError,
    SummarisedRes,
    FailedExp,  # now defined in run_types
)

# from configs import print_config, load_config, ZFILL, get_model_exp_dir, get_model_results_dir


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYSTEMD_NAME = "que-training.service"
QUE_DIR = Path(__file__).parent

QUE_NAME = "Que"
DAEMON_NAME = "Daemon"
WORKER_NAME = "Worker"
SERVER_NAME = "Server"
TRAINING_NAME = "Training"

RUN_PATH = QUE_DIR / "Runs.json"
SERVER_STATE_PATH = QUE_DIR / "Server.json"

TRAINING_LOG_PATH = QUE_DIR / "Training.log"
SERVER_LOG_PATH = QUE_DIR / "Server.log"

ARCHIVE_DIR = QUE_DIR / "old_ques"

WR_PATH = QUE_DIR / "worker.py"
WR_MODULE_PATH = f"{QUE_DIR.name}.worker"
SERVER_MODULE_PATH = f"{QUE_DIR.name}.server"

TO_RUN = "to_run"
CUR_RUN = "cur_run"
OLD_RUNS = "old_runs"
FAIL_RUNS = "fail_runs"
QUE_LOCATIONS = [TO_RUN, CUR_RUN, OLD_RUNS, FAIL_RUNS]
PROCESS_NAMES = [SERVER_NAME, DAEMON_NAME, WORKER_NAME]
SYNONYMS = {
    "new": "to_run",
    "tr": "to_run",
    "cur": "cur_run",
    "cr": "cur_run",
    "old": "old_runs",
    "or": "old_runs",
    "fail": "fail_runs",
    "fr": "fail_runs",
}

QueLocation: TypeAlias = Literal["to_run", "cur_run", "old_runs", "fail_runs"]
ProcessNames: TypeAlias = Literal["Server", "Daemon", "Worker"]


GenExp: TypeAlias = Union[ExpInfo, FailedExp, CompExpInfo]
ExpQue: TypeAlias = Union[List[ExpInfo], List[FailedExp], List[CompExpInfo]]


class AllRuns(BaseModel):
    old_runs: List[CompExpInfo]
    cur_run: List[ExpInfo]
    to_run: List[ExpInfo]
    fail_runs: List[FailedExp]


class PositionInfo(BaseModel):
    location: QueLocation
    index: int


class RangePosition(PositionInfo):
    index2: int


class SortInfo(BaseModel):
    key_set: List[str]
    reverse: bool


NO_SORT = SortInfo(key_set=[], reverse=False)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class QueException(Exception):
    pass


class QueDupExp(QueException):
    def __init__(self, message: str = "Duplicate run detected"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def __reduce__(self):
        return (self.__class__, (self.message,))


class QueEmpty(QueException):
    def __init__(self, loc: QueLocation):
        self.loc = loc
        self.message = f"{loc} is empty"
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def __reduce__(self):
        return (self.__class__, (self.loc,))


class QueIdxOOR(QueException):
    def __init__(self, loc: QueLocation, idx: int, leng: int):
        self.loc = loc
        self.idx = idx
        self.length = leng
        self.message = f"Index {idx} is out of range for {loc} (length: {leng})"
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def __reduce__(self):
        return (self.__class__, (self.loc, self.idx, self.length))


class QueIdxOORR(QueException):
    def __init__(self, loc: QueLocation, oi_idx: int, of_idx: int, leng: int):
        self.loc = loc
        self.oi_idx = oi_idx
        self.of_idx = of_idx
        self.length = leng
        self.message = (
            f"Range: {oi_idx} - {of_idx} is invalid. Length of {loc} is: {leng}"
        )
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def __reduce__(self):
        return (self.__class__, (self.loc, self.oi_idx, self.of_idx, self.length))


class QueBusy(QueException):
    def __init__(self, message: str = "Run already exists in cur_run"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def __reduce__(self):
        return (self.__class__, (self.message,))


# ---------------------------------------------------------------------------
# Kwargs
# ---------------------------------------------------------------------------
class ListManipulationKwargs(TypedDict, total=False):
    sort_keys: list[list[str]]
    reverse: bool
    filter_keys: list[list[str]]
    criterions: list[Callable[[Any], bool]]


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextmanager
def log_and_raise(logger: Logger, task: str = "Operation"):
    try:
        yield
        logger.info(f"{task} completed successfully")
    except Exception as e:
        logger.error(f"{task} failed: {e}")
        logger.error(traceback.format_exc())
        raise e


def timestamp_path(path: Union[str, Path]) -> str:
    formatted = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return str(path).replace(".json", f"_{formatted}.json")


# ---------------------------------------------------------------------------
# Que class
# ---------------------------------------------------------------------------


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

    # -----------------------------------------------------------------------
    # General helpers
    # -----------------------------------------------------------------------

    def _fetch_state(self, loc: QueLocation) -> ExpQue:
        if loc == TO_RUN:
            return self.to_run
        elif loc == CUR_RUN:
            return self.cur_run
        elif loc == FAIL_RUNS:
            return self.fail_runs
        else:
            return self.old_runs

    def _pop_run(self, loc: QueLocation, idx: int) -> GenExp:
        to_get = self._fetch_state(loc)
        if len(to_get) == 0:
            raise QueEmpty(loc)
        elif abs(idx) >= len(to_get):
            raise QueIdxOOR(loc, idx, len(to_get))
        return to_get.pop(idx)

    def _set_run(self, loc: QueLocation, idx: int, run: GenExp) -> None:
        if loc == FAIL_RUNS:
            if not self._is_failed_exp(run):
                raise TypeError("fail_runs requires a FailedExp instance")
            self.fail_runs.insert(idx, run)
        elif loc == OLD_RUNS:
            if not self._is_comp_exp_info(run):
                raise TypeError("old_runs requires a CompExpInfo instance")
            self.old_runs.insert(idx, run)
        elif loc == TO_RUN:
            self.to_run.insert(idx, run)  # type: ignore[arg-type]
        else:  # CUR_RUN
            if len(self.cur_run) != 0:
                raise QueBusy()
            self.cur_run.insert(idx, run)  # type: ignore[arg-type]

    @classmethod
    def _is_failed_exp(cls, run: Any) -> TypeGuard[FailedExp]:
        return isinstance(run, FailedExp)

    @classmethod
    def _is_comp_exp_info(cls, run: Any) -> TypeGuard[CompExpInfo]:
        return isinstance(run, CompExpInfo)

    @classmethod
    def _run_sum(cls, run: GenExp) -> Sumarised:
        """Extract a compact summary from a run model."""
        run_id = run.wandb.run_id if isinstance(run, ExpInfo) and run.wandb else None

        base = Sumarised(
            model=run.admin.model,
            exp_no=run.admin.exp_no,
            dataset=run.admin.dataset,
            split=run.admin.split,
            config_path=run.admin.config_path,
            run_id=run_id,
            best_val_acc=run.results.best_val_acc
            if isinstance(run, CompExpInfo)
            else None,
            best_val_loss=run.results.best_val_loss
            if isinstance(run, CompExpInfo)
            else None,
        )

        if cls._is_failed_exp(run):
            return SummarisedError(**base.model_dump(), error=run.error)
        elif cls._is_comp_exp_info(run):
            test = run.results.test
            return SummarisedRes(
                **base.model_dump(),
                test_top1_acc=test.top_k_per_instance_acc.top1 * 100,
                test_av_loss=test.average_loss,
            )
        return base

    def _run_to_str(self, run_sum: Sumarised) -> str:
        return (
            f"Model: {run_sum.model}, Exp No: {run_sum.exp_no}, "
            f"Dataset: {run_sum.dataset}, Split: {run_sum.split}, "
            f"Config Path: {run_sum.config_path}"
        )

    def _is_dup_exp(self, new_run: RunInfo) -> bool:
        new_sum = self._run_sum(new_run)  # type: ignore[arg-type]
        for run in self.to_run + self.old_runs + self.cur_run:  # type: ignore[operator]
            run_sum = self._run_sum(run)
            if (
                new_sum.model == run_sum.model
                and new_sum.exp_no == run_sum.exp_no
                and new_sum.dataset == run_sum.dataset
                and new_sum.split == run_sum.split
            ):
                return True
        return False

    @classmethod
    def _clean_slate(cls, run: GenExp, enum_chck: bool) -> ExpInfo:
        """Reset run to a fresh state (no error/results, recover=False, run_id=None).

        Args:
            run: Any queue run (may be FailedExp or CompExpInfo).
            enum_chck: Enumerate the checkpoint directory path.

        Returns:
            ExpInfo: A fresh run without any run-specific state.
        """
        from utils import enum_dir
        from configs import ZFILL

        new_save_path = (
            str(enum_dir(run.admin.save_path, decimals=ZFILL))
            if enum_chck
            else run.admin.save_path
        )
        new_admin = run.admin.model_copy(
            update={"recover": False, "save_path": new_save_path}
        )
        new_wandb = run.wandb.model_copy(update={"run_id": None})

        return ExpInfo.model_validate(
            {
                **run.model_dump(exclude={"error", "results"}),
                "admin": new_admin.model_dump(),
                "wandb": new_wandb.model_dump(),
            }
        )

    @classmethod
    def _get_print_stats(cls, runs: List[Sumarised]) -> Dict[str, int]:
        stats: Dict[str, int] = {
            "max_model_len": 0,
            "max_exp_no_len": 0,
            "max_run_id_len": 0,
            "max_dataset_len": 0,
            "max_split_len": 0,
            "max_config_path_len": 0,
        }
        for run in runs:
            stats["max_model_len"] = max(stats["max_model_len"], len(run.model))
            stats["max_exp_no_len"] = max(stats["max_exp_no_len"], len(run.exp_no))
            if run.run_id is not None:
                stats["max_run_id_len"] = max(stats["max_run_id_len"], len(run.run_id))
            stats["max_dataset_len"] = max(stats["max_dataset_len"], len(run.dataset))
            stats["max_split_len"] = max(stats["max_split_len"], len(run.split))
            stats["max_config_path_len"] = max(
                stats["max_config_path_len"], len(run.config_path)
            )

        if runs[0].best_val_acc is not None:
            stats["max_best_val_acc_len"] = len("Best Val Acc")
            stats["max_best_val_loss_len"] = len("Best Val Loss")

        return stats

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def load_state(self, in_path: Optional[Union[str, Path]] = None):
        if in_path is None:
            in_path = self.runs_path
        elif not Path(in_path).exists():
            self.logger.warning(
                f"No existing state found at {in_path}. Load unsuccessful."
            )
            return

        try:
            with open(in_path, "r") as f:
                data = json.load(f)
            self.to_run = [ExpInfo.model_validate(r) for r in data.get(TO_RUN, [])]
            self.cur_run = [ExpInfo.model_validate(r) for r in data.get(CUR_RUN, [])]
            self.old_runs = [
                CompExpInfo.model_validate(r) for r in data.get(OLD_RUNS, [])
            ]
            self.fail_runs = [
                FailedExp.model_validate(r) for r in data.get(FAIL_RUNS, [])
            ]
            self.logger.info(f"Loaded que state from {in_path}")
        except FileNotFoundError:
            self.logger.warning(
                f"No existing state found at {in_path}. Starting fresh."
            )
            self.to_run = []
            self.cur_run = []
            self.old_runs = []
            self.fail_runs = []

    def save_state(
        self,
        out_path: Optional[Union[str, Path]] = None,
        timestamp: bool = False,
        archive: bool = True,
    ):
        if out_path is None:
            out_path = self.runs_path
        elif Path(out_path).exists() and not timestamp:
            self.logger.warning(f"Overwriting existing state file: {out_path}")

        if archive:
            out_path = ARCHIVE_DIR / out_path

        if timestamp:
            out_path = timestamp_path(out_path)

        all_runs = {
            TO_RUN: [r.model_dump() for r in self.to_run],
            CUR_RUN: [r.model_dump() for r in self.cur_run],
            OLD_RUNS: [r.model_dump() for r in self.old_runs],
            FAIL_RUNS: [r.model_dump() for r in self.fail_runs],
        }
        with open(out_path, "w") as f:
            json.dump(all_runs, f, indent=4)
        self.logger.info(f"Saved que to {out_path}")

    # -----------------------------------------------------------------------
    # Worker helpers
    # -----------------------------------------------------------------------

    def peak_run(self, loc: QueLocation, idx: int) -> GenExp:
        to_get = self._fetch_state(loc)
        if len(to_get) == 0:
            raise QueEmpty(loc)
        elif abs(idx) >= len(to_get):
            raise QueIdxOOR(loc, idx, len(to_get))
        return to_get[idx]

    def peak_cur_run(self) -> ExpInfo:
        return self.peak_run(CUR_RUN, 0)  # type: ignore[return-value]

    def pop_cur_run(self) -> ExpInfo:
        return self._pop_run(CUR_RUN, 0)  # type: ignore[return-value]

    def set_cur_run(self, run: ExpInfo) -> None:
        self._set_run(CUR_RUN, 0, run)

    def stash_next_run(self) -> str:
        next_run = self._pop_run(TO_RUN, 0)
        sum_str = self._run_to_str(self._run_sum(next_run))
        try:
            self.set_cur_run(next_run)  # type: ignore[arg-type]
            self.logger.info(f"Stashed new run: {sum_str}")
        except QueBusy as qb:
            self.logger.error(f"Failed to stash new run: {sum_str}")
            self._set_run(TO_RUN, 0, next_run)
            raise qb
        return sum_str

    def store_fin_run(self):
        """Move finished run from cur_run to old_runs.

        NOTE: _set_run will raise TypeError if the run is not a CompExpInfo.
        The caller is responsible for ensuring results have been attached before
        calling this method.
        """
        self._set_run(OLD_RUNS, 0, self.pop_cur_run())
        self.logger.info("Stored finished run")

    def stash_failed_run(self, error: str) -> None:
        """Move the current run to fail_runs, annotated with the error message."""
        run = self.pop_cur_run()
        failed = FailedExp.model_validate({**run.model_dump(), "error": error})
        self._set_run(FAIL_RUNS, 0, failed)

    # -----------------------------------------------------------------------
    # Queue display and modification helpers
    # -----------------------------------------------------------------------

    @classmethod
    def _set_inplace(
        cls, d: Dict[Any, Any], k: Any, ks: List[Any], val: Any
    ) -> Dict[Any, Any]:
        """Recursively set a value in a nested plain dict."""
        if hasattr(d, "__setitem__"):
            if len(ks) == 0:
                d[k] = val
            else:
                next_key = ks.pop(0)
                old_val = d.get(k, {})
                d[k] = cls._set_inplace(old_val, next_key, ks, val)
        else:
            if len(ks) == 0:
                d = {k: val}
            else:
                next_key = ks.pop(0)
                d = {k: cls._set_inplace({}, next_key, ks, val)}
        return d

    @classmethod
    def set_nested(cls, d: Dict[Any, Any], ks: List[Any], val: Any) -> Dict[Any, Any]:
        """Set a value at an arbitrary depth in a plain dict using a key path."""
        return cls._set_inplace(d, ks[0], ks[1:], val)

    @classmethod
    def get_nested(cls, d: Any, ks: List[Any]) -> Any:
        """Read a value at arbitrary depth from a plain dict or pydantic model."""
        for k in ks:
            if isinstance(d, BaseModel):
                d = getattr(d, k)
            else:
                d = d[k]
        return d

    @classmethod
    def get_config(cls, next_run: RunInfo) -> str:
        return next_run.admin.config_path

    @classmethod
    def list_manipulation(
        cls,
        runs: Sequence[GenExp],
        sort_keys: list[list[str]] = [],
        reverse: bool = False,
        filter_keys: list[list[str]] = [],
        criterions: list[Callable[[Any], bool]] = [],
    ) -> Sequence[GenExp]:
        """Apply common list manipulation operations

        Args:
            runs (list[GenExp]): List of runs from any location
            sort_keys (list[list[str]], optional): List of key sets (indexing into Dict) to sort by. Defaults to [].
            reverse (bool, optional): Reverse after sort. Defaults to False.
            filter_keys (list[list[str]], optional): List of key sets (indexing into Dict) to filter by. Must match criterions. Defaults to [].
            criterions (list[Callable[[Any], bool]], optional): List of criterion to match against the values indexed by filter_keys. Defaults to [].

        Raises:
            ValueError: If filter keys are not paired with criterions

        Returns:
            list[GenExp]: _description_
        """
        if len(filter_keys) != len(criterions):
            raise ValueError("filter_key sets and criterions must be equal in length")
        elif len(filter_keys) > 0:
            for filter_key_set, crit in zip(filter_keys, criterions):
                if len(runs) == 0:
                    break

                runs = [
                    run for run in runs if crit(Que.get_nested(run, filter_key_set))
                ]

        if len(sort_keys) > 0:
            return sorted(
                runs,
                key=lambda x: tuple(
                    Que.get_nested(x, sort_key_set) for sort_key_set in sort_keys
                ),
                reverse=reverse,
            )
        elif reverse:
            return list(reversed(runs))

        return runs

    def get_val(self, run: GenExp, keys: List[str]) -> Any:
        with log_and_raise(self.logger, "get_nested"):
            return self.get_nested(run, keys)

    # -----------------------------------------------------------------------
    # Queue features
    # -----------------------------------------------------------------------

    # Direct indexing

    def create_run(
        self,
        arg_dict: AdminInfo,
        wandb_dict: WandbInfo,
        add_duplicates: bool = False,
    ) -> None:
        from configs import load_config

        with log_and_raise(self.logger, "create"):
            config: RunInfo = load_config(arg_dict)
            if self._is_dup_exp(config) and not add_duplicates:
                raise QueDupExp
            exp_info = ExpInfo.model_validate(
                {
                    **config.model_dump(),
                    "wandb": wandb_dict.model_dump(),
                }
            )
            self.to_run.append(exp_info)

    def add_run(
        self,
        arg_dict: AdminInfo,
        wandb_dict: WandbInfo,
        add_duplicates: bool = False,
    ) -> None:
        """Add a fully-tested completed run directly into old_runs."""
        from testing import full_test, load_comp_res
        from configs import get_model_exp_dir, get_model_results_dir, ZFILL, load_config

        with log_and_raise(self.logger, "add"):
            config: RunInfo = load_config(arg_dict)
            if self._is_dup_exp(config) and not add_duplicates:
                raise QueDupExp

            self.logger.debug(arg_dict.save_path[-ZFILL:])
            checknum = (
                int(arg_dict.save_path[-ZFILL:])
                if arg_dict.save_path[-1].isdigit()
                else None
            )
            res_dir = get_model_results_dir(
                get_model_exp_dir(
                    split=arg_dict.split,
                    model=arg_dict.model,
                    exp_no=int(arg_dict.exp_no),
                ),
                checkpoint_num=checknum,
            )

            try:
                results = load_comp_res(res_dir / "best_val_loss.json")
                self.logger.info("Successfully loaded results")
            except FileNotFoundError:
                results = full_test(admin=config.admin, data=config.data)
                self.logger.info("Results not found on disk — ran full_test")

            comp_run = CompExpInfo.model_validate(
                {
                    **config.model_dump(),
                    "wandb": wandb_dict.model_dump(),
                    "results": results
                    if isinstance(results, dict)
                    else results.model_dump(),
                }
            )
            self.old_runs.insert(0, comp_run)

    def recover_run(
        self,
        to_loc: QueLocation = TO_RUN,
        from_loc: QueLocation = CUR_RUN,
        index: int = 0,
        clean_slate: bool = False,
        enum_chck: bool = False,
    ) -> None:
        self.logger.debug(f"clean_slate is set to: {clean_slate}")
        with log_and_raise(self.logger, "recover"):
            run = self.peak_run(from_loc, index)

            if clean_slate:
                self.logger.debug("running _clean_slate")
                run = self._clean_slate(run, enum_chck)
            else:
                self.logger.debug("setting recover to True")
                # model_copy preserves the concrete subtype for the nested admin model
                run = run.model_copy(
                    update={"admin": run.admin.model_copy(update={"recover": True})}
                )

            if from_loc == FAIL_RUNS:
                # Strip the error field — re-validate as plain ExpInfo
                run = ExpInfo.model_validate(
                    {k: v for k, v in run.model_dump().items() if k != "error"}
                )
            elif not clean_slate and run.wandb.run_id is None:
                raise QueException("Run set to recover but no run_id present")

            _ = self._pop_run(from_loc, index)
            self._set_run(to_loc, 0, run)

        self.logger.info(
            f"Recovered Run: {self.run_str(to_loc, 0)} idx {index} from {from_loc} → {to_loc}"
        )

    def clear_runs(self, loc: QueLocation) -> None:
        to_clear = self._fetch_state(loc)
        with log_and_raise(self.logger, f"clear {loc}"):
            if len(to_clear) > 0:
                to_clear.clear()
            else:
                raise QueEmpty(loc)

    def remove_run(self, loc: QueLocation, idx: int) -> None:
        with log_and_raise(self.logger, "remove"):
            _ = self._pop_run(loc, idx)

    def shuffle(self, loc: QueLocation, o_idx: int, n_idx: int) -> None:
        with log_and_raise(self.logger, "shuffle"):
            self._set_run(loc, n_idx, self._pop_run(loc, o_idx))

    def _move(self, o_loc: QueLocation, n_loc: QueLocation, oi_idx: int) -> None:
        run = self.peak_run(o_loc, oi_idx)
        self._set_run(n_loc, 0, run)
        _ = self._pop_run(o_loc, oi_idx)

    def move(
        self,
        o_loc: QueLocation,
        n_loc: QueLocation,
        oi_idx: int,
        of_idx: Optional[int] = None,
    ) -> None:
        with log_and_raise(self.logger, "move"):
            if of_idx is None:
                self._move(o_loc, n_loc, oi_idx)
            else:
                old_location = self._fetch_state(o_loc)
                if oi_idx > of_idx:
                    oi_idx, of_idx = of_idx, oi_idx
                if abs(oi_idx) >= len(old_location) or abs(of_idx) >= len(old_location):
                    raise QueIdxOORR(o_loc, oi_idx, of_idx, len(old_location))
                for _ in range(oi_idx, of_idx + 1):
                    self._move(o_loc, n_loc, oi_idx)

    def edit_run(
        self,
        loc: QueLocation,
        idx: int,
        keys: List[str],
        value: Any,
        do_eval: bool = False,
    ) -> None:
        """Edit a single field (by key path) in a queued run.

        The run is dumped to a plain dict, mutated, then re-validated back to
        the appropriate pydantic model — so all field validators still run.
        """
        with log_and_raise(self.logger, "edit"):
            run = self.peak_run(loc, idx)
            val = ast.literal_eval(value) if do_eval else value

            run_dict = run.model_dump()
            run_dict = self.set_nested(run_dict, keys, val)

            if loc == FAIL_RUNS:
                new_run: GenExp = FailedExp.model_validate(run_dict)
            elif loc == OLD_RUNS:
                new_run = CompExpInfo.model_validate(run_dict)
            else:
                new_run = ExpInfo.model_validate(run_dict)

            _ = self._pop_run(loc, idx)
            self._set_run(loc, idx, new_run)

    # Indirect indexing

    def run_str(self, loc: QueLocation, idx: int) -> str:
        return self._run_to_str(self._run_sum(self.peak_run(loc, idx)))

    def list_runs(
        self, loc: QueLocation, **kwargs: Unpack[ListManipulationKwargs]
    ) -> ExpQue:
        return list(Que.list_manipulation(self._fetch_state(loc), **kwargs))

    def select_runs(
        self,
        loc: QueLocation,
        indexes: List[int],
        **kwargs: Unpack[ListManipulationKwargs],
    ) -> ExpQue:
        """Select runs by index after applying list manipulations."""
        runs = self.list_runs(loc, **kwargs)
        return [runs[i] for i in indexes]

    def place_runs(
        self,
        loc: QueLocation,
        runs: ExpQue,
        index: int = 0,
    ) -> None:
        """Insert runs by index. Uses 0 as default if runs is empty, and repeats last index up to lenght of runs"""
        with log_and_raise(self.logger, "place_runs"):
            for idx, run in enumerate(runs):
                self._set_run(loc, idx + index, run)

    @classmethod
    def summarise(cls, runs: ExpQue) -> List[Sumarised]:
        return [cls._run_sum(run) for run in runs]  # type: ignore[arg-type]

    def summarise_runs(
        self, loc: QueLocation, **kwargs: Unpack[ListManipulationKwargs]
    ) -> List[Sumarised]:
        return self.summarise(self.list_runs(loc, **kwargs))

    @classmethod
    def print_runs(cls, runs: List[Sumarised], exc: Optional[List[str]] = None) -> None:
        """Pretty-print an already-retrieved summary list (e.g. from a proxy)."""
        if len(runs) == 0:
            print("  No runs available")
            return

        stats = cls._get_print_stats(runs)
        has_results = runs[0].best_val_acc is not None
        has_error = isinstance(runs[0], SummarisedError)

        header_parts = [
            "Idx".ljust(5),
            "Run ID".ljust(stats.get("max_run_id_len", len("Run Id")) + 2),
            "Model".ljust(stats["max_model_len"] + 2),
            "Exp No".ljust(stats["max_exp_no_len"] + 2),
            "Dataset".ljust(stats["max_dataset_len"] + 2),
            "Split".ljust(stats["max_split_len"] + 2),
        ]
        if has_results:
            header_parts.append(
                "Best Val Acc".ljust(stats.get("max_best_val_acc_len", 4) + 2)
            )
            header_parts.append(
                "Best Val Loss".ljust(stats.get("max_best_val_loss_len", 4) + 2)
            )
        header_parts.append("Config Path".ljust(stats["max_config_path_len"] + 2))
        if has_error:
            header_parts.append("Error")

        if exc is not None:
            header_parts = [h for h in header_parts if h.strip().lower() not in exc]

        header = " | ".join(header_parts)
        print(header)
        print("-" * len(header))

        for i, run in enumerate(runs):
            row_parts = [
                str(i).ljust(5),
                (run.run_id if run.run_id is not None else "N/A").ljust(
                    stats.get("max_run_id_len", len("Run Id")) + 2
                ),
                run.model.ljust(stats["max_model_len"] + 2),
                run.exp_no.ljust(stats["max_exp_no_len"] + 2),
                run.dataset.ljust(stats["max_dataset_len"] + 2),
                run.split.ljust(stats["max_split_len"] + 2),
            ]
            if has_results:
                row_parts.append(
                    (
                        f"{run.best_val_acc:.4f}"
                        if run.best_val_acc is not None
                        else "N/A"
                    ).ljust(stats.get("max_best_val_acc_len", 4) + 2)
                )
                row_parts.append(
                    (
                        f"{run.best_val_loss:.4f}"
                        if run.best_val_loss is not None
                        else "N/A"
                    ).ljust(stats.get("max_best_val_loss_len", 4) + 2)
                )
            row_parts.append(run.config_path.ljust(stats["max_config_path_len"] + 2))
            if has_error and isinstance(run, SummarisedError):
                row_parts.append(run.error if run.error is not None else "N/A")

            if exc is not None:
                row_parts = [
                    r
                    for r, h in zip(row_parts, header_parts)
                    if h.strip().lower() not in exc
                ]
            print(" | ".join(row_parts))

    def disp_runs(
        self,
        loc: QueLocation,
        exc: Optional[List[str]] = None,
        **kwargs: Unpack[ListManipulationKwargs],
    ) -> None:
        self.print_runs(self.summarise_runs(loc, **kwargs), exc=exc)

    def disp_run(self, loc: QueLocation, idx: int) -> None:
        from configs import print_config

        print_config(self.peak_run(loc, idx))

    def copy_runs(
        self,
        o_loc: QueLocation,
        o_indexes: List[int],
        n_loc: QueLocation,
        n_idx: int = 0,
        clean_slate: bool = False,
        enum_chck: bool = True,
        **kwargs: Unpack[ListManipulationKwargs],
    ) -> None:
        with log_and_raise(self.logger, "copy"):
            runs = self.select_runs(o_loc, o_indexes, **kwargs)
            if clean_slate:
                runs = [self._clean_slate(run, enum_chck) for run in runs]

            self.place_runs(n_loc, runs, index=n_idx)

    # Meta features

    def find_runs(
        self, to_search: ExpQue, keys: List[str], criterion: Callable[[Any], bool]
    ) -> Tuple[List[int], List[GenExp]]:
        idxs, runs = [], []
        for i, run in enumerate(to_search):
            if criterion(self.get_val(run, keys)):
                idxs.append(i)
                runs.append(run)
        return idxs, runs  # type: ignore[return-value]

    def find_loc_runs(
        self,
        loc: QueLocation,
        key_set: List[List[str]],
        criterions: List[Callable[[Any], bool]],
    ) -> Tuple[List[int], List[GenExp]]:
        assert len(key_set) == len(criterions), (
            f"key_set length {len(key_set)} != criterions length {len(criterions)}"
        )
        runs: List[GenExp] = list(self._fetch_state(loc))  # type: ignore[arg-type]
        idxs: List[int] = []
        for k_lst, crit in zip(key_set, criterions):
            idxs, runs = self.find_runs(runs, k_lst, crit)  # type: ignore[arg-type]
        return idxs, runs

    def update_runs(self, key_set: List[str], transform: Callable[[Any], Any]) -> None:
        """Apply a transform to a nested field across every run in every location."""
        with log_and_raise(self.logger, "que.update_runs"):
            for run_list, model_cls in [
                (self.to_run, ExpInfo),
                (self.cur_run, ExpInfo),
                (self.fail_runs, FailedExp),
                (self.old_runs, CompExpInfo),
            ]:
                for idx, run in enumerate(run_list):
                    run_dict = run.model_dump()
                    current_val = self.get_nested(run_dict, key_set)
                    run_dict = self.set_nested(
                        run_dict, key_set, transform(current_val)
                    )
                    run_list[idx] = model_cls.model_validate(run_dict)  # type: ignore[index]


# ---------------------------------------------------------------------------
# Server state models
# ---------------------------------------------------------------------------

Worker_tasks: TypeAlias = Literal["inactive", "training", "testing"]


class WorkerStateDict(TypedDict):
    task: Worker_tasks
    current_run_id: Optional[str]
    working_pid: Optional[int]
    exception: Optional[str]


class DaemonStateDict(TypedDict):
    awake: bool
    stop_on_fail: bool
    supervisor_pid: Optional[int]


class ServerState(BaseModel):
    server_pid: Optional[int]
    daemon_state: DaemonStateDict
    worker_state: WorkerStateDict


def is_worker_state(obj: Any) -> TypeGuard[WorkerStateDict]:
    class WorkerState(BaseModel):
        task: Worker_tasks
        current_run_id: Optional[str]
        working_pid: Optional[int]
        exception: Optional[str]

    try:
        WorkerState.model_validate(obj)
        return True
    except Exception:
        return False


def is_daemon_state(obj: Any) -> TypeGuard[DaemonStateDict]:
    class DaemonState(BaseModel):
        awake: bool
        stop_on_fail: bool
        supervisor_pid: Optional[int]

    try:
        DaemonState.model_validate(obj)
        return True
    except Exception:
        return False


def is_server_state(obj: Any) -> TypeGuard[ServerState]:
    try:
        ServerState.model_validate(obj)
        return True
    except Exception:
        return False


def read_server_state(state_path: Union[Path, str] = SERVER_STATE_PATH) -> ServerState:
    """Load and validate ServerState from JSON.  Raises ValidationError if invalid."""
    with open(state_path, "r") as f:
        data = json.load(f)
    return ServerState.model_validate(data)


Process_states: TypeAlias = Union[WorkerStateDict, DaemonStateDict, ServerState]


# ---------------------------------------------------------------------------
# Protocols / Manager
# ---------------------------------------------------------------------------


class DaemonProtocol(Protocol):
    def start_supervisor(self) -> None: ...
    def stop_worker(
        self, timeout: Optional[float] = None, hard: bool = False
    ) -> None: ...
    def stop_supervisor(
        self,
        timeout: Optional[float] = None,
        hard: bool = False,
        stop_worker: bool = False,
    ) -> None: ...


class WorkerProtocol(Protocol):
    def cleanup(self) -> None: ...
    def start(self) -> None: ...


class ServerContextProtocol(Protocol):
    def save_state(self) -> None: ...
    def load_state(self) -> None: ...
    def get_state(self) -> ServerState: ...
    def set_state(
        self,
        server: Optional[ServerState],
        daemon: Optional[DaemonStateDict],
        worker: Optional[WorkerStateDict],
    ) -> None: ...


class QueManagerProtocol(Protocol):
    def get_que(self) -> Que: ...
    def get_daemon(self) -> DaemonProtocol: ...
    def get_worker(self) -> WorkerProtocol: ...
    def get_daemon_state(self) -> DaemonStateDict: ...
    def get_worker_state(self) -> WorkerStateDict: ...
    def get_server_context(self) -> ServerContextProtocol: ...


class QueManager(BaseManager):
    pass


def connect_manager(
    host="localhost", port=50000, authkey=b"abracadabra", max_retries=5, retry_delay=2
) -> "QueManagerProtocol":
    QueManager.register("get_que")
    QueManager.register("get_worker")
    QueManager.register("get_worker_state", proxytype=DictProxy)
    QueManager.register("get_daemon_state", proxytype=DictProxy)
    # QueManager.register("get_worker_state")
    # QueManager.register("get_daemon_state")
    QueManager.register("get_daemon")
    QueManager.register("get_server_context")

    for _ in range(max_retries):
        try:
            m = QueManager(address=(host, port), authkey=authkey)
            m.connect()
            return m  # type: ignore[return-value]
        except ConnectionRefusedError:
            print(f"Queue server not ready, retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    raise RuntimeError("Cannot connect to Queue server.")


def _get_basic_logger() -> Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=SERVER_LOG_PATH,
    )
    return logging.getLogger(__name__)


def main():
    logger = _get_basic_logger()
    q = Que(logger)
    q.disp_runs(OLD_RUNS)


if __name__ == "__main__":
    main()
