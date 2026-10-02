"""que
---
A lightweight in-memory queue manager for experiment configurations with
simple JSON-backed persistence.
"""

import ast
import functools
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from logging import Logger
from multiprocessing.managers import BaseManager, DictProxy
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Concatenate,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeGuard,
    TypeVar,
)

from pydantic import BaseModel, Field, TypeAdapter
from typing_extensions import TypedDict, Unpack

# locals
from src.run_types import (
    AVAIL_SPLITS,
    AdminInfo,
    CompExpInfo,
    ExpInfo,
    FailedExp,  # now defined in run_types
    RunInfo,
    Sumarised,
    SummarisedError,
    SummarisedRes,
    WandbInfo,
    strict_validate,
    # ENTITY
)

# from configs import print_config, load_config, ZFILL, get_model_exp_dir, get_model_results_dir


# ---------------------------------------------------------------------------
# Constants and types
# ---------------------------------------------------------------------------
SYSTEMD_NAME = "que-training.service"
QUE_DIR = Path(__file__).parent

QUE_NAME = "Que"
DAEMON_NAME = "Daemon"
WORKER_NAME = "Worker"
SERVER_NAME = "Server"
TRAINING_NAME = "Training"

# data and logs are kept apart from the code (both directories are gitignored)
STATE_DIR = QUE_DIR / "state"
LOG_DIR = QUE_DIR / "logs"

RUN_PATH = STATE_DIR / "Runs.json"
SERVER_STATE_PATH = STATE_DIR / "Server.json"

TRAINING_LOG_PATH = LOG_DIR / "Training.log"
SERVER_LOG_PATH = LOG_DIR / "Server.log"

ARCHIVE_DIR = STATE_DIR / "old_ques"

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


GenExp: TypeAlias = ExpInfo | FailedExp | CompExpInfo
ExpQue: TypeAlias = list[ExpInfo] | list[FailedExp] | list[CompExpInfo]


class AllRuns(BaseModel):
    old_runs: list[CompExpInfo]
    cur_run: list[ExpInfo]
    to_run: list[ExpInfo]
    fail_runs: list[FailedExp]


class PositionInfo(BaseModel):
    location: QueLocation
    index: int


class RangePosition(PositionInfo):
    index2: int


class SortInfo(BaseModel):
    key_set: list[str]
    reverse: bool


NO_SORT = SortInfo(key_set=[], reverse=False)

# Specification for filtering results


class Specification(BaseModel):
    training: dict[str, int]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def migrate_legacy_files(
    old_dir: Path = QUE_DIR, state_dir: Path = STATE_DIR, log_dir: Path = LOG_DIR
) -> list[str]:
    """Move the Que's data and log files from `old_dir`, where they lived beside the code until
    2026-10-02, into `state_dir`/`log_dir` (creating those). Idempotent.

    A file already at its new location is never overwritten: the old one is left in place and
    reported. Run only by the server at startup (see server.start_server), the files' owner --
    never by a shell, which could move files out from under a server still on the old layout.

    Returns:
        list[str]: One message per file moved or left in place, to log once logging is set up
            (it can't be before: the logs are among the files moved).
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    names = [
        *((n, state_dir) for n in ("Runs.json", "Server.json", "old_ques")),
        *((p.name, state_dir) for p in sorted(old_dir.glob("Runs_*.json"))),
        *((p.name, state_dir) for p in sorted(old_dir.glob("Server_*.json"))),
        *((n, log_dir) for n in ("Server.log", "Training.log")),
    ]
    messages = []
    for name, new_dir in names:
        old, new = old_dir / name, new_dir / name
        if not old.exists():
            continue
        if new.exists():
            messages.append(f"Not migrating {old}: {new} already exists")
            continue
        old.replace(new)
        messages.append(f"Migrated {old} -> {new}")
    return messages


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
QUE_LOGGERS = (QUE_NAME, DAEMON_NAME, SERVER_NAME, WORKER_NAME)


def add_file_handler(logger: Logger, path: str | Path) -> None:
    """Attach a handler writing `logger`'s records to `path` in LOG_FORMAT, unless `logger`
    already has one for `path`."""
    path = Path(path).resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == path:
            return
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)


def setup_server_logging(path: str | Path = SERVER_LOG_PATH) -> None:
    """Send this process's logging to `path` (default: Server.log): the Que system's own loggers
    (QUE_LOGGERS) at DEBUG, everything else (e.g. wandb) at INFO.

    Call it once in each server-side process -- the server, and the supervisor and worker it
    spawns, which start with no logging config. A single handler on the root logger means each
    record is written once. Calling it again in the same process changes nothing. Importing this
    module configures nothing, so other users of the Que (e.g. src/results) don't log here.
    """
    root = logging.getLogger()
    add_file_handler(root, path)
    root.setLevel(logging.INFO)
    # a record is checked against its own logger's level only, so these reach the root's
    # handler at DEBUG while other libraries' DEBUG records are dropped
    for name in QUE_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG)


LogName: TypeAlias = Literal["server", "training"]
LOG_PATHS: dict[LogName, Path] = {"server": SERVER_LOG_PATH, "training": TRAINING_LOG_PATH}
_LOG_TAIL_BYTES = 1024 * 1024


def read_log(path: str | Path, start: int | None = None, n: int = 10) -> tuple[str, int]:
    """Read a log file in a way that lets a caller follow it (like `tail -f`) by polling.

    Args:
        path (str | Path): The log file. A missing file reads as empty.
        start (int | None, optional): Byte offset to read on from (the offset a previous call
            returned), or None for the file's last `n` lines, searched for in its last 1 MiB.
            If the file is now shorter than `start` (rotated or cleared), it's read from the
            beginning. Defaults to None.
        n (int, optional): Number of lines when `start` is None. Defaults to 10.

    Returns:
        tuple[str, int]: The text read, and the offset to pass as `start` next time.
    """
    path = Path(path)
    if not path.exists():
        return "", 0
    with open(path, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        if start is None:
            f.seek(max(0, size - _LOG_TAIL_BYTES))
            lines = f.read().splitlines(keepends=True)[-n:] if n > 0 else []
            return b"".join(lines).decode(errors="replace"), size
        offset = start if start <= size else 0
        f.seek(offset)
        data = f.read()
    return data.decode(errors="replace"), offset + len(data)


def setup_training_logging(path: str | Path = TRAINING_LOG_PATH) -> Logger:
    """The Training logger (training/testing output, see worker.LoggerWriter), writing at INFO to
    `path` (default: Training.log) only, not to Server.log. Calling it again changes nothing."""
    logger = logging.getLogger(TRAINING_NAME)
    add_file_handler(logger, path)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
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
    """Index out of range for a location.

    `filtered` marks that `leng` is the length of a filtered view of `loc`, not of
    `loc` itself, so the message doesn't misreport the location's real size.
    """

    def __init__(self, loc: QueLocation, idx: int, leng: int, filtered: bool = False):
        self.loc = loc
        self.idx = idx
        self.length = leng
        self.filtered = filtered
        view = f"{loc} after filtering" if filtered else loc
        self.message = f"Index {idx} is out of range for {view} (length: {leng})"
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def __reduce__(self):
        return (self.__class__, (self.loc, self.idx, self.length, self.filtered))


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


class NoSweepSet(QueException):
    def __init__(self, message: str = "No sweep is currently set"):
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


def make_timestamp() -> str:
    """The current time, formatted for snapshot file names (see timestamp_path)."""
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")  # noqa: DTZ005


def timestamp_path(path: str | Path, stamp: str | None = None) -> str:
    """`path` with `_<stamp>` (default: make_timestamp()) inserted before its .json suffix."""
    stamp = make_timestamp() if stamp is None else stamp
    return str(path).replace(".json", f"_{stamp}.json")


def atomic_write_json(path: str | Path, data: Any, indent: int | None = None) -> None:
    """Write `data` to `path` as JSON, such that `path` always holds either the old or the new
    contents in full.

    Writing in place would leave a truncated, unloadable file if the process died (e.g. a power
    outage) mid-write. Instead the JSON goes to a temp file beside `path`, is fsynced, then
    renamed over `path`. An existing file's permissions are kept.
    """
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        tmp.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    # make the rename itself durable
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _logged(method: Callable[Concatenate["Que", _P], _R]) -> Callable[Concatenate["Que", _P], _R]:
    """Log a Que operation's outcome to its logger, under the method's name: success at INFO, a
    QueException (an expected, user-level error, e.g. an index out of range from the shell) as one
    WARNING line, and anything else at ERROR with its traceback. The exception is re-raised.

    An exception is only logged by the first `_logged` method it passes through, so one raised in
    a nested operation (e.g. copy_runs -> place_runs) isn't logged twice.
    """

    @functools.wraps(method)
    def wrapper(self: "Que", *args: _P.args, **kwargs: _P.kwargs) -> _R:
        name = method.__name__
        try:
            result = method(self, *args, **kwargs)
        except Exception as e:
            if not getattr(e, "_que_logged", False):
                if isinstance(e, QueException):
                    self.logger.warning(f"{name} failed: {e}")
                else:
                    self.logger.exception(f"{name} failed")
                e._que_logged = True  # type: ignore[attr-defined]
            raise
        self.logger.info(f"{name} completed successfully")
        return result

    return wrapper


def _persists(method: Callable[Concatenate["Que", _P], _R]) -> Callable[Concatenate["Que", _P], _R]:
    """Mark a Que method as a mutation: it runs under the Que's lock and is saved to disk (when
    `auto_save` is on) as soon as it returns or raises, so disk always mirrors memory.

    Only the outermost mutation saves, so one that calls others (e.g. replace_cur_run popping
    then setting cur_run) is never saved half-done. Saving on a raise too keeps
    partial changes from a failed mutation (e.g. add_new_run's fail_runs fallback) as well.
    """

    @functools.wraps(method)
    def wrapper(self: "Que", *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self._lock:
            self._mutation_depth += 1
            try:
                return method(self, *args, **kwargs)
            finally:
                self._mutation_depth -= 1
                if self._mutation_depth == 0 and self.auto_save:
                    self.save_state()

    return wrapper


# ---------------------------------------------------------------------------
# Que class
# ---------------------------------------------------------------------------


class Que:
    """The run queue: runs waiting (to_run), running (cur_run), finished (old_runs) and failed
    (fail_runs), persisted to `runs_path`.

    In the server, one Que is shared by the Worker, Daemon and Shell through the manager, which
    serves each connection on its own thread. So every mutating method is wrapped in `_persists`:
    mutations are serialised by a lock and, with `auto_save`, saved before they return -- the
    server may die at any moment, and nothing should then exist only in memory. Callers never
    need to call save_state themselves, except to write a copy elsewhere.
    """

    def __init__(
        self,
        logger: Logger | None = None,
        runs_path: str | Path = RUN_PATH,
        auto_save: bool = True,
    ) -> None:
        """
        Args:
            logger (Logger | None, optional): Defaults to the Que logger, which only writes
                somewhere once logging is set up (see setup_server_logging).
            runs_path (str | Path, optional): Where the Que is loaded from and saved to. Defaults
                to RUN_PATH, the live server's file.
            auto_save (bool, optional): Save to `runs_path` after every mutation. Turn off for a
                scratch copy of a Que you don't want written back. Defaults to True.
        """
        self.runs_path: Path = Path(runs_path)
        self.old_runs: list[CompExpInfo] = []
        self.cur_run: list[ExpInfo] = []
        self.to_run: list[ExpInfo] = []
        self.fail_runs: list[FailedExp] = []
        self.auto_save: bool = auto_save
        self.logger = logger if logger is not None else logging.getLogger(QUE_NAME)
        self._lock = threading.RLock()
        self._mutation_depth = 0
        self.load_state()

    def __getstate__(self) -> dict[str, Any]:
        # The Daemon and Worker (each holding a Que) are pickled into spawned processes, and locks
        # can't be pickled. A copy gets a fresh lock; the processes use the manager's Que proxy anyway.
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

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
        """Insert `run` at `idx` in `loc`, enforcing the run type each location is loaded back as.

        to_run/cur_run only take a plain ExpInfo: a FailedExp's error or a CompExpInfo's results
        there would make the saved Que fail validation on the next load. Use recover_run or
        copy_runs(clean_slate=True) to requeue one.
        """
        if loc in (TO_RUN, CUR_RUN) and type(run) is not ExpInfo:
            raise TypeError(
                f"{loc} requires a plain ExpInfo, got {type(run).__name__}: requeue it with "
                "recover_run or copy_runs(clean_slate=True) instead"
            )
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
    def _run_sum(cls, run: GenExp, ndigits: int | None = None) -> Sumarised:
        """Extract a compact summary from a run model. Optionally round to ndigits"""
        run_id = run.wandb.run_id if isinstance(run, ExpInfo) and run.wandb else None

        if isinstance(run, CompExpInfo):
            acc = run.results.best_val_acc
            loss = run.results.best_val_loss

            best_val_acc = round(acc, ndigits) if ndigits else acc
            best_val_loss = round(loss, ndigits) if ndigits else loss
        else:
            best_val_acc, best_val_loss = None, None

        base = Sumarised(
            model=run.admin.model,
            exp_no=run.admin.exp_no,
            dataset=run.admin.dataset,
            split=run.admin.split,
            config_path=run.admin.config_path,
            run_id=run_id,
            best_val_acc=best_val_acc,
            best_val_loss=best_val_loss,
        )

        if cls._is_failed_exp(run):
            return SummarisedError(**base.model_dump(), error=run.error)
        elif cls._is_comp_exp_info(run):
            test = run.results.test
            return SummarisedRes(
                **base.model_dump(),
                test_top1_acc=round(test.top_k_per_instance_acc.top1 * 100, ndigits),
                test_av_loss=round(test.average_loss, ndigits),
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
        from src.configs import ZFILL
        from src.utils import enum_dir

        if enum_chck:
            # probably copying a run, so make sure the previous save path is created otherwise enum_dir will fail
            Path(run.admin.save_path).mkdir(parents=True, exist_ok=True)
            new_save_path = str(enum_dir(run.admin.save_path, decimals=ZFILL))
        else:
            new_save_path = run.admin.save_path

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
    def _get_print_stats(cls, runs: list[Sumarised]) -> dict[str, int]:
        stats: dict[str, int] = {
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

    def load_state(self, in_path: str | Path | None = None):
        """Load the queue state from a JSON file. If the file does not exist, start with an empty queue.

        Loading from a file other than `runs_path` saves the loaded state to `runs_path` (with
        `auto_save`), as for any other change to the Que.

        Args:
            in_path (str | Path | None, optional): The path to the JSON file containing the queue state. Defaults to None.
        """
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
        except FileNotFoundError:
            self.logger.warning(
                f"No existing state found at {in_path}. Starting fresh."
            )
            data = {}
        # validate everything before replacing anything, so a bad file leaves the Que untouched
        to_run = [ExpInfo.model_validate(r) for r in data.get(TO_RUN, [])]
        cur_run = [ExpInfo.model_validate(r) for r in data.get(CUR_RUN, [])]
        old_runs = [CompExpInfo.model_validate(r) for r in data.get(OLD_RUNS, [])]
        fail_runs = [FailedExp.model_validate(r) for r in data.get(FAIL_RUNS, [])]
        with self._lock:
            self.to_run, self.cur_run = to_run, cur_run
            self.old_runs, self.fail_runs = old_runs, fail_runs
        if data:
            self.logger.info(f"Loaded que state from {in_path}")
        if Path(in_path) != self.runs_path and self.auto_save:
            self.save_state()

    def save_state(
        self,
        out_path: str | Path | None = None,
        timestamp: str | None = None,
        archive: bool = False,
    ):
        """Save the state of the Que to a json file (atomically, see atomic_write_json).

        Mutations already save to `runs_path` themselves (see Que), so this is only needed to
        write a copy elsewhere.

        Args:
            out_path (str | Path | None, optional): The output path. Defaults to `runs_path`.
            timestamp (str | None, optional): Insert this timestamp (see make_timestamp) into the
                output file name. Defaults to None.
            archive (bool, optional): Whether to archive the output file. Defaults to False.
        """

        if out_path is None:
            out_path = self.runs_path
        else:
            out_path = Path(out_path)
            if out_path.exists() and timestamp is None:
                self.logger.warning(f"Overwriting existing state file: {out_path}")

        if archive:
            out_path = ARCHIVE_DIR / out_path.name

        if timestamp is not None:
            out_path = timestamp_path(out_path, timestamp)

        with self._lock:
            all_runs = {
                TO_RUN: [r.model_dump() for r in self.to_run],
                CUR_RUN: [r.model_dump() for r in self.cur_run],
                OLD_RUNS: [r.model_dump() for r in self.old_runs],
                FAIL_RUNS: [r.model_dump() for r in self.fail_runs],
            }
        atomic_write_json(out_path, all_runs, indent=4)
        # the automatic save after every mutation is routine; an explicit copy is worth noting
        level = logging.DEBUG if out_path == self.runs_path else logging.INFO
        self.logger.log(level, f"Saved que to {out_path}")

    # -----------------------------------------------------------------------
    # Worker / Daemon helpers
    # -----------------------------------------------------------------------

    def len_loc(self, loc: QueLocation) -> int:
        return len(self._fetch_state(loc))

    def len_sweep_runs(self, sweep_id: str, loc: QueLocation = OLD_RUNS) -> int:
        """Number of runs in `loc` that are trials of wandb sweep `sweep_id`.

        The default, old_runs, counts the sweep's finished trials (trained and tested, including
        ones wandb stopped early): the source of truth for its progress (see ServerContext).
        """
        return len(
            self.list_runs(
                loc, filter_keys=[["wandb", "sweep_id"]], criterions=[lambda s: s == sweep_id]
            )
        )

    def peak_run(self, loc: QueLocation, idx: int) -> GenExp:
        to_get = self._fetch_state(loc)
        if len(to_get) == 0:
            raise QueEmpty(loc)
        elif abs(idx) >= len(to_get):
            raise QueIdxOOR(loc, idx, len(to_get))
        return to_get[idx]

    def peak_cur_run(self) -> ExpInfo:
        return self.peak_run(CUR_RUN, 0)  # type: ignore[return-value]

    @_persists
    def pop_cur_run(self) -> ExpInfo:
        return self._pop_run(CUR_RUN, 0)  # type: ignore[return-value]

    @_persists
    def set_cur_run(self, run: ExpInfo) -> None:
        self._set_run(CUR_RUN, 0, run)

    @_persists
    def replace_cur_run(self, run: ExpInfo) -> None:
        """Swap the run in cur_run for an updated copy (e.g. with its wandb run id), as one
        mutation -- so no save can catch cur_run empty in between."""
        _ = self.pop_cur_run()
        self.set_cur_run(run)

    @_persists
    def stash_next_run(self) -> str:
        next_run = self._pop_run(TO_RUN, 0)
        sum_str = self._run_to_str(self._run_sum(next_run))
        try:
            self.set_cur_run(next_run)  # type: ignore[arg-type]
            self.logger.info(f"Stashed new run: {sum_str}")
        except QueBusy:
            self.logger.error(f"Failed to stash new run: {sum_str}")
            self._set_run(TO_RUN, 0, next_run)
            raise
        return sum_str

    @_persists
    def store_fin_run(self, comp_run: CompExpInfo) -> None:
        """Replace the run in cur_run with its completed (tested) copy `comp_run`, in old_runs.

        Raises:
            QueEmpty: If cur_run is empty.
            TypeError: If `comp_run` isn't a CompExpInfo.
        """
        if not self._is_comp_exp_info(comp_run):
            raise TypeError("store_fin_run requires a CompExpInfo")
        _ = self.pop_cur_run()
        self._set_run(OLD_RUNS, 0, comp_run)
        self.logger.info("Stored finished run")

    @_persists
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
        cls, d: dict[Any, Any], k: Any, ks: list[Any], val: Any
    ) -> dict[Any, Any]:
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
    def set_nested(cls, d: dict[Any, Any], ks: list[Any], val: Any) -> dict[Any, Any]:
        """Set a value at an arbitrary depth in a plain dict using a key path."""
        if len(ks) == 0:
            return val
        return cls._set_inplace(d, ks[0], ks[1:], val)

    @classmethod
    def get_nested(cls, d: Any, ks: list[Any]) -> Any:
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
    def get_nested_or_none(cls, d: Any, ks: list[Any]) -> Any:
        """Attempt to read a value at arbitrary depth from a plain dict or pydantic model. Return None if any key is not found."""
        try:
            return cls.get_nested(d, ks)
        except (KeyError, AttributeError, TypeError):
            return None

    @classmethod
    def _filter_indexed_runs(
        cls,
        og_indexes: list[int],
        to_search: ExpQue,
        keys: list[str],
        criterion: Callable[[Any], bool],
    ) -> tuple[list[int], list[GenExp]]:
        """Keep the runs (and their paired original indexes) whose value at `keys`
        satisfies `criterion`. Missing keys are passed to `criterion` as None."""
        idxs, runs = [], []
        for i, run in zip(og_indexes, to_search):
            if criterion(cls.get_nested_or_none(run, keys)):
                idxs.append(i)
                runs.append(run)
        return idxs, runs

    @classmethod
    def indexed_list_manipulation(
        cls,
        runs: Sequence[GenExp],
        sort_keys: list[list[str]] | None = None,
        reverse: bool = False,
        filter_keys: list[list[str]] | None = None,
        criterions: list[Callable[[Any], bool]] | None = None,
    ) -> tuple[list[int], Sequence[GenExp]]:
        """Apply common list manipulation operations

        Args:
            runs (Sequence[GenExp]): exp configs from any location
            sort_keys (list[list[str]] | None, optional): List of key sets (indexing into Dict) to sort by. Defaults to None.
            reverse (bool, optional): Reverse after sort. Defaults to False.
            filter_keys (list[list[str]] | None, optional): List of key sets (indexing into Dict) to filter by. Must match criterions. Defaults to None.
            criterions (list[Callable[[Any], bool]] | None, optional): List of criterion to match against the values indexed by filter_keys. Defaults to None.

        Raises:
            ValueError: If filter keys are not paired with criterions

        Returns:
            tuple[list[int], Sequence[GenExp]]: original indexes, Filtered and/or sorted runs
        """
        # Set defaults
        if criterions is None:
            criterions = []
        if filter_keys is None:
            filter_keys = []
        if sort_keys is None:
            sort_keys = []

        # Preserved indexes
        original_indexes = list(range(len(runs)))

        # Filter
        if len(filter_keys) != len(criterions):
            raise ValueError("filter_key sets and criterions must be equal in length")
        elif len(filter_keys) > 0:
            for filter_key_set, crit in zip(filter_keys, criterions):
                if len(runs) == 0:
                    break

                original_indexes, runs = cls._filter_indexed_runs(
                    original_indexes, list(runs), filter_key_set, crit
                )

        # Sort
        if len(sort_keys) > 0:
            idx_runs = sorted(
                zip(original_indexes, runs),
                key=lambda x: tuple(
                    Que.get_nested(x[1], sort_key_set) for sort_key_set in sort_keys
                ),
                reverse=reverse,
            )
            return [x[0] for x in idx_runs], [x[1] for x in idx_runs]
        elif reverse:
            return list(reversed(original_indexes)), list(reversed(runs))
        else:
            return original_indexes, runs

    @classmethod
    def list_manipulation(
        cls,
        runs: Sequence[GenExp],
        **kwargs: Unpack[ListManipulationKwargs],
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
            list[GenExp]: Filtered and/or sorted runs
        """
        return cls.indexed_list_manipulation(runs, **kwargs)[1]

    # -----------------------------------------------------------------------
    # Queue features
    # -----------------------------------------------------------------------

    # Direct indexing

    @_logged
    @_persists
    def add_new_run(
        self,
        config: RunInfo,
        wandb_dict: WandbInfo,
        loc: Literal["to_run", "cur_run"] = TO_RUN,
        ndigits: int | None = 2,
    ) -> None:
        """Add a new run the the Que"""

        exp_info = ExpInfo.model_validate(
            {
                **config.model_dump(),
                "wandb": wandb_dict.model_dump(),
            }
        )
        if loc == TO_RUN:
            self.to_run.append(exp_info)
        elif loc == CUR_RUN:
            if len(self.cur_run) != 0:
                self.logger.error(
                    "Cannot add to cur_run: already occupied, added to fail_runs instead"
                )
                self.fail_runs.append(
                    FailedExp.model_validate(
                        {
                            **exp_info.model_dump(),
                            "error": "Attempted to add to cur_run but it was already occupied",
                        }
                    )
                )
                raise QueBusy
            self.cur_run.append(exp_info)
        else:
            raise ValueError(
                f"Invalid location: {loc}. Must be 'to_run' or 'cur_run'."
            )

        self.logger.debug(
            f"Added new run: {self._run_to_str(self._run_sum(exp_info, ndigits))}"
        )

    @_logged
    @_persists
    def create_run(
        self,
        arg_dict: AdminInfo,
        wandb_dict: WandbInfo,
        add_duplicates: bool = False,
    ) -> None:
        """Load and add a new run to the Que"""
        from src.configs import load_config

        config: RunInfo = load_config(arg_dict)
        if self._is_dup_exp(config) and not add_duplicates:
            raise QueDupExp

        self.add_new_run(config, wandb_dict)

    @_logged
    def add_run(
        self,
        arg_dict: AdminInfo,
        wandb_dict: WandbInfo,
        add_duplicates: bool = False,
    ) -> None:
        """Add a fully-tested completed run directly into old_runs.

        Not itself a `_persists` mutation, so the Que isn't locked through a possible full_test;
        only the final insert (via place_runs) is.
        """
        from src.configs import (
            ZFILL,
            get_model_exp_dir,
            get_model_results_dir,
            load_config,
        )
        from src.testing import full_test, load_comp_res

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
            self.logger.info("Results not found on disk — run full_test")

        comp_run = CompExpInfo.model_validate(
            {
                **config.model_dump(),
                "wandb": wandb_dict.model_dump(),
                "results": results
                if isinstance(results, dict)
                else results.model_dump(),
            }
        )
        self.place_runs(OLD_RUNS, [comp_run])

    @_logged
    @_persists
    def recover_run(
        self,
        to_loc: QueLocation = TO_RUN,
        from_loc: QueLocation = CUR_RUN,
        index: int = 0,
        clean_slate: bool = False,
        enum_chck: bool = False,
    ) -> None:
        self.logger.debug(f"clean_slate is set to: {clean_slate}")
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

    @_logged
    @_persists
    def clear_runs(self, loc: QueLocation) -> None:
        to_clear = self._fetch_state(loc)
        if len(to_clear) > 0:
            to_clear.clear()
        else:
            raise QueEmpty(loc)

    @_logged
    @_persists
    def remove_run(self, loc: QueLocation, idx: int) -> None:
        _ = self._pop_run(loc, idx)

    @_logged
    @_persists
    def shuffle(self, loc: QueLocation, o_idx: int, n_idx: int) -> None:
        self._set_run(loc, n_idx, self._pop_run(loc, o_idx))

    def _move(self, o_loc: QueLocation, n_loc: QueLocation, oi_idx: int) -> None:
        run = self.peak_run(o_loc, oi_idx)
        self._set_run(n_loc, 0, run)
        _ = self._pop_run(o_loc, oi_idx)

    @_logged
    @_persists
    def move(
        self,
        o_loc: QueLocation,
        n_loc: QueLocation,
        oi_idx: int,
        of_idx: int | None = None,
    ) -> None:
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

    @_logged
    @_persists
    def edit_run(
        self,
        loc: QueLocation,
        idx: int,
        keys: list[str],
        value: Any,
        do_eval: bool = False,
    ) -> None:
        """Edit a single field (by key path) in a queued run.

        The run is dumped to a plain dict, mutated, then re-validated back to
        the appropriate pydantic model — so all field validators still run.
        """
        run = self.peak_run(loc, idx)
        val = ast.literal_eval(value) if do_eval else value

        run_dict = run.model_dump()
        run_dict = self.set_nested(run_dict, keys, val)

        if loc == FAIL_RUNS:
            run_type = FailedExp
        elif loc == OLD_RUNS:
            run_type = CompExpInfo
        else:
            run_type = ExpInfo

        new_run = strict_validate(run_type, run_dict)

        _ = self._pop_run(loc, idx)
        self._set_run(loc, idx, new_run)

    # Indirect indexing

    def run_str(self, loc: QueLocation, idx: int, ndigits: int | None = None) -> str:
        return self._run_to_str(self._run_sum(self.peak_run(loc, idx), ndigits))

    def list_runs(
        self, loc: QueLocation, **kwargs: Unpack[ListManipulationKwargs]
    ) -> ExpQue:
        return list(Que.list_manipulation(self._fetch_state(loc), **kwargs))

    def select_runs(
        self,
        loc: QueLocation,
        indexes: list[int],
        **kwargs: Unpack[ListManipulationKwargs],
    ) -> ExpQue:
        """Select runs by index after applying list manipulations."""
        runs = self._fetch_state(loc)
        return [runs[i] for i in self.select_indexes(loc, runs, indexes, **kwargs)]

    @classmethod
    def select_indexes(
        cls,
        loc: QueLocation,
        runs: Sequence[GenExp],
        indexes: list[int],
        **kwargs: Unpack[ListManipulationKwargs],
    ) -> list[int]:
        """Map indexes into the manipulated (filtered/sorted) view of `runs` back to
        indexes into `runs` itself.

        `loc` is only used to label errors. Raises QueIdxOOR if any index falls
        outside the manipulated view.
        """
        idxs, _ = cls.indexed_list_manipulation(runs, **kwargs)
        filtered = bool(kwargs.get("filter_keys"))
        for i in indexes:
            if not -len(idxs) <= i < len(idxs):
                raise QueIdxOOR(loc, i, len(idxs), filtered)
        return [idxs[i] for i in indexes]

    @_logged
    @_persists
    def place_runs(
        self,
        loc: QueLocation,
        runs: ExpQue,
        index: int = 0,
    ) -> None:
        """Insert runs by index. Uses 0 as default if runs is empty, and repeats last index up to lenght of runs.
        `NOTE:` This method is unsafe and will drop runs if there is an error.

        """
        for idx, run in enumerate(runs):
            self._set_run(loc, idx + index, run)

    @classmethod
    def summarise(cls, runs: ExpQue, ndigits: int | None = None) -> list[Sumarised]:
        return [cls._run_sum(run, ndigits) for run in runs]  # type: ignore[arg-type]

    def summarise_runs(
        self,
        loc: QueLocation,
        ndigits: int | None = None,
        **kwargs: Unpack[ListManipulationKwargs],
    ) -> list[Sumarised]:
        return self.summarise(self.list_runs(loc, **kwargs), ndigits=ndigits)

    @classmethod
    def print_runs(cls, runs: list[Sumarised], exc: list[str] | None = None) -> None:
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
        exc: list[str] | None = None,
        **kwargs: Unpack[ListManipulationKwargs],
    ) -> None:
        self.print_runs(self.summarise_runs(loc, **kwargs), exc=exc)

    def disp_run(self, loc: QueLocation, idx: int) -> None:
        from src.configs import print_config

        print_config(self.peak_run(loc, idx))

    @_logged
    @_persists
    def copy_runs(
        self,
        o_loc: QueLocation,
        o_indexes: list[int],
        n_loc: QueLocation,
        n_idx: int = 0,
        clean_slate: bool = False,
        enum_chck: bool = True,
        **kwargs: Unpack[ListManipulationKwargs],
    ) -> None:
        runs = self.select_runs(o_loc, o_indexes, **kwargs)
        if clean_slate:
            runs = [self._clean_slate(run, enum_chck) for run in runs]

        self.place_runs(n_loc, runs, index=n_idx)

    # Meta features

    @_logged
    @_persists
    def update_runs(self, key_set: list[str], transform: Callable[[Any], Any]) -> None:
        """Apply a transform to a nested field across every run in every location."""
        for run_list, model_cls in [
            (self.to_run, ExpInfo),
            (self.cur_run, ExpInfo),
            (self.fail_runs, FailedExp),
            (self.old_runs, CompExpInfo),
        ]:
            for idx, run in enumerate(run_list):
                run_dict = run.model_dump()
                run_dict = self.set_nested(
                    run_dict, key_set, transform(self.get_nested(run_dict, key_set))
                )
                run_list[idx] = model_cls.model_validate(run_dict)  # type: ignore[index]


# ---------------------------------------------------------------------------
# Server state models
# ---------------------------------------------------------------------------

Worker_tasks: TypeAlias = Literal["inactive", "training", "testing"]


# Maintained TypedDict for dictproxy in basemanager
class WorkerStateDict(TypedDict):
    task: Annotated[Worker_tasks, Field(default="inactive")]
    current_run_id: Annotated[str | None, Field(default=None)]
    working_pid: Annotated[int | None, Field(default=None)]
    exception: Annotated[str | None, Field(default=None)]


_worker_state_adapter = TypeAdapter(WorkerStateDict)


def worker_state_validate(obj: Any) -> WorkerStateDict:
    """Validate/default `obj` (a dict, JSON payload, or manager `DictProxy`) as a `WorkerStateDict`."""
    return _worker_state_adapter.validate_python(obj)


def clear_worker_process(state: WorkerStateDict) -> None:
    """Mark the worker as not running (in place, so it works on a manager `DictProxy`).

    `exception` is kept so the last failure stays inspectable after the worker has exited.
    """
    state["task"] = "inactive"
    state["current_run_id"] = None
    state["working_pid"] = None


class SweepInfo(TypedDict):
    sweep_id: str
    sweep_project: str
    sweep_entity: str
    model: str
    dataset: str
    split: AVAIL_SPLITS
    base_config: str
    max_runs: int | None


_sweep_info_adapter = TypeAdapter(SweepInfo)


def sweep_info_validate(obj: Any) -> SweepInfo:
    """Validate `obj` (a dict, JSON payload, or manager `DictProxy`) as a fully-populated `SweepInfo`.

    Unlike `worker_state_validate`/`daemon_state_validate`, there is no partial/defaulted form of a
    `SweepInfo` -- "no sweep configured" is represented one level up as `SweepInfo | None`/`| dict`,
    so every field here is required.
    """
    return _sweep_info_adapter.validate_python(obj)


def is_sweep_complete(max_runs: int | None, completed_runs: int) -> bool:
    """Whether a sweep has run all its trials. A complete sweep stays set (so raising max_runs
    resumes it); the Daemon just stops handing it to the Worker."""
    return max_runs is not None and completed_runs >= max_runs


class DaemonStateDict(TypedDict):
    awake: Annotated[bool, Field(default=False)]
    stop_on_fail: Annotated[bool, Field(default=True)]
    supervisor_pid: Annotated[int | None, Field(default=None)]


_daemon_state_adapter = TypeAdapter(DaemonStateDict)


def daemon_state_validate(obj: Any) -> DaemonStateDict:
    """Validate/default `obj` (a dict, JSON payload, or manager `DictProxy`) as a `DaemonStateDict`."""
    return _daemon_state_adapter.validate_python(obj)


class SweepProgressDict(TypedDict):
    completed_runs: Annotated[int, Field(default=0)]


_sweep_progress_adapter = TypeAdapter(SweepProgressDict)


def sweep_progress_validate(obj: Any) -> SweepProgressDict:
    """Validate/default `obj` (a dict, JSON payload, or manager `DictProxy`) as a `SweepProgressDict`."""
    return _sweep_progress_adapter.validate_python(obj)


class ServerState(BaseModel):
    server_pid: int | None = None
    sweep: SweepInfo | dict = {}
    daemon_state: DaemonStateDict = daemon_state_validate({})
    worker_state: WorkerStateDict = worker_state_validate({})
    sweep_progress: SweepProgressDict = sweep_progress_validate({})


def read_server_state(state_path: Path | str = SERVER_STATE_PATH) -> ServerState:
    """Load and validate ServerState from JSON.  Raises ValidationError if invalid."""
    with open(state_path, "r") as f:
        data = json.load(f)

    return ServerState.model_validate(data)


Process_states: TypeAlias = WorkerStateDict | DaemonStateDict | ServerState


# ---------------------------------------------------------------------------
# Protocols / Manager
# ---------------------------------------------------------------------------


class DaemonProtocol(Protocol):
    def start_supervisor(self) -> None: ...
    def set_state(
        self, state: DaemonStateDict, awake_on_state: bool = True
    ) -> None: ...
    def stop_supervisor(
        self,
        timeout: float | None = None,
        hard: bool = False,
        stop_worker: bool = False,
    ) -> None: ...


class WorkerProtocol(Protocol):
    def cleanup(self) -> None: ...
    def start(self) -> None: ...


class ServerContextProtocol(Protocol):
    def save_state(
        self, out_path: str | Path | None = None, timestamp: str | None = None
    ) -> None: ...
    def load_state(self, in_path: str | Path | None = None) -> None: ...
    def get_state(self) -> ServerState: ...
    def set_sweep(self, sweep: SweepInfo | dict) -> None: ...
    def set_sweep_max_runs(self, max_runs: int | None) -> int | None: ...
    def sweep_completed_runs(self) -> int: ...
    def start_daemon(self) -> None: ...
    def stop_daemon(
        self, timeout: float | None = None, hard: bool = False, stop_worker: bool = False
    ) -> None: ...
    def toggle_stop_on_fail(self) -> None: ...
    def read_log(
        self, log: LogName, start: int | None = None, n: int = 10
    ) -> tuple[str, int]: ...
    def clear_log(self, log: LogName) -> None: ...

    # def set_state(
    #     self,
    #     server: ServerState | None,
    #     daemon: DaemonStateDict | None,
    #     worker: WorkerStateDict | None,
    #     sweep: SweepInfo | None
    # ) -> None: ...


class QueManagerProtocol(Protocol):
    def get_que(self) -> Que: ...
    def get_daemon(self) -> DaemonProtocol: ...
    def get_worker(self) -> WorkerProtocol: ...
    def get_sweep(self) -> SweepInfo | dict: ...
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
    QueManager.register("get_sweep", proxytype=DictProxy)
    QueManager.register("get_worker_state", proxytype=DictProxy)
    QueManager.register("get_daemon_state", proxytype=DictProxy)
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


def main():
    q = Que()
    q.disp_runs(OLD_RUNS)


if __name__ == "__main__":
    main()
