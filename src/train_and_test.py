"""Train and test one run from a config file: the job command for Que_v2 (`que.toml` at the repo root).

Unlike `training.py`, this needs no experiment number and never prompts. The run's directory is
named by the caller (`--run_name`, which Que_v2 fills from the template's `run_name`), so seeds of
one config get separate directories:

    src/runs/<split>/<model>/<run_name>/checkpoints/   training checkpoints (`best.pth`, ...)
    src/runs/<split>/<model>/<run_name>/results/       `full_test` outputs (`best_val_loss.json`, ...)
    src/runs/<split>/<model>/<run_name>/resume.json    the config and wandb run of the first attempt,
                                                      as an `ExpInfo`, for later attempts to resume
    src/runs/<split>/<model>/<run_name>/run.json       written last: the config, wandb info and
                                                      results, as a `CompExpInfo` (what the v1 Que
                                                      kept in old_runs)

A resume checkpoint is saved every epoch (only the newest is kept). If an attempt finds one, it
carries on from it, in the same wandb run, so a failed, lost or preempted run is retried by
resubmitting it unchanged. It refuses to if the config differs from the first attempt's, or if the
run has already finished.

Exit codes (Que_v2's contract): 0 when the run has been trained and tested; 75 when stopped by
SIGTERM, meaning "resume me" (the unfinished epoch is dropped, so up to one epoch is redone);
anything else on failure.
"""

import argparse
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any

from typing_extensions import Self
from wandb.sdk.wandb_run import Run

from src.configs import get_model_checkpoint_dir, load_config
from src.models import avail_models
from src.run_types import (
    ENTITY,
    PROJECT_BASE,
    RUNS_PATH,
    SEED,
    SRC_ROOT,
    AdminInfo,
    CompExpInfo,
    ExpInfo,
    RunInfo,
    WandbInfo,
)
from src.testing import full_test
from src.training import _setup_wandb, latest_resume_checkpoint, train_loop

RUN_INFO_FNAME = "run.json"
RESUME_INFO_FNAME = "resume.json"
QUE2_TAG = "que2"
EXIT_RESUME = 75
# Admin fields that may differ between attempts of one run: the config is read from a per-commit
# worktree, and `recover` is what tells the attempts apart.
_ATTEMPT_ADMIN_FIELDS = {"config_path", "recover"}


class Preempted(BaseException):
    """Raised by `SigtermGuard` on SIGTERM. A BaseException, so blanket `except Exception`
    handlers in the training code don't swallow it."""


class SigtermGuard:
    """While active, the first SIGTERM raises `Preempted` wherever the main thread is, and
    `received` records it. Que_v2 signals the whole process group, so DataLoader workers die too
    and can surface as a different exception; check `received` rather than the exception type."""

    def __init__(self) -> None:
        self.received = False
        self._previous: Callable[[int, FrameType | None], Any] | int | None = None

    def _handle(self, _signum: int, _frame: FrameType | None) -> None:
        if not self.received:
            self.received = True
            raise Preempted

    def __enter__(self) -> Self:
        self._previous = signal.signal(signal.SIGTERM, self._handle)
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        signal.signal(signal.SIGTERM, self._previous)


def get_job_exp_dir(split: str, model: str, run_name: str, runs_path: Path = RUNS_PATH) -> Path:
    """A named run's directory, beside the numbered `exp<NNN>` ones of `training.py`."""
    return runs_path / split / model / run_name


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and test one run from a config file")
    parser.add_argument("model", choices=avail_models())
    parser.add_argument("split", help="Class split, e.g. asl100")
    parser.add_argument("config_path", type=Path, help="TOML config (training/optimizer/data/...)")
    parser.add_argument("-s", "--seed", type=int, default=SEED)
    parser.add_argument(
        "-n", "--run_name", help="Run directory name. Defaults to <config stem>_s<seed>"
    )
    parser.add_argument("-p", "--project", help=f"wandb project, if not {PROJECT_BASE}-<classes>")
    parser.add_argument("-et", "--entity", default=ENTITY)
    parser.add_argument("-t", "--tags", nargs="+", default=[], help="Extra wandb tags")
    parser.add_argument("-w", "--weights_path", type=Path, help="Pretrained weights to start from")
    return parser


def build_admin(args: argparse.Namespace, runs_path: Path = RUNS_PATH) -> AdminInfo:
    """Admin info for a fresh attempt (`recover` is False), with absolute paths.

    Raises:
        FileNotFoundError: If the config or weights file does not exist.
    """
    config_path: Path = args.config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    weights_path: Path | None = args.weights_path
    if weights_path is not None:
        weights_path = weights_path.resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

    run_name: str = args.run_name or f"{config_path.stem}_s{args.seed}"
    exp_dir = get_job_exp_dir(args.split, args.model, run_name, runs_path)
    save_path = get_model_checkpoint_dir(exp_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    return AdminInfo(
        model=args.model,
        split=args.split,
        exp_no=run_name,
        recover=False,
        config_path=str(config_path),
        save_path=str(save_path),
        weight_path=None if weights_path is None else str(weights_path),
        seed=args.seed,
    )


def load_resume_info(save_path: Path) -> ExpInfo | None:
    """The first attempt's config and wandb run, if an earlier attempt left a checkpoint to resume.

    Raises:
        FileExistsError: If the run has already finished (its `run.json` exists).
    """
    exp_dir = save_path.parent
    if (exp_dir / RUN_INFO_FNAME).exists():
        raise FileExistsError(f"{exp_dir} has already finished; use another run name to rerun it")
    resume_path = exp_dir / RESUME_INFO_FNAME
    if not resume_path.exists() or latest_resume_checkpoint(save_path) is None:
        return None
    return ExpInfo.model_validate_json(resume_path.read_text())


def check_same_config(first: ExpInfo, config: RunInfo) -> None:
    """Raise ValueError if `config` differs from the first attempt's, apart from per-attempt fields."""
    exclude = {"admin": _ATTEMPT_ADMIN_FIELDS}
    before = first.model_dump(exclude={"wandb": True, **exclude})
    if before != config.model_dump(exclude=exclude):
        raise ValueError(
            f"{config.admin.exp_no}'s config differs from its first attempt's "
            f"({RESUME_INFO_FNAME}); delete the run dir or use another run name"
        )


def run_job(args: argparse.Namespace) -> None:
    """Train (or resume) and test one run, as described in the module docstring."""
    admin = build_admin(args)
    save_path = Path(admin.save_path)
    first = load_resume_info(save_path)
    if first is not None:
        admin = admin.model_copy(update={"recover": True})
    # training.py and testing.py are run from src/, so match that.
    os.chdir(SRC_ROOT)

    config = load_config(admin)
    if first is not None:
        check_same_config(first, config)
        wandb_info = first.wandb
    else:
        wandb_info = WandbInfo(
            entity=args.entity,
            project=args.project or f"{PROJECT_BASE}-{args.split[3:]}",
            tags=[QUE2_TAG, *args.tags],
        )
    run = _setup_wandb(config, wandb_info)
    run.name = f"{admin.model}_{admin.split}_{admin.exp_no}"
    wandb_info.run_id = run.id
    if first is None:
        exp = ExpInfo(**config.model_dump(), wandb=wandb_info)
        (save_path.parent / RESUME_INFO_FNAME).write_text(exp.model_dump_json(indent=2))

    train_test_and_record(config, wandb_info, run)


def train_test_and_record(config: RunInfo, wandb_info: WandbInfo, run: Run) -> None:
    """Train `config` in the already set-up wandb `run` (resuming from the newest epoch checkpoint
    when `config.admin.recover` is set), test its best checkpoint, and write the run's `run.json`.

    A resume checkpoint is saved every epoch. The working directory must already be `src/`.
    """
    admin = config.admin
    try:
        train_loop(admin.model, config, run, save_every=1, recover=admin.recover)
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish(exit_code=0)

    results = full_test(admin=admin, data=config.data, re_test=True)
    comp = CompExpInfo(**config.model_dump(), wandb=wandb_info, results=results)
    run_info_path = Path(admin.save_path).parent / RUN_INFO_FNAME
    run_info_path.write_text(comp.model_dump_json(indent=2))
    print(f"Saved {run_info_path}")


def run_preemptible(job: Callable[[argparse.Namespace], None], args: argparse.Namespace) -> None:
    """Run `job(args)` under Que_v2's exit-code contract: if it is stopped by SIGTERM, exit
    `EXIT_RESUME` so Que_v2 resumes it later; any other failure propagates."""
    with SigtermGuard() as guard:
        try:
            job(args)
        except BaseException:
            if not guard.received:
                raise
            print(f"Stopped by SIGTERM; exiting {EXIT_RESUME} to be resumed from the last epoch")
            sys.exit(EXIT_RESUME)


def main() -> None:
    run_preemptible(run_job, get_parser().parse_args())


if __name__ == "__main__":
    main()
