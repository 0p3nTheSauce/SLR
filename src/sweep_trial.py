"""Run one trial of a wandb sweep as a resumable Que_v2 job (the `sweep` template in `que.toml`).

The first attempt asks the sweep's controller for a trial (`wandb.agent(count=1)`), resolves the
trial's full config from the sweep's base config, and saves it with the trial's wandb run to the
trial's `resume.json` before training anything. Every attempt, the first included, then trains
and tests that config as `train_and_test.py` does a named run, resuming the same wandb run from
the newest epoch checkpoint. So a preempted or lost attempt picks up the same trial rather than
being handed a new one. The v1 Que did the same (`Worker._inject_sweep_config`).

Attempts of one job find each other by Que_v2's job id (`QUE_JOB_ID`), which names the trial's
directory, laid out like `train_and_test.py`'s:

    src/runs/<split>/<model>/sweep_<sweep_id>/job<QUE_JOB_ID>/

Model, split and base config come from the `command` block of the sweep's stored config
(`src.sweeping.args_from_existing_sweep`). Exit codes follow Que_v2's contract, as in
`train_and_test.py`.

The agent runs its function in a thread of its own, where SIGTERM can't interrupt it, so the
function only samples the trial (`wandb.init()`). The agent then finishes that run, and the
training resumes it on the main thread, so the run shows as finished in wandb for a moment.
"""

import argparse
import os
from pathlib import Path

import wandb
from src.configs import get_model_checkpoint_dir
from src.run_types import ENTITY, SRC_ROOT, ExpInfo, RunInfo, WandbInfo
from src.sweeping import args_from_existing_sweep, build_trial_config, get_sweep_exp_dir
from src.train_and_test import (
    QUE2_TAG,
    RESUME_INFO_FNAME,
    RUN_INFO_FNAME,
    run_preemptible,
    train_test_and_record,
)
from src.training import _setup_wandb

SWEEP_PROJECT = "Sweeps"
SWEEP_TAG = "sweep"
JOB_ID_ENV = "QUE_JOB_ID"


class NoTrialSampled(RuntimeError):
    """The sweep's agent returned without starting a trial: the sweep is finished or paused, or
    building the trial's config failed (the agent prints that exception and swallows it)."""


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run (or resume) one trial of a wandb sweep")
    parser.add_argument("sweep_id")
    parser.add_argument("-p", "--project", default=SWEEP_PROJECT)
    parser.add_argument("-et", "--entity", default=ENTITY)
    parser.add_argument(
        "-n",
        "--trial_name",
        help=f"Trial directory name, shared by every attempt. Defaults to job${JOB_ID_ENV}",
    )
    return parser


def get_trial_name(arg: str | None) -> str:
    """`arg` if given, else `job<QUE_JOB_ID>`.

    Raises:
        ValueError: If neither is set, since nothing would tie a retry to its trial.
    """
    if arg:
        return arg
    job_id = os.environ.get(JOB_ID_ENV)
    if not job_id:
        raise ValueError(f"Pass --trial_name, or run as a Que_v2 job (which sets {JOB_ID_ENV})")
    return f"job{job_id}"


def load_trial(exp_dir: Path) -> ExpInfo | None:
    """The trial an earlier attempt sampled, if any. Unlike a named run's, it is resumed even
    without a checkpoint: training from scratch is fine, sampling a different trial is not.

    Raises:
        FileExistsError: If the trial has already finished (its `run.json` exists).
    """
    if (exp_dir / RUN_INFO_FNAME).exists():
        raise FileExistsError(f"{exp_dir} has already finished")
    resume_path = exp_dir / RESUME_INFO_FNAME
    if not resume_path.exists():
        return None
    return ExpInfo.model_validate_json(resume_path.read_text())


def sample_trial(
    args: argparse.Namespace, sweep_args: argparse.Namespace, save_path: Path, trial_name: str
) -> ExpInfo:
    """Ask the sweep's controller for one trial and resolve its config (see module docstring).

    Raises:
        NoTrialSampled: If the agent didn't start a trial.
    """
    sampled: list[ExpInfo] = []

    def start_trial() -> None:
        run = wandb.init()
        config = build_trial_config(
            sweep_args.model,
            sweep_args.split,
            sweep_args.config_path,
            dict(run.config),
            save_path,
            exp_no=trial_name,
            dataset=sweep_args.dataset,
        )
        run.config.update(config.model_dump(), allow_val_change=True)
        wandb_info = WandbInfo(
            entity=args.entity,
            project=args.project,
            tags=[QUE2_TAG, SWEEP_TAG],
            run_id=run.id,
            sweep_id=args.sweep_id,
        )
        sampled.append(ExpInfo(**config.model_dump(), wandb=wandb_info))

    wandb.agent(args.sweep_id, function=start_trial, entity=args.entity, project=args.project, count=1)
    if not sampled:
        raise NoTrialSampled(f"Sweep {args.sweep_id} started no trial; see the agent's output above")
    return sampled[0]


def run_trial(args: argparse.Namespace) -> None:
    """Sample (first attempt) or reload this job's trial, then train and test it."""
    trial_name = get_trial_name(args.trial_name)
    sweep_args = args_from_existing_sweep(args.sweep_id, args.project, args.entity)
    sweep_args.config_path = sweep_args.config_path.resolve()
    exp_dir = get_sweep_exp_dir(sweep_args.split, sweep_args.model, args.sweep_id, trial_name)
    save_path = get_model_checkpoint_dir(exp_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    exp = load_trial(exp_dir)
    if exp is None:
        exp = sample_trial(args, sweep_args, save_path, trial_name)
        (exp_dir / RESUME_INFO_FNAME).write_text(exp.model_dump_json(indent=2))
    else:
        print(f"Resuming trial {exp.wandb.run_id} of sweep {args.sweep_id} from {exp_dir}")

    # training.py and testing.py are run from src/, so match that.
    os.chdir(SRC_ROOT)
    run_info = exp.model_dump(exclude={"wandb": True})
    run_info["admin"]["recover"] = True  # resume the sampled run, from a checkpoint if any
    config = RunInfo.model_validate(run_info)
    run = _setup_wandb(config, exp.wandb)
    train_test_and_record(config, exp.wandb, run)


def main() -> None:
    run_preemptible(run_trial, get_parser().parse_args())


if __name__ == "__main__":
    main()
