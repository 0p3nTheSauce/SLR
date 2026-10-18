"""Train and test one run from a config file: the job command for Que_v2 (`que.toml` at the repo root).

Unlike `training.py`, this needs no experiment number and never prompts. The run's directory is
named by the caller (`--run_name`, which Que_v2 fills from the template's `run_name`), so seeds of
one config get separate directories:

    src/runs/<split>/<model>/<run_name>/checkpoints/   training checkpoints (`best.pth`, ...)
    src/runs/<split>/<model>/<run_name>/results/       `full_test` outputs (`best_val_loss.json`, ...)
    src/runs/<split>/<model>/<run_name>/run.json       the config, wandb info and results, as a
                                                      `CompExpInfo` (what the v1 Que kept in old_runs)

Exits 0 when the run has been trained and tested, and nonzero on any failure. There is no SIGTERM
handler, so a stopped run cannot be resumed and its template must not be marked preemptible.
"""

import argparse
import os
from pathlib import Path

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
    WandbInfo,
)
from src.testing import full_test
from src.training import _setup_wandb, train_loop

RUN_INFO_FNAME = "run.json"
QUE2_TAG = "que2"


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
    """Admin info for a fresh run, with absolute paths.

    Raises:
        FileNotFoundError: If the config or weights file does not exist.
        FileExistsError: If the run directory already has checkpoints. A failed attempt leaves them
            behind, so delete the directory before resubmitting.
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
    save_path = get_model_checkpoint_dir(get_job_exp_dir(args.split, args.model, run_name, runs_path))
    if save_path.is_dir() and any(save_path.iterdir()):
        raise FileExistsError(f"{save_path} already has checkpoints; delete it to rerun")
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


def main() -> None:
    args = get_parser().parse_args()
    admin = build_admin(args)
    # training.py and testing.py are run from src/, so match that.
    os.chdir(SRC_ROOT)

    config = load_config(admin)
    wandb_info = WandbInfo(
        entity=args.entity,
        project=args.project or f"{PROJECT_BASE}-{args.split[3:]}",
        tags=[QUE2_TAG, *args.tags],
    )
    run = _setup_wandb(config, wandb_info)
    run.name = f"{admin.model}_{admin.split}_{admin.exp_no}"
    wandb_info.run_id = run.id
    try:
        train_loop(admin.model, config, run)
    except BaseException:
        run.finish(exit_code=1)
        raise
    run.finish(exit_code=0)

    results = full_test(admin=admin, data=config.data, re_test=True)
    comp = CompExpInfo(**config.model_dump(), wandb=wandb_info, results=results)
    run_info_path = Path(admin.save_path).parent / RUN_INFO_FNAME
    run_info_path.write_text(comp.model_dump_json(indent=2))
    print(f"Saved {run_info_path}")


if __name__ == "__main__":
    main()
