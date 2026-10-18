import argparse
import os
import signal
from pathlib import Path

import pytest

from src.configs import load_config
from src.run_types import CONFIGS_PATH, ExpInfo, RunInfo, WandbInfo
from src.train_and_test import (
    EXIT_RESUME,
    RESUME_INFO_FNAME,
    RUN_INFO_FNAME,
    Preempted,
    SigtermGuard,
    build_admin,
    check_same_config,
    get_parser,
    load_resume_info,
    main,
)

CONFIG = CONFIGS_PATH / "asl100/MViTv2_S_e/exp000.toml"


def _admin_args(*extra: str, config: Path = CONFIG) -> list[str]:
    return ["MViTv2_S_e", "asl100", str(config), *extra]


def test_default_run_name_is_config_stem_and_seed(tmp_path: Path) -> None:
    admin = build_admin(get_parser().parse_args(_admin_args("--seed", "3")), runs_path=tmp_path)

    assert admin.exp_no == "exp000_s3"
    assert admin.seed == 3
    assert Path(admin.save_path) == tmp_path / "asl100/MViTv2_S_e/exp000_s3/checkpoints"
    assert Path(admin.save_path).is_dir()
    assert Path(admin.config_path).is_absolute()


def test_run_name_overrides_directory(tmp_path: Path) -> None:
    admin = build_admin(get_parser().parse_args(_admin_args("-n", "mine")), runs_path=tmp_path)

    assert Path(admin.save_path).parent.name == "mine"


def test_admin_loads_with_config(tmp_path: Path) -> None:
    config = load_config(build_admin(get_parser().parse_args(_admin_args()), runs_path=tmp_path))

    assert config.admin.model == "MViTv2_S_e"


def test_missing_config(tmp_path: Path) -> None:
    args = get_parser().parse_args(_admin_args(config=tmp_path / "nope.toml"))

    with pytest.raises(FileNotFoundError):
        build_admin(args, runs_path=tmp_path)


def _config(tmp_path: Path, *extra: str) -> RunInfo:
    return load_config(build_admin(get_parser().parse_args(_admin_args(*extra)), runs_path=tmp_path))


def _first_attempt(config: RunInfo) -> ExpInfo:
    wandb = WandbInfo(entity="e", project="p", run_id="abc123")
    return ExpInfo(**config.model_dump(), wandb=wandb)


def test_no_resume_without_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    save_path = Path(config.admin.save_path)
    (save_path.parent / RESUME_INFO_FNAME).write_text(_first_attempt(config).model_dump_json())
    (save_path / "best.pth").touch()  # died in epoch 1, before any resume checkpoint

    assert load_resume_info(save_path) is None


def test_resumes_first_attempt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    save_path = Path(config.admin.save_path)
    (save_path.parent / RESUME_INFO_FNAME).write_text(_first_attempt(config).model_dump_json())
    (save_path / "checkpoint_004.pth").touch()

    first = load_resume_info(save_path)

    assert first is not None
    assert first.wandb.run_id == "abc123"
    check_same_config(first, config)  # a JSON round trip still compares equal


def test_refuses_finished_run(tmp_path: Path) -> None:
    save_path = Path(_config(tmp_path).admin.save_path)
    (save_path.parent / RUN_INFO_FNAME).touch()

    with pytest.raises(FileExistsError):
        load_resume_info(save_path)


def test_same_config_ignores_attempt_fields(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = _first_attempt(config)
    admin = config.admin.model_copy(update={"recover": True, "config_path": "/elsewhere.toml"})

    check_same_config(first, config.model_copy(update={"admin": admin}))


def test_changed_seed_is_refused(tmp_path: Path) -> None:
    first = _first_attempt(_config(tmp_path, "-n", "run"))

    with pytest.raises(ValueError, match="differs"):
        check_same_config(first, _config(tmp_path, "-n", "run", "--seed", "7"))


def test_changed_hyperparameter_is_refused(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = _first_attempt(config)
    first.optimizer.backbone_init_lr *= 2

    with pytest.raises(ValueError, match="differs"):
        check_same_config(first, config)


def test_sigterm_guard_raises_once_and_restores() -> None:
    before = signal.getsignal(signal.SIGTERM)
    with SigtermGuard() as guard:
        with pytest.raises(Preempted):
            os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)  # a second SIGTERM doesn't interrupt the unwinding
        assert guard.received

    assert signal.getsignal(signal.SIGTERM) == before


def test_main_exits_resume_on_sigterm(monkeypatch: pytest.MonkeyPatch) -> None:
    def preempted_job(_args: argparse.Namespace) -> None:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Preempted:
            raise RuntimeError("DataLoader worker killed") from None  # what may surface instead

    monkeypatch.setattr("src.train_and_test.run_job", preempted_job)
    monkeypatch.setattr("sys.argv", ["train_and_test", *_admin_args()])

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == EXIT_RESUME


def test_main_reraises_real_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_job(_args: argparse.Namespace) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("src.train_and_test.run_job", failing_job)
    monkeypatch.setattr("sys.argv", ["train_and_test", *_admin_args()])

    with pytest.raises(RuntimeError, match="boom"):
        main()
