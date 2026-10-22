import argparse
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from src import sweep_trial
from src.run_types import CONFIGS_PATH, SLR_ROOT, ExpInfo, RunInfo, WandbInfo
from src.sweep_trial import (
    NoTrialSampled,
    get_parser,
    get_trial_name,
    load_trial,
    run_trial,
    sample_trial,
)
from src.sweeping import args_from_sweep_yaml, get_sweep_exp_dir
from src.train_and_test import RESUME_INFO_FNAME, RUN_INFO_FNAME

SWEEP_DIR = CONFIGS_PATH / "sweeps/MViTv2_S_e/exp000"
SWEEP_ID = "abc123"


def _sweep_values() -> dict[str, Any]:
    """One trial's flat values: each tuned parameter at its minimum, the rest at their value."""
    params = yaml.safe_load((SWEEP_DIR / "config.yaml").read_text())["parameters"]
    return {name: p["value"] if "value" in p else p["min"] for name, p in params.items()}


def _sweep_args() -> argparse.Namespace:
    args = args_from_sweep_yaml(SWEEP_DIR / "config.yaml")
    args.config_path = SLR_ROOT / args.config_path
    return args


def _args(*extra: str) -> argparse.Namespace:
    return get_parser().parse_args([SWEEP_ID, *extra])


def _fake_agent(run_id: str, calls: list[str]) -> Callable[..., None]:
    """Stands in for wandb.agent: starts one trial whose wandb.init() returns sampled values."""

    def agent(sweep_id: str, function: Callable[[], None], **_kwargs: Any) -> None:
        calls.append(sweep_id)
        run = SimpleNamespace(id=run_id, config=_FakeConfig(_sweep_values()))
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.sweep_trial.wandb.init", lambda: run)
            function()

    return agent


class _FakeConfig(dict[str, Any]):
    def update(self, *args: Any, allow_val_change: bool = False, **kwargs: Any) -> None:
        super().update(*args, **kwargs)


class TestGetTrialName:
    def test_argument_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QUE_JOB_ID", "31")
        assert get_trial_name("mine") == "mine"

    def test_defaults_to_que_job_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QUE_JOB_ID", "31")
        assert get_trial_name(None) == "job31"

    def test_refuses_without_either(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("QUE_JOB_ID", raising=False)
        with pytest.raises(ValueError, match="QUE_JOB_ID"):
            get_trial_name(None)


class TestSampleTrial:
    def test_resolves_sampled_values(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.sweep_trial.wandb.agent", _fake_agent("run1", []))

        exp = sample_trial(_args(), _sweep_args(), tmp_path, "job31")

        assert exp.wandb == WandbInfo(
            entity=exp.wandb.entity,
            project="Sweeps",
            tags=["que2", "sweep"],
            run_id="run1",
            sweep_id=SWEEP_ID,
        )
        assert exp.admin.exp_no == "job31"
        assert exp.admin.model == "MViTv2_S_e"
        assert exp.optimizer.backbone_init_lr == _sweep_values()["backbone_init_lr"]

    def test_no_trial_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.sweep_trial.wandb.agent", lambda *_a, **_k: None)

        with pytest.raises(NoTrialSampled):
            sample_trial(_args(), _sweep_args(), tmp_path, "job31")


class TestLoadTrial:
    def _exp(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExpInfo:
        monkeypatch.setattr("src.sweep_trial.wandb.agent", _fake_agent("run1", []))
        return sample_trial(_args(), _sweep_args(), tmp_path, "job31")

    def test_nothing_to_resume(self, tmp_path: Path) -> None:
        assert load_trial(tmp_path) is None

    def test_resumes_without_checkpoint(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        exp = self._exp(tmp_path, monkeypatch)
        (tmp_path / RESUME_INFO_FNAME).write_text(exp.model_dump_json())

        assert load_trial(tmp_path) == exp

    def test_refuses_finished_trial(self, tmp_path: Path) -> None:
        (tmp_path / RUN_INFO_FNAME).write_text("{}")

        with pytest.raises(FileExistsError):
            load_trial(tmp_path)


def test_retry_resumes_the_same_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent_calls: list[str] = []
    trained: list[tuple[RunInfo, WandbInfo]] = []
    monkeypatch.chdir(tmp_path)  # run_trial moves to src/; restored after the test
    monkeypatch.setenv("QUE_JOB_ID", "31")
    monkeypatch.setattr(sweep_trial, "args_from_existing_sweep", lambda *_a: _sweep_args())
    monkeypatch.setattr(
        sweep_trial,
        "get_sweep_exp_dir",
        lambda split, model, sid, name: get_sweep_exp_dir(split, model, sid, name, runs_path=tmp_path),
    )
    monkeypatch.setattr("src.sweep_trial.wandb.agent", _fake_agent("run1", agent_calls))
    monkeypatch.setattr(sweep_trial, "_setup_wandb", lambda config, info: None)
    monkeypatch.setattr(
        sweep_trial, "train_test_and_record", lambda config, info, _run: trained.append((config, info))
    )

    run_trial(_args())
    run_trial(_args())

    assert agent_calls == [SWEEP_ID]  # the retry didn't ask for a new trial
    exp_dir = tmp_path / "asl100/MViTv2_S_e" / f"sweep_{SWEEP_ID}" / "job31"
    assert (exp_dir / RESUME_INFO_FNAME).exists()
    (first, first_info), (second, second_info) = trained
    assert first == second
    assert first.admin.recover
    assert first_info.run_id == second_info.run_id == "run1"
