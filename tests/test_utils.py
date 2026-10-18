import json
from pathlib import Path
from typing import get_args

import cv2
import numpy as np
import pytest

from src.utils import (
    CleanReason,
    apply_removals,
    decoded_frame_count,
    header_frame_count,
    plan_cleanup,
    referenced_run_ids,
    sweep_trial_ids,
)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 25, (32, 32))
    for i in range(7):
        writer.write(np.full((32, 32, 3), i * 30, dtype=np.uint8))
    writer.release()
    return path


@pytest.mark.parametrize("count", [header_frame_count, decoded_frame_count])
def test_frame_counts(video: Path, count) -> None:
    assert count(video) == 7


@pytest.mark.parametrize("count", [header_frame_count, decoded_frame_count])
def test_unopenable_video_raises(tmp_path: Path, count) -> None:
    with pytest.raises(FileNotFoundError):
        count(tmp_path / "missing.mp4")


def _touch(path: Path, size: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * size)
    return path


def _trial(sweep: Path, trial_id: str, val_loss: float | None) -> Path:
    """A finished sweep trial, tested (has `best_val_loss.json`) unless `val_loss` is None."""
    trial = sweep / trial_id
    _touch(trial / "checkpoints" / "best.pth")
    _touch(trial / "checkpoints" / "checkpoint_009.pth")
    if val_loss is not None:
        (trial / "results").mkdir()
        (trial / "results" / "best_val_loss.json").write_text(
            json.dumps({"best_val_loss": val_loss})
        )
    return trial


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    """A runs dir: a finished run, a run predating best.pth, an abandoned run and a sweep."""
    model = tmp_path / "asl100" / "S3D"
    _touch(model / "exp000" / "checkpoints" / "best.pth")
    _touch(model / "exp000" / "checkpoints" / "checkpoint_099.pth")
    _touch(model / "exp001" / "checkpoints" / "checkpoint_073.pth")
    _touch(model / "exp002" / "data_info.json")
    (model / "exp002" / "checkpoints").mkdir()
    sweep = model / "sweep_abc"
    for trial_id, loss in [
        ("best", 1.0),
        ("second", 2.0),
        ("worst", 3.0),
        ("untested", None),
    ]:
        _trial(sweep, trial_id, loss)
    return tmp_path


def _plan(runs: Path, **kwargs) -> set[tuple[str, str]]:
    args: dict = {
        "reasons": set(get_args(CleanReason)),
        "keep_top": 2,
        "keep_ids": set(),
        "protected": set(),
    } | kwargs
    return {
        (str(r.path.relative_to(runs)), r.reason) for r in plan_cleanup(runs, **args)
    }


def test_resume_checkpoints_removed_only_beside_best(runs: Path) -> None:
    plan = _plan(runs, reasons={"resume checkpoint"})
    assert (
        "asl100/S3D/exp000/checkpoints/checkpoint_099.pth",
        "resume checkpoint",
    ) in plan
    assert not any("exp001" in p for p, _ in plan)  # its checkpoint is its only weights


def test_abandoned_run_removed_whole(runs: Path) -> None:
    assert _plan(runs, reasons={"abandoned run"}) == {
        ("asl100/S3D/exp002", "abandoned run")
    }


def test_sweep_keeps_top_k_and_kept_ids(runs: Path) -> None:
    plan = _plan(runs, reasons={"sweep trial outside top-k"}, keep_ids={"untested"})
    trials = {Path(p).parts[3] for p, _ in plan}
    assert trials == {"worst"}
    assert {Path(p).name for p, _ in plan} == {"best.pth", "checkpoint_009.pth"}
    assert (runs / "asl100/S3D/sweep_abc/worst/results/best_val_loss.json").exists()


def test_untested_trials_rank_last(runs: Path) -> None:
    plan = _plan(runs, reasons={"sweep trial outside top-k"}, keep_top=3)
    assert {Path(p).parts[3] for p, _ in plan} == {"untested"}


def test_protected_runs_and_their_sweep_untouched(runs: Path) -> None:
    """A queued trial shields its sweep from ranking, but not its siblings' resume checkpoints."""
    protected = {runs / "asl100/S3D/exp000", runs / "asl100/S3D/sweep_abc/best"}
    plan = _plan(runs, protected=protected)
    assert not any("exp000" in p or "sweep_abc/best/" in p for p, _ in plan)
    assert not any(r == "sweep trial outside top-k" for _, r in plan)
    assert (
        "asl100/S3D/sweep_abc/worst/checkpoints/checkpoint_009.pth",
        "resume checkpoint",
    ) in plan


def test_each_path_planned_once(runs: Path) -> None:
    removals = plan_cleanup(runs, set(get_args(CleanReason)), 2, set(), set())
    assert len({r.path for r in removals}) == len(removals)


def test_apply_removals(runs: Path) -> None:
    apply_removals(plan_cleanup(runs, set(get_args(CleanReason)), 2, set(), set()))
    assert not (runs / "asl100/S3D/exp002").exists()
    assert (runs / "asl100/S3D/exp001/checkpoints/checkpoint_073.pth").exists()
    assert sorted(p.name for p in (runs / "asl100/S3D/sweep_abc").rglob("*.pth")) == [
        "best.pth",
        "best.pth",
    ]


def test_referenced_run_ids(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("ensemble of abcd1234, see sweep_x/efgh5678/")
    (tmp_path / "stash.json").write_text('["ijkl9012"]')
    found = referenced_run_ids(
        {"abcd1234", "efgh5678", "ijkl9012", "abcd123"}, tmp_path
    )
    assert found == {"abcd1234", "efgh5678"}


def test_sweep_trial_ids(runs: Path) -> None:
    assert sweep_trial_ids(runs) == {"best", "second", "worst", "untested"}
