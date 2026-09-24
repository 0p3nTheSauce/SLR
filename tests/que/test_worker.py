import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.que import worker as worker_module
from src.que.core import QueBusy
from src.que.worker import (
    LoggerWriter,
    SweepTrialFailed,
    Worker,
    _is_wandb_injected_stop,
)


class TestIsWandbInjectedStop:
    def test_bare_exception_is_detected(self) -> None:
        assert _is_wandb_injected_stop(Exception())

    def test_exception_with_message_is_not_detected(self) -> None:
        assert not _is_wandb_injected_stop(Exception("boom"))

    def test_exception_subclass_is_not_detected(self) -> None:
        assert not _is_wandb_injected_stop(RuntimeError())

    def test_unrelated_error_is_not_detected(self) -> None:
        assert not _is_wandb_injected_stop(ValueError("bad input"))


SWEEP_INFO: Any = {
    "sweep_id": "abc",
    "sweep_project": "p",
    "sweep_entity": "e",
    "model": "S3D",
    "dataset": "WLASL",
    "split": "asl100",
    "base_config": "base.py",
    "max_runs": None,
}


class FakeQue:
    """Just enough of the Que proxy for a sweep trial: cur_run plus fail_runs."""

    def __init__(self, add_error: Exception | None = None) -> None:
        self.cur_run: list[Any] = []
        self.fail_runs: list[str] = []
        self.add_error = add_error

    def add_new_run(self, config: Any, wandb_info: Any, loc: str) -> None:
        if self.add_error is not None:
            raise self.add_error
        self.cur_run.append(config)

    def len_loc(self, loc: str) -> int:
        assert loc == "cur_run"
        return len(self.cur_run)

    def stash_failed_run(self, error: str) -> None:
        self.cur_run.pop()  # the real one raises QueEmpty on an empty cur_run
        self.fail_runs.append(error)

    def save_state(self) -> None: ...


def _silent_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    return logger


def swallowing_agent(*args: Any, function: Any, **kwargs: Any) -> None:
    """Mimics wandb's pyagent._run_job: exceptions from the callback never propagate."""
    try:
        function()
    except Exception:  # noqa: BLE001, S110 -- swallowing is the behaviour under test
        pass


def raise_(exc: Exception) -> Any:
    raise exc


class SweepHarness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, que: FakeQue) -> None:
        self.monkeypatch = monkeypatch
        self.que = que
        self.state: dict[str, Any] = {"exception": None, "current_run_id": None}
        self.progress = {"completed_runs": 0}
        self.worker = Worker(
            server_logger=_silent_logger("test_worker_server"),
            que=que,  # type: ignore[arg-type]
            state=self.state,  # type: ignore[arg-type]
            do_traceback=False,
        )
        self.worker.training_logger = _silent_logger("test_worker_training")
        self.worker.log_adapter = LoggerWriter(_silent_logger("test_worker_training"))
        self.worker.sweep_progress = self.progress  # type: ignore[assignment]
        self.worker.server_context = SimpleNamespace(set_sweep=lambda s: None)  # type: ignore[assignment]
        self.worker.live_sweep = dict(SWEEP_INFO)

        monkeypatch.setattr(Worker, "cleanup", lambda self: None)
        monkeypatch.setattr(worker_module.gpu_manager, "wait_for_completion", lambda **kw: True)
        monkeypatch.setattr(worker_module.wandb, "agent", swallowing_agent)
        self.set_create(lambda **kw: (
            SimpleNamespace(model_dump=dict, admin=SimpleNamespace(model="S3D")),
            SimpleNamespace(id="r1", name="n", path="p", finish=lambda exit_code: None),
        ))
        self.set_train(lambda *a, **kw: None)

    def set_create(self, fn: Any) -> None:
        self.monkeypatch.setattr(worker_module, "create_sweep_run", fn)

    def set_train(self, fn: Any) -> None:
        self.monkeypatch.setattr(worker_module, "train_loop", fn)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> SweepHarness:
    return SweepHarness(monkeypatch, FakeQue())


class TestSweepTrialOutcomes:
    """wandb swallows exceptions from _sweep_train, so Worker.sweep() must re-raise them itself
    for the worker process to exit non-zero (and the Daemon's stop_on_fail to apply)."""

    def test_success_is_counted_and_does_not_raise(self, harness: SweepHarness) -> None:
        harness.worker.sweep(SWEEP_INFO)
        assert len(harness.que.cur_run) == 1
        assert harness.progress["completed_runs"] == 1
        assert harness.state["exception"] is None

    def test_hyperband_stop_is_counted_and_does_not_raise(self, harness: SweepHarness) -> None:
        harness.set_train(lambda *a, **kw: raise_(Exception()))
        harness.worker.sweep(SWEEP_INFO)
        assert len(harness.que.cur_run) == 1  # kept for testing
        assert harness.progress["completed_runs"] == 1
        assert harness.state["exception"] is None

    def test_training_crash_raises_and_stashes(self, harness: SweepHarness) -> None:
        harness.set_train(lambda *a, **kw: raise_(RuntimeError("boom")))
        with pytest.raises(SweepTrialFailed) as exc_info:
            harness.worker.sweep(SWEEP_INFO)
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert harness.que.cur_run == []
        assert harness.que.fail_runs == ["boom"]
        assert harness.state["exception"] == "boom"
        assert harness.progress["completed_runs"] == 0

    def test_create_sweep_run_crash_raises_without_stash(self, harness: SweepHarness) -> None:
        harness.set_create(lambda **kw: raise_(ValueError("bad config")))
        with pytest.raises(SweepTrialFailed):
            harness.worker.sweep(SWEEP_INFO)
        assert harness.que.fail_runs == []
        assert harness.state["exception"] == "bad config"
        assert harness.progress["completed_runs"] == 0

    def test_inject_crash_raises_without_stash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        harness = SweepHarness(monkeypatch, FakeQue(add_error=QueBusy()))
        with pytest.raises(SweepTrialFailed):
            harness.worker.sweep(SWEEP_INFO)
        assert harness.que.fail_runs == []
        assert harness.progress["completed_runs"] == 0

    def test_agent_error_outside_callback_is_not_masked(self, harness: SweepHarness) -> None:
        """With cur_run empty, stashing used to raise QueEmpty over the real error."""
        harness.monkeypatch.setattr(
            worker_module.wandb, "agent", lambda *a, **kw: raise_(ConnectionError("offline"))
        )
        with pytest.raises(ConnectionError):
            harness.worker.sweep(SWEEP_INFO)
        assert harness.que.fail_runs == []

    def test_error_from_previous_trial_does_not_leak(self, harness: SweepHarness) -> None:
        harness.set_train(lambda *a, **kw: raise_(RuntimeError("boom")))
        with pytest.raises(SweepTrialFailed):
            harness.worker.sweep(SWEEP_INFO)
        harness.set_train(lambda *a, **kw: None)
        harness.worker.sweep(SWEEP_INFO)
        assert harness.progress["completed_runs"] == 1


class FakeServerContext:
    def __init__(self) -> None:
        self.set_sweep_calls: list[dict[str, Any]] = []

    def set_sweep(self, sweep: dict[str, Any]) -> None:
        self.set_sweep_calls.append(sweep)


CAP_SWEEP_INFO: Any = {"sweep_id": "abc", "max_runs": 50}


class TestRegisterSweepTrialCompletion:
    """The cap must come from the live shared sweep, not the trial's start-time snapshot."""

    def make_worker(self, live_sweep: dict[str, Any], completed: int) -> Worker:
        worker = Worker(
            server_logger=_silent_logger("test_worker_server"),
            que=None,  # type: ignore[arg-type]
            state={},  # type: ignore[arg-type]
        )
        worker.training_logger = _silent_logger("test_worker_training")
        worker.live_sweep = live_sweep
        worker.sweep_progress = {"completed_runs": completed}
        worker.server_context = FakeServerContext()  # type: ignore[assignment]
        return worker

    def test_raised_cap_mid_trial_keeps_sweep(self) -> None:
        worker = self.make_worker({"sweep_id": "abc", "max_runs": 60}, completed=49)
        worker._register_sweep_trial_completion(CAP_SWEEP_INFO)
        assert worker.sweep_progress == {"completed_runs": 50}
        assert worker.server_context.set_sweep_calls == []  # type: ignore[union-attr]

    def test_lowered_cap_mid_trial_clears_sweep(self) -> None:
        worker = self.make_worker({"sweep_id": "abc", "max_runs": 40}, completed=39)
        worker._register_sweep_trial_completion(CAP_SWEEP_INFO)
        assert worker.server_context.set_sweep_calls == [{}]  # type: ignore[union-attr]

    def test_unlimited_never_clears(self) -> None:
        worker = self.make_worker({"sweep_id": "abc", "max_runs": None}, completed=999)
        worker._register_sweep_trial_completion(CAP_SWEEP_INFO)
        assert worker.server_context.set_sweep_calls == []  # type: ignore[union-attr]

    @pytest.mark.parametrize("live_sweep", [{}, {"sweep_id": "other", "max_runs": 5}])
    def test_cleared_or_replaced_sweep_not_counted(self, live_sweep: dict[str, Any]) -> None:
        worker = self.make_worker(live_sweep, completed=3)
        worker._register_sweep_trial_completion(CAP_SWEEP_INFO)
        assert worker.sweep_progress == {"completed_runs": 3}
        assert worker.server_context.set_sweep_calls == []  # type: ignore[union-attr]
