import logging
from typing import Any

import pytest

from src.que.worker import Worker, _is_wandb_injected_stop


class TestIsWandbInjectedStop:
    def test_bare_exception_is_detected(self) -> None:
        assert _is_wandb_injected_stop(Exception())

    def test_exception_with_message_is_not_detected(self) -> None:
        assert not _is_wandb_injected_stop(Exception("boom"))

    def test_exception_subclass_is_not_detected(self) -> None:
        assert not _is_wandb_injected_stop(RuntimeError())

    def test_unrelated_error_is_not_detected(self) -> None:
        assert not _is_wandb_injected_stop(ValueError("bad input"))


def _silent_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    return logger


class FakeServerContext:
    def __init__(self) -> None:
        self.set_sweep_calls: list[dict[str, Any]] = []

    def set_sweep(self, sweep: dict[str, Any]) -> None:
        self.set_sweep_calls.append(sweep)


SWEEP_INFO: Any = {"sweep_id": "abc", "max_runs": 50}


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
        worker._register_sweep_trial_completion(SWEEP_INFO)
        assert worker.sweep_progress == {"completed_runs": 50}
        assert worker.server_context.set_sweep_calls == []  # type: ignore[union-attr]

    def test_lowered_cap_mid_trial_clears_sweep(self) -> None:
        worker = self.make_worker({"sweep_id": "abc", "max_runs": 40}, completed=39)
        worker._register_sweep_trial_completion(SWEEP_INFO)
        assert worker.server_context.set_sweep_calls == [{}]  # type: ignore[union-attr]

    def test_unlimited_never_clears(self) -> None:
        worker = self.make_worker({"sweep_id": "abc", "max_runs": None}, completed=999)
        worker._register_sweep_trial_completion(SWEEP_INFO)
        assert worker.server_context.set_sweep_calls == []  # type: ignore[union-attr]

    @pytest.mark.parametrize("live_sweep", [{}, {"sweep_id": "other", "max_runs": 5}])
    def test_cleared_or_replaced_sweep_not_counted(self, live_sweep: dict[str, Any]) -> None:
        worker = self.make_worker(live_sweep, completed=3)
        worker._register_sweep_trial_completion(SWEEP_INFO)
        assert worker.sweep_progress == {"completed_runs": 3}
        assert worker.server_context.set_sweep_calls == []  # type: ignore[union-attr]
