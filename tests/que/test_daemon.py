import logging
from typing import Any

import pytest

from src.que.daemon import Daemon, sweep_to_hand_off

SWEEP: dict[str, Any] = {"sweep_id": "abc", "max_runs": 50}


class TestSweepToHandOff:
    def test_hands_off_active_sweep(self) -> None:
        assert sweep_to_hand_off(SWEEP, completed_runs=10, to_run_len=0) == SWEEP

    def test_no_sweep_set(self) -> None:
        assert sweep_to_hand_off({}, completed_runs=0, to_run_len=0) is None

    def test_queued_runs_take_priority(self) -> None:
        assert sweep_to_hand_off(SWEEP, completed_runs=10, to_run_len=1) is None

    @pytest.mark.parametrize("completed", [50, 51])
    def test_complete_sweep_not_handed_off(self, completed: int) -> None:
        assert sweep_to_hand_off(SWEEP, completed_runs=completed, to_run_len=0) is None

    def test_raised_cap_resumes_complete_sweep(self) -> None:
        raised = SWEEP | {"max_runs": 60}
        assert sweep_to_hand_off(raised, completed_runs=50, to_run_len=0) == raised

    def test_unlimited_sweep_always_handed_off(self) -> None:
        unlimited = SWEEP | {"max_runs": None}
        assert sweep_to_hand_off(unlimited, completed_runs=10_000, to_run_len=0) == unlimited


class TestNextSweepLogging:
    @pytest.fixture
    def daemon(self) -> Daemon:
        daemon = Daemon.__new__(Daemon)  # skip __init__: only logging state is needed
        daemon.logger = logging.getLogger("test_daemon")
        daemon.logger.propagate = False
        daemon._logged_complete = None
        return daemon

    def test_logs_completion_once_per_cap(
        self, daemon: Daemon, caplog: pytest.LogCaptureFixture
    ) -> None:
        daemon.logger.addHandler(caplog.handler)
        with caplog.at_level(logging.INFO, logger="test_daemon"):
            daemon._next_sweep(SWEEP, 50, 0)
            daemon._next_sweep(SWEEP, 50, 0)
            daemon._next_sweep(SWEEP | {"max_runs": 60}, 60, 0)
        assert [r.message.split(" (")[0] for r in caplog.records] == [
            "Sweep abc complete",
            "Sweep abc complete",
        ]
        assert "(50/50)" in caplog.records[0].message
        assert "(60/60)" in caplog.records[1].message
        daemon.logger.removeHandler(caplog.handler)
