import logging
import multiprocessing as mp
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from src.que import daemon as daemon_module
from src.que.core import WorkerStateDict
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


class FakeProcess:
    """Stands in for multiprocessing.Process: 'runs' the worker instantly, exiting 0."""

    started: ClassVar[list[Any]] = []
    exitcode: ClassVar[int] = 0

    def __init__(self, target: Any, args: tuple[Any, ...] = ()) -> None:
        self.args = args
        self.pid = None  # so the supervisor's exit-time _hard_stop has nothing to kill

    def start(self) -> None:
        FakeProcess.started.append(self.args)

    def join(self, timeout: float | None = None) -> None:
        pass


class TestSupervise:
    """Drives Daemon.supervise() in-process against a fake manager."""

    @pytest.fixture
    def setup(self, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
        stop = mp.Event()
        env = SimpleNamespace(
            to_run=[0],  # to_run length per supervisor iteration; stops after the last
            sweep={},
            completed=0,  # the sweep's finished trials in the Que
            saves=[],  # the daemon state at each server-state save
            daemon_state={"awake": True, "stop_on_fail": True, "supervisor_pid": 1},
            # what the manager's worker-state proxy points at, i.e. the real shared state
            shared_worker_state=WorkerStateDict(
                task="training", current_run_id="abc", working_pid=123, exception=None
            ),
        )

        def len_loc(loc: str) -> int:
            if len(env.to_run) == 1:
                stop.set()
            return env.to_run.pop(0)

        def save_state() -> None:
            env.saves.append(dict(env.daemon_state))

        manager = SimpleNamespace(
            get_daemon_state=lambda: env.daemon_state,
            get_que=lambda: SimpleNamespace(len_loc=len_loc),
            get_worker_state=lambda: env.shared_worker_state,
            get_sweep=lambda: env.sweep,
            get_server_context=lambda: SimpleNamespace(
                sweep_completed_runs=lambda: env.completed, save_state=save_state
            ),
        )
        monkeypatch.setattr(daemon_module, "connect_manager", lambda: manager)
        monkeypatch.setattr(daemon_module, "Process", FakeProcess)
        FakeProcess.started = []
        FakeProcess.exitcode = 0

        logger = logging.getLogger("test_daemon")
        logger.propagate = False
        # the supervisor's copy of the Worker: its state is NOT the shared one
        worker = SimpleNamespace(state=dict(env.shared_worker_state), start=None, cleanup=lambda: None)
        env.daemon = Daemon(
            worker=worker,  # type: ignore[arg-type]
            logger=logger,
            stop_worker_event=mp.Event(),
            stop_daemon_event=stop,
            state={"awake": False, "stop_on_fail": True, "supervisor_pid": None},
            idle_poll_interval=0.0,
        )
        env.daemon._reattach_server_logger = lambda: None
        return env

    def test_idles_without_launching_worker_when_no_work(self, setup: SimpleNamespace) -> None:
        setup.to_run = [0, 0, 0]
        setup.daemon.supervise()
        assert FakeProcess.started == []

    def test_idles_when_sweep_complete(self, setup: SimpleNamespace) -> None:
        setup.to_run = [0, 0]
        setup.sweep = {"sweep_id": "abc", "max_runs": 0}
        setup.daemon.supervise()
        assert FakeProcess.started == []

    def test_logs_idle_once_per_stretch(
        self, setup: SimpleNamespace, caplog: pytest.LogCaptureFixture
    ) -> None:
        setup.to_run = [0, 0, 1, 0, 0]
        setup.daemon.logger.addHandler(caplog.handler)
        with caplog.at_level(logging.INFO, logger="test_daemon"):
            setup.daemon.supervise()
        setup.daemon.logger.removeHandler(caplog.handler)
        assert len(FakeProcess.started) == 1
        assert sum("waiting for work" in r.message for r in caplog.records) == 2

    def test_launches_worker_for_queued_run(self, setup: SimpleNamespace) -> None:
        setup.to_run = [1]
        setup.daemon.supervise()
        assert FakeProcess.started == [(None,)]

    def test_hands_off_sweep_when_to_run_empty(self, setup: SimpleNamespace) -> None:
        setup.to_run = [0]
        setup.sweep = {"sweep_id": "abc", "max_runs": 5}
        setup.daemon.supervise()
        assert FakeProcess.started == [(setup.sweep,)]

    def test_clears_shared_worker_state_after_worker_exits(self, setup: SimpleNamespace) -> None:
        setup.to_run = [1]
        setup.daemon.supervise()
        assert setup.shared_worker_state == WorkerStateDict(
            task="inactive", current_run_id=None, working_pid=None, exception=None
        )

    def test_complete_sweep_counted_from_que_is_not_handed_off(self, setup: SimpleNamespace) -> None:
        """On 2026-10-02 a stale counter (44/50) let the Daemon hand out a 51st trial."""
        setup.to_run = [0, 0]
        setup.sweep = {"sweep_id": "abc", "max_runs": 50}
        setup.completed = 50
        setup.daemon.supervise()
        assert FakeProcess.started == []

    def test_saves_server_state_after_each_worker(self, setup: SimpleNamespace) -> None:
        setup.to_run = [1, 1, 0]
        setup.daemon.supervise()
        assert len(FakeProcess.started) == 2
        assert len(setup.saves) == 3  # one per worker, plus the supervisor's exit

    def test_exit_saves_daemon_as_not_awake(self, setup: SimpleNamespace) -> None:
        """Otherwise a daemon that stopped (e.g. stop_on_fail after a failed run) would still be
        'awake' on disk, and resumed by a server restarted after an outage."""
        setup.to_run = [1]
        FakeProcess.exitcode = 1
        setup.daemon.supervise()
        assert setup.saves[-1]["awake"] is False
