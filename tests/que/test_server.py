import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.que.core import NoSweepSet, ServerState
from src.que.server import ServerContext


def make_context(sweep: dict[str, Any], completed: int) -> SimpleNamespace:
    """Stand-in for ServerContext: its constructor installs signal handlers and real logging."""
    logger = logging.getLogger("test_server")
    logger.propagate = False
    return SimpleNamespace(
        sweep=sweep, sweep_progress={"completed_runs": completed}, server_logger=logger
    )


class TestSetSweepMaxRuns:
    def test_updates_cap_and_keeps_progress(self) -> None:
        ctx = make_context({"sweep_id": "abc", "max_runs": 50}, completed=49)
        previous = ServerContext.set_sweep_max_runs(ctx, 60)  # type: ignore[arg-type]
        assert previous == 50
        assert ctx.sweep["max_runs"] == 60
        assert ctx.sweep_progress["completed_runs"] == 49

    def test_can_set_unlimited(self) -> None:
        ctx = make_context({"sweep_id": "abc", "max_runs": 50}, completed=10)
        ServerContext.set_sweep_max_runs(ctx, None)  # type: ignore[arg-type]
        assert ctx.sweep["max_runs"] is None

    def test_can_cap_unlimited_sweep(self) -> None:
        ctx = make_context({"sweep_id": "abc", "max_runs": None}, completed=37)
        previous = ServerContext.set_sweep_max_runs(ctx, 40)  # type: ignore[arg-type]
        assert previous is None
        assert ctx.sweep["max_runs"] == 40

    def test_cap_at_or_below_completed_is_allowed(self) -> None:
        """It just marks the sweep complete -- the Daemon stops handing it out."""
        ctx = make_context({"sweep_id": "abc", "max_runs": 50}, completed=37)
        ServerContext.set_sweep_max_runs(ctx, 30)  # type: ignore[arg-type]
        assert ctx.sweep["max_runs"] == 30

    @pytest.mark.parametrize("max_runs", [0, -1])
    def test_rejects_cap_below_one(self, max_runs: int) -> None:
        ctx = make_context({"sweep_id": "abc", "max_runs": 50}, completed=0)
        with pytest.raises(ValueError):
            ServerContext.set_sweep_max_runs(ctx, max_runs)  # type: ignore[arg-type]
        assert ctx.sweep["max_runs"] == 50

    def test_raises_without_sweep(self) -> None:
        with pytest.raises(NoSweepSet):
            ServerContext.set_sweep_max_runs(make_context({}, completed=0), 10)  # type: ignore[arg-type]


class TestRegisterSweepTrial:
    def test_counts_trial_of_active_sweep(self) -> None:
        ctx = make_context({"sweep_id": "abc", "max_runs": 50}, completed=4)
        assert ServerContext.register_sweep_trial(ctx, "abc") == 5  # type: ignore[arg-type]
        assert ctx.sweep_progress["completed_runs"] == 5

    def test_does_not_clear_sweep_at_cap(self) -> None:
        ctx = make_context({"sweep_id": "abc", "max_runs": 5}, completed=4)
        ServerContext.register_sweep_trial(ctx, "abc")  # type: ignore[arg-type]
        assert ctx.sweep["sweep_id"] == "abc"

    @pytest.mark.parametrize("sweep", [{}, {"sweep_id": "other", "max_runs": 5}])
    def test_ignores_cleared_or_replaced_sweep(self, sweep: dict[str, Any]) -> None:
        ctx = make_context(sweep, completed=3)
        assert ServerContext.register_sweep_trial(ctx, "abc") is None  # type: ignore[arg-type]
        assert ctx.sweep_progress["completed_runs"] == 3


class TestLoadState:
    """The saved state is a snapshot, possibly taken mid-run before the server died."""

    SAVED = ServerState(
        sweep={"sweep_id": "abc", "max_runs": 50},
        daemon_state={"awake": False, "stop_on_fail": False, "supervisor_pid": 111},
        worker_state={
            "task": "training", "current_run_id": "old", "working_pid": 222, "exception": "boom"
        },
        sweep_progress={"completed_runs": 7},
    )

    @pytest.fixture
    def ctx(self, tmp_path: Path) -> SimpleNamespace:
        state_path = tmp_path / "Server.json"
        state_path.write_text(self.SAVED.model_dump_json())
        ctx = make_context({}, completed=0)
        ctx.state_path = state_path
        ctx.daemon = SimpleNamespace(
            state={"awake": False, "stop_on_fail": True, "supervisor_pid": None}
        )
        ctx.worker = SimpleNamespace(
            state={"task": "inactive", "current_run_id": None, "working_pid": None, "exception": None}
        )
        ctx.loaded = None
        ctx._keep_live_fields = lambda state: ServerContext._keep_live_fields(ctx, state)  # type: ignore[arg-type]
        ctx._set_state = lambda state: setattr(ctx, "loaded", state)
        return ctx

    def test_keeps_live_process_fields(self, ctx: SimpleNamespace) -> None:
        ServerContext.load_state(ctx)  # type: ignore[arg-type]
        assert ctx.loaded.daemon_state == {
            "awake": False, "stop_on_fail": True, "supervisor_pid": None
        }
        assert ctx.loaded.worker_state == {
            "task": "inactive", "current_run_id": None, "working_pid": None, "exception": "boom"
        }

    def test_restores_sweep(self, ctx: SimpleNamespace) -> None:
        ServerContext.load_state(ctx)  # type: ignore[arg-type]
        assert ctx.loaded.sweep == self.SAVED.sweep
        assert ctx.loaded.sweep_progress == {"completed_runs": 7}

    def test_reads_given_path(self, ctx: SimpleNamespace, tmp_path: Path) -> None:
        other = tmp_path / "other.json"
        other.write_text(ServerState(sweep_progress={"completed_runs": 3}).model_dump_json())
        ServerContext.load_state(ctx, other)  # type: ignore[arg-type]
        assert ctx.loaded.sweep_progress == {"completed_runs": 3}

    def test_missing_file_loads_nothing(self, ctx: SimpleNamespace, tmp_path: Path) -> None:
        ServerContext.load_state(ctx, tmp_path / "missing.json")  # type: ignore[arg-type]
        assert ctx.loaded is None
