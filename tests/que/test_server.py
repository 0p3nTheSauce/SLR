import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.que.core import NoSweepSet
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
