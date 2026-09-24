import pickle
from types import SimpleNamespace
from typing import Any

import pytest

from src.que.core import MaxRunsTooLow, NoSweepSet
from src.que.server import ServerContext


def make_context(sweep: dict[str, Any], completed: int) -> SimpleNamespace:
    """Stand-in for ServerContext: its constructor installs signal handlers and real logging."""
    return SimpleNamespace(sweep=sweep, sweep_progress={"completed_runs": completed})


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

    @pytest.mark.parametrize("max_runs", [37, 10, 0, -1])
    def test_rejects_cap_not_above_completed(self, max_runs: int) -> None:
        ctx = make_context({"sweep_id": "abc", "max_runs": 50}, completed=37)
        with pytest.raises(MaxRunsTooLow):
            ServerContext.set_sweep_max_runs(ctx, max_runs)  # type: ignore[arg-type]
        assert ctx.sweep["max_runs"] == 50

    def test_raises_without_sweep(self) -> None:
        with pytest.raises(NoSweepSet):
            ServerContext.set_sweep_max_runs(make_context({}, completed=0), 10)  # type: ignore[arg-type]


def test_max_runs_too_low_pickle_round_trip() -> None:
    """Raised inside the manager process, so it must survive pickling back to the shell."""
    err = pickle.loads(pickle.dumps(MaxRunsTooLow(5, 7)))
    assert (err.max_runs, err.completed_runs) == (5, 7)
    assert str(err) == str(MaxRunsTooLow(5, 7))
