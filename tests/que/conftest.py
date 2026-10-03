"""Shared Que fixtures: a Que persisted under `tmp_path` (see factories.py for runs to fill it with),
and a real ServerContext on files under `tmp_path`."""

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import pytest
from factories import MakeQue, silent_logger

from src.que import server as server_module
from src.que.core import Que
from src.que.server import ServerContext

StartServer: TypeAlias = Callable[[], ServerContext]


@pytest.fixture
def runs_path(tmp_path: Path) -> Path:
    return tmp_path / "Runs.json"


@pytest.fixture
def make_que(runs_path: Path) -> MakeQue:
    """Builds a Que backed by `runs_path`; calling it again simulates a restart (reload from disk)."""
    return lambda: Que(logger=silent_logger("test_que"), runs_path=runs_path)


@pytest.fixture
def que(make_que: MakeQue) -> Que:
    return make_que()


@pytest.fixture
def start_server(monkeypatch: pytest.MonkeyPatch, runs_path: Path, tmp_path: Path) -> StartServer:
    """Builds a real ServerContext on files under `tmp_path`; calling it again simulates the
    server restarting (e.g. under systemd after a power outage) on whatever reached disk.

    Signal handlers and the start method are left alone (the latter is the `spawn_method`
    fixture's job, for tests that need it).
    """
    monkeypatch.setattr(server_module.signal, "signal", lambda *args: None)
    monkeypatch.setattr(server_module.mp, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ServerContext,
        "_setup_logging",
        lambda self: tuple(silent_logger(f"test_{n}") for n in ("que", "daemon", "server", "worker")),
    )
    return lambda: ServerContext(server_state_path=tmp_path / "Server.json", runs_path=runs_path)
