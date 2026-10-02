"""Shared Que fixtures: a Que persisted under `tmp_path` (see factories.py for runs to fill it with)."""

from pathlib import Path

import pytest
from factories import MakeQue, silent_logger

from src.que.core import Que


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
