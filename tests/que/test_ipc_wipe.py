"""The Que must survive its POSIX IPC being wiped from under it.

systemd-logind (RemoveIPC=yes, the default) deletes all of a user's named semaphores in /dev/shm
when their last login session ends, even while their services are running. On 2026-10-03 this
happened mid-way through a sweep trial: the trial finished, but the next worker the supervisor
spawned died unpickling the server's stop Event (`SemLock._rebuild` -> FileNotFoundError),
because spawned processes open the server's semaphores by name.

These tests run the real spawn/pickle boundaries (supervisor -> worker, server -> supervisor)
against a real manager, wiping the semaphores the server created (only those, so a live Que
server's aren't touched) before each spawn.
"""

import _multiprocessing
import contextlib
import multiprocessing as mp
import socket
import threading
from collections.abc import Callable, Iterator
from functools import partial
from multiprocessing import synchronize
from typing import Any, cast

import pytest
from factories import StartServer, silent_logger

from src.que import core
from src.que import daemon as daemon_module
from src.que import worker as worker_module
from src.que.core import Que, QueManager
from src.que.server import ServerContext, register_context
from src.que.worker import Worker

AUTHKEY = b"test"
JOIN_TIMEOUT = 120.0  # spawned children import torch/wandb, which is slow on a cold cache


class _TestManager(QueManager):
    """Keeps the test's registrations off QueManager (see register_context)."""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _in_child(port: int, target: Callable[..., None], args: tuple[Any, ...]) -> None:
    """Runs `target(*args)` in a spawned child, pointed at the test manager and not logging to
    the Que's real log files. `target` and `args` arrive pickled, like the Daemon's own spawns."""
    connect = partial(core.connect_manager, port=port, authkey=AUTHKEY, max_retries=1)
    daemon_module.connect_manager = connect
    worker_module.connect_manager = connect
    daemon_module.setup_server_logging = lambda: None
    worker_module.setup_server_logging = lambda: None
    worker_module.setup_training_logging = lambda: silent_logger("test_training")
    target(*args)


def _check_stop_event(self: Worker) -> None:
    """Stands in for Worker.train: the spawned worker must see the stop event the server set."""
    assert self.stop_event is not None and self.stop_event.is_set()


def _run_worker(port: int, worker: Worker) -> None:
    worker_module.Worker.train = _check_stop_event  # type: ignore[method-assign]
    _in_child(port, worker.start, (None,))


@pytest.fixture
def spawn_method() -> Iterator[None]:
    """Use `spawn`, as the server does (pytest runs under the platform default, `fork`)."""
    previous = mp.get_start_method(allow_none=True)
    mp.set_start_method("spawn", force=True)
    yield
    mp.set_start_method(previous, force=True)


@pytest.fixture
def server_semaphores(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Names of the named semaphores created in this (the server's) process from now on."""
    names: list[str] = []
    make_name = synchronize.SemLock._make_name  # type: ignore[attr-defined]

    def recording_make_name() -> str:
        names.append(make_name())
        return names[-1]

    monkeypatch.setattr(synchronize.SemLock, "_make_name", staticmethod(recording_make_name))
    return names


def wipe(names: list[str]) -> None:
    """What logind's RemoveIPC does to these semaphores when the user's last session ends."""
    for name in names:
        try:
            _multiprocessing.sem_unlink(name)  # type: ignore[attr-defined]
        except FileNotFoundError:
            pass


@pytest.fixture
def served(
    spawn_method: None, server_semaphores: list[str], start_server: StartServer
) -> Iterator[tuple[ServerContext, int]]:
    """A real ServerContext served by a manager on a free port, as `python -m que.server` does."""
    ctx = start_server()
    register_context(ctx, manager_cls=_TestManager)
    port = _free_port()
    server = _TestManager(address=("localhost", port), authkey=AUTHKEY).get_server()

    def serve() -> None:
        with contextlib.suppress(SystemExit):  # serve_forever always ends with sys.exit
            server.serve_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    yield ctx, port
    server.stop_event.set()  # type: ignore[attr-defined]
    thread.join(timeout=5)


def test_worker_spawns_after_ipc_wipe(
    served: tuple[ServerContext, int], server_semaphores: list[str]
) -> None:
    ctx, port = served
    ctx.daemon.stop_worker_event.set()
    wipe(server_semaphores)

    # what Daemon.supervise does to launch a worker
    process = mp.Process(target=_run_worker, args=(port, ctx.worker))
    process.start()
    process.join(JOIN_TIMEOUT)

    assert process.exitcode == 0


def test_supervisor_spawns_and_stops_after_ipc_wipe(
    served: tuple[ServerContext, int],
    server_semaphores: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx, port = served
    ctx.daemon.idle_poll_interval = 0.1  # nothing queued: the supervisor idles until stopped
    ctx.daemon.logger = silent_logger("test_daemon")
    wipe(server_semaphores)

    def spawn_via_child(target: Callable[..., None], args: tuple[Any, ...]) -> mp.Process:
        return mp.Process(target=_in_child, args=(port, target, args))

    monkeypatch.setattr(daemon_module, "Process", spawn_via_child)
    # the supervisor polls the Que for work once it's up; it's served from this process
    polled = threading.Event()
    len_loc = Que.len_loc
    monkeypatch.setattr(Que, "len_loc", lambda self, loc: polled.set() or len_loc(self, loc))

    ctx.daemon.start_supervisor()
    supervisor = cast(mp.Process, ctx.daemon.supervisor_process)
    for _ in range(int(JOIN_TIMEOUT * 10)):
        if polled.wait(0.1) or not supervisor.is_alive():
            break
    ctx.daemon.stop_supervisor(timeout=JOIN_TIMEOUT)

    assert polled.is_set()
    assert supervisor.exitcode == 0
