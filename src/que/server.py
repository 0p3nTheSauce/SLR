import argparse
import logging

# from multiprocessing.managers import BaseManager
import multiprocessing as mp
import os
import signal
import sys
from logging import Logger
from multiprocessing import Event
from multiprocessing.managers import DictProxy
from pathlib import Path

from src.que.core import (
    DAEMON_NAME,
    QUE_NAME,
    RUN_PATH,
    # ProcessNames,
    SERVER_NAME,
    SERVER_STATE_PATH,
    WORKER_NAME,
    DaemonStateDict,
    NoSweepSet,
    Que,
    QueManager,
    ServerState,
    SweepInfo,
    SweepProgressDict,
    WorkerStateDict,
    atomic_write_json,
    is_sweep_complete,
    migrate_legacy_files,
    read_server_state,
    setup_server_logging,
    # Process_states
    timestamp_path,
)
from src.que.daemon import Daemon
from src.que.worker import Worker


class ServerContext:
    """
    Holds the Singleton instances of the Daemon, Worker, and State.
    This prevents relying on loose global variables.

    The server can die at any moment (e.g. a power outage, after which systemd restarts it), so
    its state is saved whenever it changes: the Que saves itself (see Que), and the server state
    (sweep config, daemon flags) is saved by the methods that change it. Sweep progress isn't
    stored at all -- it's derived from the Que (see sweep_completed_runs), so it can't drift.
    """

    def __init__(
        self,
        save_on_shutdown: bool = True,
        cleanup_timeout: float = 10.0,
        stop_on_fail: bool = True,
        awake: bool = False,
        server_state_path: str | Path = SERVER_STATE_PATH,
        runs_path: str | Path = RUN_PATH,
    ):
        # context attributes
        self.save_on_shutdown: bool = save_on_shutdown
        self.cleanup_timeout: float = cleanup_timeout
        self.state_path: str | Path = server_state_path

        # # Pids
        self.server_pid: int | None = os.getpid()

        # spawn for CUDA context
        mp.set_start_method("spawn", force=True)

        # signal handlers for systemd
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # logging
        que_logger, daemon_logger, server_logger, worker_logger = self._setup_logging()
        self.server_logger = server_logger
        self.server_logger.info(self.seperator("Server starting up"))

        # Events for controlling Daemon and Worker
        self.stop_worker_event = Event()
        self.stop_daemon_event = Event()

        # Classes
        self.sweep: dict = {}
        self.que = Que(logger=que_logger, runs_path=runs_path)
        self.worker = Worker(
            server_logger=worker_logger,
            que=self.que,
            stop_event=self.stop_worker_event,
            state=WorkerStateDict(
                task="inactive",
                current_run_id=None,
                working_pid=None,
                exception=None,
            ),
        )
        self.daemon = Daemon(
            worker=self.worker,
            logger=daemon_logger,
            stop_daemon_event=self.stop_daemon_event,
            stop_worker_event=self.stop_worker_event,
            state=DaemonStateDict(
                awake=awake,
                stop_on_fail=stop_on_fail,
                supervisor_pid=None,
            ),
        )
        self.load_state()

    def seperator(self, r_str: str) -> str:
        sep = ""

        if r_str:
            sep += ("\n" * 2) + ("-" * 10) + ("\n")
            sep += f"{r_str:^10}"
            sep += ("\n" * 2) + ("-" * 10) + ("\n")
        else:
            sep += "\n"
        return sep.title()

    def _setup_logging(self) -> tuple[Logger, Logger, Logger, Logger]:
        """Sets up logging to Server.log, returning the Que, Daemon, Server and Worker loggers."""
        setup_server_logging()

        que_logger = logging.getLogger(QUE_NAME)
        dn_logger = logging.getLogger(DAEMON_NAME)
        server_logger = logging.getLogger(SERVER_NAME)
        worker_logger = logging.getLogger(WORKER_NAME)

        return que_logger, dn_logger, server_logger, worker_logger

    def _handle_shutdown(self, signum, frame):
        """Handle SIGTERM/SIGINT for graceful shutdown"""
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        self.server_logger.info(
            f"Received {signal_name}, initiating graceful shutdown..."
        )

        try:
            self.server_pid = None
            self.server_logger.info("Stopping daemon and worker...")
            self.daemon.stop_supervisor(
                timeout=self.cleanup_timeout, hard=False, stop_worker=True
            )

            if self.save_on_shutdown:
                self.server_logger.info("Saving server state...")
                self.daemon.state["awake"] = (
                    False  # if being stopped by signal, probably don't want to be awake when restarted
                )
                self.server_pid = None  # similarly, we have no need to save an old pid
                self.save_state()

            self.server_logger.info("Graceful shutdown complete")
        except Exception:
            self.server_logger.exception(
                "Error during shutdown",
            )
        finally:
            sys.exit(0)

    def get_state(self) -> ServerState:
        return ServerState(
            server_pid=self.server_pid,
            sweep=self.sweep,
            daemon_state=self.daemon.get_state(),
            worker_state=self.worker.get_state(),
            sweep_progress=SweepProgressDict(completed_runs=self.sweep_completed_runs()),
        )

    def sweep_completed_runs(self) -> int:
        """The active sweep's finished trials: its runs in old_runs (0 if no sweep is set).

        Counted from the Que on every call rather than stored, so it always matches the Que --
        including after shell edits (remove, recover, ...) and unclean restarts. A trial that
        fails (in training or testing) lands in fail_runs instead, so doesn't count.
        """
        if not self.sweep:
            return 0
        return self.que.len_sweep_runs(self.sweep["sweep_id"])

    def set_sweep(self, sweep: SweepInfo | dict) -> None:
        """Set (or, with `{}`, clear) the active sweep, and save the server state.

        Trials of the sweep already in old_runs count towards its progress, so re-setting a
        sweep resumes it where it left off.

        Args:
            sweep (SweepInfo | dict): Sweep information to set.
        """
        self.sweep.clear()
        self.sweep.update(sweep)
        self.save_state()

    def set_sweep_max_runs(self, max_runs: int | None) -> int | None:
        """Change the active sweep's trial cap, and save the server state.

        The Daemon checks the cap before handing out each trial, so the change applies from the
        next trial on. A cap at or below the completed count marks the sweep complete (the
        running trial, if any, still finishes); raising it again resumes the sweep.

        Args:
            max_runs (int | None): New cap, or None for unlimited.

        Raises:
            NoSweepSet: If no sweep is set.
            ValueError: If `max_runs` is less than 1.

        Returns:
            int | None: The previous cap.
        """
        if not self.sweep:
            raise NoSweepSet()
        if max_runs is not None and max_runs < 1:
            raise ValueError(f"max_runs must be at least 1, got {max_runs}")
        previous: int | None = self.sweep["max_runs"]
        self.sweep["max_runs"] = max_runs
        self.save_state()
        return previous

    def start_daemon(self) -> None:
        """Start the Daemon's supervisor (see Daemon.start_supervisor), and save the server state.

        Starting sets the daemon 'awake', which is what makes the server resume it on restart
        after an outage (see load_state), so it has to reach disk straight away.
        """
        try:
            self.daemon.start_supervisor()
        finally:
            self.save_state()

    def stop_daemon(
        self, timeout: float | None = None, hard: bool = False, stop_worker: bool = False
    ) -> None:
        """Stop the Daemon's supervisor (see Daemon.stop_supervisor), and save the server state,
        so a server restarted after an outage doesn't resume it."""
        try:
            self.daemon.stop_supervisor(timeout=timeout, hard=hard, stop_worker=stop_worker)
        finally:
            self.save_state()

    def toggle_stop_on_fail(self) -> None:
        self.daemon.state["stop_on_fail"] = not self.daemon.state["stop_on_fail"]
        self.save_state()

    def _set_state(
        self,
        server: ServerState | None = None,
        daemon: DaemonStateDict | None = None,
        worker: WorkerStateDict | None = None,
    ) -> None:
        """Used by load_state to set the state of the server, daemon, sweep, and worker. This is separate from the set_state methods of the individual components to allow for a more centralized state management.

        Args:
            server (ServerState | None, optional): Server state to set. Defaults to None.
            daemon (DaemonStateDict | None, optional): Daemon state to set. Defaults to None.
            worker (WorkerStateDict | None, optional): Worker state to set. Defaults to None.
        """
        if server is not None:
            # do not reset server_pid after loading
            self.sweep.clear()
            self.sweep.update(server.sweep)
            self.daemon.set_state(server.daemon_state)
            self.worker.set_state(server.worker_state)
        if daemon is not None:
            self.daemon.set_state(daemon)
        if worker is not None:
            self.worker.set_state(worker)

    def save_state(
        self, out_path: str | Path | None = None, timestamp: str | None = None
    ) -> None:
        """Save the server state (atomically, see atomic_write_json).

        Args:
            out_path (str | Path | None, optional): Defaults to `self.state_path`.
            timestamp (str | None, optional): Insert this timestamp (see make_timestamp) into
                the output file name. Defaults to None.
        """
        if out_path is None:
            out_path = self.state_path
        elif Path(out_path).exists() and timestamp is None:
            self.server_logger.warning(f"Overwriting existing state file: {out_path}")

        if timestamp is not None:
            out_path = timestamp_path(out_path, timestamp)

        atomic_write_json(out_path, self.get_state().model_dump())
        # saved on every change (see ServerContext); an explicit copy is worth noting
        level = logging.DEBUG if out_path == self.state_path else logging.INFO
        self.server_logger.log(level, f"Saved state to: {out_path}")

    def load_state(self, in_path: str | Path | None = None) -> None:
        """Load a saved server state (default: `self.state_path`).

        The saved `stop_on_fail` and process fields (supervisor/worker pids, worker task and run
        id) are not restored: those describe processes, not configuration, and the ones in the
        file may be long gone (e.g. a snapshot taken mid-run before the server died), so the
        current values are kept. An 'awake' daemon relaunches its supervisor (see Daemon.set_state).
        The saved sweep progress isn't restored either: it's derived from the Que, and only
        checked against it (see _check_sweep_progress). Loading from a file other than
        `self.state_path` saves the loaded state to it, as for any other change.
        """
        in_path = self.state_path if in_path is None else in_path
        if not Path(in_path).exists():
            self.server_logger.warning(
                f"No existing state found at {in_path}. Load unsuccessful."
            )
            return

        try:
            state = read_server_state(in_path)
            self._keep_live_fields(state)
            self._check_sweep_progress(state)
            self._set_state(state)
            self.server_logger.info(f"Loaded state from: {in_path}")
            if Path(in_path) != Path(self.state_path):
                self.save_state()
        except Exception as e:
            self.server_logger.warning(
                f"Ran into an error when loading state: {e}\nloading abandoned",
                exc_info=True,
            )

    def _keep_live_fields(self, state: ServerState) -> None:
        """Overwrite the fields of a loaded `state` that load_state must not restore with their
        current values."""
        state.daemon_state["stop_on_fail"] = self.daemon.state["stop_on_fail"]
        state.daemon_state["supervisor_pid"] = self.daemon.state["supervisor_pid"]
        state.worker_state["task"] = self.worker.state["task"]
        state.worker_state["current_run_id"] = self.worker.state["current_run_id"]
        state.worker_state["working_pid"] = self.worker.state["working_pid"]

    def _check_sweep_progress(self, state: ServerState) -> None:
        """Warn if a loaded `state`'s saved sweep progress differs from the Que's count, which is
        the one used (see sweep_completed_runs), and log if the sweep is complete.

        A difference means the server died between a trial finishing and the state being saved
        (or Runs.json was edited); before progress was derived from the Que, it left a 44/50 sweep
        whose 50 trials were all in old_runs (2026-10-02).
        """
        if not state.sweep:
            return
        sweep_id = state.sweep["sweep_id"]
        saved = state.sweep_progress["completed_runs"]
        completed = self.que.len_sweep_runs(sweep_id)
        if saved != completed:
            self.server_logger.warning(
                f"Sweep {sweep_id}: saved progress was {saved} trials, but old_runs holds "
                f"{completed}; using the Que's count"
            )
        max_runs = state.sweep.get("max_runs")
        if is_sweep_complete(max_runs, completed):
            self.server_logger.info(f"Sweep {sweep_id} is complete ({completed}/{max_runs})")


# --- Registration Logic ---


def setup_manager(stop_on_fail: bool = True):
    """
    Configures the QueManager with the ServerContext.

    """

    # NOTE: Additions to this function must be mirrored in connect_manager() in core.py

    context = ServerContext(stop_on_fail=stop_on_fail)

    QueManager.register(
        "get_que",
        callable=lambda: context.que,
    )

    QueManager.register(
        "get_server_context",
        callable=lambda: context,
    )

    QueManager.register(
        "get_daemon",
        callable=lambda: context.daemon,
    )

    QueManager.register(
        "get_daemon_state",
        callable=lambda: context.daemon.state,
        proxytype=DictProxy,
    )

    QueManager.register(
        "get_worker",
        callable=lambda: context.worker,
    )

    QueManager.register(
        "get_worker_state",
        callable=lambda: context.worker.state,
        proxytype=DictProxy,
    )

    QueManager.register(
        "get_sweep",
        callable=lambda: context.sweep,
        proxytype=DictProxy,
    )



# --- Server Startup ---


def get_server_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="queShell command line arguments")

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host IP. If localhost, then connects to local manager. If remote, will establish SSH tunnel and connect to manager through that (default: localhost)",
    )
    parser.add_argument(
        "--port_server",
        type=int,
        default=50000,
        help="Remote port for SSH tunnel (default: 50000)",
    )
    parser.add_argument(
        "--authkey",
        type=str,
        default="abracadabra",  # for testing, should be changed back to None for production
        help="Authentication key for connecting to the manager (default: None, will prompt for password)",
    )
    parser.add_argument(
        "--stop_on_fail",
        "-f",
        action="store_true",
        help="Stop daemon if a run fails (default: False)",
    )

    return parser


def start_server(
    stop_on_fail: bool = True,
    address: tuple[str, int] = ("localhost", 50000),
    authkey: bytes = b"abracadabra",
):
    migrations = migrate_legacy_files()
    setup_server_logging()
    for message in migrations:
        logging.getLogger(SERVER_NAME).info(message)
    setup_manager(stop_on_fail=stop_on_fail)

    # Note: We bind to localhost for security, change to 0.0.0.0 to expose externally
    m = QueManager(address=address, authkey=authkey)
    s = m.get_server()

    print(
        f"Object Server started on {address[0]}:{address[1]} with authkey: {authkey.decode()}"
    )

    try:
        s.serve_forever()
    except KeyboardInterrupt:
        print("Server shutdown by user")


if __name__ == "__main__":
    parser = get_server_parser()
    args = parser.parse_args()

    start_server(
        stop_on_fail=args.stop_on_fail,
        address=(args.host, args.port_server),
        authkey=args.authkey.encode(),
    )
