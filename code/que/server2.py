from multiprocessing import Event

# from multiprocessing.managers import BaseManager
import multiprocessing as mp
import logging
from logging import Logger
import json
import os
import signal
import sys
from typing import Optional, Union, Tuple, Any, TypeGuard
from pathlib import Path

from .core import (
    timestamp_path,
    Que,
    QueManager,
    # ProcessNames,
    SERVER_LOG_PATH,
    SERVER_STATE_PATH,
    QUE_NAME,
    SERVER_NAME,
    DAEMON_NAME,
    TRAINING_NAME,
    TRAINING_LOG_PATH,
    WORKER_NAME,
    ServerState,
    WorkerState,
    DaemonState,
    read_server_state,
    ServerStateHandler,
    # Process_states
)
from .daemon2 import Daemon, DaemonState
from .worker2 import Worker, WorkerState


class ServerContext:
    """
    Holds the Singleton instances of the Daemon, Worker, and State.
    This prevents relying on loose global variables.
    """

    def __init__(
        self,
        save_on_shutdown: bool = True,
        cleanup_timeout: float = 10.0,
        stop_on_fail: bool = True,
        awake: bool = False,
        server_state_path: Union[str, Path] = SERVER_STATE_PATH,
    ):
        # context attributes
        self.save_on_shutdown: bool = save_on_shutdown
        self.cleanup_timeout: float = cleanup_timeout
        self.state_path: Union[str, Path] = server_state_path

        # # Pids
        self.server_pid: Optional[int] = os.getpid()

        # spawn for CUDA context
        mp.set_start_method("spawn", force=True)

        # signal handlers for systemd
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # logging
        que_logger, daemon_logger, server_logger, worker_logger, training_logger = (
            self._setup_logging()
        )
        self.server_logger = server_logger

        # Events for controlling Daemon and Worker
        self.stop_worker_event = Event()
        self.stop_daemon_event = Event()


        # Classes
        self.load_state()
        self.que = Que(logger=que_logger)
        self.worker = Worker(
            server_logger=worker_logger, training_logger=training_logger, que=self.que, stop_event=self.stop_worker_event,
            state=WorkerState(
            task='inactive',
            current_run_id=None,
            working_pid=None,
            exception=None
        )
        )
        self.daemon = Daemon(
            worker=self.worker,
            logger=daemon_logger,
            stop_daemon_event=self.stop_daemon_event,
            stop_worker_event=self.stop_worker_event,
            state=DaemonState(
            awake=awake,
            stop_on_fail=stop_on_fail,
            supervisor_pid=None
        )
        )

    def _setup_training_logger(self) -> logging.Logger:
        """Sets up a dedicated training logger with its own file handler."""
        training_logger = logging.getLogger(TRAINING_NAME)

        # Create a separate file handler for the training logger
        training_file_handler = logging.FileHandler(TRAINING_LOG_PATH)
        training_file_handler.setLevel(logging.INFO)
        training_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        training_file_handler.setFormatter(training_formatter)

        # Add the handler to the training logger
        training_logger.addHandler(training_file_handler)

        # IMPORTANT: Prevent the training logger from propagating to the root logger
        # This stops it from also writing to server.log
        training_logger.propagate = False
        return training_logger

    def _setup_logging(self) -> Tuple[Logger, Logger, Logger, Logger, Logger]:
        """Sets up loggers for the server components."""
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=SERVER_LOG_PATH,
        )

        que_logger = logging.getLogger(QUE_NAME)
        dn_logger = logging.getLogger(DAEMON_NAME)
        server_logger = logging.getLogger(SERVER_NAME)
        worker_logger = logging.getLogger(WORKER_NAME)
        training_logger = self._setup_training_logger()

        return que_logger, dn_logger, server_logger, worker_logger, training_logger

    def _handle_shutdown(self, signum, frame):
        """Handle SIGTERM/SIGINT for graceful shutdown"""
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        self.server_logger.info(
            f"Received {signal_name}, initiating graceful shutdown..."
        )

        try:
            if self.save_on_shutdown:
                self.server_logger.info("Saving server state...")
                self.save_state()

            self.server_logger.info("Stopping daemon and worker...")
            self.daemon.stop_supervisor(
                timeout=self.cleanup_timeout, hard=False, stop_worker=True
            )

            self.server_logger.info("Graceful shutdown complete")
        except Exception as e:
            self.server_logger.error(f"Error during shutdown: {e}", exc_info=True)
        finally:
            sys.exit(0)

    def get_state(self) -> ServerState:
        return ServerState(
            server_pid=self.server_pid,
            daemon_state=self.daemon.get_state(),
            worker_state=self.worker.get_state(),
        )

    def set_state(
        self,
        server: Optional[ServerState] = None,
        daemon: Optional[DaemonState] = None,
        worker: Optional[WorkerState] = None,
    ) -> None:
        if server is not None:
            self.server_pid = server["server_pid"]
            daemon = server["daemon_state"]
            worker = server["worker_state"]
        if daemon is not None:
            self.daemon.set_state(daemon)
        if worker is not None:
            self.worker.set_state(worker)

    def save_state(
        self, out_path: Optional[Union[str, Path]] = None, timestamp: bool = False
    ):
        if out_path is None:
            out_path = self.state_path
        elif Path(out_path).exists() and not timestamp:
            self.server_logger.warning(f"Overwriting existing state file: {out_path}")

        if timestamp:
            out_path = timestamp_path(out_path)

        with open(out_path, "w") as f:
            json.dump(self.get_state(), f)

        self.server_logger.info(f"Saved state to: {out_path}")

    def load_state(self, in_path: Optional[Union[str, Path]] = None) -> None:
        if in_path is None:
            in_path = self.state_path
        elif not Path(in_path).exists():
            self.server_logger.warning(
                f"No existing state found at {in_path}. Load unsuccessful."
            )
            return

        try:
            self.set_state(read_server_state(self.state_path))
            self.server_logger.info(f"Loaded state from: {self.state_path}")
        except Exception as e:
            self.server_logger.warning(
                f"Ran into an error when loading state: {e}\nloading abandoned"
            )


# --- Registration Logic ---


def setup_manager():
    """
    Configures the QueManager with the ServerContext.

    """

    # NOTE: Additions to this function must be mirrored in connect_manager() in core.py

    context = ServerContext()

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
        "get_worker",
        callable=lambda: context.worker,
    )


# --- Server Startup ---


def start_server():
    setup_manager()

    # Note: We bind to localhost for security, change to 0.0.0.0 to expose externally
    m = QueManager(address=("localhost", 50000), authkey=b"abracadabra")
    s = m.get_server()

    print("Object Server started on localhost:50000")
    print("Exposed Objects: ServerStateHandler, ServerController")

    try:
        s.serve_forever()
    except KeyboardInterrupt:
        print("Server shutdown by user")


if __name__ == "__main__":
    start_server()
