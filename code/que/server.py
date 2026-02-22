import argparse
import getpass
from multiprocessing import Event
from multiprocessing.managers import DictProxy
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
    # Process_states
)
from .daemon import Daemon, DaemonState
from .worker import Worker, WorkerState


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
        que_logger, daemon_logger, server_logger, worker_logger = (
            self._setup_logging()
        )
        self.server_logger = server_logger

        # Events for controlling Daemon and Worker
        self.stop_worker_event = Event()
        self.stop_daemon_event = Event()


        # Classes
        self.que = Que(logger=que_logger)
        self.worker = Worker(
            server_logger=worker_logger, que=self.que, stop_event=self.stop_worker_event,
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
        self.load_state()

    def _setup_logging(self) -> Tuple[Logger, Logger, Logger, Logger]:
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
                self.daemon.state["awake"] = False #if being stopped by signal, probably don't want to be awake when restarted
                self.server_pid = None #similarly, we have no need to save an old pid
                self.save_state()
            
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
            self.daemon.set_state(server["daemon_state"])
            self.worker.set_state(server["worker_state"])
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
            # self.set_state(read_server_state(self.state_path))
            state = read_server_state(self.state_path)
            state["server_pid"] = self.server_pid #do not reset server_pid after loading
            self.set_state(state)
            self.server_logger.info(f"Loaded state from: {self.state_path}")
        except Exception as e:
            self.server_logger.warning(
                f"Ran into an error when loading state: {e}\nloading abandoned",
                exc_info=True
            )


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
        default='abracadabra', #for testing, should be changed back to None for production
        help="Authentication key for connecting to the manager (default: None, will prompt for password)",
    )
    parser.add_argument(
        "--stop_on_fail",
        action="store_true",
        help="Whether the daemon should stop itself if a run fails (default: False)",
    )

    return parser

def start_server(stop_on_fail: bool = True, address: Tuple[str, int] = ("localhost", 50000), authkey: bytes = b"abracadabra"):
    setup_manager(stop_on_fail=stop_on_fail)

    # Note: We bind to localhost for security, change to 0.0.0.0 to expose externally
    m = QueManager(address=address, authkey=authkey)
    s = m.get_server()

    print(f"Object Server started on {address[0]}:{address[1]} with authkey: {authkey.decode()}")

    try:
        s.serve_forever()
    except KeyboardInterrupt:
        print("Server shutdown by user")


if __name__ == "__main__":
    parser = get_server_parser()
    args = parser.parse_args()
    
    
    start_server(stop_on_fail=args.stop_on_fail, address=(args.host, args.port_server), authkey=args.authkey.encode())
