from multiprocessing import Event
import multiprocessing as mp
import logging
from logging import Logger
import json
import os
import signal
import sys
from typing import Dict, Optional, TypedDict, Union, Tuple, Any, TypeGuard
from pathlib import Path

from .core import (
    timestamp_path,
    Que,
    QueManager,
    ProcessNames,
    SERVER_LOG_PATH,
    SERVER_STATE_PATH,
    QUE_NAME,
    SERVER_NAME,
    DAEMON_NAME,
    TRAINING_NAME,
    TRAINING_LOG_PATH,
    WORKER_NAME,
    ServerState,
)
from .daemon import Daemon, DaemonState
from .worker import Worker, WorkerState



def is_server_state(val: Any) -> TypeGuard[ServerState]:
    """
    Type guard to check if an arbitrary value is structurally
    compatible with the ServerState TypedDict.
    """
    # 1. Check if the value is a dictionary
    if not isinstance(val, dict):
        return False

    # 2. Check for the presence of all required keys
    # Since all keys are technically *optional* in the Python dictionary sense
    # but *required* by TypedDict (unless explicitly marked NotRequired),
    # we check for all keys listed in the TypedDict.
    required_keys = ServerState.__annotations__.keys()
    if not all(key in val for key in required_keys):
        return False

    # 3. Check the type of each value
    # We use .get() here defensively, although the previous check makes it safe
    # to use val[key].

    # Check 'pid' and 'worker_pid' (Optional[int])
    if not (val.get("pid") is None or isinstance(val["pid"], int)):
        return False

    if not (val.get("worker_pid") is None or isinstance(val["worker_pid"], int)):
        return False

    # Check 'stop_on_fail' and 'awake' (bool)
    if not isinstance(val.get("stop_on_fail"), bool):
        return False

    if not isinstance(val.get("awake"), bool):
        return False

    # If all checks pass, it is a DaemonState
    return True

def read_server_state(state_path: Union[Path, str] = SERVER_STATE_PATH) -> ServerState:
    with open(state_path, "r") as f:
        data = json.load(f)
    if is_server_state(data):
        return data
    else:
        raise ValueError(
            f"Data read from: {state_path} is not compatible with DaemonState"
        )


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
        #context attributes
        self.save_on_shutdown: bool = save_on_shutdown
        self.cleanup_timeout: float = cleanup_timeout
        self.stop_on_fail: bool = stop_on_fail
        self.awake: bool = awake
        self.state_path: Union[str, Path] = server_state_path
        
        
        # Pids
        self.server_pid: Optional[int] = os.getpid()
        self.daemon_pid: Optional[int] = None
        self.worker_pid: Optional[int] = None

        # spawn for CUDA context
        mp.set_start_method("spawn", force=True)

        # signal handlers for systemd
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # logging
        que_logger, daemon_logger, server_logger, worker_logger = self._setup_logging()
        self.server_logger = server_logger

        # Events for controlling Daemon and Worker
        self.stop_worker_event = Event()
        self.stop_daemon_event = Event()

        # Classes
        self.load_state()
        self.que = Que(logger=que_logger)
        self.worker = Worker(server_logger=worker_logger)
        self.daemon = Daemon(
            awake=self.awake,
            stop_on_fail=self.stop_on_fail,
            worker=self.worker,
            logger=daemon_logger,
            stop_daemon_event=self.stop_daemon_event,
            stop_worker_event=self.stop_worker_event,
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

    def _setup_logging(self) -> Tuple[Logger, Logger, Logger, Logger]:
        """Sets up loggers for the server components."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=SERVER_LOG_PATH,
        )

        que_logger = logging.getLogger(QUE_NAME)
        dn_logger = logging.getLogger(DAEMON_NAME)
        server_logger = logging.getLogger(SERVER_NAME)
        worker_logger = logging.getLogger(WORKER_NAME)
        # training_logger = self._setup_training_logger()

        return que_logger, dn_logger, server_logger, worker_logger

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

    def save_state(
        self, out_path: Optional[Union[str, Path]] = None, timestamp: bool = False
    ):
        if out_path is None:
            out_path = self.state_path
        elif Path(out_path).exists() and not timestamp:
            self.server_logger.warning(f"Overwriting existing state file: {out_path}")

        if timestamp:
            out_path = timestamp_path(out_path)
        state: ServerState = {
            "server_pid": self.server_pid,
            "daemon_pid": self.daemon_pid,
            "worker_pid": self.worker_pid,
            "stop_on_fail": self.stop_on_fail,
            "awake": self.awake,
        }
        with open(out_path, "w") as f:
            json.dump(state, f)
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
            state = read_server_state(self.state_path)
            self.server_pid = state["server_pid"]
            self.worker_pid = state["worker_pid"]
            self.stop_on_fail = state["stop_on_fail"]
            self.awake = state["awake"]
            self.server_logger.info(f"Loaded state from: {self.state_path}")
        except Exception as e:
            self.server_logger.warning(
                f"Ran into an error when loading state: {e}\nloading from scratch"
            )
            self.server_pid = None
            self.daemon_pid = None
            self.worker_pid = None
            self.stop_on_fail = False
            self.awake = False

    def get_state(self) -> ServerState:
        return {
            "server_pid": self.server_pid,
            "daemon_pid": self.daemon_pid,
            "worker_pid": self.worker_pid,
            "stop_on_fail": self.stop_on_fail,
            "awake": self.awake,
        }

    def set_state(self, state: ServerState) -> None:
        self.server_pid = state["server_pid"]
        self.daemon_pid = state["daemon_pid"]
        self.worker_pid = state["worker_pid"]
        self.stop_on_fail = state["stop_on_fail"]
        self.awake = state["awake"]

    def start(self):
        self.daemon.start_supervisor()

    def stop_worker(self, timeout: Optional[float] = None, hard: bool = False):
        self.daemon.stop_worker(timeout=timeout, hard=hard)

    def stop_supervisor(
        self,
        timeout: Optional[float] = None,
        hard: bool = False,
        stop_worker: bool = False,
    ):
        self.daemon.stop_supervisor(timeout=timeout, hard=hard, stop_worker=stop_worker)

    def get_pid(self, process: ProcessNames) -> Optional[int]:
        if process == SERVER_NAME:
            return self.server_pid
        elif process == DAEMON_NAME:
            return self.daemon_pid
        elif process == WORKER_NAME:
            return self.worker_pid
        else:
            raise ValueError(f"Unknown process name: {process}")

    def set_pid(self, process: ProcessNames, pid: Optional[int]) -> None:
        if process == SERVER_NAME:
            self.server_pid = pid
        elif process == DAEMON_NAME:
            self.daemon_pid = pid
        elif process == WORKER_NAME:
            self.worker_pid = pid
        else:
            raise ValueError(f"Unknown process name: {process}")
        
    def get_stop_on_fail(self) -> bool:
        return self.stop_on_fail

    def set_stop_on_fail(self, stop_on_fail: bool) -> None:
        self.stop_on_fail = stop_on_fail

    def get_awake(self) -> bool:
        return self.awake

    def set_awake(self, awake: bool) -> None:
        self.awake = awake

    def clear_cuda_memory(self) -> None:
        self.worker.cleanup()


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
        callable=lambda: context.worker
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
