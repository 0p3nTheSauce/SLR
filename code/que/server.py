from multiprocessing import Event
import multiprocessing as mp
import logging
import signal
import sys
from typing import Dict, Optional, TypedDict

from .core import (
    Que,
    QueManager,
    SERVER_LOG_PATH,
    QUE_NAME,
    SERVER_NAME,
    DAEMON_NAME,
    TRAINING_NAME,
    TRAINING_LOG_PATH,
    WORKER_NAME,
    ServerState,
    ServerStateHandler
)
from .daemon import Daemon
from .worker import Worker

class LoggingDict(TypedDict):
    que: logging.Logger
    daemon: logging.Logger
    server: logging.Logger
    worker: logging.Logger
    # training: logging.Logger

class ServerContext:
    """
    Holds the Singleton instances of the Daemon, Worker, and State.
    This prevents relying on loose global variables.
    """

    def __init__(self, save_on_shutdown=True, cleanup_timeout=30.0):
        self.save_on_shutdown = save_on_shutdown
        self.cleanup_timeout = cleanup_timeout
        
        # spawn for CUDA context
        mp.set_start_method('spawn', force=True)

        # signal handlers for systemd
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # logging
        logging_dict = self._setup_logging()
        self.server_logger = logging_dict["server"]

        # Events for controlling Daemon and Worker
        self.stop_worker_event = Event()
        self.stop_daemon_event = Event()

        # Classes
        self.que = Que(logger=logging_dict["que"])
        self.worker = Worker(server_logger=logging_dict["worker"])
        self.state_handler = ServerStateHandler(logger=logging_dict["server"])
        self.daemon = Daemon(
            worker=self.worker,
            logger=logging_dict["daemon"],
            local_state=self.state_handler,
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

    def _setup_logging(self) -> LoggingDict:
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

        return LoggingDict(
            que=que_logger,
            daemon=dn_logger,
            server=server_logger,
            worker=worker_logger,
            # training=training_logger,
        )

    def _handle_shutdown(self, signum, frame):
        """Handle SIGTERM/SIGINT for graceful shutdown"""
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        self.server_logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        
        try:
            if self.save_on_shutdown:
                self.server_logger.info("Saving server state...")
                self.state_handler.save_state()

            self.server_logger.info("Stopping daemon and worker...")
            self.daemon.stop_supervisor(
                timeout=self.cleanup_timeout, 
                hard=False, 
                stop_worker=True
            )
            
            self.server_logger.info("Graceful shutdown complete")
        except Exception as e:
            self.server_logger.error(f"Error during shutdown: {e}", exc_info=True)
        finally:
            sys.exit(0)
        


class ServerController:
    """
    The Object Server wrapper.
    Instead of registering functions, we register this class.
    """

    def __init__(self, context: ServerContext):
        self.ctx = context

    def save_state(self):
        self.ctx.state_handler.save_state()

    def load_state(self):
        self.ctx.state_handler.load_state()

    def start(self):
        self.ctx.daemon.start_supervisor()

    def stop_worker(self, timeout: Optional[float] = None, hard: bool = False):
        self.ctx.daemon.stop_worker(timeout=timeout, hard=hard)

    def stop_supervisor(self, timeout: Optional[float] = None, hard: bool = False, stop_worker: bool = False):
        self.ctx.daemon.stop_supervisor(timeout=timeout, hard=hard, stop_worker=stop_worker)

    def get_state(self) -> ServerState:
        return self.ctx.state_handler.get_state()

    def set_stop_on_fail(self, value: bool) -> None:
        self.ctx.state_handler.set_stop_on_fail(value)

    def set_awake(self, value: bool) -> None:
        self.ctx.state_handler.set_awake(value)

    def clear_cuda_memory(self) -> None:
        self.ctx.worker.cleanup()


# --- Registration Logic ---


def setup_manager():
    """
    Configures the QueManager with the ServerContext.
    """
    context = ServerContext()

    # 2. Register ServerController (Object Server)
    QueManager.register("ServerController", callable=lambda: ServerController(context))

    # 3. Register Shared Que Proxy
    QueManager.register(
        "get_que",
        callable=lambda: context.que,
    )

    # 4. Register shared Server State Proxy
    QueManager.register(
        "get_server_state_handler",
        callable=lambda: context.state_handler,
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
