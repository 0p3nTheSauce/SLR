from multiprocessing import Event
import logging
from typing import Optional

from .core import (
    Que,
    QueManager,
    SR_LOG_PATH,
    QUE_NAME,
    DN_NAME,
    WR_LOG_PATH,
    WORKER_NAME,
    DaemonState,
)
from .daemon import Daemon, DaemonStateHandler
from .worker import Worker


class ServerContext:
    """
    Holds the Singleton instances of the Daemon, Worker, and State.
    This prevents relying on loose global variables.
    """

    def __init__(self):
        # Setup Logging
        # logging.basicConfig(level=logging.INFO)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=SR_LOG_PATH,
        )

        que_logger = logging.getLogger(QUE_NAME)
        dn_logger = logging.getLogger(DN_NAME)
        dn_state_logger = logging.getLogger(f"{DN_NAME} State")
        wr_logger = logging.getLogger(WORKER_NAME)

        # Create a separate file handler for the worker logger
        worker_file_handler = logging.FileHandler(WR_LOG_PATH)
        worker_file_handler.setLevel(logging.INFO)
        worker_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        worker_file_handler.setFormatter(worker_formatter)

        # Add the handler to the worker logger
        wr_logger.addHandler(worker_file_handler)

        # IMPORTANT: Prevent the worker logger from propagating to the root logger
        # This stops it from also writing to server.log
        wr_logger.propagate = False

        # Initialize Logic
        self.que = Que(logger=que_logger)
        self.worker = Worker(server_logger=wr_logger, training_logger=wr_logger)
        self.stop_worker_event = Event()
        self.stop_daemon_event = Event()

        # State and Daemon
        self.daemon_state = DaemonStateHandler(logger=dn_state_logger)
        self.daemon = Daemon(
            worker=self.worker,
            logger=dn_logger,
            local_state=self.daemon_state,
            stop_daemon_event=self.stop_daemon_event,
            stop_worker_event=self.stop_worker_event,
        )


class DaemonController:
    """
    The Object Server wrapper.
    Instead of registering functions, we register this class.
    """

    def __init__(self, context: ServerContext):
        self.ctx = context

    def save_state(self):
        self.ctx.daemon_state.to_disk()

    def load_state(self):
        self.ctx.daemon_state.from_disk()

    def start(self):
        self.ctx.daemon.start_supervisor()

    def stop_worker(self, timeout: Optional[float] = None, hard: bool = False):
        self.ctx.daemon.stop_worker(timeout=timeout, hard=hard)

    def stop_supervisor(self, timeout: Optional[float] = None, hard: bool = False, and_worker: bool = False):
        self.ctx.daemon.stop_supervisor(timeout=timeout, hard=hard, and_worker=and_worker)

    def get_state(self) -> DaemonState:
        return self.ctx.daemon_state.get_state()

    def set_stop_on_fail(self, value: bool) -> None:
        self.ctx.daemon_state.set_stop_on_fail(value)

    def set_awake(self, value: bool) -> None:
        self.ctx.daemon_state.set_awake(value)


# --- Registration Logic ---


def setup_manager():
    """
    Configures the QueManager with the ServerContext.
    """
    # Initialize the context once (Singleton pattern)
    context = ServerContext()

    # 2. Register DaemonController (Object Server)
    # Allows client to call: manager.DaemonController().start()
    QueManager.register("DaemonController", callable=lambda: DaemonController(context))

    # 3. Register Shared Que Proxy
    QueManager.register(
        "get_que",
        callable=lambda: context.que,
    )

    # 4. Register shared Daemon State Proxy
    QueManager.register(
        "get_daemon_state",
        callable=lambda: context.daemon_state,
    )


# --- Server Startup ---


def start_server():
    setup_manager()

    # Note: We bind to localhost for security, change to 0.0.0.0 to expose externally
    m = QueManager(address=("localhost", 50000), authkey=b"abracadabra")
    s = m.get_server()

    print("Object Server started on localhost:50000")
    print("Exposed Objects: DaemonStateHandler, DaemonController")

    try:
        s.serve_forever()
    except KeyboardInterrupt:
        print("Server shutdown by user")


if __name__ == "__main__":
    start_server()
