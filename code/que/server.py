from multiprocessing import Event
import multiprocessing as mp
import logging
import signal
import sys
from typing import Optional

from .core import (
    Que,
    QueManager,
    SR_LOG_PATH,
    QUE_NAME,
    DN_NAME,
    WR_LOG_PATH,
    WORKER_NAME,
    ServerState,
    ServerStateHandler
)
from .daemon import Daemon
from .worker import Worker


class ServerContext:
    """
    Holds the Singleton instances of the Daemon, Worker, and State.
    This prevents relying on loose global variables.
    """

    def __init__(self, save_on_shutdown=True, cleanup_timeout=30.0):
        self.save_on_shutdown = save_on_shutdown
        self.cleanup_timeout = cleanup_timeout
        
        
        #for handling CUDA context
        mp.set_start_method('spawn', force=True)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)


        # Setup Logging
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

        # Store logger for signal handler
        self.main_logger = logging.getLogger("ServerContext")

        # Initialize Logic
        self.que = Que(logger=que_logger)
        self.worker = Worker(server_logger=wr_logger, training_logger=wr_logger)
        self.stop_worker_event = Event()
        self.stop_daemon_event = Event()

        # State and Daemon
        self.state_handler = ServerStateHandler(logger=dn_state_logger)
        self.daemon = Daemon(
            worker=self.worker,
            logger=dn_logger,
            local_state=self.state_handler,
            stop_daemon_event=self.stop_daemon_event,
            stop_worker_event=self.stop_worker_event,
        )

    def _handle_shutdown(self, signum, frame):
        """Handle SIGTERM/SIGINT for graceful shutdown"""
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        self.main_logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        
        try:
            if self.save_on_shutdown:
                self.main_logger.info("Saving server state...")
                self.state_handler.save_state()
            
            self.main_logger.info("Stopping daemon and worker...")
            self.daemon.stop_supervisor(
                timeout=self.cleanup_timeout, 
                hard=False, 
                and_worker=True
            )
            
            self.main_logger.info("Graceful shutdown complete")
        except Exception as e:
            self.main_logger.error(f"Error during shutdown: {e}", exc_info=True)
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

    def stop_supervisor(self, timeout: Optional[float] = None, hard: bool = False, and_worker: bool = False):
        self.ctx.daemon.stop_supervisor(timeout=timeout, hard=hard, and_worker=and_worker)

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
    # Initialize the context once (Singleton pattern)
    context = ServerContext()

    # 2. Register ServerController (Object Server)
    # Allows client to call: manager.ServerController().start()
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
    
    #5. Register stop_worker_event (Test)
    # QueManager.register(
    #     "get_worker_stop_event",
    #     callable=lambda: context.stop_worker_event,
    # )
    


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
