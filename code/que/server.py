from typing import Protocol, TYPE_CHECKING, Optional
from multiprocessing.managers import BaseManager, DictProxy

from multiprocessing import Process, Event
import time
import logging
from logging import Logger

from .core import Que, SR_LOG_PATH, QUE_NAME, DN_NAME, SR_NAME, WR_NAME
from .daemon import Daemon, DaemonStateHandler, DaemonState
from .worker import Worker


if TYPE_CHECKING:
    class DaemonControllerProtocol(Protocol):
        def start(self) -> None: ...
        def stop(self) -> None: ...

    class QueManagerProtocol(Protocol):
        def get_shared_dict(self) -> dict: ...
        # Notice we now return Objects, not just void functions
        def DaemonStateHandler(self) -> DaemonStateHandler: ...
        def DaemonController(self) -> DaemonControllerProtocol: ...

class QueManager(BaseManager): 
    pass


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
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=SR_LOG_PATH
        )

        que_logger = logging.getLogger(QUE_NAME)
        dn_logger = logging.getLogger(DN_NAME)
        dn_state_logger = logging.getLogger(f"{DN_NAME} State")
        wr_logger = logging.getLogger(WR_NAME)



        # Initialize Logic
        self.que = Que(logger=que_logger)
        self.worker = Worker(que=self.que, logger=wr_logger)
        self._shared_dict = {'t1': 0, 't2': 1, 't3': 2}
        self.stop_event = Event()


        # State and Daemon
        self.daemon_state = DaemonStateHandler(logger=dn_state_logger)
        self.daemon = Daemon(worker=self.worker, logger=dn_logger, state_proxy=self.daemon_state, stop_event=self.stop_event)
     

class DaemonController:
    """
    The Object Server wrapper.
    Instead of registering functions, we register this class.
    """
    def __init__(self, context: ServerContext):
        self.ctx = context

    def start(self):
        self.ctx.daemon.start_supervisor()

    def stop(self):
        self.ctx.daemon.stop_supervisor()

# --- Registration Logic ---

def setup_manager():
    """
    Configures the QueManager with the ServerContext.
    """
    # Initialize the context once (Singleton pattern)
    context = ServerContext()

    # 1. Register DaemonStateHandler (Object Server)
    # Allows client to call: manager.DaemonStateHandler().get_state()
    QueManager.register(
        'DaemonStateHandler', 
        callable=lambda: context.daemon_state
    )

    # 2. Register DaemonController (Object Server)
    # Allows client to call: manager.DaemonController().start()
    QueManager.register(
        'DaemonController',
        callable=lambda: DaemonController(context)
    )

    # 3. Register Shared Dict (Resource)
    QueManager.register(
        'get_shared_dict', 
        callable=lambda: context._shared_dict, 
        proxytype=DictProxy
    )

# --- Server Startup ---

def start_server():
    setup_manager()
    
    # Note: We bind to localhost for security, change to 0.0.0.0 to expose externally
    m = QueManager(address=('localhost', 50000), authkey=b'abracadabra')
    s = m.get_server()
    
    print("Object Server started on localhost:50000")
    print("Exposed Objects: DaemonStateHandler, DaemonController")
    
    try:
        s.serve_forever()
    except KeyboardInterrupt:
        print("Server shutdown by user")

# --- Client Connection Helper ---

def connect_manager(max_retries=5, retry_delay=2) -> "QueManagerProtocol":
    """
    Useful helper for clients to connect to the QueManager server.
    
    :param max_retries: Maximum number of connection attempts
    :param retry_delay: Delay between retries in seconds
    :return: Connected QueManager instance
    :rtype: QueManagerProtocol
    """
    QueManager.register('DaemonStateHandler')
    QueManager.register('DaemonController')
    QueManager.register('get_shared_dict')

    for _ in range(max_retries):
        try:
            m = QueManager(address=('localhost', 50000), authkey=b'abracadabra')
            m.connect()
            return m # type: ignore
        except ConnectionRefusedError:
            print(f"Queue server not ready, retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            
    raise RuntimeError("Cannot connect to Queue server.")

if __name__ == '__main__':
    start_server()

