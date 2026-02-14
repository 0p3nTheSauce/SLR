from multiprocessing.managers import BaseManager
from typing import Protocol
from .core import Que
from .daemon import Daemon
from .worker import Worker
from .server import ServerContext
import time

class QueManagerProtocol(Protocol):
    def get_que(self) -> Que: ...
    #Testing 
    def get_daemon(self) -> Daemon: ...
    def get_worker(self) -> Worker: ...
    def get_server_context(self) -> ServerContext: ...

class QueManager(BaseManager):
    pass


def connect_manager(max_retries=5, retry_delay=2) -> "QueManagerProtocol":
    """
    Useful helper for clients to connect to the QueManager server.

    :param max_retries: Maximum number of connection attempts
    :param retry_delay: Delay between retries in seconds
    :return: Connected QueManager instance
    :rtype: QueManagerProtocol
    """
    QueManager.register("ServerController")
    QueManager.register("get_que")
    QueManager.register("get_server_state_handler")
    #Testing
    QueManager.register("get_daemon") 
    QueManager.register("get_server_context")
    

    for _ in range(max_retries):
        try:
            m = QueManager(address=("localhost", 50000), authkey=b"abracadabra")
            m.connect()
            return m  # type: ignore
        except ConnectionRefusedError:
            print(f"Queue server not ready, retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    raise RuntimeError("Cannot connect to Queue server.")