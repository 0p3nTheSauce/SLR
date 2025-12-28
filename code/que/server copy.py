from typing import Protocol, TYPE_CHECKING, Optional
from multiprocessing.managers import BaseManager, DictProxy

from multiprocessing import Process
import time
import logging
from logging import Logger

from .core import Que, SR_LOG_PATH, QUE_NAME, DN_NAME, SR_NAME, WR_NAME
from .daemon import Daemon, DaemonInterface, DaemonStateHandler, DaemonState
from .worker import Worker


if TYPE_CHECKING:
    class QueManagerProtocol(Protocol):
        def get_shared_dict(self) -> dict: ...
        def start_daemon(self) -> None: ...
        def stop_daemon(self) -> None: ...
        def get_daemon_state(self) -> DaemonState: ...


class QueManager(BaseManager): 
    pass


def connect_manager(max_retries=5, retry_delay=2) -> "QueManagerProtocol":
    """Connect to the Queue manager (returns manager, not Que instance)"""
    QueManager.register('get_shared_dict')
    QueManager.register('start_daemon')
    QueManager.register('stop_daemon')
    QueManager.register('get_daemon_state')
    
    
    
    for _ in range(max_retries):
        try:
            m = QueManager(address=('localhost', 50000), authkey=b'abracadabra')
            m.connect()
            return m  # type: ignore
        except ConnectionRefusedError:
            print(f"Queue server not ready, retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            
    raise RuntimeError(
        "Cannot connect to Queue server. "
        "Start it with: python Que_server.py"
    )

def start_server():
    m = QueManager(address=('localhost', 50000), authkey=b'abracadabra')
    s = m.get_server()
    print("Debug server started on localhost:50000")
    try:
        s.serve_forever()
    except KeyboardInterrupt:
        print("Debug server shutdown by user")
    except Exception as e:
        print(f' Debug Server failed due to {e}')
        return

    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=SR_LOG_PATH
)

logger = logging.getLogger(SR_NAME)
que_logger = logging.getLogger(QUE_NAME)
dn_logger = logging.getLogger(DN_NAME)
dn_state_logger = logging.getLogger(f"{DN_NAME} State")
dn_int_logger = logging.getLogger(f"{DN_NAME} Interface")
wr_logger = logging.getLogger(WR_NAME)

que = Que(logger=que_logger)
worker = Worker(que=que, logger=wr_logger)

daemon_state = DaemonStateHandler(logger=dn_state_logger)
daemon = Daemon(worker=worker, logger=dn_logger, state_proxy=daemon_state)
daemon_interface = DaemonInterface(logger=dn_int_logger, state_proxy=daemon_state)
    
_shared_dict = {
    't1': 0,
    't2': 1,
    't3': 2,
}

def get_shared_dict():
    """Return the shared dict instance"""
    return _shared_dict

def start_daemon():
    """Start the daemon process"""
    daemon_interface.start_daemon(daemon)
    
def stop_daemon():
    """Stop the daemon process"""
    daemon_interface.stop_daemon()
    
def get_daemon_state():
    """Get the current state of the daemon"""
    return daemon_state.get_state()

QueManager.register('get_shared_dict', callable=get_shared_dict, proxytype=DictProxy)
QueManager.register('start_daemon', callable=start_daemon)
QueManager.register('stop_daemon', callable=stop_daemon)
QueManager.register('get_daemon_state', callable=get_daemon_state)
    
if __name__ == '__main__':
    start_server()
    