from typing import Protocol, TYPE_CHECKING, Optional
from multiprocessing.managers import BaseManager, DictProxy

from multiprocessing import Process
import time
import logging
from logging import Logger

from .core import Que, SR_LOG_PATH, QUE_NAME, DN_NAME, SR_NAME, WR_NAME
from .daemon import Daemon
from .worker import Worker


if TYPE_CHECKING:
    class QueManagerProtocol(Protocol):
        # def get_Que(self) -> Que: ...
        # def get_Daemon(self) -> Daemon: ..
        def get_shared_dict(self) -> dict: ...
        

class QueManager(BaseManager): 
    pass


def connect_manager(max_retries=5, retry_delay=2) -> "QueManagerProtocol":
    """Connect to the Queue manager (returns manager, not Que instance)"""
    # QueManager.register('get_Que')
    # QueManager.register('get_Daemon')
    QueManager.register('get_shared_dict')
    
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
# que_logger = logging.getLogger(QUE_NAME)
# dn_logger = logging.getLogger(DN_NAME)
# wr_logger = logging.getLogger(WR_NAME)

    
_shared_dict = {
    't1': 0,
    't2': 1,
    't3': 2,
}

def get_shared_dict():
    """Return the shared dict instance"""
    return _shared_dict

QueManager.register('get_shared_dict', callable=get_shared_dict, proxytype=DictProxy)

    
if __name__ == '__main__':
    start_server()
    