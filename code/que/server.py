from typing import Protocol, TYPE_CHECKING
from multiprocessing.managers import BaseManager
import time
import logging

from .core import Que, SR_LOG_PATH, QUE_NAME, DN_NAME, SR_NAME
from .daemon import Daemon


if TYPE_CHECKING:
    class QueManagerProtocol(Protocol):
        def get_Que(self) -> Que: ...
        def get_Daemon(self) -> Daemon: ...


class QueManager(BaseManager): 
    pass


def connect_manager(max_retries=5, retry_delay=2) -> "QueManagerProtocol":
    """Connect to the Queue manager (returns manager, not Que instance)"""
    QueManager.register('get_Que')
    QueManager.register('get_Daemon')
    
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
    

def main():
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=SR_LOG_PATH
    )

    logger = logging.getLogger(SR_NAME)
    que_logger = logging.getLogger(QUE_NAME)
    dn_logger = logging.getLogger(DN_NAME)
    
    
    # Initialize instances
    que_instance = Que(que_logger)
    daemon_instance = Daemon(dn_logger)

    def get_Que():
        return que_instance
    
    def get_Daemon():
        return daemon_instance
    
    QueManager.register('get_Que', callable=get_Que)
    QueManager.register('get_Daemon', callable=get_Daemon)
    
    m = QueManager(address=('localhost', 50000), authkey=b'abracadabra')
    s = m.get_server()
    logger.info("Queue server started on localhost:50000")
    print("Queue server started on localhost:50000")
    try:
        s.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutdown by user")
    except Exception as e:
        logger.critical(f' Server failed due to {e}')
        que_instance.save_state()
        return
if __name__ == '__main__':
    main()