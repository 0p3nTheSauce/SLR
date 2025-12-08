from typing import Protocol, TYPE_CHECKING, cast
from multiprocessing.managers import BaseManager
import time
import logging

from .core import Que, SR_LOG_PATH, QUE_NAME, DN_NAME, SR_NAME
from .daemon import Daemon



if TYPE_CHECKING:
    class QueManagerProtocol(Protocol):
        def get_Que(self) -> Que: ...
        def get_Daemon(self) -> Daemon: ...
        def reload_Que(self, preserve_state: bool = True) -> None: ...
        def reload_Daemon(self) -> None: ...


class QueManager(BaseManager): 
    pass


def connect_manager(max_retries=5, retry_delay=2) -> "QueManagerProtocol":
    """Connect to the Queue manager (returns manager, not Que instance)"""
    QueManager.register('get_Que')
    QueManager.register('get_Daemon')
    QueManager.register('reload_Que')
    QueManager.register('reload_Daemon')
    
    for _ in range(max_retries):
        try:
            m = QueManager(address=('localhost', 50000), authkey=b'abracadabra')
            m.connect()
            return m #type: ignore
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
        filename=SR_LOG_PATH  # Optional: log to file
    )

    logger = logging.getLogger(SR_NAME)
    que_logger = logging.getLogger(QUE_NAME)
    dn_logger = logging.getLogger(DN_NAME)
    
    
    state = {'Que_instance': Que(que_logger),
             'Daemon_instance': Daemon(dn_logger)}

    def get_Que():
        return state['Que_instance']
    
    def get_Daemon():
        return state['Daemon_instance']
    
    def reload_Que(preserve_state=True):
        """Hot reload the Que instance"""
        old_Que = cast(Que, state['Que_instance'])
        
        if preserve_state:
            old_Que.save_state()
            state['Que_instance'] = Que(que_logger) #automatically loads saved state
            logger.info("Reloaded successfully (state preserved)")
        else:
            state['Que_instance'] = Que(que_logger)
            logger.info("Reloaded successfully (fresh instance)")
            
    def reload_Daemon():
        """Hot reload Daemon instance"""
        old_Daemon = cast(Daemon, state['Daemon_instance'])
        old_Daemon.stop_worker()
        
        state['Daemon_instance'] = Daemon(dn_logger)
        
    QueManager.register('get_Que', callable=get_Que)
    QueManager.register('get_Daemon', callable=get_Daemon)
    QueManager.register('reload_Que', callable=reload_Que)
    QueManager.register('reload_Daemon', callable=reload_Daemon)
    
    m = QueManager(address=('localhost', 50000), authkey=b'abracadabra')
    s = m.get_server()
    print("Queue server started on localhost:50000")
    s.serve_forever()
    
if __name__ == '__main__':
    main()