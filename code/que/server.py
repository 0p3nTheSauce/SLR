from multiprocessing.managers import BaseManager
from typing import Protocol, TYPE_CHECKING
from .core import que

import time

if TYPE_CHECKING:
    class QueueManagerProtocol(Protocol):
        def get_que(self) -> que: ...
        def reload_que(self, preserve_state: bool = True) -> str: ...

class QueueManager(BaseManager): 
    pass


def connect_manager(max_retries=5, retry_delay=2) -> "QueueManagerProtocol":
    """Connect to the queue manager (returns manager, not que instance)"""
    QueueManager.register('get_que')
    QueueManager.register('reload_que')
    
    for _ in range(max_retries):
        try:
            m = QueueManager(address=('localhost', 50000), authkey=b'abracadabra')
            m.connect()
            return m #type: ignore
        except ConnectionRefusedError:
            print(f"Queue server not ready, retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            
    raise RuntimeError(
        "Cannot connect to queue server. "
        "Start it with: python que_server.py"
    )
        
        
def main():
    
    state = {'que_instance': que()}

    def get_que():
        return state['que_instance']
    
    def reload_que(preserve_state=True):
        """Hot reload the queue instance"""
        old_que = state['que_instance']
        
        if preserve_state:
            old_que.save_state()
            state['que_instance'] = que() #automatically loads saved state
            return "Reloaded successfully (state preserved)"
        else:
            state['que_instance'] = que()
            return "Reloaded successfully (fresh instance)"
    
    QueueManager.register('get_que', callable=get_que)
    QueueManager.register('reload_que', callable=reload_que)
    
    m = QueueManager(address=('localhost', 50000), authkey=b'abracadabra')
    s = m.get_server()
    print("Queue server started on localhost:50000")
    s.serve_forever()
    
if __name__ == '__main__':
    main()