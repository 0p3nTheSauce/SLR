from multiprocessing.managers import BaseManager
import subprocess 
import time
import logging
import sys
from pathlib import Path
from typing import Protocol, TYPE_CHECKING
from .core import Que, SR_LOG_PATH, WR_PATH, WR_LOG_PATH


logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	filename=SR_LOG_PATH  # Optional: log to file
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    class QueueManagerProtocol(Protocol):
        def get_que(self) -> Que: ...
        def reload_que(self, preserve_state: bool = True) -> str: ...

class QueueManager(BaseManager): 
    pass


def connect_manager(max_retries=5, retry_delay=2) -> "QueueManagerProtocol":
    """Connect to the Queue manager (returns manager, not Que instance)"""
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
        "Cannot connect to Queue server. "
        "Start it with: python Que_server.py"
    )
    
class daemon:
    def __init__(self, worker_path: str | Path = WR_PATH, worker_log: str | Path = WR_LOG_PATH) -> None:
        self.worker_path = worker_path
        self.worker_log = worker_log
        self.worker = self.run_worker()
        
    def run_worker(self) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, '-u', '-m', self.worker_path],
            stdout=open(self.worker_log, 'a'), #this does not allow us to close the file, but that's okay - Luke         
            stderr=subprocess.STDOUT,
            bufsize=0
        )
    # return_code = proc.wait()
    
    

def main():
    
    state = {'Que_instance': Que(logger)}

    def get_Que():
        return state['Que_instance']
    
    def reload_Que(preserve_state=True):
        """Hot reload the Queue instance"""
        old_Que = state['Que_instance']
        
        if preserve_state:
            old_Que.save_state()
            state['Que_instance'] = Que(logger) #automatically loads saved state
            logger.info("Reloaded successfully (state preserved)")
        else:
            state['Que_instance'] = Que(logger)
            logger.info("Reloaded successfully (fresh instance)")
    
    QueueManager.register('get_Que', callable=get_Que)
    QueueManager.register('reload_Que', callable=reload_Que)
    
    m = QueueManager(address=('localhost', 50000), authkey=b'abracadabra')
    s = m.get_server()
    print("Queue server started on localhost:50000")
    s.serve_forever()
    
if __name__ == '__main__':
    main()