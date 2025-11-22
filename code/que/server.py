from multiprocessing.managers import BaseManager
from quewing import que
import time

class QueueManager(BaseManager): 
    pass

def connect_que(max_retries=5, retry_delay=2) -> que:
    QueueManager.register('get_que')
    
    for _ in range(max_retries):
        try:
            m = QueueManager(address=('localhost', 50000), authkey=b'abracadabra')
            m.connect()
            return m.get_que()  # type: ignore
        except ConnectionRefusedError:
            print(f"Queue server not ready, retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            
    raise RuntimeError(
        "Cannot connect to queue server. "
        "Start it with: python que_server.py"
    )
        
        
def main():
    que_instance = que()
    QueueManager.register('get_que', callable=lambda: que_instance)  # Returns INSTANCE
    # QueueManager.register('get_que', callable=lambda: que)
    m = QueueManager(address=('localhost', 50000), authkey=b'abracadabra')
    s = m.get_server()
    s.serve_forever()
    
if __name__ == '__main__':
    main()