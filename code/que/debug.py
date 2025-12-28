from .server import connect_manager
import multiprocessing as mp

def shared_dict1():
    """Test the used of a dictionary hosted by a multiprocessing.Manager
    Process 1
    
    """
    server = connect_manager()
    shared_dict = server.get_shared_dict()
    print("Initial shared dict:", shared_dict)
    for key in shared_dict:
        shared_dict[key] += 10
    print("Updated shared dict:", shared_dict)
    
def shared_dict2():
    """Test the used of a dictionary hosted by a multiprocessing.Manager
    Process 2
    
    """
    server = connect_manager()
    shared_dict = server.get_shared_dict()
    print("Accessed shared dict:", shared_dict)
    for key in shared_dict:
        shared_dict[key] *= 2
    print("Modified shared dict:", shared_dict)
    
def process_opener():
    p1 = mp.Process(target=shared_dict1)
    p2 = mp.Process(target=shared_dict2)
    
    p1.start()
    p1.join()
    
    p2.start()
    p2.join()
    
if __name__ == '__main__':
    process_opener()