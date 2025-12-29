from .server import connect_manager
import multiprocessing as mp
import time
def shared_dict1():
    """Test the used of a dictionary hosted by a multiprocessing.Manager
    Process 1
    
    """
    server = connect_manager()
    shared_dict = server.get_shared_dict()
    print("Initial shared dict:", shared_dict)
    for key in shared_dict.keys():
        shared_dict[key] += 10
    print("Updated shared dict:", shared_dict)
    
def shared_dict2():
    """Test the used of a dictionary hosted by a multiprocessing.Manager
    Process 2
    
    """
    server = connect_manager()
    shared_dict = server.get_shared_dict()
    print("Accessed shared dict:", shared_dict)
    for key in shared_dict.keys():
        shared_dict[key] *= 2
    print("Modified shared dict:", shared_dict)
    
def process_opener():
    p1 = mp.Process(target=shared_dict1)
    p2 = mp.Process(target=shared_dict2)
    
    p1.start()
    p1.join()
    
    p2.start()
    p2.join()
    
def client_logic1():
    manager = connect_manager()

    # 1. Get the Controller Object
    controller = manager.DaemonController()
    
    # 2. Get the State Object
    state_handler = manager.DaemonStateHandler()

    print("Initial State:", state_handler.get_state())

    # 3. Operate on the Controller
    print("Starting daemon...")
    controller.start()
    
    print("Updated State:", state_handler.get_state())
    
def client_logic2():
    manager = connect_manager()

    # 1. Get the Controller Object
    controller = manager.DaemonController()
    
    # 2. Get the State Object
    state_handler = manager.DaemonStateHandler()

    print("Current State:", state_handler.get_state())

    # 3. Operate on the Controller
    print("Stopping daemon...")
    controller.stop()
    
    print("Final State:", state_handler.get_state())

    
if __name__ == '__main__':
    # process_opener()
    # idle_daemon()
    # client_logic1()
    client_logic2()