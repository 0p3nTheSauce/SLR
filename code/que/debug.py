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
    
def idle_daemon():
    """Test the Daemon process from the server
    
    """
    server = connect_manager()
    server.start_daemon()

def daemon_state_stop():
    """Test the Daemon process from the server:
    Check if daemon is idle, then stop it.
    
    """
    server = connect_manager()
    state = server.get_daemon_state()
    print("Daemon state :", state)
    if state['awake']:
        server.stop_daemon()
        time.sleep(2)  # Wait for daemon to stop
        state = server.get_daemon_state()
        print("Daemon state after stopping:", state)

def idle_daemon_state_stop():
    """Test the Daemon process from the server:
    Start the daemon, check its state, then stop it.
    
    """
    server = connect_manager()
    server.start_daemon()
    time.sleep(2)  # Wait for daemon to start
    state = server.get_daemon_state()
    print("Daemon state after starting:", state)
    server.stop_daemon()
    time.sleep(2)  # Wait for daemon to stop
    state = server.get_daemon_state()
    print("Daemon state after stopping:", state)
    
if __name__ == '__main__':
    # process_opener()
    # idle_daemon()
    daemon_state_stop()