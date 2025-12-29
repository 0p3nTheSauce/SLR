from .server import connect_manager
import multiprocessing as mp
import time

    
def client_logic1():
    manager = connect_manager()

    # 1. Get the Controller Object
    controller = manager.DaemonController()
    


    print("Initial State:", controller.get_state())

    # 3. Operate on the Controller
    print("Starting daemon...")
    controller.start()
    
    print("Updated State:", controller.get_state())
    
def client_logic2():
    manager = connect_manager()

    # 1. Get the Controller Object
    controller = manager.DaemonController()
    


    print("Current State:", controller.get_state())

    # 3. Operate on the Controller
    print("Stopping daemon...")
    controller.stop()
    
    print("Final State:", controller.get_state())

def que_client():
    manager = connect_manager()

    # 1. Get the Controller Object
    controller = manager.DaemonController()
    

    print("Current State:", controller.get_state())

    # 3. Operate on the Controller
    print("Stopping daemon...")
    controller.stop()
    
    print("Final State:", controller.get_state())

    
if __name__ == '__main__':
    # process_opener()
    # idle_daemon()
    # client_logic1()
    client_logic2()