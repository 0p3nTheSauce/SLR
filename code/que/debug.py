# from .server import connect_manager
import multiprocessing as mp
import time
from .shell import QueShell
from que.shell import QueShell
from .core import Que, connect_manager, _get_basic_logger
from .tmux import tmux_manager
from typing import cast    
import torch
import torch.nn as nn
import gc


def client_logic1():
    manager = connect_manager()

    # 1. Get the Controller Object
    controller = manager.ServerController()
    


    print("Initial State:", controller.get_state())

    # 3. Operate on the Controller
    print("Starting daemon...")
    controller.start()
    
    print("Updated State:", controller.get_state())
    
def client_logic2():
    manager = connect_manager()

    # 1. Get the Controller Object
    controller = manager.ServerController()
    


    print("Current State:", controller.get_state())

    # 3. Operate on the Controller
    print("Stopping daemon...")
    controller.stop_supervisor()
    
    print("Final State:", controller.get_state())

def que_client():
    manager = connect_manager()

    #get the shared que
    que = manager.get_que()
    cast(Que, que)
    print("Current Que State:")
    Que.print_runs(que.list_runs('cur_run'))

def tmuxer():
    tman = tmux_manager()
    tman.join_session()

def timestamp():
    q = Que(_get_basic_logger())
    q.save_state(timestamp=True)
    
def check_gpu_memory():
    """Helper to check GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

def gpu_worker_fixed():
    """Simulates PyTorch GPU work with proper cleanup"""
    print("Worker: Starting GPU work...")
    
    try:
        model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000)
        ).cuda()
        
        data = torch.randn(100, 1000).cuda()
        
        for i in range(10):
            output = model(data)
            loss = output.sum()
            loss.backward()
        
        print("Worker: Work complete")
        check_gpu_memory()
        
    finally:
        # Proper cleanup
        print("Worker: Cleaning up...")
        # del model, data
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()

def sim_leak():
    # CRITICAL: Use 'spawn' method
    # mp.set_start_method('spawn', force=True)
    
    print("=== Initial State ===")
    # check_gpu_memory()
    
    print("\n=== Starting Process (with fix) ===")
    p = mp.Process(target=gpu_worker_fixed)
    p.start()
    p.join()
    
    print("\n=== Starting Process (with fix2) ===")
    p = mp.Process(target=gpu_worker_fixed)
    p.start()
    p.join()


    print("\n=== Memory should be cleared! ===")
    print("\n=== After Process Completes ===")
    time.sleep(1)
    check_gpu_memory()
    
def constant():
    from .core import SERVER_MODULE_PATH
    print(SERVER_MODULE_PATH)
    
def activate_conda_env(env_name: str):
    tman = tmux_manager()
    tman.activate_conda_env(env_name)

def show_help():
    
    server = connect_manager()
    shell = QueShell(server)
    daemon_parser = shell._get_daemon_parser()
    print(daemon_parser.description)



def reconnect():
    server = connect_manager()
    server_controller = server.ServerController()
    print(server_controller.get_state())
    ready = input("Retry? (y/n): ")
    if ready.lower() == 'y':
        _cleanup(server_controller)
        print("Reconnecting...")
        reconnect()
    else:
        print("Exiting.")


def _cleanup(old_server_controller=None):
    """Properly disconnect old proxies and reconnect"""
    # Step 1: Try to clean up old proxy connections
    if old_server_controller is not None:
        try:
            # Close the underlying connection
            old_server_controller._close()
        except:
            pass
    
def server_context_daemon_start():
    manager = connect_manager()
    
    
    context = manager.get_server_context()
    context.start()
    
def server_context_daemon_stop(t:int):
    manager = connect_manager()
    
    context = manager.get_server_context()
    context.stop_supervisor(stop_worker=True, timeout=t)
    
def server_controller_stop():
    manager = connect_manager()
    
    server_controller = manager.ServerController()
    server_controller.stop_supervisor(stop_worker=True)
    
if __name__ == '__main__':
    # process_opener()
    # idle_daemon()
    # client_logic1()
    # client_logic2()
    # que_client()
    # tmuxer()
    # timestamp()
    # sim_leak()
    # constant()
    # activate_conda_env("wlasl")
    # show_help()
    # reconnect()
    # server_context_daemon_start()
    server_context_daemon_stop(t=1)
    # server_controller_stop()
    
    