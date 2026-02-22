import shlex
import configs
# from .server import connect_manager
import multiprocessing as mp
import time
from .shell import QueShell
# from que.shell import QueShell
from .core import Que, connect_manager, _get_basic_logger
from .tmux import tmux_manager
from typing import Optional, cast    
import torch
import torch.nn as nn
import gc
import getpass
import subprocess
import time
from pathlib import Path


def client_logic1():
    manager = connect_manager()

    # 1. Get the Controller Object
    controller = manager.get_server_context()
    daemon = manager.get_daemon()


    print("Initial State:", controller.get_state())

    # 3. Operate on the Controller
    print("Starting daemon...")
    daemon.start_supervisor()
    
    print("Updated State:", controller.get_state())
    
def client_logic2():
    manager = connect_manager()

    # 1. Get the Controller Object
    controller = manager.get_server_context()
    daemon = manager.get_daemon()


    print("Current State:", controller.get_state())

    # 3. Operate on the Controller
    print("Stopping daemon...")
    daemon.stop_supervisor()
    
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
    server_controller = server.get_server_context()
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
        except Exception as _:
            pass
    
def daemon_start_supervisor():
    manager = connect_manager()
    
    daemon = manager.get_daemon()
    daemon.start_supervisor()
    
def daemon_stop_supervisor(t:float, hard:bool=False):
    manager = connect_manager()
    
    daemon = manager.get_daemon()
    daemon.stop_supervisor(stop_worker=True, timeout=t, hard=hard)
    
    
def try_except_finally():
    try:
        print("In try block")
        raise ValueError("An error occurred")
    except ValueError as ve:
        print(f"Caught exception: {ve}")
    finally:
        print("In finally block")

def test_create():
    que = Que(_get_basic_logger())
    arg = "R(2+1)D_18 asl100 6"
    args = shlex.split(arg)
    maybe_args = configs.take_args(sup_args=args)
    if isinstance(maybe_args, tuple):
        admin_info, wandb_info = maybe_args
        que.create_run(admin_info, wandb_info)
    else:
        print("oops")


def test_is_daemon_state():
    from .core import is_daemon_state, DaemonState
    
    eg = DaemonState(
        awake=True,
        stop_on_fail=True,
        supervisor_pid=None
    )
    
    beg = {
        'stop_on_fail': True,
        'supervisor_pid': 1
    }
    
    beg1 = {
        'awake': 1,
        'stop_on_fail': True,
        'supervisor_pid': 1
    }
    
    print(is_daemon_state(eg))
    print(is_daemon_state(beg))
    print(is_daemon_state(beg1))
    
    
def test_shared_dict_proc1():
    manager = connect_manager()
    wstate = manager.get_worker_state()
    print('before changing')
    print(wstate)
    print('after changing')
    wstate['current_run_id'] = 'debugging'
    print(wstate)
    
def test_shared_dict_proc2():
    manager = connect_manager()
    wstate = manager.get_worker_state()
    print('before changing')
    print(wstate)
    print('after changing')
    wstate['current_run_id'] = 'debugging_2'
    print(wstate)
    
def test_shared_dict_proc_opener():
    p1 = mp.Process(target=test_shared_dict_proc1)
    p1.start()
    p1.join()
    p2 = mp.Process(target=test_shared_dict_proc2)
    p2.start()
    p2.join()
    
def reset_state():
    manager = connect_manager()
    server = manager.get_server_context()
    server.set_state(server=None, daemon=None, worker={'task': 'inactive', 'current_run_id': None, 'working_pid': None, 'exception': None})
    







def connect_manager_ssh( 
    host: Optional[str] = None,
    ssh_user: Optional[str] = None,
    ssh_key: Optional[Path] = None,
    port: int = 50000,
    authkey: Optional[bytes] = None,
    max_retries=5,
    retry_delay=2,):
    
    if authkey is None:
        password = getpass.getpass("Queue server password: ")
        authkey = password.encode()

    if ssh_user is None:
        ssh_user = Path.home().name

    ssh_cmd = [
        "ssh",
        "-N",                          # don't execute a command, just tunnel
        "-L", f"50000:localhost:{port}", # local port -> remote port
        "-o", "ExitOnForwardFailure=yes",
    ]
    if ssh_key:
        ssh_cmd += ["-i", ssh_key.expanduser().as_posix()]
    ssh_cmd.append(f"{ssh_user}@{host}")

    tunnel = subprocess.Popen(ssh_cmd)
    time.sleep(2)  # give the tunnel a moment to establish

    try:
        manager = connect_manager(host="127.0.0.1", port=50000, authkey=authkey, max_retries=max_retries, retry_delay=retry_delay)
        return manager, tunnel
    except Exception:
        tunnel.terminate()
        raise

def ssh_connect_and_test(
    host: Optional[str] = None,
    ssh_user: Optional[str] = None,
    ssh_key: Optional[Path] = None,
    port: int = 50000,
    authkey: Optional[bytes] = None,
    max_retries=5,
    retry_delay=2,
):
    manager, tunnel = connect_manager_ssh(
        host=host,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        port=port,
        authkey=authkey,
        max_retries=max_retries,
        retry_delay=retry_delay
    )
    try:
        print("Connected to manager via SSH tunnel!")
        # You can add more tests here to interact with the manager
        que_shell = QueShell(manager)
        try:
            que_shell.cmdloop()
        except KeyboardInterrupt:
            print("\n[INFO] Exiting queShell due to keyboard interrupt.")
    finally:
        tunnel.terminate()
        print("SSH tunnel closed.")


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
    # server_context_daemon_stop(t=1)
    # server_controller_stop()
    # daemon_start_supervisor()
    # daemon_stop_supervisor(t=1, hard=True)
    # try_except_finally()
    # test_create()
    # test_is_daemon_state()
    # test_shared_dict_proc_opener()
    # reset_state()
    # key_path = Path.home() / ".ssh" / "ed25519"
    
    # if key_path.exists():
    #     ssh_connect_and_test(
    #         ssh_key=str(key_path)
    #     )
    # else:   
    #     print("SSH key not found, falling back to password auth")    
    #     ssh_connect_and_test()
        
    ssh_connect_and_test(host='146.231.88.174', authkey=b'abracadabra')