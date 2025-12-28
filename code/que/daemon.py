from typing import Optional, Union, TypeGuard, Dict, Any, TypedDict
from pathlib import Path
from multiprocessing import Process
from logging import Logger
import json
import os


from .core import WR_LOG_PATH, WR_PATH, WR_MODULE_PATH, DN_LOG_PATH, DN_NAME, DAEMON_STATE_PATH
from .worker import Worker


class DaemonState(TypedDict):
    pid: Optional[int]
    worker_pid: Optional[int]
    stop_on_fail: bool
    awake: bool
    
def is_daemon_state(val: Any) -> TypeGuard[DaemonState]:
    """
    Type guard to check if an arbitrary value is structurally
    compatible with the DaemonState TypedDict.
    """
    # 1. Check if the value is a dictionary
    if not isinstance(val, dict):
        return False

    # 2. Check for the presence of all required keys
    # Since all keys are technically *optional* in the Python dictionary sense
    # but *required* by TypedDict (unless explicitly marked NotRequired),
    # we check for all keys listed in the TypedDict.
    required_keys = DaemonState.__annotations__.keys()
    if not all(key in val for key in required_keys):
        return False

    # 3. Check the type of each value
    # We use .get() here defensively, although the previous check makes it safe
    # to use val[key].

    # Check 'pid' and 'worker_pid' (Optional[int])
    if not (val.get('pid') is None or isinstance(val['pid'], int)):
        return False

    if not (val.get('worker_pid') is None or isinstance(val['worker_pid'], int)):
        return False

    # Check 'stop_on_fail' and 'awake' (bool)
    if not isinstance(val.get('stop_on_fail'), bool):
        return False

    if not isinstance(val.get('awake'), bool):
        return False

    # If all checks pass, it is a DaemonState
    return True
    
def read_daemon_state(state_path: Union[Path, str] = DAEMON_STATE_PATH) -> DaemonState:
    with open(state_path, 'r') as f:
        data = json.load(f)
    if is_daemon_state(data):
        return data
    else:
        raise ValueError(f'Data read from: {state_path} is not compatible with DaemonState')

default_state: DaemonState = {
    'pid': None,
    'worker_pid': None,
    'stop_on_fail': False,
    'awake': False 
}

class DaemonStateHandler:
    def __init__(
        self,
        logger: Logger,
        pid: Optional[int] = None,
        worker_pid: Optional[int] = None,
        stop_on_fail: bool = False,
        awake: bool = False,
        state_path: Union[Path, str] = DAEMON_STATE_PATH,
    ) -> None:
        self.logger = logger
        self.pid: Optional[int] = pid
        self.worker_pid: Optional[int] = worker_pid
        self.stop_on_fail: bool = stop_on_fail
        self.awake: bool = awake
        self.state_path: Path = Path(state_path)
        self.load_state()
        
    def load_state(self) -> None:
        try:
            state = read_daemon_state(self.state_path)
            self.pid = state['pid']
            self.worker_pid = state['worker_pid']
            self.stop_on_fail = state['stop_on_fail']
            self.awake = state['awake']
            self.logger.info(f'Loaded state from: {self.state_path}')
        except Exception as e:
            self.logger.warning(
                f'Ran into an error when loading state: {e}\n'
                "loading from scratch"
            )
            self.pid = None
            self.worker_pid = None
            self.stop_on_fail = False
            self.awake = False
            
    def save_state(self) -> None:
        state: DaemonState = {
            'pid': self.pid,
            'worker_pid': self.worker_pid,
            'stop_on_fail': self.stop_on_fail,
            'awake': self.awake
        }
        with open(self.state_path, 'w') as f:
            json.dump(state, f)
        self.logger.info(f'Saved state to: {self.state_path}')

    def get_pid(self) -> Optional[int]:
        return self.pid
    
    def set_pid(self, pid: Optional[int]) -> None:
        self.pid = pid
        
    def get_worker_pid(self) -> Optional[int]:
        return self.worker_pid
    
    def set_worker_pid(self, worker_pid: Optional[int]) -> None:
        self.worker_pid = worker_pid
        
    def get_stop_on_fail(self) -> bool:
        return self.stop_on_fail
    
    def set_stop_on_fail(self, stop_on_fail: bool) -> None:
        self.stop_on_fail = stop_on_fail
        
    def get_awake(self) -> bool:
        return self.awake
    
    def set_awake(self, awake: bool) -> None:
        self.awake = awake


class Daemon:
    def __init__(
        self,
        worker: Worker,
        logger: Logger,
        state_proxy: DaemonStateHandler,
    ) -> None:
        self.worker: Worker = worker
        self.worker_process: Optional[Process] = None
        self.logger: Logger = logger
        self.state_proxy: DaemonStateHandler = state_proxy
        #setup initial state
        self.state_proxy.set_pid(None) #if just inistalised, then no pid
        self.state_proxy.set_worker_pid(None)
    
    def start(self):
        """Start the training cycle"""
        self.logger.info("Daemon started")
        self.state_proxy.set_pid(os.getpid())
        self.state_proxy.set_awake(True)
        self.state_proxy.save_state()
        while True:
            try:
                self.worker_process = Process(target=self.worker.start)
                self.worker_process.start()
                self.state_proxy.set_worker_pid(self.worker_process.pid)
                self.logger.info(f"Started worker process with PID: {self.worker_process.pid}")
                self.worker_process.join()
                self.logger.info("Worker process has finished")
            except Exception as e:
                self.logger.error(f"stopping because of error: {e}")
                break
    
    
class DaemonInterface:
    def __init__(self, logger: Logger, state_proxy: DaemonStateHandler) -> None:
        self.logger = logger
        self.daemon_process: Optional[Process] = None
        #Daemon state variables
        self.state_proxy: DaemonStateHandler = state_proxy
        
    def load_state(self):
        """Load the daemon state from file"""
        self.state_proxy.load_state()
        
    def save_state(self):
        """Save the daemon state to file"""
        self.state_proxy.save_state()
        
    def start_daemon(self, daemon: Daemon):
        """Start the daemon process"""
        self.daemon_process = Process(target=daemon.start)
        self.daemon_process.start()
        self.daemon_pid = self.daemon_process.pid
        self.logger.info(f'Started daemon process with PID: {self.daemon_pid}')
        self.save_state()
        
"""
Things to do (Test after each):



- Add methods to stop the daemon and worker processes
- use a shared dictionary (proxy held by the server) to store state
- Add methods to check the state of the daemon and worker processes

"""

