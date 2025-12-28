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

class Daemon:
    def __init__(
        self,
        worker: Worker,
        logger: Logger,
        state_path: Union[Path, str] = DAEMON_STATE_PATH,
    ) -> None:
        self.worker: Worker = worker
        self.worker_process: Optional[Process] = None
        self.logger: Logger = logger
        self.state_path: Path = Path(state_path)
        self.state: DaemonState = default_state.copy()
        self.load_state()

    def load_state(self):
        try:
            self.state = read_daemon_state(self.state_path)
            self.state['pid'] = os.getpid()
            self.state['worker_pid'] = None
            self.logger.info(f'Loaded state from: {self.state_path}')
        except Exception as e:
            self.logger.warning(
                f'Ran into an error when loading state: {e}\n'
                "loading from scratch"
            )
            self.state = default_state.copy()
            self.state['pid'] = os.getpid()

    def start(self):
        """Start the training cycle"""
        self.state['awake'] = True
        self.state['pid'] = os.getpid()
        while True:
            try:
                self.worker_process = Process(target=self.worker.start)
                self.worker_process.start()
                self.state['worker_pid'] = self.worker_process.pid
                self.logger.info(f"Started worker process with PID: {self.worker_process.pid}")
                self.worker_process.join()
                self.logger.info("Worker process has finished")
            except Exception as e:
                self.logger.error(f"stopping because of error: {e}")
                break
    
    
class DaemonInterface:
    def __init__(self, logger: Logger, state_path: Union[Path, str] = DAEMON_STATE_PATH) -> None:
        self.logger = logger
        self.state_path: Path = Path(state_path)
        self.daemon_process: Optional[Process] = None
        #Daemon state variables
        self.awake: bool = False 
        self.stop_on_fail: bool = False
        self.worker_pid: Optional[int] = None
        self.daemon_pid: Optional[int] = None
        self.load_state()

    def get_state(self) -> DaemonState:
        """Return state as dictionary"""
        return DaemonState(
            pid=self.pid,
            worker_pid=self.worker_pid,
            stop_on_fail=self.stop_on_fail,
            awake=self.awake
        )
        
    def set_state(self, state: DaemonState):
        """Set state from dictionary"""
        self.awake = state['awake']
        self.stop_on_fail = state['stop_on_fail']
        self.worker_pid = state['worker_pid']
        self.pid = state['pid']
    
    def load_state(self):
        """Read state from file"""
        try:
            state = read_daemon_state(self.state_path)
            self.set_state(state)
            self.logger.info(f'Loaded state from: {self.state_path}')
        except Exception as e:
            self.logger.warning(
                f'Ran into an error when loading state: {e}\n'
                "loading from scratch"
            )
            self.awake = False
            self.stop_on_fail = False
            self.worker_pid = None
            self.pid = None
        
    def save_state(self):
        """Save state to file"""
        with open(self.state_path, 'w') as f:
            json.dump(self.get_state(), f)
        self.logger.info(f'Saved state to: {self.state_path}')
        
    def start_daemon(self, daemon: Daemon):
        """Start the daemon process"""
        self.daemon_process = Process(target=daemon.start)
        self.daemon_process.start()
        self.daemon_pid = self.daemon_process.pid
        self.logger.info(f'Started daemon process with PID: {self.daemon_pid}')
        self.save_state()
        
"""
Things to do:
- Refactor using a new DaemonState class with saving/loading methods
- Add methods to stop the daemon and worker processes
- use a shared dictionary (proxy held by the server) to store state
- Add methods to check the state of the daemon and worker processes

"""

