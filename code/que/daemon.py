from typing import Optional, Union, TypeGuard, Dict, Any, TypedDict
from pathlib import Path
from multiprocessing import Process
from logging import Logger
import json
import os
import signal

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
        
    def get_state(self) -> DaemonState:
        return {
            'pid': self.pid,
            'worker_pid': self.worker_pid,
            'stop_on_fail': self.stop_on_fail,
            'awake': self.awake
        }
        
    def set_state(self, state: DaemonState) -> None:
        self.pid = state['pid']
        self.worker_pid = state['worker_pid']
        self.stop_on_fail = state['stop_on_fail']
        self.awake = state['awake']

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
    def __init__(self, worker, logger, state_proxy):
        self.worker = worker
        self.logger = logger
        self.state_proxy = state_proxy
        self.worker_process: Optional[Process] = None

    def handle_signal(self, signum, frame):
        """Signal handler only sets the flag and kills the sub-process"""
        self.logger.info("Termination signal received...")
        self.state_proxy.set_awake(False)
        if self.worker_process and self.worker_process.is_alive():
            self.worker_process.terminate()

    def start(self):
        # Attach the signal handler
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal) # Handle Ctrl+C too

        self.state_proxy.set_pid(os.getpid())
        self.state_proxy.set_awake(True)

        while self.state_proxy.get_awake(): # Check the flag here
            try:
                self.worker_process = Process(target=self.worker.start)
                self.worker_process.start()
                self.state_proxy.set_worker_pid(self.worker_process.pid)
                
                # Wait for worker, but check 'running' flag periodically
                while self.worker_process.is_alive():
                    self.worker_process.join(timeout=1.0)
                    if not self.state_proxy.get_awake():
                        break

            except Exception as e:
                self.logger.error(f"Error: {e}")
                break

        self.cleanup()

    def cleanup(self):
        """Standardized cleanup after the loop breaks"""
        if self.worker_process and self.worker_process.is_alive():
            self.worker_process.join()
        self.state_proxy.set_pid(None)
        self.state_proxy.set_awake(False)
        self.logger.info("Daemon shutdown complete")
    
    
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
        
    def get_state(self) -> DaemonState:
        """Get the current daemon state"""
        return self.state_proxy.get_state()
    
    def set_state(self, state: DaemonState):
        """Set the current daemon state"""
        self.state_proxy.set_state(state)
        
    def start_daemon(self, daemon: Daemon):
        """Start the daemon process"""
        self.daemon_process = Process(target=daemon.start)
        self.daemon_process.start()
        self.daemon_pid = self.daemon_process.pid
        self.logger.info(f'Started daemon process with PID: {self.daemon_pid}')
        self.save_state()
        
    def stop_daemon(self):
        """Stop the daemon process"""
        if self.daemon_process is not None:
            self.daemon_process.terminate()
            self.logger.info(f'Stopped daemon process with PID: {self.daemon_pid}')
            self.state_proxy.set_pid(None)
            self.state_proxy.set_worker_pid(None)
            self.state_proxy.set_awake(False)
            self.save_state()
        else:
            self.logger.warning('No daemon process to stop')

"""
Things to do (Test after each):
- Add methods to stop the daemon and worker processes
- Add methods to check the state of the daemon and worker processes

"""

