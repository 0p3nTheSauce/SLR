from typing import Optional, Union, TypeGuard, Dict, Any, TypedDict
from pathlib import Path
from multiprocessing import Process, Event
from multiprocessing.synchronize import Event as EventClass
from logging import Logger
import json
import os
import time

from .core import (
    WR_LOG_PATH,
    WR_PATH,
    WR_MODULE_PATH,
    DN_LOG_PATH,
    DN_NAME,
    DAEMON_STATE_PATH,
)
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
    if not (val.get("pid") is None or isinstance(val["pid"], int)):
        return False

    if not (val.get("worker_pid") is None or isinstance(val["worker_pid"], int)):
        return False

    # Check 'stop_on_fail' and 'awake' (bool)
    if not isinstance(val.get("stop_on_fail"), bool):
        return False

    if not isinstance(val.get("awake"), bool):
        return False

    # If all checks pass, it is a DaemonState
    return True


def read_daemon_state(state_path: Union[Path, str] = DAEMON_STATE_PATH) -> DaemonState:
    with open(state_path, "r") as f:
        data = json.load(f)
    if is_daemon_state(data):
        return data
    else:
        raise ValueError(
            f"Data read from: {state_path} is not compatible with DaemonState"
        )


default_state: DaemonState = {
    "pid": None,
    "worker_pid": None,
    "stop_on_fail": False,
    "awake": False,
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
        self.from_disk()

    def from_disk(self) -> None:
        try:
            state = read_daemon_state(self.state_path)
            self.pid = state["pid"]
            self.worker_pid = state["worker_pid"]
            self.stop_on_fail = state["stop_on_fail"]
            self.awake = state["awake"]
            self.logger.info(f"Loaded state from: {self.state_path}")
        except Exception as e:
            self.logger.warning(
                f"Ran into an error when loading state: {e}\nloading from scratch"
            )
            self.pid = None
            self.worker_pid = None
            self.stop_on_fail = False
            self.awake = False

    def to_disk(self) -> None:
        state: DaemonState = {
            "pid": self.pid,
            "worker_pid": self.worker_pid,
            "stop_on_fail": self.stop_on_fail,
            "awake": self.awake,
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f)
        self.logger.info(f"Saved state to: {self.state_path}")

    def get_state(self) -> DaemonState:
        return {
            "pid": self.pid,
            "worker_pid": self.worker_pid,
            "stop_on_fail": self.stop_on_fail,
            "awake": self.awake,
        }

    def set_state(self, state: DaemonState) -> None:
        self.pid = state["pid"]
        self.worker_pid = state["worker_pid"]
        self.stop_on_fail = state["stop_on_fail"]
        self.awake = state["awake"]

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
        stop_event: EventClass,
    ) -> None:
        self.worker = worker
        self.logger = logger
        self.state_proxy = state_proxy
        self.stop_event = stop_event
        self.worker_process: Optional[Process] = None
        self.supervisor_process: Optional[Process] = None
        
        if self.state_proxy.get_awake():
            self.logger.info("Daemon state is 'awake', starting supervisor...")
            self.start_supervisor()

    def monitor_worker(self) -> bool:
        """
        Monitor the worker process until it exits.
        If it exits with a non-zero code and 'stop_on_fail' is False,
        restart the worker. If 'stop_on_fail' is True, return False to
        indicate no restart should occur.
        """
        assert isinstance(self.worker_process, Process)
        self.worker_process.join()

        # If worker died naturally (crash or finish)
        exit_code = self.worker_process.exitcode
        if exit_code == 0:
            self.logger.info("Worker process completed successfully.")
        else:
            self.logger.warning(f"Worker process ended with exit code: {exit_code}")

            if self.state_proxy.get_stop_on_fail():
                self.logger.info("stop_on_fail is True. Not restarting.")
                return False
            
            # Small backoff before restarting to prevent rapid looping on hard crashes
            if not self.stop_event.is_set():
                self.logger.info("Restarting worker in 1 second...")
                time.sleep(1.0)
                
        return True

    def hard_cleanup(self) -> None:
        """
        Forcefully terminate the worker and supervisor processes if they are running.
        """
        if self.worker_process and self.worker_process.is_alive():
            self.logger.info("Forcefully terminating worker process...")
            self.worker_process.terminate()
            self.worker_process.join()

        if self.supervisor_process and self.supervisor_process.is_alive():
            self.logger.info("Forcefully terminating supervisor process...")
            self.supervisor_process.terminate()
            self.supervisor_process.join()


    def supervise(self) -> None:
        #TODO: add max retries
        """
        This runs inside the CHILD process.
        The worker process is started and monitored here. After it completes successfully, it is restarted.
        If it crashes and 'stop_on_fail' is True, the supervisor exits without restarting.
        """
        self.state_proxy.set_pid(os.getpid())
        self.state_proxy.to_disk() 

        self.logger.info(f"Supervisor loop started. PID: {os.getpid()}")

        while not self.stop_event.is_set():
            try:
                self.worker_process = Process(target=self.worker.start, args=(self.stop_event,))
                self.worker_process.start()
                
                self.state_proxy.set_worker_pid(self.worker_process.pid)
                self.state_proxy.to_disk() 

                self.logger.info(f"Worker started with PID: {self.worker_process.pid}")

                if not self.monitor_worker():
                    break                

            except Exception as e:
                self.logger.error(f"Supervisor error: {e}")
                if self.stop_event.is_set():
                    break
                time.sleep(1.0) # Prevent tight loop on error

        # Cleanup before process exit
        self.state_proxy.set_pid(None)
        self.state_proxy.set_worker_pid(None)
        self.state_proxy.to_disk() # Save final state to disk
        self.logger.info("Supervisor process exiting.")

    def start_supervisor(self) -> None:
        """Start the supervisor process"""
        if self.supervisor_process and self.supervisor_process.is_alive():
            self.logger.warning("Supervisor is already running.")
            return

        self.stop_event.clear() # Reset event in case it was set previously
        self.state_proxy.set_awake(True)
        self.state_proxy.to_disk()

        self.supervisor_process = Process(target=self.supervise)
        self.supervisor_process.start()
         
        self.logger.info(f"Supervisor launched (Child PID: {self.supervisor_process.pid})")

    def stop_supervisor(self, timeout: float = 5.0, hard: bool = True) -> None:
        """Gracefully stop the supervisor process"""
        if self.supervisor_process and self.supervisor_process.is_alive():
            self.logger.info("Signaling worker and supervisor to stop...")
            
            # 1. Signal the event
            self.stop_event.set()
            
            # 2. Wait for it to finish gracefully
            self.supervisor_process.join(timeout=timeout)
            
            # 3. Force kill if it's stuck (optional safety net)
            if hard:
                self.hard_cleanup()
            
            self.state_proxy.set_awake(False)
            self.state_proxy.to_disk()
            self.logger.info("Supervisor stopped.")
        else:
            self.logger.warning("No supervisor process to stop")


"""
Things to do (Test after each):
- Add methods to stop the daemon and worker processes
- Add methods to check the state of the daemon and worker processes

"""
