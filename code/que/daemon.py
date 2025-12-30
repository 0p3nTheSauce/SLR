from typing import Optional, Union, TypeGuard, Dict, Any, TypedDict
from pathlib import Path
from multiprocessing import Process, Event
from multiprocessing.synchronize import Event as EventClass
from logging import Logger
import json
import os
import time

from .core import (
    connect_manager,
    DaemonState,
    DaemonStateHandler,
    WR_LOG_PATH,
    WR_PATH,
    WR_MODULE_PATH,
    DN_LOG_PATH,
    DN_NAME,
    DAEMON_STATE_PATH,
)
from .worker import Worker



class Daemon:
    def __init__(
        self,
        worker: Worker,
        logger: Logger,
        local_state: DaemonStateHandler,
        stop_event: EventClass,
    ) -> None:
        self.worker = worker
        self.logger = logger
        self.local_state = local_state
        self.stop_event = stop_event
        self.worker_process: Optional[Process] = None
        self.supervisor_process: Optional[Process] = None
        
        if self.local_state.awake:
            self.logger.info("Daemon state is 'awake', starting supervisor...")
            self.start_supervisor()

    def monitor_worker(self, state_proxy: DaemonStateHandler) -> bool:
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

            if state_proxy.get_stop_on_fail():
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
        manager = connect_manager()
        state_proxy = manager.get_daemon_state()
        state_proxy.set_pid(os.getpid())
        state_proxy.to_disk() 

        self.logger.info(f"Supervisor loop started. PID: {os.getpid()}")

        while not self.stop_event.is_set():
            try:
                self.worker_process = Process(target=self.worker.start, args=(self.stop_event,))
                self.worker_process.start()
                
                state_proxy.set_worker_pid(self.worker_process.pid)
                state_proxy.to_disk() 

                self.logger.info(f"Worker started with PID: {self.worker_process.pid}")

                if not self.monitor_worker(state_proxy):
                    break                

            except Exception as e:
                self.logger.error(f"Supervisor error: {e}")
                if self.stop_event.is_set():
                    break
                time.sleep(1.0) # Prevent tight loop on error

        # Cleanup before process exit
        state_proxy.set_pid(None)
        state_proxy.set_worker_pid(None)
        state_proxy.set_awake(False)
        state_proxy.to_disk() # Save final state to disk
        self.logger.info("Supervisor process exiting.")

    def start_supervisor(self) -> None:
        """Start the supervisor process"""
        if self.supervisor_process and self.supervisor_process.is_alive():
            self.logger.warning("Supervisor is already running.")
            return

        self.stop_event.clear() # Reset event in case it was set previously
        self.local_state.set_awake(True)#child inherits local state
        self.local_state.to_disk() 

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
            
            self.local_state.awake = False
            self.local_state.pid = None
            self.local_state.worker_pid = None
            
            self.local_state.to_disk()
            self.logger.info("Supervisor stopped.")
        else:
            self.logger.warning("No supervisor process to stop")


