from typing import Optional, TypedDict
from multiprocessing import Process
from multiprocessing.synchronize import Event as EventClass
from logging import Logger
import os
import time
from .core import (
    DAEMON_NAME,
    WORKER_NAME,
    connect_manager,
    DaemonState,
)
from .worker2 import Worker



class Daemon:
    def __init__(
        self,
        worker: Worker,
        logger: Logger,
        stop_worker_event: EventClass,
        stop_daemon_event: EventClass,
        state: DaemonState,
    ) -> None:
        self.worker = worker
        self.logger = logger
        self.stop_worker_event = stop_worker_event
        self.stop_daemon_event = stop_daemon_event
        self.worker_process: Optional[Process] = None
        self.supervisor_process: Optional[Process] = None
        self.state = state
        
        self.logger.info("Daemon initialized")

        if self.awake:
            self.logger.info("Daemon state is 'awake', starting supervisor...")
            self.start_supervisor()

    def get_state(self) -> DaemonState:
        return self.state

    def set_state(self, state: DaemonState) -> None:
        self.state = state

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

            if self.state['stop_on_fail']:
                self.logger.info("stop_on_fail is True. Not restarting.")
                return False

            # Small backoff before restarting to prevent rapid looping on hard crashes
            if not self.stop_daemon_event.is_set():
                self.logger.info("Restarting worker in 1 second...")
                time.sleep(1.0)
            else:
                self.logger.info("Stop event detected, not restarting worker.")
                return False

        return True

    def hard_cleanup(self, supervisor: bool = True, worker: bool = True) -> None:
        """
        Forcefully terminate the worker and supervisor processes if they are running.
        """
        if worker and self.worker_process and self.worker_process.is_alive():
            self.logger.info("Forcefully terminating worker process...")
            self.worker_process.terminate()
            self.worker_process.join()

        if (
            supervisor
            and self.supervisor_process
            and self.supervisor_process.is_alive()
        ):
            self.logger.info("Forcefully terminating supervisor process...")
            self.supervisor_process.terminate()
            self.supervisor_process.join()

    def supervise(self) -> None:
        """
        This runs inside the CHILD process.
        The worker process is started and monitored here. After it completes successfully, it is restarted.
        If it crashes and 'stop_on_fail' is True, the supervisor exits without restarting.
        """
        
        self.logger.info(f"Supervisor loop started. PID: {os.getpid()}")

        while not self.stop_daemon_event.is_set():
            try:
                # self.worker_process = Process(
                #     target=self.worker.start, args=(self.stop_worker_event,)
                # )
                self.worker_process = Process(
                    target=self.worker.start,
                )
                self.worker_process.start()

                self.logger.info(f"Worker started with PID: {self.worker_process.pid}")

                if not self.monitor_worker():
                    break

                self.worker.cleanup()
                
            except Exception as e:
                self.logger.error(f"Supervisor error: {e}")
                if self.stop_daemon_event.is_set():
                    break
                time.sleep(1.0)  # Prevent tight loop on error

        self.logger.info("Supervisor process exiting.")

    def start_supervisor(self) -> None:
        """Start the supervisor process"""
        if self.supervisor_process and self.supervisor_process.is_alive():
            self.logger.warning("Supervisor is already running.")
            return

        self.stop_daemon_event.clear()  # Reset event in case it was set previously
        self.stop_worker_event.clear()
        self.state['awake'] = True
        

        self.supervisor_process = Process(target=self.supervise)
        self.supervisor_process.start()
        self.state['supervisor_pid'] = self.supervisor_process.pid
        self.logger.info(
            f"Supervisor launched (Child PID: {self.supervisor_process.pid})"
        )

    def stop_worker(self, timeout: Optional[float] = None, hard: bool = False) -> None:
        self.logger.info("Signaling worker to stop...")

        # 1. Signal the event
        self.stop_worker_event.set()
        
        if self.worker_process is not None and self.worker_process.is_alive():

            # 2. Wait for it to finish gracefully
            self.worker_process.join(timeout=timeout)

            # 3. Force kill if it's stuck (optional safety net)
            if hard:
                self.hard_cleanup(supervisor=False, worker=True)

            if not self.worker_process.is_alive():
                self.worker_process = None
                self.worker.state['working_pid'] = None

            self.logger.info("Worker stopped.")
        else:
            self.logger.warning("No worker process to stop")
            self.worker_process = None
            self.worker.state['working_pid'] = None

    def stop_supervisor(
        self,
        timeout: Optional[float] = None,
        hard: bool = False,
        stop_worker: bool = False,
    ) -> None:
        """Gracefully stop the supervisor process"""
        
        if stop_worker:
            self.stop_worker(timeout=timeout, hard=hard)

        
        if self.supervisor_process and self.supervisor_process.is_alive():
            self.logger.info("Signaling supervisor to stop...")

            # 1. Signal the event
            self.stop_daemon_event.set()
                
            self.supervisor_process.join(timeout=timeout)

            # 3. Force kill if it's stuck (optional safety net)
            if hard:
                self.hard_cleanup()

            self.awake = False
            
            if not self.supervisor_process.is_alive():
                self.supervisor_process = None
                self.supervisor_pid = None
                
            self.logger.info("Supervisor stopped.")

        else:
            self.logger.warning("No supervisor process to stop")
            self.supervisor_process = None
            self.supervisor_pid = None
            
        


# --- Manager Registration ---
