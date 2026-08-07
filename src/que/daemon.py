import logging
import os
import random
import signal
import string
import time
from logging import Logger
from multiprocessing import Process
from multiprocessing.synchronize import Event as EventClass
from typing import Literal

# locals
from src.que.core import (
    DAEMON_NAME,
    SERVER_LOG_PATH,
    DaemonStateDict,
    SweepInfo,
    connect_manager,
)
from src.que.worker import Worker


def generate_run_id(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


class Daemon:
    def __init__(
        self,
        worker: Worker,
        logger: Logger,
        stop_worker_event: EventClass,
        stop_daemon_event: EventClass,
        state: DaemonStateDict,
    ) -> None:
        self.worker = worker
        self.logger = logger
        self.stop_worker_event = stop_worker_event
        self.stop_daemon_event = stop_daemon_event
        self.worker_process: Process | None = None
        self.supervisor_process: Process | None = None
        self.logger.info("Daemon initialized")
        self.set_state(state)

    def _reattach_server_logger(self):
        """Re-attach the server log file handler in a spawned child process."""
        logger = logging.getLogger(DAEMON_NAME)
        if not logger.handlers:  # avoid duplicate handlers on repeated calls
            handler = logging.FileHandler(SERVER_LOG_PATH)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        self.logger = logger

    def get_state(self) -> DaemonStateDict:
        return self.state

    def set_state(self, state: DaemonStateDict, awake_on_state: bool = True) -> None:
        if not hasattr(self, "state"):
            self.state = state

        self.state.update(state)
        if self.state["awake"] and awake_on_state:
            self.logger.info("Daemon state is 'awake', starting supervisor...")
            self.logger.warning(
                "There was likely a power outage/system failure which triggered this"
            )
            # it is likely cur_run will have a run in it, so we must recover this run
            # add the sweep id from saved file
            self.start_supervisor(recover_run=True)

    def monitor_worker(self) -> bool:
        """
        Monitor the worker process until it exits.
        If it exits with a non-zero code and 'stop_on_fail' is False,
        restart the worker. If 'stop_on_fail' is True, return False to
        indicate no restart should occur.
        """
        assert isinstance(self.worker_process, Process)
        self.worker_process.join()

        self.worker.state["working_pid"] = None
        # If worker died naturally (crash or finish)
        exit_code = self.worker_process.exitcode
        if exit_code == 0:
            self.logger.info("Worker process completed successfully.")
        else:
            self.logger.warning(f"Worker process ended with exit code: {exit_code}")

            if self.state["stop_on_fail"]:
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


    def supervise(
        self,
        recover_run: bool = False,
        retries: int = 5,
    ) -> None:
        """This runs inside the CHILD process. It is naturally triggered by start_supervisor
        The worker process is started and monitored here. After it completes successfully, it is restarted.
        If it crashes and 'stop_on_fail' is True, the supervisor exits without restarting.

        Args:
            recover_run (bool, optional): Wether to recover the last failed run. Defaults to False.
            sweep_id (Optional[str], optional): Wether to initialise a sweep. Defaults to None.
        """
        # Initialise shared state through proxy
        manager = connect_manager()
        self.state = manager.get_daemon_state()
        sweep = manager.get_sweep()
        # handle automatic recovery
        if recover_run:
            self.que = manager.get_que()
            self.que.recover_run()

        self._reattach_server_logger()
        self.logger.info(f"Supervisor loop started. PID: {os.getpid()}")

        cnt = 0
        worker_pid = None
        while not self.stop_daemon_event.is_set():
            
            try:
                self.worker_process = Process(
                    target=self.worker.start,args=(sweep,)
                )
                self.worker_process.start()
                worker_pid = self.worker_process.pid
                self.logger.info(f"Worker started with PID: {worker_pid}")

                if not self.monitor_worker():
                    break

                self.worker.cleanup()

            except Exception as e:  # noqa: BLE001
                self.logger.error(f"Supervisor error: {e}")
                cnt += 1
                if cnt > retries:
                    self.logger.error("Too many consecutive errors. Exiting supervisor.")
                    break
                time.sleep(1.0)  # Prevent tight loop on error

        self.logger.info("Supervisor process exiting.")
        self._hard_stop(name="worker", pid=worker_pid, proc=self.worker_process)
        self._reset_process_state(worker=True, supervisor=True)

    def start_supervisor(self, recover_run: bool = False) -> None:
        """
        Start the supervisor process

        :param self: Daemon
        :param recover_run: Boolean flag to first recover the run in cur_run, before launching the worker.
        :type recover_run: bool
        """
        if self.supervisor_process and self.supervisor_process.is_alive():
            self.logger.warning("Supervisor is already running.")
            return

        self.stop_daemon_event.clear()  # Reset event in case it was set previously
        self.stop_worker_event.clear()
        self.state["awake"] = True

        self.supervisor_process = Process(target=self.supervise, args=(recover_run,))
        self.supervisor_process.start()
        self.state["supervisor_pid"] = self.supervisor_process.pid
        self.logger.info(
            f"Supervisor launched (Child PID: {self.supervisor_process.pid})"
        )

    def _kill(self, pid: int, sig: int = signal.SIGTERM) -> bool:
        """Send a signal to a process and handle exceptions."""
        try:
            os.kill(pid, sig)
            self.logger.info(f"Sent signal {sig} to process {pid}.")
            return True
        except ProcessLookupError:
            self.logger.warning(f"Process {pid} does not exist.")
            return True
        except PermissionError:
            self.logger.error(f"No permission to send signal {sig} to process {pid}.")
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to send signal {sig} to process {pid}: {e}")
        return False

    def pid_is_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # process exists but we don't have permission to signal it

    def _try_join(
        self, process: Process | None, timeout: float | None = None
    ) -> bool:
        """Attempt to join a process and return whether it joined"""
        if process is not None:
            process.join(timeout=timeout)
            if process.is_alive():
                self.logger.warning(f"Process {process.pid} did not exit in time.")
                return False
            else:
                self.logger.info(f"Process {process.pid} exited successfully.")
                return True
        return False  # No process to join

    def hard_cleanup(self, pid: int, timeout: int = 10) -> bool:
        """
        Forcefully terminate the pid. Returns its success as a boolean
        """
        if self.pid_is_alive(pid):
            time.sleep(timeout)
        else:
            return True

        if self.pid_is_alive(pid) and not self._kill(pid, sig=signal.SIGTERM):
            time.sleep(timeout)
        else:
            return True

        return not (self.pid_is_alive(pid) and not self._kill(pid, sig=signal.SIGKILL))

    def _reset_process_state(self, worker: bool = False, supervisor: bool = False):
        if worker:
            self.worker_process = None
            self.worker.state["working_pid"] = None
            self.worker.state["task"] = "inactive"
            self.worker.state["current_run_id"] = None

        if supervisor:
            self.state["awake"] = False
            self.supervisor_process = None
            self.state["supervisor_pid"] = None

    def _hard_stop(
        self,
        name: Literal["worker", "daemon"],
        pid: int | None,
        proc: Process | None,
    ):
        self.logger.info(f"Killing {name}...")

        if proc is not None:
            pid = proc.pid
            proc.join(1)

        if pid is not None:
            # 3. Force kill if it's stuck (optional safety net)
            if self.hard_cleanup(pid):
                self.logger.info(f"{name} cleaned up successfully.")
                self._reset_process_state(
                    worker=name == "worker", supervisor=name == "daemon"
                )
            else:
                self.logger.error(f"Failed to kill {name} process: {pid}")
        else:
            self.logger.info(f"No {name} process/pid to kill")

    def stop_proc(
        self,
        name: Literal["worker", "daemon"],
        timeout: float | None = None,
        hard: bool = False,
    ) -> None:
        """Gracefully stop a process"""

        self.logger.info(f"Signaling {name} to stop...")

        if name == "worker":
            self.stop_worker_event.set()
            proc = self.worker_process
            pid = self.worker.state["working_pid"]
            w, d = True, False
        elif name == "daemon":
            self.stop_daemon_event.set()
            proc = self.supervisor_process
            pid = self.state["supervisor_pid"]
            w, d = False, True
        else:
            self.logger.error(f"Invalid name for stop_proc: {name}")
            return

        if proc is not None:
            if self._try_join(proc, timeout=timeout):
                self.logger.info(f"{name} joined successfully.")
                self._reset_process_state(worker=w, supervisor=d)
                return
            else:
                self.logger.warning(
                    f"{name} handle exists but failed to join in timeout"
                )
        else:
            time.sleep(timeout if timeout else 1)

        if pid is not None and self.pid_is_alive(pid):
            self.logger.warning(f"{name} is still alive: {pid}")
            if hard:
                self._hard_stop(name, pid, proc)
        else:
            self.logger.info(f"{name} pid is not alive: {pid}")
            self._reset_process_state(worker=w, supervisor=d)

    def stop_supervisor(
        self,
        timeout: float | None = None,
        hard: bool = False,
        stop_worker: bool = False,
    ) -> None:
        """Gracefully stop the supervisor process"""

        self.state["awake"] = False

        if stop_worker:
            self.stop_proc(name="worker", timeout=timeout, hard=hard)

        self.stop_proc(name="daemon", timeout=timeout, hard=hard)


# --- Manager Registration ---
