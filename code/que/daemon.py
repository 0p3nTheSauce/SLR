from typing import Optional, Union, TypeGuard, Dict, Any
import logging
from logging import Logger
from pathlib import Path
import sys
import time
from .core import WR_LOG_PATH, WR_PATH, WR_MODULE_PATH

import subprocess
from typing import TextIO


def _proc_is_running(proc: Optional[subprocess.Popen]) -> TypeGuard[subprocess.Popen]:
    if proc is None:
        return False
    return proc.poll() is None

def _log_is_open(log: Optional[TextIO]) -> TypeGuard[TextIO]:
    return log is not None and not log.closed
    
class Daemon:
    def __init__(
        self,
        logger: Logger,
        worker_path: Union[str, Path] = WR_MODULE_PATH,
        worker_log_path: Union[str, Path] = WR_LOG_PATH,
        # daemon_log_path: Union[str, Path] = DN_LOG_PATH
    ) -> None:
        self.worker_path: str | Path = worker_path
        self.worker_log_path: str | Path = worker_log_path
        self.worker_log_file: Optional[TextIO] = None
        self.worker_process: Optional[subprocess.Popen] = None
        self.logger = logger

    def start_worker(self) -> None:
        """Start the worker if not already running"""
        if _proc_is_running(self.worker_process):
            self.logger.warning("Worker already running")
            return

        self.worker_log_file = open(self.worker_log_path, "a")
        
        try:
            self.worker_process = subprocess.Popen(
                [sys.executable, "-u", "-m", str(self.worker_path)],
                stdout=self.worker_log_file,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            self.logger.info(f"Worker started with PID: {self.worker_process.pid}")
            
            # Give it a moment to crash if there's an immediate problem
            time.sleep(0.1)
            
            # Check if it's still running
            if self.worker_process.poll() is not None:
                # Process already exited
                return_code = self.worker_process.returncode
                self.logger.error(f"Worker failed to start (exit code: {return_code})")
                self.worker_process = None
                self.worker_log_file.close()
                self.worker_log_file = None
                raise RuntimeError(f"Worker process terminated immediately with exit code {return_code}")
                
        except FileNotFoundError as e:
            self.logger.error(f"Failed to start worker: {e}")
            if self.worker_log_file and not self.worker_log_file.closed:
                self.worker_log_file.close()
            self.worker_log_file = None
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error starting worker: {e}")
            if self.worker_log_file and not self.worker_log_file.closed:
                self.worker_log_file.close()
            self.worker_log_file = None
            raise

    def stop_worker(self, timeout: int = 10) -> bool:
        """Stop the worker gracefully, force kill if necessary"""
        if not _proc_is_running(self.worker_process):
            self.logger.warning("No worker process to stop")
            return False
        
        try:
            self.logger.info("Attempting graceful worker shutdown...")
            self.worker_process.terminate()
            self.worker_process.wait(timeout=timeout)
            self.logger.info("Worker stopped gracefully")
            return True
        except subprocess.TimeoutExpired:
            self.logger.warning("Worker didn't stop gracefully, force killing...")
            self.worker_process.kill()
            self.worker_process.wait()
            self.logger.info("Worker force killed")
            return True
        finally:
            self.worker_process = None
            if not _log_is_open(self.worker_log_file):
                self.logger.error('Worker log file was not open while worker is running')
            else:
                self.worker_log_file.close()
            self.worker_log_file = None

    def restart_worker(self) -> None:
        """Restart the worker process"""
        self.logger.info("Restarting worker...")
        self.stop_worker()
        time.sleep(1)  # Brief delay before restart
        self.start_worker()

    def get_worker_status(self) -> Dict[str, Any]:
        """Get detailed worker status information"""
        if not _proc_is_running(self.worker_process):
            return {
                "running": False,
                "pid": None,
                "return_code": self.worker_process.returncode
                if self.worker_process
                else None,
            }
        return {"running": True, "pid": self.worker_process.pid, "return_code": None}

    def tail_worker_log(self, lines: int = 10) -> list[str]:
        """Get last N lines from worker log"""
        try:
            with open(self.worker_log_path, "r") as f:
                all_lines = f.readlines()
                return all_lines[-lines:]
        except FileNotFoundError:
            self.logger.warning(f"Log file not found: {self.worker_log_path}")
            return []

    def clear_worker_log(self) -> None:
        """Clear the worker log file"""
        try:
            open(self.worker_log_path, "w").close()
            self.logger.info(f"Cleared worker log: {self.worker_log_path}")
        except Exception as e:
            self.logger.error(f"Failed to clear log: {e}")

