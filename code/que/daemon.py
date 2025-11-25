import logging
from pathlib import Path
import sys
import time
from .core import WR_LOG_PATH, WR_PATH
from typing import Optional
import subprocess


class Daemon:
    def __init__(self, worker_path: str | Path = WR_PATH, worker_log: str | Path = WR_LOG_PATH) -> None:
        self.worker_path = worker_path
        self.worker_log = worker_log
        self.worker_process: Optional[subprocess.Popen] = None
        self.logger = logging.getLogger(__name__)
        
    def run_worker(self) -> subprocess.Popen:
        """Start a new worker process"""
        return subprocess.Popen(
            [sys.executable, '-u', '-m', str(self.worker_path)],
            stdout=open(self.worker_log, 'a'),
            stderr=subprocess.STDOUT,
            bufsize=0, 
        )
    
    def start_worker(self) -> None:
        """Start the worker if not already running"""
        if self.is_worker_running():
            self.logger.warning("Worker already running")
            return
        
        self.worker_process = self.run_worker()
        self.logger.info(f"Worker started with PID: {self.worker_process.pid}")
    
    def stop_worker(self, timeout: int = 10) -> bool:
        """Stop the worker gracefully, force kill if necessary"""
        if not self.is_worker_running():
            self.logger.warning("No worker process to stop")
            return False
        assert self.worker_process is not None #checked by is worker running
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
    
    def restart_worker(self) -> None:
        """Restart the worker process"""
        self.logger.info("Restarting worker...")
        self.stop_worker()
        time.sleep(1)  # Brief delay before restart
        self.start_worker()
    
    def is_worker_running(self) -> bool:
        """Check if worker process is currently running"""
        if self.worker_process is None:
            return False
        
        poll_result = self.worker_process.poll()
        return poll_result is None
    
    def get_worker_status(self) -> dict:
        """Get detailed worker status information"""
        if not self.is_worker_running():
            return {
                'running': False,
                'pid': None,
                'return_code': self.worker_process.returncode if self.worker_process else None
            }
        assert self.worker_process is not None #checked by is worker running
        return {
            'running': True,
            'pid': self.worker_process.pid,
            'return_code': None
        }
    
    def monitor_worker(self, restart_on_crash: bool = True) -> None:
        """Monitor worker and optionally restart on crash"""
        while True:
            if not self.is_worker_running():
                if self.worker_process is not None:
                    # Worker crashed
                    return_code = self.worker_process.returncode
                    self.logger.error(f"Worker crashed with return code: {return_code}")
                    
                    if restart_on_crash:
                        self.logger.info("Attempting to restart worker...")
                        self.start_worker()
                    else:
                        break
                else:
                    # Worker never started
                    self.logger.info("Starting worker...")
                    self.start_worker()
            
            time.sleep(5)  # Check every 5 seconds
    
    def tail_worker_log(self, lines: int = 10) -> list[str]:
        """Get last N lines from worker log"""
        try:
            with open(self.worker_log, 'r') as f:
                all_lines = f.readlines()
                return all_lines[-lines:]
        except FileNotFoundError:
            self.logger.warning(f"Log file not found: {self.worker_log}")
            return []
    
    def clear_worker_log(self) -> None:
        """Clear the worker log file"""
        try:
            open(self.worker_log, 'w').close()
            self.logger.info(f"Cleared worker log: {self.worker_log}")
        except Exception as e:
            self.logger.error(f"Failed to clear log: {e}")
            
    def health_check(self) -> dict:
        """Comprehensive health check of daemon and worker"""
        return {
            'worker': self.get_worker_status(),
            'log_file_exists': Path(self.worker_log).exists(),
            'worker_script_exists': Path(self.worker_path).exists()
        }
    
    def safe_shutdown(self) -> None:
        """Safely shut down daemon and worker"""
        self.logger.info("Initiating safe shutdown...")
        self.stop_worker(timeout=30)  # Longer timeout for graceful shutdown
        self.logger.info("Daemon shutdown complete")