import inspect
from typing import (
    Optional,
    List,
    Callable,
)
from logging import Logger, INFO, ERROR
import subprocess

from contextlib import contextmanager
#locals 
from .core import (
    SESH_NAME,
)



class tmux_manager:
    def __init__(
        self,
        sesh_name: str = SESH_NAME,
        logger: Optional[Logger] = None,
    ) -> None:
        self.sesh_name = sesh_name
        self.logger = logger

        with self.subprocess_error_handler("Initial tmux session check"):
            res = self.check_tmux_session()
            if res is None:
                self.print_logger("Tmux session not found, creating new session...")
                res = self.setup_tmux_session()
                if res is not None:
                    self.print_logger("Tmux session created.")


    def print_logger(self, msg: str, level: int = INFO, include_caller: bool = False) -> None:
        """Print to logger if exists, else print to console

        Args:
            msg (str): Message to print
            level (int): Logging level (default: INFO)
            include_caller (bool): Whether to prepend caller function name
        """
        if include_caller:
            caller = inspect.currentframe()
            assert caller is not None
            caller = caller.f_back
            assert caller is not None
            caller = caller.f_code.co_name
            msg = f"[{caller}] {msg}"
        
        if self.logger is not None:
            self.logger.log(level, msg)
        else:
            print(msg)
            
    @contextmanager
    def subprocess_error_handler(self, cmd: str):
        """Context manager to handle subprocess errors

        Args:
            cmd (str): The command being run
        """
        try:
            yield
        except subprocess.CalledProcessError as e:
            self.print_logger(f"Error occurred while running command: {cmd}", ERROR, include_caller=True)
            self.print_logger(f"Error details: {e.stderr}", ERROR, include_caller=True)

    def setup_tmux_session(self) -> Optional[subprocess.CompletedProcess[bytes]]:
        """
        Create the train tmux session
        
        :return: CompletedProcess if successful, None if failed
        :rtype: CompletedProcess[bytes] | None
        """
        #tmux new-session \; split-window -h
        create_sesh_cmd = [
            "tmux", 
            "new",
            "-d",
            "-s",
            self.sesh_name
        ]
            
        with self.subprocess_error_handler(" ".join(create_sesh_cmd)):
            return subprocess.run(create_sesh_cmd, check=True)

    def check_tmux_session(self) -> Optional[subprocess.CompletedProcess[bytes]]:
        """
        Check if the tmux session exists

        :return: CompletedProcess if successful, None if failed
        :rtype: CompletedProcess[bytes] | None
        """
        tmux_cmd = ["tmux", "has-session", "-t", f"{self.sesh_name}"]
        with self.subprocess_error_handler(" ".join(tmux_cmd)):
            return subprocess.run(tmux_cmd, check=True)
        
    def join_session(self) -> Optional[subprocess.CompletedProcess[bytes]]:
        """
        Join the tmux session

        :return: CompletedProcess if successful, None if failed
        :rtype: CompletedProcess[bytes] | None
        """

        tmux_cmd = ["tmux", "attach-session", "-t", f"{self.sesh_name}"]
    
        with self.subprocess_error_handler(" ".join(tmux_cmd)):
            return subprocess.run(tmux_cmd, check=True)

    def _send(
        self, cmd: str
    ) -> Optional[subprocess.CompletedProcess[bytes]]:  # use with caution
        """Send a command to the given window

        Args:
            cmd (str): The command as you would type in the terminal

        Returns:
            Optional[subprocess.CompletedProcess[bytes]]: The return object of the completed process, or None if failure.
        """
        tmux_cmd = [
            "tmux",
            "send-keys",
            "-t",
            f"{self.sesh_name}",
            cmd,
            "Enter",
        ]
        with self.subprocess_error_handler(" ".join(tmux_cmd)):
            return subprocess.run(tmux_cmd, check=True)
    
    def activate_conda_env(self, env_name: str) -> Optional[subprocess.CompletedProcess[bytes]]:
        """Activate the given conda environment in the tmux session

        Args:
            env_name (str): The name of the conda environment to activate

        Returns:
            Optional[subprocess.CompletedProcess[bytes]]: The return object of the completed process, or None if failure.
        """
        cmd = f"conda activate {env_name}"
        return self._send(cmd)
            
    def start_python_module(self, executble_path: str) -> Optional[subprocess.CompletedProcess[bytes]]:
        """Start the tmux session with the given command

        Args:
            exec_path (str): The command to execute in the tmux session

        Returns:
            Optional[subprocess.CompletedProcess[bytes]]: The return object of the completed process, or None if failure.
        """
        cmd = f"python -m {executble_path}"
        return self._send(cmd)  
    
    
    