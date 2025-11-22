from typing import (
    Optional,
    List,
)
from pathlib import Path
import subprocess
#locals 
from .core import (
    WR_NAME,
    DN_NAME,
    SESH_NAME,
    WR_PATH,
    SR_NAME,
    
)


class tmux_manager:
    def __init__(
        self,
        wr_name: str = WR_NAME,
        dn_name: str = DN_NAME,
        sr_name: str = SR_NAME,
        sesh_name: str = SESH_NAME,
        exec_path: str = WR_PATH,
    ) -> None:
        self.wr_name = wr_name
        self.dn_name = dn_name
        self.sr_name = sr_name
        self.sesh_name = sesh_name
        self.exec_path = exec_path

        ep = Path(exec_path)
        if (not ep.exists()) or (not ep.is_file()):
            # raise ValueError(f"Executable: {exec_path} does not exist")
            print(Warning(f"Executable: {exec_path} does not exist"))

    def setup_tmux_session(self) -> Optional[list[subprocess.CompletedProcess[bytes]]]:
        """Create the que_training tmux session is set up, with windows daemon and worker

        Returns:
			Optional[list[subprocess.CompletedProcess[bytes]]]: A list of successful process outputs, or None if one or both failed.
        """

        create_sesh_cmd = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            self.sesh_name,  # -d for detach
            "-n",
            f"{self.dn_name}",
        ]
        create_wWndw_cmd = [  # daemon window created in first command
            "tmux",
            "new-window",
            "-t",
            self.sesh_name,
            "-n",
            self.wr_name,
        ]

        try:
            o1 = subprocess.run(create_sesh_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(
                "setup_tmux_session ran into an error when creating the session and daemon window: "
            )
            print(e.stderr)
            return
        try:
            o2 = subprocess.run(create_wWndw_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(
                "setup_tmux_session ran into an error when creating the worker window: "
            )
            print(e.stderr)
            return
        return [o1, o2]

    def check_tmux_session(self) -> Optional[list[subprocess.CompletedProcess[bytes]]]:
        """Verify that the que_training tmux session is set up, with windows daemon and worker

        Returns:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Optional[list[subprocess.CompletedProcess[bytes]]]: A list of successful process outputs, or None if one or both failed.
        """
        window_names = [self.dn_name, self.wr_name]
        results = []
        for win_name in window_names:
            tmux_cmd = ["tmux", "has-session", "-t", f"{self.sesh_name}:{win_name}"]
            try:
                results.append(
                    subprocess.run(tmux_cmd, check=True, capture_output=True, text=True)
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"check_tmux_session ran into an error when checking the {win_name} window: "
                )
                print(e.stderr)
                return
        return results

    def join_session(self, wndw: str):
        avail_wndws = [self.dn_name, self.wr_name]
        if wndw not in avail_wndws:
            print(
                f"Window {wndw} not one of validated windows: {', '.join(avail_wndws)}"
            )
            return None

        tmux_cmd = ["tmux", "attach-session", "-t", f"{self.sesh_name}:{wndw}"]
        try:
            return subprocess.run(tmux_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("join_session ran into an error when spawning the worker process: ")
            print(e.stderr)
            return None

    def switch_to_window(self):
        tmux_cmd = ["tmux", "select-window", "-t", f"{self.sesh_name}:{self.wr_name}"]
        try:
            _ = subprocess.run(tmux_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(
                "switch_to_window ran into an error when spawning the worker process: "
            )
            print(e.stderr)

    def _send(
        self, cmd: str, wndw: str
    ) -> Optional[subprocess.CompletedProcess[bytes]]:  # use with caution
        """Send a command to the given window

        Args:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        cmd (str): The command as you would type in the terminal
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        wndw (str): The tmux window

        Returns:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Optional[subprocess.CompletedProcess[bytes]]: The return object of the completed process, or None if failure.
        """
        avail_wndws = [self.dn_name, self.wr_name]
        if wndw not in avail_wndws:
            print(
                f"Window {wndw} not one of validated windows: {', '.join(avail_wndws)}"
            )
            return None
        tmux_cmd = [
            "tmux",
            "send-keys",
            "-t",
            f"{self.sesh_name}:{wndw}",
            cmd,
            "Enter",
        ]
        try:
            return subprocess.run(tmux_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("Send ran into an error when spawning the worker process: ")
            print(e.stderr)

    def start(
        self, mode: str, setting: str, ext_args: Optional[List[str]] = None
    ) -> Optional[subprocess.CompletedProcess[bytes]]:
        """Wrapper for send, specialised to starting the worker or daemon

        Args:
        	mode (str): The mode for quefeather (worker or daemon)
        	setting (str): The setting for the given mode (e.g. sMonitor)

        Raises:
        	ValueError: If mode is not the same as the initialised self variable

        Returns:
        	Optional[subprocess.CompletedProcess[bytes]]: The output of the completed process if successful, otherwise None.
        """
        add_args = [] if ext_args is None else ext_args

        if mode == self.dn_name or mode == self.wr_name:
            cmd = f"{self.exec_path} {mode} {setting}"
            for arg in add_args:
                cmd += arg
            return self._send(cmd, mode)
        else:
            raise ValueError(f"Unknown mode: {mode}")