from typing import (
    Optional,
    List
)
import subprocess
#locals 
from .core import (
    WORKER_NAME,
    SERVER_NAME,
    SESH_NAME,
)


class tmux_manager:
    def __init__(
        self,
        worker_name: str = WORKER_NAME,
        server_name: str = SERVER_NAME,
        sesh_name: str = SESH_NAME,
    ) -> None:
        self.worker_name = worker_name
        self.server_name = server_name
        self.sesh_name = sesh_name
        
        try:
            res = self.check_tmux_session_panes()
            if res is None:
                print("Tmux session not found, creating new session...")
                res = self.setup_tmux_session_panes()
                if res is not None:
                    print("Tmux session created.")
        except Exception as e:
            print(f"Error during tmux session check/setup: {e}")
        

    def setup_tmux_session_panes(self) -> Optional[List[subprocess.CompletedProcess[bytes]]]:
        """
        Create the que_training tmux session is set up, with panes for daemon and worker
        
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
        split_window_cmd = [
            "tmux",
            "split-window",
            "-h", 
            "-t",
            f"{self.sesh_name}"
        ]
            
        try:
            o1 = subprocess.run(create_sesh_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(
                "setup_tmux_session ran into an error when creating the session"
            )
            print(e.stderr)
            return
        
        try:
            o2 = subprocess.run(split_window_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(
                "setup_tmux_session ran into an error when splitting the window: "
            )
            print(e.stderr)
            return
        
        return [o1, o2]

    def check_tmux_session_panes(self) -> Optional[list[subprocess.CompletedProcess[bytes]]]:
        
        results = []
        
        tmux_cmd = ["tmux", "has-session", "-t", f"{self.sesh_name}"]
        try:
            results.append(
                subprocess.run(tmux_cmd, check=True, capture_output=True, text=True)
            )
        except subprocess.CalledProcessError as e:
            print(
                f"check_tmux_session ran into an error when checking the seshion {self.sesh_name}: "
            )
            print(e.stderr)
            return
        return results
    
    def join_session_pane(self) -> Optional[subprocess.CompletedProcess[str]]:
        tmux_cmd = ["tmux", "attach-session", "-t", f"{self.sesh_name}"]
        try:
            return subprocess.run(tmux_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("join_session ran into an error when spawning the worker process: ")
            print(e.stderr)
            return 




    # def _send(
    #     self, cmd: str, wndw: str
    # ) -> Optional[subprocess.CompletedProcess[bytes]]:  # use with caution
    #     """Send a command to the given window

    #     Args:
    #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     cmd (str): The command as you would type in the terminal
    #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     wndw (str): The tmux window

    #     Returns:
    #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     Optional[subprocess.CompletedProcess[bytes]]: The return object of the completed process, or None if failure.
    #     """
    #     avail_wndws = [self.server_name, self.worker_name]
    #     if wndw not in avail_wndws:
    #         print(
    #             f"Window {wndw} not one of validated windows: {', '.join(avail_wndws)}"
    #         )
    #         return None
    #     tmux_cmd = [
    #         "tmux",
    #         "send-keys",
    #         "-t",
    #         f"{self.sesh_name}:{wndw}",
    #         cmd,
    #         "Enter",
    #     ]
    #     try:
    #         return subprocess.run(tmux_cmd, check=True)
    #     except subprocess.CalledProcessError as e:
    #         print("Send ran into an error when spawning the worker process: ")
    #         print(e.stderr)

    # def setup_tmux_session_windows(self) -> Optional[list[subprocess.CompletedProcess[bytes]]]:
    #     """Create the que_training tmux session is set up, with windows daemon and worker

    #     Returns:
	# 		Optional[list[subprocess.CompletedProcess[bytes]]]: A list of successful process outputs, or None if one or both failed.
    #     """

    #     create_sesh_cmd = [
    #         "tmux",
    #         "new-session",
    #         "-d",
    #         "-s",
    #         self.sesh_name,  # -d for detach
    #         "-n",
    #         f"{self.server_name}",
    #     ]
    #     create_wWndw_cmd = [  # daemon window created in first command
    #         "tmux",
    #         "new-window",
    #         "-t",
    #         self.sesh_name,
    #         "-n",
    #         self.worker_name,
    #     ]

    #     try:
    #         o1 = subprocess.run(create_sesh_cmd, check=True)
    #     except subprocess.CalledProcessError as e:
    #         print(
    #             "setup_tmux_session ran into an error when creating the session and daemon window: "
    #         )
    #         print(e.stderr)
    #         return
    #     try:
    #         o2 = subprocess.run(create_wWndw_cmd, check=True)
    #     except subprocess.CalledProcessError as e:
    #         print(
    #             "setup_tmux_session ran into an error when creating the worker window: "
    #         )
    #         print(e.stderr)
    #         return
    #     return [o1, o2]

    # def check_tmux_session_windows(self) -> Optional[list[subprocess.CompletedProcess[bytes]]]:
    #     """Verify that the que_training tmux session is set up, with windows daemon and worker

    #     Returns:
    #         Optional[list[subprocess.CompletedProcess[bytes]]]: A list of successful process outputs, or None if one or both failed.
    #     """
    #     window_names = [self.server_name, self.worker_name]
    #     results = []
    #     for win_name in window_names:
    #         tmux_cmd = ["tmux", "has-session", "-t", f"{self.sesh_name}:{win_name}"]
    #         try:
    #             results.append(
    #                 subprocess.run(tmux_cmd, check=True, capture_output=True, text=True)
    #             )
    #         except subprocess.CalledProcessError as e:
    #             print(
    #                 f"check_tmux_session ran into an error when checking the {win_name} window: "
    #             )
    #             print(e.stderr)
    #             return
    #     return results

    # def join_session_windows(self, wndw: str):
    #     avail_wndws = [self.server_name, self.worker_name]
    #     if wndw not in avail_wndws:
    #         print(
    #             f"Window {wndw} not one of validated windows: {', '.join(avail_wndws)}"
    #         )
    #         return None

    #     tmux_cmd = ["tmux", "attach-session", "-t", f"{self.sesh_name}:{wndw}"]
    #     try:
    #         return subprocess.run(tmux_cmd, check=True)
    #     except subprocess.CalledProcessError as e:
    #         print("join_session ran into an error when spawning the worker process: ")
    #         print(e.stderr)
    #         return None
