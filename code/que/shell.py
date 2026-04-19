import webbrowser
import cmd as cmdLib
import shlex
from typing import Callable, Optional, List, Any
import argparse
import configs
import time
from contextlib import contextmanager

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm
from rich.text import Text
from rich.syntax import Syntax
from rich import box
import io
import sys
import json
from pathlib import Path
import subprocess
import getpass
import traceback

# locals
from utils import gpu_manager
from .core import (
    TO_RUN,
    GenExp,
    CUR_RUN,
    QUE_LOCATIONS,
    SYNONYMS,
    connect_manager,
    ServerState,
    # WorkerState,
    # DaemonState,
    TRAINING_LOG_PATH,
    SERVER_LOG_PATH,
    RUN_PATH,
    QueManagerProtocol,
    QueDupExp,
)
from configs import get_avail_splits, ENTITY, PROJECT_BASE, get_train_parser, ZFILL
from .tmux import tmux_manager


# ---------------------------------------------------------------------------
# Criterion parsing
# ---------------------------------------------------------------------------

SAFE_GLOBALS = {
    "__builtins__": {},  # block all builtins
    # whitelist only what criteria might legitimately need
    "len": len,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "isinstance": isinstance,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
}

def parse_criterion(expr: str) -> Callable[[Any], bool]:
    """Evaluate a lambda string in a restricted namespace."""
    result = eval(expr, SAFE_GLOBALS)  # noqa: S307
    if not callable(result):
        raise ValueError(f"Criterion must be callable, got: {type(result)}")
    return result # type: ignore[return-value]



class QueShell(cmdLib.Cmd):
    avail_locs = QUE_LOCATIONS + list(SYNONYMS.keys())
    # recover_locs = [FAIL_RUNS, CUR_RUN] maybe make this if you feel like it

    def __init__(
        self,
        server: QueManagerProtocol,
        auto_save: bool = True,
    ) -> None:
        super().__init__()
        # Pretty stuff
        self.console = Console()
        self._show_banner()
        self.prompt = "\x01\033[1;36m\x02(que)$\x01\033[0m\x02 "
        self.intro = ""
        # Core Objects
        self.tmux_man = tmux_manager()
        self.auto_save = auto_save
        # - proxy objects
        self.que = server.get_que()
        self.daemon = server.get_daemon()
        self.worker = server.get_worker()
        self.server_context = server.get_server_context()

        # - parsing
        self._parser_factories = {
            "create": lambda: configs.get_train_parser(
                prog="create", desc="Create a new training run"
            ),
            "add": self._get_add_parser,
            "remove": self._get_remove_parser,
            "clear": self._get_clear_parser,
            "list": self._get_list_parser,
            "quit": self._get_quit_parser,
            "shuffle": self._get_shuffle_parser,
            "move": self._get_move_parser,
            "edit": self._get_edit_parser,
            "copy": self._get_copy_parser,
            "display": self._get_display_parser,
            "daemon": self._get_daemon_parser,
            "server": self._get_server_parser,
            "worker": self._get_worker_parser,
            "logs": self._get_log_parser,
            "load": self._get_load_parser,
            "save": self._get_save_parser,
            "recover": self._get_recover_parser,
            "wandb": self._get_wandb_parser,
        }

    # Exception handling

    @contextmanager
    def unwrap_exception(self, message: str = "", error: str = ""):
        try:
            yield
            if message:
                self.console.print(f"[bold green]✓ {message} [/bold green]")
        except Exception as e:
            if error:
                self.console.print(
                    f"[bold red]✗ {error} : {e} [/bold red]", style="red"
                )
            else:
                self.console.print(f"[bold red]✗ {e} [/bold red]", style="red")

            # NOTE: cool idea to make this optional
            # Print full traceback for debugging
            self.console.print("[dim]" + traceback.format_exc() + "[/dim]")

    def _reconnect_proxies(self) -> None:
        """Reconnect the server controller and que proxies"""
        self.server_context._close()  # type: ignore
        self.que._close()  # type: ignore
        self.daemon._close()  # type: ignore
        self.worker._close()  # type: ignore
        server = connect_manager()
        self.server_context = server.get_server_context()
        self.que = server.get_que()
        self.daemon = server.get_daemon()
        self.worker = server.get_worker()

    # Cmd overrides

    def onecmd(self, line):
        """Override to handle connection errors gracefully"""
        try:
            return super().onecmd(line)
        except (EOFError, ConnectionError, BrokenPipeError, OSError) as e:
            self.console.print(
                f"\n[bold yellow][WARNING][/bold yellow] Connection lost: {e}"
            )
            print("Attempting to reconnect...")
            try:
                self._reconnect_proxies()
                self.console.print("[bold green][OK][/bold green] Reconnected!\n")
            except Exception as reconnect_error:
                self.console.print(
                    f"[bold red][ERROR][/bold red] Reconnection failed: {reconnect_error}"
                )
                return False

            # Retry is now outside the inner try/except, so if it fails
            # it loops back to the top-level handler on the next onecmd call
            # instead of being misreported as a reconnection failure.
            try:
                return super().onecmd(line)
            except (EOFError, ConnectionError, BrokenPipeError, OSError):
                self.console.print(
                    "[bold yellow][WARNING][/bold yellow] Command failed after reconnect — server may still be starting. Try again."
                )
                return False

    def _show_banner(self):
        """Display a fancy welcome banner"""
        banner = Text()
        banner.append("╔═══════════════════════════════════════╗\n", style="bold cyan")
        banner.append("║          ", style="bold cyan")
        banner.append("QueShell", style="bold yellow")
        banner.append("                     ║\n", style="bold cyan")
        banner.append("║   ", style="bold cyan")
        banner.append("Queue Management System", style="bold white")
        banner.append("             ║\n", style="bold cyan")
        banner.append("╚═══════════════════════════════════════╝", style="bold cyan")

        self.console.print(banner)
        self.console.print(
            "\nType [bold cyan]help[/bold cyan] or [bold cyan]?[/bold cyan] to list commands.\n"
        )

    def do_help(self, arg):
        """Override help to provide Rich formatted help"""
        if arg:
            parser = self._get_parser(arg)
            if parser:
                # Capture help text and display with Rich

                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                parser.print_help()
                sys.stdout = old_stdout
                help_text = buffer.getvalue()

                syntax = Syntax(
                    help_text, "text", theme="monokai", background_color="default"
                )
                self.console.print(
                    Panel(syntax, title=f"Help: {arg}", border_style="cyan")
                )
            else:
                super().do_help(arg)
        else:
            # Show custom help menu
            self._show_help_menu()

    def _show_help_menu(self):
        """Display a beautiful help menu"""
        table = Table(title="Available Commands", box=box.ROUNDED, border_style="cyan")
        table.add_column("Command", style="bold yellow", width=12)
        table.add_column("Description", style="white")

        for key, value in self._parser_factories.items():
            cmd_parser = value()
            desc = (
                cmd_parser.description if cmd_parser.description else "No description"
            )
            table.add_row(key, desc)

        self.console.print(table)
        self.console.print(
            "\n[dim]Tip: Use 'help <command>' for detailed information about a specific command[/dim]\n"
        )

    def do_quit(self, arg):
        """Exit the shell with style"""

        parsed_args = self._parse_args_or_cancel("quit", arg)
        if parsed_args is None:
            return

        if not parsed_args.no_save:
            with self.console.status("[bold green]Saving state...", spinner="dots"):
                self.do_save("que")
                self.do_save("server")
                time.sleep(0.5)  # Brief pause for visual feedback
        else:
            self.console.print("[yellow]Exiting without saving[/yellow]")

        self.console.print(
            Panel(
                "[bold green]Thank you for using queShell![/bold green]\n[dim]Queue management made beautiful ✨[/dim]",
                border_style="green",
            )
        )
        return True

    def do_exit(self, arg):
        """Exit the shell"""
        return self.do_quit(arg)

    def do_EOF(self, arg):
        """Exit on Ctrl+D"""
        self.console.print()
        return self.do_quit(arg)

    # Que based

    def do_save(self, arg):
        """Save state with visual feedback"""
        parsed_args = self._parse_args_or_cancel("save", arg)
        if parsed_args is None:
            return

        if parsed_args.command == "que":
            with self.unwrap_exception(
                "Queue state saved to file", "Failed to save que state"
            ):
                self.que.save_state(
                    out_path=parsed_args.Output_Path, timestamp=parsed_args.Timestamp
                )
        elif parsed_args.command == "server":
            with self.unwrap_exception(
                "Server state saved to file", "Failed to save server state"
            ):
                self.server_context.save_state()
        else:
            raise ValueError(
                "neither Que nor Server specified, this should not be possible"
            )
        # self.console.print("[bold green]✓[/bold green] Queue state saved to file")

    def do_load(self, arg):
        """Load state with visual feedback"""
        parsed_args = self._parse_args_or_cancel("load", arg)
        if parsed_args is None:
            return

        if parsed_args.command == "que":
            with self.unwrap_exception(
                "Que state loaded from file", "Failed to load Que state from file"
            ):
                self.que.load_state(parsed_args.Input_Path)
        elif parsed_args.command == "server":
            with self.unwrap_exception(
                "Server state loaded from file", "Failed to load server state from file"
            ):
                self.server_context.load_state()
        else:
            raise ValueError(
                "neither Que nor Server specified, this should not be possible"
            )

        # self.console.print("[bold green]✓[/bold green] Queue state loaded from file")

    def do_recover(self, arg):
        """Recover a run with status indication"""
        parsed_args = self._parse_args_or_cancel("recover", arg)
        if parsed_args is None:
            return
        with self.unwrap_exception("Run recovered successfully"):
            with self.console.status("[bold yellow]Recovering run...", spinner="dots"):
                self.que.recover_run(
                    from_loc=parsed_args.o_location,
                    to_loc=parsed_args.n_location,
                    index=parsed_args.index,
                    clean_slate=parsed_args.clean_slate,
                )

    def do_clear(self, arg):
        """Clear runs with confirmation"""
        parsed_args = self._parse_args_or_cancel("clear", arg)
        if parsed_args is None:
            return

        # Confirmation prompt
        if Confirm.ask(
            f"[bold red]Clear all runs from {parsed_args.location}?[/bold red]"
        ):
            with self.unwrap_exception(f"Cleared runs from {parsed_args.location}"):
                self.que.clear_runs(parsed_args.location)
            # self.console.print(f"[bold green]✓[/bold green] Cleared runs from {parsed_args.location}")
        else:
            self.console.print("[yellow]Clear cancelled[/yellow]")

    def do_list(self, arg):
        """Display runs in a beautiful table"""
        parsed_args = self._parse_args_or_cancel("list", arg)
        if parsed_args is None:
            return

        runs = None

        with self.unwrap_exception("", "Failed to list runs"):
            runs = self.que.list_runs(
                parsed_args.location, parsed_args.sort_keys, parsed_args.reverse
            )

        with self.unwrap_exception("", "Failed to filter runs with criterion"):
            if bool(parsed_args.filter_keys) != bool(parsed_args.criterion):
                parser.error("--filter_keys and --criterion must be used together")
            elif parsed_args.filter_keys:
                criterion = parse_criterion(parsed_args.criterion)  
                _, runs = self.que.find_runs(runs, parsed_args.filter_keys, criterion)

        runs = self.que.summarise(runs)

        if not runs:
            self.console.print(
                Panel(
                    f"[yellow]No runs found in {parsed_args.location}[/yellow]",
                    border_style="yellow",
                )
            )
            return

        # Create a styled table
        table = Table(
            title=f"Runs in {parsed_args.location}",
            box=box.ROUNDED,
            border_style="cyan",
            show_header=True,
            header_style="bold magenta",
        )

        dict_runs = list(map(lambda x: x.model_dump(), runs))

        table.add_column("Index", style="cyan", justify="right", width=8)
        for header in dict_runs[0].keys():
            table.add_column(header.capitalize(), style="white")

        # runs are a list of Summarised dicts

        for idx, row in enumerate(dict_runs):
            row_values = []
            for value in row.values():
                value_str = str(value)
                row_values.append(value_str)
            table.add_row(str(idx).zfill(3), *row_values)

        self.console.print(table)

    def do_remove(self, arg):
        """Remove a run with confirmation"""
        parsed_args = self._parse_args_or_cancel("remove", arg)
        if parsed_args is None:
            return

        if Confirm.ask(
            f"[bold red]Remove run {parsed_args.index} from {parsed_args.location}?[/bold red]"
        ):
            with self.unwrap_exception(
                f"Removed run {parsed_args.index} from {parsed_args.location}"
            ):
                self.que.remove_run(parsed_args.location, parsed_args.index)
            # self.console.print(f"[bold green]✓[/bold green] Removed run {parsed_args.index} from {parsed_args.location}")
        else:
            self.console.print("[yellow]Remove cancelled[/yellow]")

    def _unpack_keys(self, run: GenExp, key_set: List[str]) -> Any:
        unpack = run.model_dump()
        for k in key_set:
            unpack = unpack[k]
        return unpack

    def do_display(self, arg):
        """Display run details in a styled panel"""
        parsed_args = self._parse_args_or_cancel("display", arg)
        if parsed_args is None:
            return

        with self.unwrap_exception("", "Display failed"):
            if parsed_args.list_key_set:
                run = self.que.list_runs(
                    parsed_args.location, parsed_args.list_key_set, parsed_args.reverse
                )[parsed_args.index]

            else:
                run = self.que.peak_run(parsed_args.location, parsed_args.index)
            title = f"Run {parsed_args.index} in {parsed_args.location}"
            if parsed_args.display_keys is not None:
                run = self._unpack_keys(run, parsed_args.display_keys)
                title = f"Run {parsed_args.index} in {parsed_args.location}: {', '.join(parsed_args.display_keys)}"

            # Format as JSON-like syntax
            if isinstance(run, dict):
                run_json = json.dumps(run, indent=2)
            else:
                run_json = json.dumps(run.model_dump(), indent=2)
            syntax = Syntax(run_json, "json", theme="monokai", line_numbers=True)

            self.console.print(
                Panel(
                    syntax,
                    title=title,
                    border_style="cyan",
                    padding=(1, 2),
                )
            )

    def do_shuffle(self, arg):
        """Reposition with visual confirmation"""
        parsed_args = self._parse_args_or_cancel("shuffle", arg)
        if parsed_args is None:
            return

        with self.unwrap_exception(
            f"Moved run from position {parsed_args.o_index} to {parsed_args.n_index} in {parsed_args.location}"
        ):
            self.que.shuffle(
                parsed_args.location, parsed_args.o_index, parsed_args.n_index
            )

    def do_move(self, arg):
        """Move run with visual feedback"""
        parsed_args = self._parse_args_or_cancel("move", arg)
        if parsed_args is None:
            return

        with self.unwrap_exception(
            f"Moved run from {parsed_args.o_location} to {parsed_args.n_location}"
        ):
            self.que.move(
                parsed_args.o_location,
                parsed_args.n_location,
                parsed_args.oi_index,
                parsed_args.of_index,
            )

    def do_create(self, arg):
        """Create with progress indication"""
        args = shlex.split(arg)

        try:
            maybe_args = configs.take_args(sup_args=args)
        except (SystemExit, ValueError):
            self.console.print("[red]Create cancelled (incorrect arguments)[/red]")
            return

        if isinstance(maybe_args, tuple):
            admin_info, wandb_info = maybe_args
        else:
            self.console.print("[yellow]Create cancelled (by user)[/yellow]")
            return

        with self.unwrap_exception(
            "Run created successfully", "Failed to create new run"
        ):
            try:
                self.que.create_run(admin_info, wandb_info)
            except QueDupExp:
                # ask if want to create anyway
                if Confirm.ask(
                    "[bold yellow]A run with the same config already exists. Create duplicate?[/bold yellow]"
                ):
                    self.que.create_run(admin_info, wandb_info, add_duplicates=True)

    def do_add(self, arg):
        """Add run with feedback"""
        args = shlex.split(arg)
        parser = self._get_add_parser()
        parsed_args = parser.parse_args(args)
        parsed_args.no_enum_chck = True  # bypass enum check
        # print(parsed_args.checkpoint_num)
        try:
            maybe_args = configs.take_args(parsed_args=parsed_args)
        except (SystemExit, ValueError):
            self.console.print("[red]Add cancelled (incorrect arguments)[/red]")
            return

        if isinstance(maybe_args, tuple):
            admin_info, wandb_info = maybe_args

            # add correct checkpoint num to save path
            checknum = (
                str(parsed_args.checkpoint_num).zfill(ZFILL)
                if parsed_args.checkpoint_num is not None
                else ""
            )
            admin_info.save_path = str(admin_info.save_path) + checknum

            # check that checkpoint exists
            if not Path(admin_info.save_path).exists():
                self.console.print(
                    f"[red]Add cancelled (save path: {admin_info.save_path} does not exist)[/red]"
                )
                return
        else:
            self.console.print("[yellow]Add cancelled (by user)[/yellow]")
            return

        with self.unwrap_exception("Run added successfully", "Failed to add run"):
            with self.console.status("[bold green]Adding run...", spinner="dots"):
                self.que.add_run(admin_info, wandb_info)

    def do_edit(self, arg):
        """Edit with visual confirmation"""
        parsed_args = self._parse_args_or_cancel("edit", arg)
        if parsed_args is None:
            return

        with self.unwrap_exception("Edit successful", "Edit failed"):
            self.que.edit_run(
                parsed_args.location,
                parsed_args.index,
                parsed_args.edit_keys,
                parsed_args.value,
                parsed_args.do_eval,
            )

    def do_copy(self, arg):
        """Copy a run"""
        parsed_args = self._parse_args_or_cancel("copy", arg)
        if parsed_args is None:
            return
        if parsed_args.o_indexes is None or len(parsed_args.o_indexes) == 0:
            o_idxs = [0]
        else:
            o_idxs = list(map(int, parsed_args.o_indexes))

        with self.unwrap_exception("Copy successful", "Copy failed"):
            self.que.copy_runs(
                parsed_args.o_location,
                o_idxs,
                parsed_args.n_location,
                parsed_args.n_index,
                parsed_args.clean_slate,
            )

    def do_wandb(self, arg):
        """Open the wandb page for a run"""
        parsed_args = self._parse_args_or_cancel("wandb", arg)
        if parsed_args is None:
            return

        url = "https://wandb.ai/"

        if parsed_args.location is not None and parsed_args.index is not None:
            run = self.que.peak_run(loc=parsed_args.location, idx=parsed_args.index)
            wandb_info = run.wandb
            url = url + f"{wandb_info.entity}/{wandb_info.project}/{wandb_info.run_id}"
        else:
            url = url + f"{parsed_args.entity}/{parsed_args.project}"

        with self.unwrap_exception("Wandb opened successfully", "Opening wandb failed"):
            webbrowser.open(url)

    #   Worker

    def do_worker(self, arg):
        parsed_args = self._parse_args_or_cancel("worker", arg)
        if parsed_args is None:
            return

        if parsed_args.command == "clear_mem":
            self.worker.cleanup()
            self.console.print("[bold green]Cleared CUDA memory[/bold green]")
            used, total = gpu_manager.get_gpu_memory_usage()
            self.console.print(f"CUDA memory: {used}/{total} GiB")
        else:
            self.console.print(
                f"[bold red]Command not recognised: {parsed_args.command}[/bold red]"
            )

    # Daemon

    def do_daemon(self, arg):
        """Interact with the worker"""
        parsed_args = self._parse_args_or_cancel("daemon", arg)
        if parsed_args is None:
            return
        elif parsed_args.command == "start":
            with self.unwrap_exception(
                "Worker process started", "Failed to start worker"
            ):
                self.daemon.start_supervisor()
        elif parsed_args.command == "stop":
            if parsed_args.supervisor:
                with self.unwrap_exception(
                    "Supervisor process stopped", "Failed to stop supervisor"
                ):
                    self.daemon.stop_supervisor(
                        timeout=parsed_args.timeout,
                        hard=parsed_args.hard,
                        stop_worker=parsed_args.worker,
                    )
            else:
                with self.unwrap_exception(
                    "Worker process stopped", "Failed to stop worker"
                ):
                    self.daemon.stop_worker(
                        timeout=parsed_args.timeout, hard=parsed_args.hard
                    )

    # Server

    def _pretty_status(self, status: ServerState):

        # Main status table
        table = Table(
            title="Server Status", show_header=False, box=None, padding=(0, 2)
        )
        table.add_column("Section", style="bold cyan", width=20)
        table.add_column("Details")

        # Server section
        server_pid = status.server_pid
        server_status = Text()
        if server_pid:
            server_status.append("Running ", style="bold green")
            server_status.append(f"(PID: {server_pid})", style="dim")
        else:
            server_status.append("Not Running", style="bold red")
        table.add_row("Server", server_status)

        # Daemon section
        daemon_state = status.daemon_state
        daemon_table = Table(show_header=False, box=None, padding=(0, 1))
        daemon_table.add_column(style="yellow", width=15)
        daemon_table.add_column()

        awake_icon = "✓" if daemon_state.awake else "✗"
        awake_style = "green" if daemon_state.awake else "red"
        daemon_table.add_row("Awake:", Text(awake_icon, style=awake_style))

        stop_icon = "✓" if daemon_state.stop_on_fail else "✗"
        daemon_table.add_row(
            "Stop on Fail:",
            Text(stop_icon, style="yellow" if daemon_state.stop_on_fail else "dim"),
        )

        if daemon_state.supervisor_pid:
            daemon_table.add_row("Supervisor PID:", str(daemon_state.supervisor_pid))

        table.add_row("Daemon", daemon_table)

        # Worker section
        worker_state = status.worker_state
        worker_table = Table(show_header=False, box=None, padding=(0, 1))
        worker_table.add_column(style="magenta", width=15)
        worker_table.add_column()

        task_style = "bold green" if worker_state.task == "training" else "dim"
        worker_table.add_row("Task:", Text(worker_state.task, style=task_style))

        if worker_state.current_run_id:
            worker_table.add_row("Run ID:", worker_state.current_run_id)

        if worker_state.working_pid:
            worker_table.add_row("Worker PID:", str(worker_state.working_pid))

        if worker_state.exception:
            error_text = Text(worker_state.exception, style="bold red")
            worker_table.add_row("Exception:", error_text)

        table.add_row("Worker", worker_table)

        self.console.print(table)

    def do_server(self, arg):
        parsed_args = self._parse_args_or_cancel("server", arg)
        if parsed_args is None:
            return

        if parsed_args.command == "save":
            with self.unwrap_exception(
                "Server state saved", "Failed to save server state"
            ):
                self.server_context.save_state()
        elif parsed_args.command == "load":
            with self.unwrap_exception(
                "Server state loaded", "Failed to load server state"
            ):
                self.server_context.load_state()
        elif parsed_args.command == "status":
            self._pretty_status(self.server_context.get_state())

    # Subprocess

    def do_attach(self, arg):
        """Attach to the que_training tmux session"""
        with self.unwrap_exception(
            "Attached to tmux session", "Failed to attach to tmux session"
        ):
            self.tmux_man.join_session()

    def do_logs(self, arg):
        """Tail the worker or daemon logs"""
        parsed_args = self._parse_args_or_cancel("logs", arg)
        if parsed_args is None:
            return

        if parsed_args.worker:
            log_file = str(TRAINING_LOG_PATH)  # your constant
        elif parsed_args.server:
            log_file = str(SERVER_LOG_PATH)  # your constant
        else:
            raise ValueError("Please specify --worker or --server")

        if parsed_args.clear:
            if Confirm.ask(f"[bold red]Clear all logs in {log_file}?[/bold red]"):
                with self.unwrap_exception(
                    f"Cleared {log_file}", f"Failed to clear log file: {log_file}"
                ):
                    with open(log_file, "w") as f:
                        f.truncate(0)
                return
            else:
                self.console.print("[yellow]Action cancelled[/yellow].")
                return

        try:
            subprocess.run(["tail", "-f", log_file])
        except KeyboardInterrupt:
            self.console.print("\n[cyan]Stopped tailing log file[/cyan]")
        except FileNotFoundError:
            self.console.print(f"[red]Error: Log file not found at {log_file}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error reading log file: {e}[/red]")

    # Helper functions for parsing

    def _apply_synonyms(self, parsed_args):
        """Apply synonyms to location arguments"""
        if hasattr(parsed_args, "o_location"):
            parsed_args.o_location = SYNONYMS.get(
                parsed_args.o_location.lower(), parsed_args.o_location
            )
        if hasattr(parsed_args, "n_location"):
            parsed_args.n_location = SYNONYMS.get(
                parsed_args.n_location.lower(), parsed_args.n_location
            )
        if hasattr(parsed_args, "location") and parsed_args.location is not None:
            parsed_args.location = SYNONYMS.get(
                parsed_args.location.lower(), parsed_args.location
            )
        return parsed_args

    def _parse_args_or_cancel(self, cmd: str, arg: str) -> Optional[argparse.Namespace]:
        """
        Parse generic arguments

        :param self: QueShell instance
        :param cmd: The name of the command, i.e. x in do_x function name
        :type cmd: str
        :param arg: Arg as collected by cmdlib (passed to do_* function)
        :type arg: str
        :return: Returns parsed args in argparse format, otherwise None if failure.
        :rtype: Namespace | None
        """
        args = shlex.split(arg)
        parser = self._get_parser(cmd)
        if parser:
            try:
                return self._apply_synonyms(parser.parse_args(args))
            except (SystemExit, ValueError):
                self.console.print(f"[yellow]{cmd} cancelled[/yellow]")
                return None
        else:
            self.console.print(f"[red]{cmd} not found[/red]")

    def _get_parser(self, cmd: str) -> Optional[argparse.ArgumentParser]:
        """Get argument parser for a given command"""
        factory = self._parser_factories.get(cmd)
        return factory() if factory else None

    # --- Argument factory methods ---

    def _add_genkey_arg(
        self,
        parser: argparse.ArgumentParser,
        long_name: str = "--keys",
        short_name: str = "-k",
        help: str = "List of keys to unpack the nested run by",
    ) -> argparse.ArgumentParser:
        """General method for adding a keys argument with customizable flags and help text
        For example, --sort_keys, -s or --filter_keys, -f
        """
        parser.add_argument(
            long_name,
            short_name,
            nargs="+",
            type=str,
            help=help,
        )
        return parser

    def _add_value_args(
        self, parser: argparse.ArgumentParser, help: str = "Value to assign"
    ) -> argparse.ArgumentParser:
        """Positional `value` and its companion `--do_eval / -de` flag.
        Used wherever a run field is being assigned a new value."""
        parser.add_argument("value", type=str, help=help)
        parser.add_argument(
            "--do_eval",
            "-de",
            action="store_true",
            help="Evaluate the provided value to a Python type rather than treating it as a raw string.",
        )
        return parser

    def _add_criterion_args(
        self,
        parser: argparse.ArgumentParser,
        help: str = "criterion to filter runs by, evaluates boolean expression with run fields as variables, e.g. --criterion 'accuracy > 0.8' (requires --filter_keys / -f to specify which keys to filter by)",
    ) -> argparse.ArgumentParser:
        """--criterion / -c: criterion to filter runs by, evaluates boolean expression with run fields as variables, e.g. --criterion 'accuracy > 0.8' (requires --filter_keys / -f to specify which keys to filter by)"""
        parser.add_argument(
            "--criterion",
            "-c",
            help=help,
        )
        return parser

    def _add_location_arg(
        self,
        parser: argparse.ArgumentParser,
        name: str = "location",
        positional: bool = True,
        help: str = "Queue location",
    ) -> argparse.ArgumentParser:
        """Positional or optional location argument drawn from avail_locs"""
        if positional:
            parser.add_argument(name, choices=self.avail_locs, help=help)
        else:
            parser.add_argument(
                f"--{name}",
                f"-{''.join(w[0] for w in name.split('_'))}",
                choices=self.avail_locs,
                help=help,
            )
        return parser

    def _add_index_arg(
        self,
        parser: argparse.ArgumentParser,
        name: str = "index",
        default: int = 0,
        required: bool = True,
    ) -> argparse.ArgumentParser:
        """Positional or optional integer index argument"""
        flag = f"--{name}"
        short = f"-{''.join(w[0] for w in name.split('_'))}"
        if required:
            parser.add_argument(name, type=int)
        else:
            parser.add_argument(flag, short, type=int, default=default)
        return parser

    def _add_reverse_arg(
        self, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """--reverse / -r: sort in descending order"""
        parser.add_argument(
            "--reverse",
            "-r",
            action="store_true",
            help="Sort in descending order",
        )
        return parser

    def _add_clean_slate_arg(
        self, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """--clean_slate / -cs: strip results/recover status, keep config only"""
        parser.add_argument(
            "--clean_slate",
            "-cs",
            action="store_true",
            help="Preserve only the config, not results, wandb or recover status",
        )
        return parser

    def _add_graceful_stop_args(
        self, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """--timeout / -to and --hard / -hd: used wherever a process is being stopped"""
        parser.add_argument(
            "--timeout",
            "-to",
            type=int,
            default=10,
            help="Timeout in seconds before force kill (default: 10)",
        )
        parser.add_argument(
            "--hard",
            "-hd",
            action="store_true",
            help="Force kill the process after timeout",
        )
        return parser

    def _add_o_location_arg(
        self, parser: argparse.ArgumentParser, default=None
    ) -> argparse.ArgumentParser:
        """--o_location / -ol: origin location"""
        kwargs = dict(type=str, choices=self.avail_locs, help="Origin location")
        if default is not None:
            kwargs["default"] = default
        parser.add_argument("--o_location", "-ol", **kwargs)  # type: ignore
        return parser

    def _add_n_location_arg(
        self, parser: argparse.ArgumentParser, default=None
    ) -> argparse.ArgumentParser:
        """--n_location / -nl: destination location"""
        kwargs = dict(type=str, choices=self.avail_locs, help="Destination location")
        if default is not None:
            kwargs["default"] = default
        parser.add_argument("--n_location", "-nl", **kwargs)  # type: ignore
        return parser



    def _add_checkpoint_num_arg(
        self,
        parser: argparse.ArgumentParser,
        help: str = "Checkpoint number to add when adding a run",
        default=None,
    ) -> argparse.ArgumentParser:
        """--checkpoint_num / -cpn: checkpoint number to add when adding a run"""
        parser.add_argument(
            "--checkpoint_num",
            "-cpn",
            type=int,
            default=default,
            help=help,
        )
        return parser

    # Que

    def _get_save_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Save the state of the Que or Daemon", prog="save"
        )
        subparsers = parser.add_subparsers(
            dest="command", required=True, help="Target to save"
        )

        # Que Subparser
        que_parser = subparsers.add_parser("que", help="Save Que state")
        que_parser.add_argument(
            "--Timestamp", "-t", action="store_true", help="Timestamp the output file"
        )
        que_parser.add_argument(
            "--Output_Path",
            "-op",
            type=str,
            default=RUN_PATH,
            help=f"Output path (default: {RUN_PATH})",
        )

        # Daemon Subparser
        subparsers.add_parser("server", help="Save Server state")

        return parser

    def _get_load_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Load the state of the Que or Daemon", prog="load"
        )
        subparsers = parser.add_subparsers(
            dest="command", required=True, help="Target to load"
        )

        # Que Subparser
        que_load = subparsers.add_parser("que", help="Load Que state")
        que_load.add_argument(
            "--Input_Path",
            "-ip",
            type=str,
            default=RUN_PATH,
            help=f"Input path (default: {RUN_PATH})",
        )

        # Daemon Subparser
        subparsers.add_parser("server", help="Load Server state")
        # TODO: Maybe add this if desired

        return parser

    def _get_quit_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Exit queShell", prog="quit")
        parser.add_argument("-ns", "--no_save", action="store_true")
        return parser

    def _get_add_parser(self) -> argparse.ArgumentParser:
        train_parser = get_train_parser(
            prog="add", desc="Add a completed training run to old_runs"
        )
        return self._add_checkpoint_num_arg(train_parser)

    def _get_list_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="List runs", prog="list")
        self._add_location_arg(parser)
        self._add_genkey_arg(
            parser,
            long_name="--sort_keys",
            short_name="-s",
            help="List of keys to sort the list by",
        )
        self._add_reverse_arg(parser)
        self._add_genkey_arg(
            parser,
            long_name="--filter_keys",
            short_name="-f",
            help="List of keys to filter the list by (only display runs where these keys are not None)",
        )
        self._add_criterion_args(parser)
        return parser

    def _get_clear_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Clear runs", prog="clear")
        self._add_location_arg(parser)
        return parser

    def _get_remove_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Remove a run", prog="remove")
        self._add_location_arg(parser)
        self._add_index_arg(parser)
        return parser

    def _get_shuffle_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Reposition a run", prog="shuffle")
        self._add_location_arg(parser)
        parser.add_argument("o_index", type=int)
        parser.add_argument("n_index", type=int)
        return parser

    def _get_display_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Display run config", prog="display"
        )
        self._add_location_arg(parser)
        self._add_index_arg(parser)
        self._add_genkey_arg(
            parser,
            long_name="--display_keys",
            short_name="-dk",
            help="List of keys to display within the run",
        )
        self._add_genkey_arg(
            parser,
            long_name="--sort_keys",
            short_name="-sk",
            help="List of keys to sort the list by",
        )
        self._add_reverse_arg(parser)

        return parser

    def _get_edit_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Edit run", prog="edit")
        self._add_location_arg(parser)
        self._add_index_arg(parser)
        self._add_genkey_arg(
            parser,
            long_name="--edit_keys",
            short_name="-ek",
            help="List of keys to edit within the run",
        )
        self._add_value_args(parser)
        return parser

    def _get_copy_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Copy run", prog="copy")
        self._add_location_arg(
            parser, name="o_location", positional=True, help="Origin location"
        )
        self._add_location_arg(
            parser, name="n_location", positional=True, help="Destination location"
        )
        self._add_clean_slate_arg(parser)
        self._add_reverse_arg(parser)
        self._add_genkey_arg(
            parser,
            long_name="--sort_keys",
            short_name="-s",
            help="List of keys to sort the list by",
        )
        parser.add_argument(
            "--o_indexes", "-i", nargs="+", type=str, help="List of indexes to copy"
        )
        parser.add_argument("-ni", "--n_index", type=int, default=0, help="New index")
        return parser

    def _get_recover_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Recover a failed run", prog="recover"
        )
        self._add_o_location_arg(parser, default=CUR_RUN)
        self._add_n_location_arg(parser, default=TO_RUN)
        self._add_index_arg(parser, required=False, default=0)
        self._add_clean_slate_arg(parser)
        return parser

    def _get_move_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Move run between locations", prog="move"
        )
        self._add_location_arg(
            parser, name="o_location", positional=True, help="Origin location"
        )
        self._add_location_arg(
            parser, name="n_location", positional=True, help="Destination location"
        )
        parser.add_argument("oi_index", type=int)
        parser.add_argument("--of_index", "-of", type=int, default=None)
        return parser

    # Other

    def _get_daemon_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Interact with the worker process", prog="daemon"
        )

        subparsers = parser.add_subparsers(
            dest="command", required=True, help="Daemon commands"
        )

        # Start
        subparsers.add_parser("start", help="Start the supervisor")

        # Stop with timeout
        stop_parser = subparsers.add_parser(
            "stop", help="Stop the worker gracefully, force kill if necessary"
        )
        stop_parser.add_argument(
            "--worker", "-w", action="store_true", help="Stop the worker process"
        )
        stop_parser.add_argument(
            "--supervisor",
            "-s",
            action="store_true",
            help="Stop the supervisor process",
        )
        self._add_graceful_stop_args(stop_parser)

        return parser

    def _get_server_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Interact with the server context", prog="server"
        )

        subparsers = parser.add_subparsers(
            dest="command", required=True, help="Server commands"
        )

        # Save state
        subparsers.add_parser("save", help="Save Server state to disk")
        # Load state
        subparsers.add_parser("load", help="Load Server state from disk")

        # Status
        subparsers.add_parser("status", help="Get worker status information")

        return parser

    def _get_worker_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Interact with the worker", prog="worker"
        )

        subparsers = parser.add_subparsers(
            dest="command", required=True, help="Server commands"
        )

        # Clear memory
        subparsers.add_parser("cleanup", help="Clear CUDA memory")

        return parser

    def _get_log_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Interact with Que log files", prog="logs"
        )

        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--worker", "-w", action="store_true", help="Tail the Worker.log file"
        )
        group.add_argument(
            "--server", "-s", action="store_true", help="Tail the Server.log file"
        )

        parser.add_argument(
            "--clear",
            "-c",
            action="store_true",
            help="Clear the log file instead of tailing",
        )

        return parser

    def _get_wandb_parser(self) -> argparse.ArgumentParser:
        likely_projects = [
            f"{PROJECT_BASE}-{split[3:]}" for split in get_avail_splits()
        ]
        parser = argparse.ArgumentParser(
            description="Open the wandb page for a run, or project", prog="wandb"
        )
        parser.add_argument("--location", "-l", choices=self.avail_locs)
        parser.add_argument("--index", "-i", type=int)
        parser.add_argument(
            "--project",
            "-p",
            type=str,
            help="Wandb project name. Probably one of: " + ", ".join(likely_projects),
            default="projects",
        )
        parser.add_argument(
            "--entity",
            "-e",
            type=str,
            help=f"Wandb entity name. Default: {ENTITY}",
            default=ENTITY,
        )
        return parser


def ssh_tunnel_maker(
    host_ip: str,
    ssh_user: Optional[str] = None,
    ssh_key: Optional[Path] = None,
    port_client: int = 50000,
    port_server: int = 50000,
) -> subprocess.Popen:
    """Open an ssh tunnel with the specified host, user, and key, forwarding
    port_client to port_server on the remote host. Returns the subprocess.Popen
    object for the tunnel.

    Args:
        host_ip (str): Host IP address.
        ssh_user (Optional[str], optional): The user profile on the server, otherwise uses the current logged in user for this session. Defaults to None.
        ssh_key (Optional[Path], optional): Path to ssh_key. Defaults to None, will ask for password if not provided.
        port_client (int, optional): Client side port. Defaults to 50000.
        port_server (int, optional): Server side port. Defaults to 50000.

    Returns:
        subprocess.Popen: Opened subprocess for the ssh tunnel, which can be terminated with .terminate() when the tunnel is no longer needed.
    """

    if ssh_user is None:
        ssh_user = Path.home().name

    ssh_cmd = [
        "ssh",
        "-N",  # don't execute a command, just tunnel
        "-L",
        f"{port_client}:localhost:{port_server}",  # local port -> remote port
        "-o",
        "ExitOnForwardFailure=yes",
    ]
    if ssh_key:
        ssh_cmd += ["-i", ssh_key.expanduser().as_posix()]
    ssh_cmd.append(f"{ssh_user}@{host_ip}")

    tunnel = subprocess.Popen(ssh_cmd)
    time.sleep(1)  # give the tunnel a moment to establish

    try:
        return tunnel
    except Exception:
        tunnel.terminate()
        raise


@contextmanager
def tunnel_handler(tunnel: Optional[subprocess.Popen]):
    """Context manager to handle the lifecycle of an ssh tunnel subprocess. Ensures that the tunnel is properly terminated when the context is exited, even if an exception occurs.

    Args:
        tunnel (Optional[subprocess.Popen]): The subprocess.Popen object representing the ssh tunnel. Can be None if no tunnel was created.

    Yields:
        None
    """
    try:
        yield
    finally:
        if tunnel is not None:
            tunnel.terminate()
            tunnel.wait()  # Wait for the subprocess to terminate to avoid zombies


def get_queshell_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="queShell command line arguments")

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host IP. If localhost, then connects to local manager. If remote, will establish SSH tunnel and connect to manager through that (default: localhost)",
    )
    parser.add_argument(
        "--ssh_user",
        type=str,
        default=getpass.getuser(),
        help="SSH username (default: current user)",
    )
    parser.add_argument(
        "--ssh_key",
        type=Path,
        default=None,
        help="Path to SSH private key (default: None, will use default SSH keys or password authentication)",
    )
    parser.add_argument(
        "--port_client",
        type=int,
        default=50000,
        help="Local port for SSH tunnel (default: 50000)",
    )
    parser.add_argument(
        "--port_server",
        type=int,
        default=50000,
        help="Remote port for SSH tunnel (default: 50000)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum number of connection retries (default: 5)",
    )
    parser.add_argument(
        "--retry_delay",
        type=int,
        default=2,
        help="Delay in seconds between connection retries (default: 2)",
    )
    return parser


if __name__ == "__main__":
    parser = get_queshell_parser()
    args = parser.parse_args()

    if args.host != "localhost":
        if args.ssh_key is not None:
            args.ssh_key = args.ssh_key.expanduser()
            if not args.ssh_key.exists():
                print(f"SSH key not found at {args.ssh_key}")
                raise ValueError("SSH key not found")
        else:
            id_rsa = Path.home() / ".ssh" / "id_rsa"  # default SSH key path
            ed25519 = Path.home() / ".ssh" / "id_ed25519"
            if id_rsa.exists():
                args.ssh_key = id_rsa
            elif ed25519.exists():
                args.ssh_key = ed25519
            else:
                print(
                    "No SSH key provided and no default keys found, will attempt password authentication"
                )

        try:
            tunnel = ssh_tunnel_maker(
                host_ip=args.host,
                ssh_user=args.ssh_user,
                ssh_key=args.ssh_key,
                port_client=args.port_client,
                port_server=args.port_server,
            )
            print("SSH tunnel established successfully")
        except Exception as e:
            print(f"Failed to establish SSH tunnel: {e}")
            raise

    else:
        tunnel = None  # No tunnel needed for localhost

    with tunnel_handler(tunnel):
        try:
            que_shell = QueShell(connect_manager())
            que_shell.cmdloop()
        except KeyboardInterrupt:
            print("\n[INFO] Exiting queShell without saving due to keyboard interrupt.")
