from .core import QUE_LOCATIONS, SYNONYMS, connect_manager, DaemonState, WR_LOG_PATH, SR_LOG_PATH, RUN_PATH, DAEMON_STATE_PATH

from .tmux import tmux_manager
import cmd as cmdLib
import shlex
from typing import Optional
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

import subprocess

class QueShell(cmdLib.Cmd):
    def __init__(
        self,
        auto_save: bool = True,
    ) -> None:
        super().__init__()
        self.console = Console()
        self.tmux_man = tmux_manager()
        self.auto_save = auto_save
        self.server = connect_manager()
        self.que = self.server.get_que() #proxy object
        self.daemon_controller = self.server.DaemonController() #object server (hold processes)
        # Display welcome banner
        self._show_banner()

        # Override prompt with rich styling
        # self.prompt = "(que)$"  # We'll handle this with rich
        self.prompt = "\x01\033[1;36m\x02(que)$\x01\033[0m\x02 "
        self.intro = ""  # We'll handle this with rich

    def _show_banner(self):
        """Display a fancy welcome banner"""
        banner = Text()
        banner.append("╔═══════════════════════════════════════╗\n", style="bold cyan")
        banner.append("║        ", style="bold cyan")
        banner.append("queShell v2.0", style="bold yellow")
        banner.append("                  ║\n", style="bold cyan")
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

        commands = [
            ("help", "Show this help message or detailed help for a command"),
            ("list", "Display runs in a given location"),
            ("create", "Create a new run and add to queue"),
            ("add", "Add a completed run to old runs"),
            ("remove", "Remove a run from the queue"),
            ("display", "Show detailed config for a run"),
            ("edit", "Edit a run's configuration"),
            ("move", "Move a run between locations"),
            ("shuffle", "Reposition a run within a location"),
            ("clear", "Clear all runs from a location"),
            ("daemon", "Start/manage the daemon process"),
            ("logs", "View the Que log files"),
            ("attach", "Attach to tmux session"),
            ("save", "Save queue state to file"),
            ("load", "Load queue state from file"),
            ("recover", "Recover from a failed run"),
            ("quit/exit", "Exit queShell"),
        ]

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        self.console.print(table)
        self.console.print(
            "\n[dim]Tip: Use 'help <command>' for detailed information about a specific command[/dim]\n"
        )

    @contextmanager
    def unwrap_exception(self, message: str, error: str = ""):
        try:
            yield
            self.console.print(f"[bold green]✓ {message} [/bold green]")
        except Exception as e:
            if error:
                self.console.print(
                    f"[bold red]✗ {error} : {e} [/bold red]", style="red"
                )
            else:
                self.console.print(f"[bold red]✗ {e} [/bold red]", style="red")

    def do_quit(self, arg):
        """Exit the shell with style"""

        parsed_args = self._parse_args_or_cancel("quit", arg)
        if parsed_args is None:
            return

        if not parsed_args.no_save:
            with self.console.status("[bold green]Saving state...", spinner="dots"):
                self.do_save('que')
                self.do_save('daemon')
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

    #Que based


    def do_save(self, arg):
        """Save state with visual feedback"""
        parsed_args = self._parse_args_or_cancel("save", arg)
        if parsed_args is None:
            return

        if parsed_args.command == 'que':
            with self.unwrap_exception("Queue state saved to file", "Failed to save que state"):
                self.que.save_state(out_path=parsed_args.Output_Path, timestamp=parsed_args.Timestamp)
        elif parsed_args.command == 'daemon':
            with self.unwrap_exception("Daemon state saved to file", "Failed to save daemon state"):
                self.daemon_controller.save_state()
        else:
            raise ValueError('neither Que nor Daemon specified, this should not be possible')
        # self.console.print("[bold green]✓[/bold green] Queue state saved to file")

    def do_load(self, arg):
        """Load state with visual feedback"""
        with self.console.status("[bold cyan]Loading state...", spinner="dots"):
            with self.unwrap_exception("Que state loaded from file"):
                    self.que.load_state()
            with self.unwrap_exception("Daemon state loaded from file"):
                    self.daemon_controller.load_state()        
                
        # self.console.print("[bold green]✓[/bold green] Queue state loaded from file")

    def do_recover(self, arg):
        """Recover a run with status indication"""
        with self.unwrap_exception("Run recovered successfully"):
            with self.console.status("[bold yellow]Recovering run...", spinner="dots"):
                self.que.recover_run()

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

        runs = self.que.list_runs(parsed_args.location)

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

        table.add_column("Index", style="cyan", justify="right", width=8)
        for header in runs[0].keys():
            table.add_column(header.capitalize(), style="white")

        # runs are a list of Summarised dicts

        for idx, row in enumerate(runs):
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

    def do_display(self, arg):
        """Display run details in a styled panel"""
        parsed_args = self._parse_args_or_cancel("display", arg)
        if parsed_args is None:
            return
        try:
            run = self.que.peak_run(parsed_args.location, parsed_args.index)
        
            # Format as JSON-like syntax

            run_json = json.dumps(run, indent=2)
            syntax = Syntax(run_json, "json", theme="monokai", line_numbers=True)

            self.console.print(
                Panel(
                    syntax,
                    title=f"Run {parsed_args.index} in {parsed_args.location}",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
        except Exception as e:
            self.console.print(
                f"[red]Display failed : {e}[/red]"
            )

    def do_shuffle(self, arg):
        """Reposition with visual confirmation"""
        parsed_args = self._parse_args_or_cancel("shuffle", arg)
        if parsed_args is None:
            return

        with self.unwrap_exception(
            f"Moved run from position {parsed_args.o_index} to {parsed_args.n_index} in {parsed_args.location}"
        ):
            self.que.shuffle(parsed_args.location, parsed_args.o_index, parsed_args.n_index)

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

        with self.console.status("[bold cyan]Creating run...", spinner="dots"):
            with self.unwrap_exception("Run created successfully", "Failed to create new run"):
                self.que.create_run(admin_info, wandb_info)
        # self.console.print("[bold green]✓[/bold green] Run created successfully")

    def do_add(self, arg):
        """Add run with feedback"""
        args = shlex.split(arg)

        try:
            maybe_args = configs.take_args(sup_args=args, ask_bf_ovrite=False)
        except (SystemExit, ValueError):
            self.console.print("[red]Add cancelled (incorrect arguments)[/red]")
            return

        if isinstance(maybe_args, tuple):
            admin_info, wandb_info = maybe_args
        else:
            self.console.print("[yellow]Add cancelled (by user)[/yellow]")
            return

        with self.console.status("[bold cyan]Adding run...", spinner="dots"):
            self.que.add_run(admin_info, wandb_info)
        self.console.print("[bold green]✓[/bold green] Run added to old_runs")

    def do_edit(self, arg):
        """Edit with visual confirmation"""
        parsed_args = self._parse_args_or_cancel("edit", arg)
        if parsed_args is None:
            return

        self.que.edit_run(
            parsed_args.location,
            parsed_args.index,
            parsed_args.key1,
            parsed_args.value,
            parsed_args.key2,
        )
        self.console.print(
            f"[bold green]✓[/bold green] Edited run {parsed_args.index} in {parsed_args.location}"
        )

    #Tmux
    def do_attach(self, arg):
        """Attach to the que_training tmux session"""
        with self.unwrap_exception("Attached to tmux session", "Failed to attach to tmux session"):
            self.tmux_man.join_session_pane()

    #Daemon based
    
    def _pretty_status(self, status: DaemonState):
        if status["awake"]:
            self.console.print("Daemon is currently: [bold green]Awake[/bold green]")
        else:
            self.console.print("Daemon is currently: [bold yellow]Asleep[/bold yellow]")
        if status["stop_on_fail"]:
            self.console.print("Stop on fail is: [bold red]Enabled[/bold red]")
        else:
            self.console.print("Stop on fail is: [bold green]Disabled[/bold green]")
        if status["pid"]:
            self.console.print(f"Daemon PID: [bold cyan]{status['pid']}[/bold cyan]")
        else:
            self.console.print("Daemon PID: [bold yellow]N/A[/bold yellow]")
        if status["worker_pid"]:
            self.console.print(f"Worker PID: [bold cyan]{status['worker_pid']}[/bold cyan]")
        else:
            self.console.print("Worker PID: [bold yellow]N/A[/bold yellow]")
            
    def do_daemon(self, arg):
        """Interact with the worker"""
        parsed_args = self._parse_args_or_cancel("daemon", arg)
        if parsed_args is None:
            return
        
        if parsed_args.command == 'save':
            with self.unwrap_exception("Daemon state saved", "Failed to save daemon state"):
                self.daemon_controller.save_state()
        elif parsed_args.command == 'load':
            with self.unwrap_exception("Daemon state loaded", "Failed to load daemon state"):
                self.daemon_controller.load_state()
        elif parsed_args.command == 'start':
            with self.unwrap_exception("Worker process started", "Failed to start worker"):
                self.daemon_controller.start()
        elif parsed_args.command == 'stop':
            if parsed_args.supervisor:
                with self.unwrap_exception("Supervisor process stopped", "Failed to stop supervisor"):
                    self.daemon_controller.stop_supervisor(timeout=parsed_args.timeout, hard=parsed_args.hard, and_worker=parsed_args.worker)
            else:
                with self.unwrap_exception("Worker process stopped", "Failed to stop worker"):
                    self.daemon_controller.stop_worker(timeout=parsed_args.timeout, hard=parsed_args.hard)
        elif parsed_args.command == 'status':
            self._pretty_status(self.daemon_controller.get_state())
        elif parsed_args.command == 'stop-on-fail':
            if parsed_args.value == 'on':
                parsed_args.value = True
            else:
                parsed_args.value = False
            self.daemon_controller.set_stop_on_fail(parsed_args.value)
            self.console.print(f"[bold green]✓[/bold green] Set stop on fail to {parsed_args.value}")
        elif parsed_args.command == 'awake':
            if parsed_args.value == 'on':
                parsed_args.value = True
            self.daemon_controller.set_awake(parsed_args.value)
            self.console.print(f"[bold green]✓[/bold green] Set awake to {parsed_args.value}")
        else:
            self.console.print(f"[bold red]Command not recognised: {parsed_args.command}[/bold red]")            
    
    def do_logs(self, arg):
        """Tail the worker or daemon logs"""
        parsed_args = self._parse_args_or_cancel("logs", arg)
        if parsed_args is None:
            return
        
        if parsed_args.worker:
            log_file = str(WR_LOG_PATH)  # your constant
        elif parsed_args.server:
            log_file = str(SR_LOG_PATH)  # your constant
        else:
            print("Please specify --worker or --server")
            return
        
        try:
            # Use tail -f to follow the log file
            subprocess.run(["tail", "-f", log_file])
        except KeyboardInterrupt:
            # Allow user to exit with Ctrl+C
            print("\nStopped tailing log file")
        except FileNotFoundError:
            print(f"Error: Log file not found at {log_file}")
        except Exception as e:
            print(f"Error reading log file: {e}")
        
    
    
    #Helper function

    avail_locs = QUE_LOCATIONS + list(SYNONYMS.keys())

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
        if hasattr(parsed_args, "location"):
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
        parsers = {
            "create": lambda: configs.get_train_parser(
                prog="create", desc="Create a new training run"
            ),
            "add": lambda: configs.get_train_parser(
                prog="add", desc="Add a completed training run to old_runs"
            ),
            "remove": self._get_remove_parser,
            "clear": self._get_clear_parser,
            "list": self._get_list_parser,
            "quit": self._get_quit_parser,
            "shuffle": self._get_shuffle_parser,
            "move": self._get_move_parser,
            "edit": self._get_edit_parser,
            "display": self._get_display_parser,
            "daemon": self._get_daemon_parser,
            "logs": self._get_log_parser,
            "load": self._get_load_parser,
            "save": self._get_save_parser
        }
        if cmd in parsers:
            return parsers[cmd]()
        return None

    #Que

    def _get_save_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Save the state of the Que or Daemon", prog="save"
        )
        subparsers = parser.add_subparsers(dest="command", required=True, help="Target to save")

        # Que Subparser
        que_parser = subparsers.add_parser("que", aliases=["-q"], help="Save Que state")
        que_parser.add_argument("--Timestamp", "-t", action="store_true", help='Timestamp the output file')
        que_parser.add_argument("--Output_Path", '-op', type=str, default=RUN_PATH, 
                                help=f'Output path (default: {RUN_PATH})')

        # Daemon Subparser
        daemon_parser = subparsers.add_parser("daemon", aliases=["-d"], help="Save Daemon state")
        #TODO: Maybe add this if desired
        return parser
    
    def _get_load_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Load the state of the Que or Daemon", prog="load"
        )
        subparsers = parser.add_subparsers(dest="target", required=True, help="Target to load")

        # Que Subparser
        que_load = subparsers.add_parser("que", aliases=["-q"], help="Load Que state")
        que_load.add_argument("--Input_Path", "-ip", type=str, default=RUN_PATH,
                            help=f'Input path (default: {RUN_PATH})')

        # Daemon Subparser
        daemon_load = subparsers.add_parser("daemon", aliases=["-d"], help="Load Daemon state")
        #TODO: Maybe add this if desired

        return parser

    def _get_move_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Move run between locations", prog="move"
        )
        parser.add_argument("o_location", choices=self.avail_locs)
        parser.add_argument("n_location", choices=self.avail_locs)
        parser.add_argument("oi_index", type=int)
        parser.add_argument("-of", "--of_index", type=int, default=None)
        return parser

    def _get_shuffle_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Reposition a run", prog="shuffle")
        parser.add_argument("location", choices=self.avail_locs)
        parser.add_argument("o_index", type=int)
        parser.add_argument("n_index", type=int)
        return parser

    def _get_remove_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Remove a run", prog="remove")
        parser.add_argument("location", choices=self.avail_locs)
        parser.add_argument("index", type=int)
        return parser

    def _get_display_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Display run config", prog="display"
        )
        parser.add_argument("location", choices=self.avail_locs)
        parser.add_argument("index", type=int)
        return parser

    def _get_clear_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Clear runs", prog="clear")
        parser.add_argument("location", choices=self.avail_locs)
        return parser

    def _get_list_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="List runs", prog="list")
        parser.add_argument("location", choices=self.avail_locs)
        return parser

    def _get_quit_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Exit queShell", prog="quit")
        parser.add_argument("-ns", "--no_save", action="store_true")
        return parser

    def _get_edit_parser(self) -> argparse.ArgumentParser:
        # opts_keys = list(map(str, self.que.old_runs[0].keys()))
        parser = argparse.ArgumentParser(description="Edit run", prog="edit")
        parser.add_argument("location", choices=self.avail_locs)
        parser.add_argument("index", type=int)
        # parser.add_argument("key1", type=str, choices=opts_keys)
        parser.add_argument("key1", type=str)
        parser.add_argument("value", type=str)
        parser.add_argument("-k2", "--key2", type=str, default=None)
        return parser

    
    #Daemon
    
    def _get_daemon_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Interact with the worker process", 
            prog="daemon"
        )
        
        subparsers = parser.add_subparsers(dest='command', required=True, help='Daemon commands')
        
        #Save state
        subparsers.add_parser('save', help='Save daemon state to disk')
        #Load state
        subparsers.add_parser('load', help='Load daemon state from disk')
        
        # Start
        subparsers.add_parser('start', help='Start the worker process')
        
        # Stop with timeout
        stop_parser = subparsers.add_parser('stop', help='Stop the worker gracefully, force kill if necessary')
        stop_parser.add_argument('--worker', '-w', action='store_true', help='Stop the worker process')
        stop_parser.add_argument('--supervisor', '-s', action='store_true', help='Stop the supervisor process. Include -w to stop worker as well, otherwise wait for trainloop to complete')
        stop_parser.add_argument('--timeout', '-to', type=int, default=10, help='Timeout in seconds (default: 10)')
        stop_parser.add_argument('--hard', '-hd', action='store_true', help='Force kill the worker after timeout')
        
        # Status
        subparsers.add_parser('status', help='Get worker status information')
        
        #Set stop on fail
        set_stop_on_fail_parser = subparsers.add_parser('stop-on-fail', help='Set stop on fail option')
        set_stop_on_fail_parser.add_argument('value', choices=['on', 'off'], help='Boolean value to set')
        
        #Set awake
        set_awake_parser = subparsers.add_parser('set-awake', help='Set awake option')
        set_awake_parser.add_argument('value', choices=['on', 'off'], help='Boolean value to set')
        
        
        return parser

    def _get_log_parser(self) -> argparse.ArgumentParser:
    
        parser = argparse.ArgumentParser(description="Interact with Que log files", prog="logs")
        
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--worker", "-w", action='store_true', help='Tail the Worker.log file')
        group.add_argument("--server", "-s", action='store_true', help='Tail the Server.log file')
        
        return parser
        

    #Tmux
    


"""To dos:
- There are some functions which do not have exception handling - add those
- Add probe to check server is running/start server + restart server

"""
    
    
if __name__ == "__main__":
    que_shell = QueShell()
    que_shell.cmdloop()
