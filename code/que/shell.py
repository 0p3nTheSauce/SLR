from .core import que, QUE_LOCATIONS, SYNONYMS, WR_PATH, DN_PATH, DN_LOG_PATH
# from .tmux import tmux_manager
from .server import connect_manager
import cmd as cmdLib
import shlex
from typing import Optional
import argparse
import configs
import time
from pathlib import Path
import subprocess


from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import  Confirm
from rich.text import Text
from rich.syntax import Syntax
from rich import box
import io
import sys
import json

class queShell(cmdLib.Cmd):
	def __init__(
		self,
		dn_log_path: str | Path = DN_LOG_PATH,
		auto_save: bool = True,
	) -> None:
		super().__init__()
		self.console = Console()
		self.server = connect_manager()
		self.que = self.server.get_que()
		self.dn_log_path = DN_LOG_PATH
		
		# Display welcome banner
		self._show_banner()

		self.auto_save = auto_save
		
		# Override prompt with rich styling
		self.prompt = ""  # We'll handle this with rich
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
		self.console.print("\nType [bold cyan]help[/bold cyan] or [bold cyan]?[/bold cyan] to list commands.\n")

	def cmdloop(self, intro=None):
		"""Override cmdloop to use Rich-styled prompt with readline support"""
		
		self.preloop()
		if intro is not None:
			self.intro = intro
		if self.intro:
			self.console.print(self.intro)
		
		stop = None
		while not stop:
			try:
				# Print prompt with Rich, but use standard input() for readline support
				self.console.print("[bold cyan](que)$[/bold cyan] ", end="")
				line = input()
				line = self.precmd(line)
				stop = self.onecmd(line)
				stop = self.postcmd(stop, line)
			except KeyboardInterrupt:
				self.console.print("\n[yellow]Use 'quit' or 'exit' to leave queShell[/yellow]")
			except EOFError:
				self.console.print()
				stop = self.do_quit("")
		self.postloop()

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
				
				syntax = Syntax(help_text, "text", theme="monokai", background_color="default")
				self.console.print(Panel(syntax, title=f"Help: {arg}", border_style="cyan"))
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
			("worker", "Start/manage the worker process"),
			("attach", "Attach to tmux session"),
			("save", "Save queue state to file"),
			("load", "Load queue state from file"),
			("recover", "Recover from a failed run"),
			("quit/exit", "Exit queShell"),
		]
		
		for cmd, desc in commands:
			table.add_row(cmd, desc)
		
		self.console.print(table)
		self.console.print("\n[dim]Tip: Use 'help <command>' for detailed information about a specific command[/dim]\n")

	def do_quit(self, arg):
		"""Exit the shell with style"""
		args = shlex.split(arg) if arg else []
		parser = self._get_quit_parser()
		try:
			parsed_args = parser.parse_args(args)
		except SystemExit:
			self.console.print("[yellow]Quit cancelled[/yellow]")
			return

		if not parsed_args.no_save:
			with self.console.status("[bold green]Saving state...", spinner="dots"):
				self.do_save(arg)
				time.sleep(0.5)  # Brief pause for visual feedback
		else:
			self.console.print("[yellow]Exiting without saving[/yellow]")

		self.console.print(Panel(
			"[bold green]Thank you for using queShell![/bold green]\n[dim]Queue management made beautiful ✨[/dim]",
			border_style="green"
		))
		return True

	def do_exit(self, arg):
		"""Exit the shell"""
		return self.do_quit(arg)

	def do_EOF(self, arg):
		"""Exit on Ctrl+D"""
		self.console.print()
		return self.do_quit(arg)

	def do_save(self, arg):
		"""Save state with visual feedback"""
		self.que.save_state()
		self.console.print("[bold green]✓[/bold green] Queue state saved to file")

	def do_load(self, arg):
		"""Load state with visual feedback"""
		with self.console.status("[bold cyan]Loading state...", spinner="dots"):
			self.que.load_state()
		self.console.print("[bold green]✓[/bold green] Queue state loaded from file")

	def do_recover(self, arg):
		"""Recover a run with status indication"""
		with self.console.status("[bold yellow]Recovering run...", spinner="dots"):
			self.que.recover_run()
		self.console.print("[bold green]✓[/bold green] Run recovered successfully")

	def do_clear(self, arg):
		"""Clear runs with confirmation"""
		parsed_args = self._parse_args_or_cancel("clear", arg)
		if parsed_args is None:
			return

		# Confirmation prompt
		if Confirm.ask(f"[bold red]Clear all runs from {parsed_args.location}?[/bold red]"):
			self.que.clear_runs(parsed_args.location)
			self.console.print(f"[bold green]✓[/bold green] Cleared runs from {parsed_args.location}")
		else:
			self.console.print("[yellow]Clear cancelled[/yellow]")

	def do_list(self, arg):
		"""Display runs in a beautiful table"""
		parsed_args = self._parse_args_or_cancel("list", arg)
		if parsed_args is None:
			return

		runs = self.que.list_runs(parsed_args.location)
		
		if not runs:
			self.console.print(Panel(
				f"[yellow]No runs found in {parsed_args.location}[/yellow]",
				border_style="yellow"
			))
			return

		# Create a styled table
		table = Table(
			title=f"Runs in {parsed_args.location}",
			box=box.ROUNDED,
			border_style="cyan",
			show_header=True,
			header_style="bold magenta"
		)
		
		table.add_column("Index", style="cyan", justify="right", width=8)
		for header in runs[0].keys():
			table.add_column(header.capitalize(), style="white")

		#runs are a list of Summarised dicts

		for idx, row in enumerate(runs):
			row_values = []
			for value in row.values():
				value_str = str(value)
				if len(value_str) > 60:
					value_str = value_str[:57] + "..."
				row_values.append(value_str)
			table.add_row(str(idx).zfill(3), *row_values)
			
		
		self.console.print(table)

	def do_remove(self, arg):
		"""Remove a run with confirmation"""
		parsed_args = self._parse_args_or_cancel("remove", arg)
		if parsed_args is None:
			return

		if Confirm.ask(f"[bold red]Remove run {parsed_args.index} from {parsed_args.location}?[/bold red]"):
			self.que.remove_run(parsed_args.location, parsed_args.index)
			self.console.print(f"[bold green]✓[/bold green] Removed run {parsed_args.index} from {parsed_args.location}")
		else:
			self.console.print("[yellow]Remove cancelled[/yellow]")

	def do_display(self, arg):
		"""Display run details in a styled panel"""
		parsed_args = self._parse_args_or_cancel("display", arg)
		if parsed_args is None:
			return

		run = self.que._peak_run(parsed_args.location, parsed_args.index)
		if run:
			# Format as JSON-like syntax

			run_json = json.dumps(run, indent=2)
			syntax = Syntax(run_json, "json", theme="monokai", line_numbers=True)
			
			self.console.print(Panel(
				syntax,
				title=f"Run {parsed_args.index} in {parsed_args.location}",
				border_style="cyan",
				padding=(1, 2)
			))
		else:
			self.console.print(f"[red]Run {parsed_args.index} not found in {parsed_args.location}[/red]")

	def do_shuffle(self, arg):
		"""Reposition with visual confirmation"""
		parsed_args = self._parse_args_or_cancel("shuffle", arg)
		if parsed_args is None:
			return

		self.que.shuffle(parsed_args.location, parsed_args.o_index, parsed_args.n_index)
		self.console.print(
			f"[bold green]✓[/bold green] Moved run from position {parsed_args.o_index} "
			f"to {parsed_args.n_index} in {parsed_args.location}"
		)

	def do_move(self, arg):
		"""Move run with visual feedback"""
		parsed_args = self._parse_args_or_cancel("move", arg)
		if parsed_args is None:
			return

		self.que.move(
			parsed_args.o_location,
			parsed_args.n_location,
			parsed_args.oi_index,
			parsed_args.of_index,
		)
		self.console.print(
			f"[bold green]✓[/bold green] Moved run from {parsed_args.o_location} "
			f"to {parsed_args.n_location}"
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
			self.que.create_run(admin_info, wandb_info)
		self.console.print("[bold green]✓[/bold green] Run created successfully")

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
		self.console.print(f"[bold green]✓[/bold green] Edited run {parsed_args.index} in {parsed_args.location}")

	# def do_daemon(self, arg):


	# Helper functions 
	
	def run_daemon(self) -> subprocess.Popen:
		"""Start the daemon process"""
		return subprocess.Popen(
			[sys.executable, "-u", "-m", "que.daemon"],
			stdout=open(self.dn_log_path, 'w'),
			stderr=subprocess.STDOUT,

		)
		
		
	
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
		"""Parse arguments or return None if parsing fails"""
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
			"create": lambda: configs.get_train_parser(prog="create", desc="Create a new training run"),
			"add": lambda: configs.get_train_parser(prog="add", desc="Add a completed training run to old_runs"),
			"remove": self._get_remove_parser,
			"clear": self._get_clear_parser,
			"list": self._get_list_parser,
			"quit": self._get_quit_parser,
			"shuffle": self._get_shuffle_parser,
			"move": self._get_move_parser,
			"attach": self._get_attach_parser,
			"daemon": self._get_daemon_parser,
			"worker": self._get_worker_parser,
			"edit": self._get_edit_parser,
			"display": self._get_display_parser,
		}
		if cmd in parsers:
			return parsers[cmd]()
		return None

	def _get_daemon_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(description="Start the que daemon with a given setting", prog="daemon")
		parser.add_argument("setting", choices=["sWatch", "sMonitor", "monitorO", "idle", "idle_log"])
		parser.add_argument("-re", "--recover", action="store_true")
		parser.add_argument("-ri", "--run_id", type=str, default=None)
		return parser

	def _get_worker_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(description="Start the que worker", prog="worker")
		parser.add_argument("setting", choices=["work", "idle", "idle_log", "idle_gpu"])
		return parser

	def _get_attach_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(description="Attach to tmux session")
		parser.add_argument("window", choices=["worker", "daemon"])
		return parser

	def _get_move_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(description="Move run between locations", prog="move")
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
		parser = argparse.ArgumentParser(description="Display run config", prog="display")
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
		opts_keys = list(map(str, self.que.old_runs[0].keys()))
		parser = argparse.ArgumentParser(description="Edit run", prog="edit")
		parser.add_argument("location", choices=self.avail_locs)
		parser.add_argument("index", type=int)
		parser.add_argument("key1", type=str, choices=opts_keys)
		parser.add_argument("value", type=str)
		parser.add_argument("-k2", "--key2", type=str, default=None)
		return parser





if __name__ == "__main__":
	que_shell = queShell()
	que_shell.cmdloop()