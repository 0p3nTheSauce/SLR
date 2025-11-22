from .core import que, QUE_LOCATIONS, SYNONYMS, DN_NAME, WR_NAME, SESH_NAME, WR_PATH, RUN_PATH
from .tmux import tmux_manager
from .server import connect_que
import cmd as cmdLib
import shlex
from typing import Optional
import argparse
import configs
import time
from pathlib import Path


class queShell(cmdLib.Cmd):
	intro = "queShell: Type help or ? to list commands.\n"
	prompt = "(que)$ "
	avail_locs = QUE_LOCATIONS + list(SYNONYMS.keys())

	def __init__(
		self,
		dn_name: str = DN_NAME,
		wr_name: str = WR_NAME,
		sesh_name: str = SESH_NAME,
		exec_path: str = WR_PATH,
		run_path: str | Path = RUN_PATH,
		verbose: bool = True,
		auto_save: bool = True,
	) -> None:
		super().__init__()
		self.que = connect_que()
		self.tmux_man = tmux_manager(
			wr_name=wr_name, dn_name=dn_name, sesh_name=sesh_name, exec_path=exec_path
		)
		check = self.tmux_man.check_tmux_session()
		if check is None:
			check = self.tmux_man.setup_tmux_session()
		if check is None:
			raise ValueError("Failed to start tmux manager, exiting")
		self.dn_name = dn_name
		self.wr_name = wr_name
		self.auto_save = auto_save

	# queShell based

	def do_help(self, arg):
		"""Override help to provide detailed argparse help"""
		parser = self._get_parser(arg)
		if parser:
			parser.print_help()
		else:
			super().do_help(arg)

	def do_quit(self, arg):
		"""Exit the shell"""
		args = shlex.split(arg)
		parser = self._get_quit_parser()
		try:
			parsed_args = parser.parse_args(args)
		except SystemExit as _:
			print("Quit cancelled")
			return

		if not parsed_args.no_save:
			self.do_save(arg)
		else:
			self.que.print_v("Exiting without saving")

		print("Goodbye!")
		return True

	def do_exit(self, arg):
		"""Exit the shell"""
		return self.do_quit(arg)

	def do_EOF(self, arg):
		"""Exit on Ctrl+D"""
		print()  # Print newline for clean exit
		return self.do_quit(arg)

	# que based functions

	def do_save(self, arg):
		"""Save state of que to queRuns.json"""
		self.que.print_v("Que saved to file")
		self.que.save_state()

	def do_load(self, arg):  # happens automatically anyway
		"""Load state of que from queRuns.json"""
		self.que.print_v("Que loaded from file")
		self.que.load_state()

	def do_recover(self, arg):
		"""Recover a run, at the moment implemented for restarting the daemon script"""
		self.que.recover_run()
		self.que.print_v("Recovered run")

	def do_clear(self, arg):
		"""Clear past or future runs"""
		parsed_args = self._parse_args_or_cancel("clear", arg)
		if parsed_args is None:
			return

		self.que.clear_runs(parsed_args.location)

	def do_list(self, arg):
		"""Summarise to a list of runs, in a given location"""
		parsed_args = self._parse_args_or_cancel("list", arg)
		if parsed_args is None:
			return

		# self.que.disp_runs(parsed_args.location)
		que.disp_runs(self.que.list_runs(parsed_args.location), parsed_args.location)

	def do_remove(self, arg):
		"""Remove a run from a given que"""
		parsed_args = self._parse_args_or_cancel("remove", arg)
		if parsed_args is None:
			return

		self.que.remove_run(parsed_args.location, parsed_args.index)

	def do_display(self, arg):
		"""Display a run config for a given que"""
		parsed_args = self._parse_args_or_cancel("display", arg)
		if parsed_args is None:
			return

		self.que.disp_run(parsed_args.location, parsed_args.index)

	def do_shuffle(self, arg):
		"""Reposition a run in the que"""
		parsed_args = self._parse_args_or_cancel("shuffle", arg)
		if parsed_args is None:
			return

		self.que.shuffle(parsed_args.location, parsed_args.o_index, parsed_args.n_index)

	def do_move(self, arg):
		"""Moves a run between locations in que"""
		parsed_args = self._parse_args_or_cancel("move", arg)
		if parsed_args is None:
			return

		self.que.move(
			parsed_args.o_location,
			parsed_args.n_location,
			parsed_args.oi_index,
			parsed_args.of_index,
		)

	def do_create(self, arg):
		"""Create a new run and add it to the queue"""
		args = shlex.split(arg)

		try:
			maybe_args = configs.take_args(sup_args=args)
		except (SystemExit, ValueError) as _:
			print("Create cancelled (incorrect arguments)")
			return

		if isinstance(maybe_args, tuple):
			admin_info, wandb_info = maybe_args
		else:
			print("Create cancelled (by user)")
			return

		self.que.create_run(admin_info, wandb_info)
  
	def do_add(self, arg):
		"""Add a completed run to the old runs que"""
		args = shlex.split(arg)

		try:
			maybe_args = configs.take_args(sup_args=args, ask_bf_ovrite=False)
		except (SystemExit, ValueError) as _:
			print("Add cancelled (incorrect arguments)")
			return

		if isinstance(maybe_args, tuple):
			admin_info, wandb_info = maybe_args
		else:
			print("Add cancelled (by user)")
			return

		self.que.add_run(admin_info, wandb_info)
  

	def do_edit(self, arg):
		"""Edit a run in a given location"""
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

	# process based functions

	# tmux
	def do_attach(self, arg):
		"""Attaches to one of the validated tmux sessions"""
		parsed_args = self._parse_args_or_cancel("attach", arg)
		if parsed_args is None:
			return

		self.tmux_man.join_session(parsed_args.window)

	def do_daemon(self, arg):
		"""Start the daemon with the given setting"""
		parsed_args = self._parse_args_or_cancel("daemon", arg)
		if parsed_args is None:
			return

		add_args = []
		if parsed_args.setting == "recover":
			add_args.append(f" -os {parsed_args.o_setting}")
			if parsed_args.run_id:
				add_args.append(f" -ri {parsed_args.run_id}")

		ext_args = None if len(add_args) == 0 else add_args

		# make sure que is consistent before and after starting daemon
		self.que.save_state()
		self.tmux_man.start(self.dn_name, parsed_args.setting, ext_args=ext_args)
		# give daemon some time
		time.sleep(5)
		self.que.load_state()

	def do_worker(self, arg):
		"""Start the worker with the given setting"""
		parsed_args = self._parse_args_or_cancel("worker", arg)
		if parsed_args is None:
			return

		self.tmux_man.start(self.wr_name, parsed_args.setting)

	# helper functions

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
		"""Parse arguments or return None if parsing fails/is cancelled"""
		args = shlex.split(arg)
		parser = self._get_parser(cmd)
		# assert isinstance(parser, argparse.ArgumentParser), f"{cmd} cannot use this generic parser"
		if parser:
			try:
				return self._apply_synonyms(parser.parse_args(args))
			except (SystemExit, ValueError):
				print(f"{cmd} cancelled")
				return None
		else:
			print(f"{cmd} not found")

	def _get_parser(self, cmd: str) -> Optional[argparse.ArgumentParser]:
		"""Get argument parser for a given command"""
		parsers = {
			"create": lambda: configs.get_train_parser(
				prog="create",
				desc="Create a new training run",
			),
			"add": lambda: configs.get_train_parser(
				prog="add",
				desc="Add a completed training run to old_runs",
			),
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
			parser = parsers[cmd]()
			# assert isinstance(parser, argparse.ArgumentParser), f"{cmd} parser invalid"
			return parser
		return None

	def _get_daemon_parser(self) -> argparse.ArgumentParser:
		"""Get parser for daemon command"""
		parser = argparse.ArgumentParser(
			description="Start the que daemon with a given setting", prog="daemon"
		)
		parser.add_argument(
			"setting",
			choices=["sWatch", "sMonitor", "monitorO", "idle", "idle_log"],
			help="Operation of daemon:  worker here, worker in seperate window, tail log file, worker idle here, worker idle and log",
		)
		parser.add_argument(
			"-re", "--recover", action="store_true", help="Recover from run failure"
		)
		parser.add_argument(
			"-ri",
			"--run_id",
			type=str,
			help="The run id, if needed. Otherwise keeps the run id written to Temp",
			default=None,
		)
		return parser

	def _get_worker_parser(self) -> argparse.ArgumentParser:
		"""Get parser for worker command"""
		parser = argparse.ArgumentParser(
			description="Start the que worker with a given setting", prog="worker"
		)
		parser.add_argument(
			"setting",
			choices=["work", "idle", "idle_log", "idle_gpu"],
			help="Operation of worker: do its main job, idle here, idle in log, idle on GPU",
		)
		return parser

	def _get_attach_parser(self) -> argparse.ArgumentParser:
		"""Get parser for attach command"""
		parser = argparse.ArgumentParser(
			description="Attach to the daemon or worker tmux session"
		)
		parser.add_argument(
			"window", choices=["worker", "daemon"], help="Tmux window to attach to"
		)
		return parser

	def _get_move_parser(self) -> argparse.ArgumentParser:
		"""Get parser for move command"""
		parser = argparse.ArgumentParser(
			description="Moves a run between locations in que", prog="move"
		)
		parser.add_argument(
			"o_location", choices=self.avail_locs, help="Original location"
		)
		parser.add_argument("n_location", choices=self.avail_locs, help="New location")
		parser.add_argument(
			"oi_index", type=int, help="Index of run in original location"
		)
		parser.add_argument(
			"-of",
			"--of_index",
			type=int,
			help="Final original index if specifying a range",
			required=False,
			default=None,
		)
		return parser

	def _get_shuffle_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(
			description="Repositions a run from the que", prog="shuffle"
		)
		parser.add_argument(
			"location", choices=self.avail_locs, help="Location of the run"
		)
		parser.add_argument(
			"o_index", type=int, help="Original position of run in location"
		)
		parser.add_argument("n_index", type=int, help="New position of run in location")
		return parser

	def _get_remove_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(
			description="Remove a run in a given que", prog="remove"
		)
		parser.add_argument(
			"location", choices=self.avail_locs, help="Location of the run"
		)
		parser.add_argument("index", type=int, help="Position of run in location")
		return parser

	def _get_display_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(
			description="Display the config of a run in a given que", prog="display"
		)
		parser.add_argument(
			"location", choices=self.avail_locs, help="Location of the run"
		)
		parser.add_argument("index", type=int, help="Position of run in location")
		return parser

	def _get_clear_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(
			description="Clear future or past runs", prog="clear"
		)
		parser.add_argument(
			"location",
			choices=self.avail_locs,
			help="Location of the run",
		)
		return parser

	def _get_list_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(
			description="Summarise to a list of runs, in a given location", prog="list"
		)
		parser.add_argument(
			"location", choices=self.avail_locs, help="Location of the run"
		)

		return parser

	def _get_quit_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(
			description="Exit queShell", prog="<quit|exit>"
		)
		parser.add_argument(
			"-ns", "--no_save", action="store_true", help="Don't autosave on exit"
		)

		return parser

	def _get_edit_parser(self) -> argparse.ArgumentParser:
		opts_keys = list(map(str, self.que.old_runs[0].keys()))
		parser = argparse.ArgumentParser(description="Edit run", prog="<edit>")
		parser.add_argument(
			"location", choices=self.avail_locs, help="Location of the run"
		)
		parser.add_argument("index", type=int, help="Position of run in location")
		parser.add_argument(
			"key1",
			type=str,
			help="First key in dictionary",
			choices=opts_keys,
		)
		parser.add_argument(
			"value",
			type=str,
			help="Other types not implemented yet",
		)
		parser.add_argument(
			"-k2", "--key2", type=str, help="Optional second key", default=None
		)

		return parser

if __name__ == "__main__":
	que_shell = queShell()
	que_shell.cmdloop()