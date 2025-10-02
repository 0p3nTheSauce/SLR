from pathlib import Path
import argparse
import json
from typing import Optional, List, Literal
import subprocess
import time
import cmd as cmdLib
import shlex
from filelock import FileLock
import sys
# locals
import configs
import utils

# constants
SESH_NAME = "que_training"
DN_NAME = "daemon"
WR_NAME = "worker"
MR_NAME = "monitor"
WR_PATH = "./quefeather.py"
TMP_PATH = "./queTemp.json"
RUN_PATH = "./queRuns.json"
LOG_PATH = "./queLogs.log"
IMP_PATH = "./info/wlasl_implemented_info.json"


def retrieve_Data(path: Path) -> dict:
	"""Retrieves data from a given path."""
	with open(path, "r") as file:
		data = json.load(file)
	return data


def store_Data(path: Path, data: dict):
	"""Stores data to a given path."""
	with open(path, "w") as file:
		json.dump(data, file, indent=4)


class que:
	def __init__(
		self,
		runs_path: str | Path,
		implemented_path: str | Path,
		verbose: bool = True,
	) -> None:
		self.runs_path = Path(runs_path)
		implemented_path = Path(implemented_path)
		assert implemented_path.exists(), f"{implemented_path} does not exist"
		implemented_info = retrieve_Data(implemented_path)
		assert implemented_info, "No implemented info found"
		self.imp_models = implemented_info["models"]
		self.imp_splits = implemented_info["splits"]
		self.verbose = verbose
		self.old_runs = []
		self.to_run = []
		self.load_state()

	@classmethod
	def get_config(cls, next_run: dict) -> str:
		admin = next_run["config"]["admin"]
		return admin["config_path"]

	def print_v(self, message: str) -> None:
		"""Prints a message if verbose is True."""
		if self.verbose:
			print(message)

	def save_state(self):
		all_runs = {"old_runs": self.old_runs, "to_run": self.to_run}
		store_Data(self.runs_path, all_runs)
		self.print_v(f"Saved state to {self.runs_path}")

	def load_state(self, all_runs: Optional[dict] = None) -> dict:
		"""Loads state from queRuns.json, or dictionary, returns all_runs"""
		if all_runs:
			self.old_runs = all_runs["old_runs"]
			self.to_run = all_runs["to_run"]
			return all_runs

		if self.runs_path.exists():
			data = retrieve_Data(self.runs_path)
			self.old_runs = data.get("old_runs", [])
			self.to_run = data.get("to_run", [])
			self.print_v(f"Loaded state from {self.runs_path}")
		else:
			self.print_v(
				f"No existing state found at {self.runs_path}. Starting fresh."
			)
			self.old_runs = []
			self.to_run = []
			data = {"old_runs": [], "to_run": []}
		return data

	def fetch_state(self) -> dict:
		"""Return all_runs (without loading from file)"""
		return {
			'to_run' : self.to_run,
			'old_runs' : self.old_runs
		}

	def get_next_run(self) -> dict | None:
		"""Retrieves the next run from the queue, and moves the run to old_runs"""
		if self.to_run:
			next_run = self.to_run.pop(0)
			self.old_runs.append(next_run)
			self.print_v(f"Retrieved next run: {next_run}")
			return next_run
		else:
			self.print_v("No runs in the queue.")
			return None

	def clear_runs(self, loc: Literal['to_run', 'old_runs', 'all']): 
		"""reset the runs queue"""
		#NOTE: might change this to use location : str instead
		past = loc == 'old_runs'
		future = loc == 'to_run'
		past, future = loc == 'all', loc == 'all'
		if past:
			self.old_runs = []
		if future:
			self.to_run = []

		if past or future:
			self.print_v("Successfully cleared")
		else:
			self.print_v("No runs cleared")

	def list_configs(self, loc:Literal['to_run', 'old_runs'],
					disp: bool = True) -> List[str]:
		all_runs = self.fetch_state()
		to_disp = all_runs[loc]
		self.print_v(f'{loc}:')
		if len(to_disp) == 0:
			print("runs are finished")
		conf_list = []
		for i, run in enumerate(to_disp):
			config = self.get_config(run)
			if disp:
				print(f'{config} : {i}') 
			conf_list.append(config)
		return conf_list

	def list_runs_o(self, loc:Literal['to_run', 'old_runs'],
					disp: bool = False) -> List[str]:
		"""Summarise to a list of runs, in a given location

		Args:
			loc (Literal[&#39;to_run&#39;, &#39;old_runs&#39;]): Location to list
			disp (bool, optional): Print list, with indexes. Defaults to False.

		Returns:
			List[str]: Summarised run info 
		"""

		all_runs = self.fetch_state()
		to_disp = all_runs[loc]
		self.print_v(f'{loc}:')
		if len(to_disp) == 0:
			print("runs are finished")
		conf_list = []
		for i, run in enumerate(to_disp):
			r_str = self.str_run(run)
			if disp:
				print(f'{r_str} : {i}') 
			conf_list.append(r_str)
		return conf_list

	def list_runs(self, loc: Literal['to_run', 'old_runs'],
              disp: bool = False) -> List[str]:
		"""Summarise to a list of runs, in a given location

		Args:
			loc (Literal['to_run', 'old_runs']): Location to list
			disp (bool, optional): Print list, with indexes. Defaults to False.

		Returns:
			List[str]: Summarised run info 
		"""
		
		all_runs = self.fetch_state()
		to_disp = all_runs[loc]
		
		# Nicer header
		loc_display = loc.replace('_', ' ').title()
		self.print_v(f'\n=== {loc_display} ===')
		
		if len(to_disp) == 0:
			self.print_v("  No runs available\n")
			return []
		
		# Extract run info
		runs_info = [self._run_sum(run) for run in to_disp]
		
		# Calculate column widths
		max_model = max(len(r['model']) for r in runs_info)
		max_exp = max(len(str(r['exp_no'])) for r in runs_info)
		max_split = max(len(r['split']) for r in runs_info)
		
		conf_list = []
		for i, info in enumerate(runs_info):
			# Format with padding for alignment
			r_str = (f"{info['model']:<{max_model}}  "
					f"ex: {info['exp_no']:<{max_exp}}  "
					f"sp: {info['split']:<{max_split}}  "
					f"cf: {info['config_path']}")
			
			if disp:
				print(f'  [{i:2d}] {r_str}')
			conf_list.append(r_str)
		
		if disp:
			print()  # Add spacing after list
		
		return conf_list

	def str_run(self, run: dict) -> str:
		admin = run['config']['admin']
		return f"{admin['model']} ex: {admin['exp_no']} sp: {admin['split']} cf: {admin['config_path']}"
	
	def _run_sum(self, run: dict) -> dict:
		"""Extract key details from a run configuration.
		
		Args:
			run: Dictionary containing run configuration with admin details
			
		Returns:
			Dictionary with model, exp_no, split, and config_path
		"""
		admin = run['config']['admin']
		return {
			'model': admin['model'],
			'exp_no': admin['exp_no'],
			'split': admin['split'],
			'config_path': admin['config_path']
		}
 
	# High level functions taking multistep input

	def create_run(self,
				arg_dict: dict,
				tags: list[str],
				output: str,
				save_path: str,
				project: str,
				entity: str,
				ask: bool = True) -> None: 
		"""Create and add a new training run entry

		Args:
			arg_dict (dict): Arguments used by training function
			tags (list[str]): Wandb tags
			output (str): Experiment directory
			save_path (str): Checkpoint directory
			project (str): Wandb project
			entity (str): Wandb entity
			ask (bool, optional): Pre-check run before creation. Defaults to True.
		"""
     
		config = configs.load_config(arg_dict, verbose=True)

		if ask:
			configs.print_config(config)

		model_specifics = self.imp_models[config["admin"]["model"]]
  
		if ask:
			proceed = (
				utils.ask_nicely(
					message="Confirm: y/n: ",
					requirment=lambda x: x.lower() in ["y", "n"],
					error="y or n: ",
				).lower()
				== "y"
			)
		else:
			proceed = True

		if proceed:
			info = {
				"model_info": model_specifics,
				"config": config,
				"entity": entity,
				"project": project,
				"tags": tags,
				"output": output,
				"save_path": save_path,
			}
			self.to_run.append(info)
			self.print_v(f"Added new run: {info}")
		else:
			self.print_v("Training cancelled by user")


	def remove_run(self, loc: str, idx: int) -> Optional[dict]:
		"""Removes a run from the que 
  			Args:
	 			loc: to_run or old_runs
				idx: index of run
		  	Returns:
		   		rem: the removed run, if successful"""
	
		all_runs = self.fetch_state()

		if loc in all_runs.keys():
			if 0 <= abs(idx) < len(all_runs[loc]):
				rem = all_runs[loc].pop(idx)
				self.load_state(all_runs)
				return rem
			else:
				print(
					f"Index: {idx} out of range for to_run of length {len(all_runs[loc])}"
				)

		else:
			print(f"Location: {loc} not one of available keys: {all_runs.keys()}")

	def shuffle_configs(self):
		if len(self.to_run) == 0:
			print("Warning, to_run is empty")
			return

		for i, run in enumerate(self.to_run):
			print(f"{self.get_config(run)} : {i}")

		def requirement(x: str) -> bool:
			return (x.isdigit() and 0 <= int(x) < len(self.to_run)) or x == "q"

		idx_or_q = utils.ask_nicely(
			"select index to move (or q to quit): ",
			requirement,
			"Invalid input, please enter a valid index or 'q' to quit.",
		)

		if idx_or_q == "q":
			self.print_v("No runs moved")
			return
		else:
			idx = int(idx_or_q)

		newpos_or_q = utils.ask_nicely(
			"select new position (or q to quit): ",
			requirement,
			"Invalid input, please enter a valid index or 'q' to quit.",
		)

		if newpos_or_q == "q":
			self.print_v("No runs moved")
			return
		else:
			newpos = int(newpos_or_q)

		srun = self.to_run.pop(idx)
		if not srun:
			print(f"Invalid idx for list of configs len: {len(self.to_run)}")
			return

		self.to_run.insert(newpos, srun)

		self.print_v(
			f"Run: {self.get_config(srun)} \
		successfully moved to position: {newpos}"
		)

	def return_old(self, num_from_end: Optional[int] = None):
		"""Return the runs in old_runs to to_run. Adds them at the beggining of to_runs.
		Default behavior is to ask, otherwise moves multiple
		runs from the end specified by the num_from_end."""

		if len(self.old_runs) == 0:
			print("Warning there are no old_runs")
			return

		if num_from_end is None:
			for i, run in enumerate(self.old_runs):
				admin = run["config"]["admin"]
				print(f"{admin['config_path']} : {i}")

			to_move = []
			print("press q to quit")
			while True:
				ans = utils.ask_nicely(
					message="select index, or q: ",
					requirment=lambda x: x.isdigit() or x == "q",
					error="Invalid, select index, or q: ",
				)
				if ans == "q":
					break
				srun = self.old_runs.pop(int(ans))
				if not srun:
					print(f"Invalid idx for list of configs len: {len(self.old_runs)}")
				else:
					self.print_v(f"Moving: {self.get_config(srun)}, to to_run")
					to_move.append(srun)

			self.to_run = to_move + self.to_run

		else:
			old2mv = []  # return runs to the begining of to_run
			for _ in range(num_from_end):
				old2mv.append(self.old_runs.pop(-1))
			self.to_run = old2mv + self.to_run
			self.print_v(f"moved {num_from_end} from old")


# TODO: put this in a class


def join_session(
	wndw_name: str,
	sesh_name: str,
) -> None:
	"""Join a tmux session and optionally target a specific window."""
	tmux_cmd = ["tmux", "attach-session", "-t", f"{sesh_name}:{wndw_name}"]
	try:
		_ = subprocess.run(tmux_cmd, check=True)
	except subprocess.CalledProcessError as e:
		print("join_session ran into an error when spawning the worker process: ")
		print(e.stderr)


def switch_to_window(wndw_name: str, sesh_name: str) -> None:
	"""Switch to specific window by index."""
	tmux_cmd = ["tmux", "select-window", "-t", f"{sesh_name}:{wndw_name}"]
	try:
		_ = subprocess.run(tmux_cmd, check=True)
	except subprocess.CalledProcessError as e:
		print("switch_to_window ran into an error when spawning the worker process: ")
		print(e.stderr)


def send(
	cmd: str,
	wndw_name: str,
	sesh_name: str,
) -> None:
	tmux_cmd = ["tmux", "send-keys", "-t", f"{sesh_name}:{wndw_name}", cmd, "Enter"]
	try:
		subprocess.run(tmux_cmd, check=True)
	except subprocess.CalledProcessError as e:
		print("Send ran into an error when spawning the worker process: ")
		print(e.stderr)


def print_tmux(
	message: str,
	wndw_name: str,
	sesh_name: str,
) -> None:
	send(cmd=f"echo {message} ", wndw_name=wndw_name, sesh_name=sesh_name)


def seperator(title: str = "Next Run"):
	"""This prints out a seperator between training runs"""
	sep = ""
	if title:
		sep += ("\n" * 2) + ("-" * 10) + ("\n")
		sep += f"{title:^10}"
		sep += ("\n" * 2) + ("-" * 10) + ("\n")
	else:
		sep += "\n"
	return sep


def idle(message: str) -> str:
	# testing if blocking
	print(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
	for i in range(10):
		print(f"Idling: {i}")
		print(message)
		time.sleep(10)
	print(f"Finishign at {time.strftime('%Y-%m-%d %H:%M:%S')}")
	return message


class worker:
	def __init__(
		self,
		exec_path: str = WR_PATH,
		temp_path: str = TMP_PATH,
		log_path: str = LOG_PATH,
		wr_name: str = WR_NAME,
		sesh_name: str = SESH_NAME,
	):
		self.temp_path = Path(temp_path)
		self.exec_path = exec_path
		self.log_path = log_path
		self.wr_name = wr_name
		self.sesh_name = sesh_name

	def start_here(self, next_run: dict, args: Optional[list[str]] = None) -> None:
		"""Blocking start which prints worker output in daemon terminal"""

		store_Data(self.temp_path, next_run)
		cmd = [self.exec_path, self.wr_name]
		if args:
			cmd.extend(args)
		with subprocess.Popen(
			cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
		) as proc:
			if proc.stdout:
				for line in proc.stdout:
					print(line.strip())

	def start_log(
		self, next_run: dict, args: Optional[list[str]] = None
	) -> subprocess.Popen:
		"""Non-blocking start which prints worker output to temp.json, and passes the process"""
		store_Data(self.temp_path, next_run)
		cmd = [self.exec_path, self.wr_name]
		if args:
			cmd.extend(args)

		return subprocess.Popen(
			cmd, stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT
		)

	def monitor_log(self):
		send(f"tail -f {self.log_path}", self.wr_name, self.sesh_name)


class daemon:
	"""Class for the queue daemon process. The function works in a fetch execute repeat
	cycle. The function reads runs from the queRuns.json file, then writes them to the
	queTemp.json file for the worker process to find. The daemon spawns the worker, then
	waits for it to complete before proceeding. The daemon runs in it's own tmux session,
	while the worker outputs to a log file.

	Args:
			name: of own tmux window
			w_name: of worker tmux window (monitor)
			sesh: tmux session name
			runs_path: to queRuns.json
			temp_path: to queTemp.json
			imp_path: to implemented_info.json
			exec_path: to quefeather.py
			verbose: speaks to you
			wr: supply its worker
			q: supply its que
	"""

	def __init__(
		self,
		name: str = DN_NAME,
		w_name: str = WR_NAME,
		sesh: str = SESH_NAME,
		runs_path: str = RUN_PATH,
		temp_path: str = TMP_PATH,
		imp_path: str = IMP_PATH,
		exec_path: str = WR_PATH,
		verbose: bool = True,
		wr: Optional[worker] = None,
		q: Optional[que] = None,
	) -> None:
		self.name = name
		self.w_name = w_name
		self.sesh = sesh
		self.runs_path = Path(runs_path)
		self.temp_path = Path(temp_path)
		if wr:
			self.worker = wr
		else:
			self.worker = worker(
				exec_path=exec_path, temp_path=temp_path, wr_name=w_name, sesh_name=sesh
			)
		if q:
			self.que = q
		else:
			self.que = que(
				runs_path=runs_path, implemented_path=imp_path, verbose=verbose
			)
		self.verbose = verbose

	def print_v(self, message: str) -> None:
		"""Prints a message if verbose is True."""
		if self.verbose:
			print(message)

	def start_n_watch(self):
		"""Start process in this terminal and watch"""
		while True:
			self.que.load_state()
			next_run = self.que.get_next_run()
			self.que.save_state()

			if next_run is None:
				self.print_v("No more runs to execute")
				break

			self.print_v(seperator(que.get_config(next_run)))

			self.worker.start_here(next_run)

	def start_n_monitor_simple(self):
		"""Start process and use existing tmux monitoring"""
		while True:
			self.que.load_state()
			next_run = self.que.get_next_run()
			self.que.save_state()

			if next_run is None:
				self.print_v("No more runs to execute")
				break

			self.print_v(seperator(que.get_config(next_run)))

			# Start process in background
			proc = self.worker.start_log(next_run)

			# Start monitoring in tmux (non-blocking)
			self.worker.monitor_log()

			# Wait for completion
			return_code = proc.wait()

			if return_code == 0:
				self.print_v("Process completed successfully")
			else:
				self.print_v(f"Process failed with return code: {return_code}")


class queShell(cmdLib.Cmd):
	intro = "QueShell: Type help or ? to list commands.\n"
	prompt = "(QueShell)$ "

	def __init__(
		self,
		dn_name: str = DN_NAME,
		wr_name: str = WR_NAME,
		sesh_name: str = SESH_NAME,
		run_path: str = RUN_PATH,
		temp_path: str = TMP_PATH,
		imp_path: str = IMP_PATH,
		exec_path: str = WR_PATH,
		verbose: bool = True,
		auto_save: bool = False,
	) -> None:
		super().__init__()
		self.que = que(run_path, imp_path, verbose)
		self.daemon = daemon(
			name=dn_name,
			w_name=wr_name,
			sesh=sesh_name,
			runs_path=run_path,
			temp_path=temp_path,
			imp_path=imp_path,
			exec_path=exec_path,
			q=self.que,
		)
		self.auto_save = auto_save

	# queShell based

	def do_help(self, arg):
		"""Override help to provide detailed argparse help"""

		#TODO: seeing a pattern here
  
		if arg == "create":
			parser = configs.take_args(
				self.que.imp_splits,
				self.que.imp_models,
				return_parser_only=True,
				prog="create",
				desc="Create a new training run",
			)
			assert isinstance(parser, argparse.ArgumentParser)
			parser.print_help()
		elif arg == "remove": 
			parser = self._get_remove_run_parser()
			parser.print_help()
		elif arg == "clear":
			parser = self._get_clear_runs_parser()
			parser.print_help()
		elif arg == "list_runs":
			parser = self._get_list_parser()
			parser.print_help()
		else:
			super().do_help(arg)

	def do_quit(self, arg):
		"""Exit the shell"""
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

	#	-	Low level 
 
	def do_save(self, arg):
		"""Save state of que to queRuns.json"""
		self.que.save_state()

	def do_load(self, arg):  # happens automatically anyway
		"""Load state of que from queRuns.json"""
		self.que.load_state()
  
	#	-	High level
  
	def do_clear(self, arg):
		"""Clear past or future runs"""
		args = shlex.split(arg)
		parser = self._get_clear_runs_parser()
		try:
			parsed_args = parser.parse_args(args)
		except SystemExit as _:
			print("Clear cancelled")
			return

		self.que.clear_runs(parsed_args.location)

	def do_list_runs(self, arg):
		"""Summarise to a list of runs, in a given location"""
		args = shlex.split(arg)
		parser = self._get_list_parser()
		try:
			parsed_args = parser.parse_args(args)
		except SystemExit as _:
			print("List cancelled")
			return 

		self.que.list_runs(parsed_args.location, disp=True)

	def do_remove(self, arg):
		"""Remove a run from the past or future"""
		args = shlex.split(arg)
		parser = self._get_remove_run_parser()
		try:
			parsed_args = parser.parse_args(args)
		except SystemExit as _:
			print("Remove cancelled")
			return

		self.que.remove_run(parsed_args.location, parsed_args.index)
  
	def do_create(self, arg):
		"""Create a new run and add it to the queue"""
		args = shlex.split(arg)

		try: 
			maybe_args = configs.take_args(
				self.que.imp_splits, self.que.imp_models.keys(), args	
	   		)
		except (SystemExit, ValueError) as _:
			print("Create cancelled (incorrect arguments)")
			return
   
		if isinstance(maybe_args, tuple):
			arg_dict, tags, output, save_path, project, entity = maybe_args
		else:
			print("Create cancelled (by user)")
			return
  
		self.que.create_run(arg_dict, tags, output, save_path, project, entity)
  
	#helper functions 
 
	#	-	Parser getters
 
	def _get_remove_run_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(description="Remove a run from future or past runs",
								prog='remove')
		parser.add_argument(
			'location',
			choices=["to_run", "old_runs"],
			help="Location of the run"
		)
		parser.add_argument(
			'index',
			type=int,
			help="Position of run in location"
		)
		return parser

	def _get_clear_runs_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(description="Clear future or past runs",
								prog='clear')
		parser.add_argument(
			'location',
			choices=["to_run", "old_runs", "all"],
			help="Location of the run"
		)
		return parser

	def _get_list_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(description="Summarise to a list of runs, in a given location",
								prog='list_runs')
		parser.add_argument(
			'location',
			choices=["to_run", "old_runs"],
			help="Location of the run"
		)
  
		return parser

	#	- 	Misc
	
	def _auto_save(self):
		if self.auto_save:
			self.que.save_state()

if __name__ == "__main__":
	# try:
	# 	_ = join_session(WR_NAME, SESH_NAME)
	# except subprocess.CalledProcessError as e:
	# 	print("Daemon ran into an error when spawning the worker process: ")
	# 	print(e.stderr)
	#place lock on queRuns.json 
 
	lock = FileLock(f'{RUN_PATH}.lock', timout=1)
	try:
		with lock:
			print(f'Acquired lock on {RUN_PATH}')
			queShell().cmdloop()
	except TimeoutError:
		print(f'Error: {RUN_PATH} is currently in used by another user')
		sys.exit(1)
	finally:
		print(f"Released lock on {RUN_PATH}")
     
