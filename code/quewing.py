#!/home/luke/miniconda3/envs/wlasl/bin/python
from pathlib import Path
import argparse
import json
from typing import Optional, List, Literal, TypeAlias, Tuple, Dict
import subprocess
import time
import cmd as cmdLib
import shlex
from filelock import FileLock
import sys
import wandb
import torch

# locals
import configs
import utils
from utils import gpu_manager
from training import train_loop, wandb_manager

# constants
SESH_NAME = "que_training"
DN_NAME = "daemon"
WR_NAME = "worker"
MR_NAME = "monitor"
WR_PATH = "./quefeather.py"
TMP_PATH = "./queTemp.json"
RUN_PATH = "./queRuns.json"
LOG_PATH = "./queLogs.log"
TO_RUN = "to_run"
OLD_RUNS = "old_runs"
# List for argparse choices
QUE_LOCATIONS = [TO_RUN, OLD_RUNS]
SYNONYMS = {
			'new': 'to_run',
			'tr': 'to_run',
			'old': 'old_runs',
			'or': 'old_runs',
		}
QueLocation: TypeAlias = Literal["to_run", "old_runs"]


def retrieve_Data(path: Path) -> dict:
	"""Retrieves data from a given path."""
	with open(path, "r") as file:
		data = json.load(file)
	return data


def store_Data(path: Path, data: dict):
	"""Stores data to a given path."""
	with open(path, "w") as file:
		json.dump(data, file, indent=4)

#NOTE: at some point it might be nice to do away with the temp file
class que:
	def __init__(
		self,
		runs_path: str | Path,
		verbose: bool = True,
		auto_save: bool = False
	) -> None:
		self.runs_path = Path(runs_path)
		self.imp_splits = configs.get_avail_splits()
		self.verbose = verbose
		self.old_runs = []
		self.to_run = []
		self.auto_save = auto_save
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
		all_runs = {OLD_RUNS: self.old_runs, TO_RUN: self.to_run}
		store_Data(self.runs_path, all_runs)
		# self.print_v(f"Saved state to {self.runs_path}")

	def load_state(self, all_runs: Optional[dict] = None) -> dict:
		"""Loads state from queRuns.json, or dictionary, returns all_runs"""
		if all_runs:
			self.old_runs = all_runs[OLD_RUNS]
			self.to_run = all_runs[TO_RUN]
			return all_runs

		if self.runs_path.exists():
			data = retrieve_Data(self.runs_path)
			self.old_runs = data.get(OLD_RUNS, [])
			self.to_run = data.get(TO_RUN, [])
			# self.print_v(f"Loaded state from {self.runs_path}")
		else:
			self.print_v(
				f"No existing state found at {self.runs_path}. Starting fresh."
			)
			self.old_runs = []
			self.to_run = []
			data = {OLD_RUNS: [], TO_RUN: []}
		return data

	def fetch_state(self, loc: QueLocation) -> list:
		"""Return reference to the specified list"""
		return self.to_run if loc == TO_RUN else self.old_runs

	def run_sum(self, run: dict, exc: Optional[List[str]] = None) -> dict:
		"""Extract key details from a run configuration.

		Args:
			run: Dictionary containing run configuration with admin details
			exc: Optional list of keys to exclude from the summary

		Returns:
			Dictionary with model, exp_no, split, and config_path
		"""
		admin = run["admin"]

		dic = {}

		if "run_id" in run:
			dic["run_id"] = run["run_id"]

		dic.update(
			{
				"model": admin["model"],
				"split": admin["split"],
				"exp_no": admin["exp_no"],
				"config_path": admin["config_path"],
			}
		)

		if exc:
			for key in exc:
				if key in dic:
					dic.pop(key)

		return dic

	def get_runs_info(
		self, run_confs: List[Dict]
	) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
		"""Get summarised run info, and stats for printing

		Args:
						run_confs (List[Dict]): A list of run configs (to_run or old_runs)

		Returns:
						Tuple[List[Dict], Dict]: List of summary dictionaries, dictionary of max lengths
		"""

		runs_info = [self.run_sum(run) for run in run_confs]

		# Calculate column widths
		max_model = max(len(r["model"]) for r in runs_info)
		max_exp = max(len(str(r["exp_no"])) for r in runs_info)
		max_split = max(len(r["split"]) for r in runs_info)

		stats = {}

		if "run_id" in runs_info[0]:
			max_id = max(len(r["run_id"]) for r in runs_info)
			stats["max_id"] = max_id

		stats.update(
			{"max_model": max_model, "max_exp": max_exp, "max_split": max_split}
		)

		return runs_info, stats

	def run_str(
		self, r_info: Dict[str, str], stats: Optional[Dict[str, int]] = None
	) -> str:
		"""Convert a run to summarised string representation

		Args:
						r_info (Dict): Summarised run info.
						stats (Optional[Dict[str, int]], optional): Max lengths for alignment. Defaults to None.

		Returns:
						str: Summarised string representation of run info
		"""

		if stats is None:
			stats = {
				"max_id": 0,
				"max_model": 0,
				"max_exp": 0,
				"max_split": 0,
			}

		r_str = ""

		if "run_id" in r_info:
			r_str += f"Run ID: {r_info['run_id']:<{stats['max_id']}}  "

		r_str += (
			f"{r_info['model']:<{stats['max_model']}}  "
			f"Split: {r_info['split']:<{stats['max_split']}}  "
			f"Exp: {r_info['exp_no']:<{stats['max_exp']}}  "
			f"Config: {r_info['config_path']}"
		)

		return r_str

	def get_next_run(self) -> Optional[dict]:
		"""Retrieves the next run from the que, and saves state"""
		self.load_state()
		if self.to_run:
			next_run = self.to_run.pop(0)
			self.save_state()
			self.print_v(f"Retrieved next run: {self.run_str(self.run_sum(next_run))}")
			return next_run
		else:
			self.print_v("No runs in the queue.")
			return

	def store_old_run(self, old_run: Dict):
		"""Saves run to OLD_RUNS, and save state"""
		self.load_state()
		self.old_runs.insert(0, old_run)
		self.save_state()
		self.print_v("Saved to old_runs")

	def clear_runs(self, loc: QueLocation):
		"""reset the runs queue"""
		past = loc == OLD_RUNS
		future = loc == TO_RUN
		if past:
			self.old_runs = []
		if future:
			self.to_run = []

		if past or future:
			self.print_v("Successfully cleared")
		else:
			self.print_v("No runs cleared")

	def list_runs(self, loc: QueLocation, disp: bool = False) -> List[str]:
		"""Summarise to a list of runs, in a given location

		Args:
						loc (QueLocation): Location to list
						disp (bool, optional): Print list, with indexes. Defaults to False.

		Returns:
						List[str]: Summarised run info
		"""

		to_disp = self.fetch_state(loc)

		# Nicer header
		loc_display = loc.replace("_", " ").title()

		if disp:
			print(f"\n=== {loc_display} ===")

		if len(to_disp) == 0:
			print("  No runs available\n")
			return []

		# Extract run info
		runs_info, stats = self.get_runs_info(to_disp)

		conf_list = []
		for i, info in enumerate(runs_info):
			# Format with padding for alignment
			r_str = self.run_str(info, stats)
			if disp:
				print(f"  [{i:2d}] {r_str}")
			conf_list.append(r_str)

		if disp:
			print()  # Add spacing after list

		return conf_list

	def _is_dup_exp(self, new_run: dict) -> bool:
		"""Check if new_run already exists in to_run or old_runs"""
		exc = ["run_id", "config_path"]
		new_sum = self.run_sum(new_run, exc)

		for run in self.to_run + self.old_runs:
			if self.run_sum(run, exc) == new_sum:
				return True
		return False

	def create_run(
		self,
		arg_dict: dict,
		tags: list[str],
		project: str,
		entity: str,
		ask: bool = True,
	) -> None:
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

		try:
			config = configs.load_config(arg_dict)
		except ValueError:
			print(f"{arg_dict['config_path']} not found")
			self.print_v("Training cancelled")
			return

		if self._is_dup_exp(config):
			print(f"Duplicate run detected: {self.run_str(self.run_sum(config))}")
			self.print_v("Training cancelled")
			return

		if ask:
			configs.print_config(config)

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

			config['wandb'] = {
				'entity': entity,
				'project': project,
				'tags': tags
			}
						
			self.to_run.append(config)
			self.print_v(f"Added new run: {self.run_str(self.run_sum(config))}")
		else:
			self.print_v("Training cancelled by user")

	def remove_run(self, loc: QueLocation, idx: int) -> Optional[dict]:
		"""Removes a run from the que
		Args:
						loc: TO_RUN or OLD_RUNS
						idx: index of run
		Returns:
						rem: the removed run, if successful"""

		to_remove = self.fetch_state(loc)

		if abs(idx) < len(to_remove):
			self.print_v(f"Successfully removed entry from {loc}")
			return to_remove.pop(idx)
		else:
			print(f"Index: {idx} out of range for len({loc}) = {len(to_remove)}")

	def shuffle(self, loc: QueLocation, o_idx: int, n_idx: int):
		"""Repositions a run from the que
		Args:
						loc: TO_RUN or OLD_RUNS
						o_idx: original index of run
						n_idx: new index of run
		"""

		to_shuffle = self.fetch_state(loc)
		if len(to_shuffle) == 0:
			print(f"{loc} is empty")
			return

		if abs(o_idx) < len(to_shuffle):
			srun = to_shuffle.pop(o_idx)
		else:
			print(f"{o_idx} out of range for len({loc}) - 1 = {len(to_shuffle) - 1}")
			return

		if not abs(n_idx) <= len(to_shuffle):
			print(f"Warning: {n_idx} out of range for len({loc}) = {len(to_shuffle)}")

		to_shuffle.insert(n_idx, srun)

		self.list_runs(loc, disp=self.verbose)

	def move(
		self,
		o_loc: QueLocation,
		n_loc: QueLocation,
		oi_idx: int,
		of_idx: Optional[int] = None
	):
		"""Moves a run between locations in que (at beginning)

		Args:
						o_loc (QueLocation): Old location
						n_loc (QueLocation): New location
						oi_idx (int): Old initial index
						of_idx (int): Old final index, if specifying a range.
		"""

		old_location = self.fetch_state(o_loc)
		new_location = self.fetch_state(n_loc)
	
	
		if of_idx is None:

			if abs(oi_idx) < len(old_location):
				run = old_location.pop(oi_idx)
			else:
				print(
					f"{oi_idx} is out of range. Length of {o_loc} is only: {len(old_location)}"
				)
				return

			new_location.insert(0, run)
			
		else:
			
			if abs(oi_idx) < len(old_location) and abs(of_idx) < len(old_location): 
				tomv = []
				for i in range(oi_idx, of_idx+1):    
					tomv.append(old_location.pop(oi_idx))
			else:
				print(
					f"Range: {oi_idx} - {of_idx} is an invalid range. Length of {o_loc} is: {len(old_location)}"
				)
				return 
			
			for run in tomv:
				new_location.insert(0, run)
					
		
		self.print_v("Successfully added\n")

		self.list_runs(n_loc, disp=self.verbose)


class tmux_manager:
	def __init__(
		self,
		wr_name: str = WR_NAME,
		dn_name: str = DN_NAME,
		sesh_name: str = SESH_NAME,
		exec_path: str = WR_PATH,
	) -> None:
		self.wr_name = wr_name
		self.dn_name = dn_name
		self.sesh_name = sesh_name
		self.exec_path = exec_path
		
		ep = Path(exec_path)
		if (not ep.exists()) or (not ep.is_file()):
			raise ValueError(f"Executable: {exec_path} does not exist")

	def setup_tmux_session(self) -> Optional[list[subprocess.CompletedProcess[bytes]]]:
		"""Create the que_training tmux session is set up, with windows daemon and worker

		Returns:
			Optional[list[subprocess.CompletedProcess[bytes]]]: A list of successful process outputs, or None if one or both failed.
		"""
	
		create_sesh_cmd = [
			'tmux', 'new-session', '-d', '-s', self.sesh_name, # -d for detach 
			'-n', f'{self.dn_name}'
		]
		create_wWndw_cmd = [ #daemon window created in first command
			'tmux', 'new-window', '-t', self.sesh_name, '-n', self.wr_name
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
			tmux_cmd = ['tmux', 'has-session', '-t', f'{self.sesh_name}:{win_name}']
			try:
				results.append(subprocess.run(tmux_cmd, check=True, capture_output=True, text=True))
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
			print(f"Window {wndw} not one of validated windows: {', '.join(avail_wndws)}")
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

	def _send(self, cmd: str, wndw: str) -> Optional[subprocess.CompletedProcess[bytes]]:  # use with caution
		"""Send a command to the given window

		Args:
			cmd (str): The command as you would type in the terminal
			wndw (str): The tmux window

		Returns:
			Optional[subprocess.CompletedProcess[bytes]]: The return object of the completed process, or None if failure.
		"""
		avail_wndws = [self.dn_name, self.wr_name]
		if wndw not in avail_wndws:
			print(f"Window {wndw} not one of validated windows: {', '.join(avail_wndws)}")
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
			
	def start(self, mode: str, setting: str, ext_args: Optional[List[str]] = None) -> Optional[subprocess.CompletedProcess[bytes]]:
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
		


class worker:
	def __init__(
		self,
		exec_path: str = WR_PATH,
		temp_path: str = TMP_PATH,
		log_path: str = LOG_PATH,
		wr_name: str = WR_NAME,
		sesh_name: str = SESH_NAME,
		debug: bool = True,
		verbose: bool = True,
	):
		self.temp_path = Path(temp_path)
		self.exec_path = exec_path
		self.log_path = log_path
		self.wr_name = wr_name
		self.sesh_name = sesh_name
		self.debug = debug
		self.verbose = verbose

	def print_v(self, message: str) -> None:
		if self.verbose:
			print(message)

	def work(self):
		gpu_manager.wait_for_completion()

		info = retrieve_Data(self.temp_path)

		if not info:
			# empty temp file
			raise ValueError(
				f"Tried to read next run from {self.temp_path} but it was empty"
			)
			
		wandb_info = info['wandb']
		entity = wandb_info['entity']
		project = wandb_info['project']
		tags = wandb_info['tags']
		
		admin = info["admin"]

		# setup wandb run
		run_name = f"{admin['model']}_{admin['split']}_exp{admin['exp_no']}"

		if admin["recover"]:
			if "run_id" in info:
				run_id = info["run_id"]
			else:
				run_id = wandb_manager.get_run_id(
					run_name, entity, project, idx=-1
				)  # probably want the last one

			self.print_v(f"Resuming run with ID: {run_id}")

			run = wandb.init(
				entity=entity,
				project=project,
				id=run_id,
				resume="must",
				name=run_name,
				tags=tags,
				config=info,
			)
		else:
			self.print_v(f"Starting new run with name: {run_name}")
			run = wandb.init(
				entity=entity, project=project, name=run_name, tags=tags, config=info
			)

		#save run_id for recovering
		wandb_info["run_id"] = run.id
		info["wandb"] = wandb_info
		
		self.print_v("writing my id to temp file")
		store_Data(self.temp_path, info)
		
		self.print_v(f"Run ID: {run.id}")
		self.print_v(f"Run name: {run.name}")  # Human-readable name
		self.print_v(f"Run path: {run.path}")  # entity/project/run_id format

		train_loop(admin["model"], run, recover=admin["recover"])
		run.finish()

		

	def idle(
		self,
		message: str,
		wait: Optional[int] = None,
		cycles: Optional[int] = None,
	) -> str:
		gpu_manager.wait_for_completion(
			check_interval=5,  # seconds,
			confirm_interval=1,  # second
			num_checks=10,  # confirm consistency over 10 seconds
			verbose=self.verbose,
		)
		t = wait if wait else 1
		c = cycles if cycles else 10
		print("\n" * 2)
		print("-" * 10)
		print(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
		for i in range(c):
			print(f"Idling: {i}")
			print(message)
			time.sleep(t)
		print(f"Finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
		return message

	def idle_log(
		self, message: str, wait: Optional[int] = None, cycles: Optional[int] = None
	):
		gpu_manager.wait_for_completion(
			check_interval=5,  # seconds,
			confirm_interval=1,  # second
			num_checks=10,  # confirm consistency over 10 seconds
			verbose=self.verbose,
		)
		t = wait if wait else 1
		c = cycles if cycles else 10
		with open(self.log_path, "a") as f:
			f.write("\n" * 2)
			f.write("-" * 10)
			f.write("\n")
			f.write(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
			print(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
			f.flush()

			for i in range(c):
				f.write(f"Idling: {i}\n")
				print(f"Idling: {i}")
				f.write(message + "\n")
				time.sleep(t)
				f.flush()

			f.write(f"Finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
			print(f"Finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
			f.flush()

	def sim_gpu_usage(self):
		print("about to work")
		x = torch.rand(10000, 10000, device="cuda")
		y = torch.rand(10000, 10000, device="cuda")
		z = torch.rand(10000, 10000, device="cuda")
		for i in range(100):
			z = x @ y
			time.sleep(2)
			print(f"Idling: {i}")
		print("bunch of work done")
		return z.cpu()


class daemon:
	"""Class for the queue daemon process. The function works in a fetch execute repeat
	cycle. The function reads runs from the queRuns.json file, then writes them to the
	queTemp.json file for the worker process to find. The daemon spawns the worker, then
	waits for it to complete before proceeding. The daemon runs in it's own tmux session,
	while the worker outputs to a log file.

	Args:
					name: of own tmux window
					wr_name: of worker tmux window (monitor)
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
		wr_name: str = WR_NAME,
		sesh: str = SESH_NAME,
		runs_path: str = RUN_PATH,
		temp_path: str = TMP_PATH,
		exec_path: str = WR_PATH,
		verbose: bool = True,
		wr: Optional[worker] = None,
		q: Optional[que] = None,
		tm: Optional[tmux_manager] = None,
		stp_on_fail: bool = True,
	) -> None:
		self.name = name
		self.wr_name = wr_name
		self.sesh = sesh
		self.runs_path = Path(runs_path)
		self.temp_path = Path(temp_path)
		if wr:
			self.worker = wr
		else:
			self.worker = worker(
				exec_path=exec_path,
				temp_path=temp_path,
				wr_name=wr_name,
				sesh_name=sesh,
			)
		if q:
			self.que = q
		else:
			self.que = que(runs_path=runs_path, verbose=verbose)
		if tm:
			self.tmux_man = tm
		else:
			self.tmux_man = tmux_manager(wr_name=wr_name, dn_name=name, sesh_name=sesh, exec_path=exec_path)
		self.verbose = verbose
		self.stp_on_fail = stp_on_fail

	def print_v(self, message: str) -> None:
		"""Prints a message if verbose is True."""
		if self.verbose:
			print(message)

	def seperator(self, run: Dict) -> str:
		sep = ""
		r_str = self.que.run_str(self.que.run_sum(run))

		if r_str:
			sep += ("\n" * 2) + ("-" * 10) + ("\n")
			sep += f"{r_str:^10}"
			sep += ("\n" * 2) + ("-" * 10) + ("\n")
		else:
			sep += "\n"
		return sep.title()

	def start_n_watch(self):
		"""Start process in this terminal and watch"""
		while True:
			next_run = self.que.get_next_run()

			if next_run is None:
				self.print_v("No more runs to execute")
				break

			self.print_v(self.seperator(next_run))

			store_Data(self.temp_path, next_run)

			# worker
			self.worker_here()

			# retrieve run_id
			fin_run = retrieve_Data(self.temp_path)

			self.que.store_old_run(fin_run)

	def start_n_monitor(self):
		"""Start process and use existing tmux monitoring"""
		while True:
			next_run = self.que.get_next_run()

			if next_run is None:
				self.print_v("No more runs to execute")
				break

			self.print_v(self.seperator(next_run))

			store_Data(self.temp_path, next_run)

			# Start process in background
			proc = self.worker_log()

			# Start monitoring in tmux (non-blocking)
			self.monitor_log()

			# Wait for completion
			return_code = proc.wait()

			if return_code == 0:
				self.print_v("Process completed successfully")
			else:
				self.print_v(f"Process failed with return code: {return_code}")
				if self.stp_on_fail:
					break
			# retrieve run_id
			fin_run = retrieve_Data(self.temp_path)

			self.que.store_old_run(fin_run)

	def start_idle(self):
		"""Start process in this terminal and watch"""
		try:
			while True:
				self.print_v("Starting new idle process\n")
				self.worker_idle_here()
		except KeyboardInterrupt:
			self.print_v("Finished idling, bye")

	def start_idle_log(self):
		"""Start process and use existing tmux monitoring"""
		try:
			while True:
				self.print_v("Starting new idle process\n")
				proc = self.worker_idle_log()
				self.monitor_log()
				return_code = proc.wait()
				if return_code == 0:
					self.print_v("Process completed successfully")
				else:
					self.print_v(f"Process failed with return code: {return_code}")
					if self.stp_on_fail:
						break

		except KeyboardInterrupt:
			self.print_v("Finished idling, bye")

	def monitor_log(self):
		self.tmux_man._send(f"tail -f {self.worker.log_path}", self.wr_name)

	def worker_here(self, args: Optional[list[str]] = None) -> None:
		"""Blocking start which prints worker output in daemon terminal"""

		cmd = [self.worker.exec_path, self.wr_name, "work"]
		if args:
			cmd.extend(args)
		subprocess.run(cmd, check=True)

	def worker_log(self, args: Optional[list[str]] = None) -> subprocess.Popen:
		"""Non-blocking start which prints worker output to LOG_PATH, and passes the process"""

		cmd = [self.worker.exec_path, self.wr_name, "work"]
		if args:
			cmd.extend(args)

		return subprocess.Popen(
			cmd, stdout=open(self.worker.log_path, "w"), stderr=subprocess.STDOUT
		)

	def worker_idle_here(self, args: Optional[list[str]] = None):
		"""Blocking start which prints worker output in daemon terminal"""

		cmd = [self.worker.exec_path, self.wr_name, "idle"]
		if args:
			cmd.extend(args)
		subprocess.run(cmd, check=True)

	def worker_idle_log(self, args: Optional[list[str]] = None):
		"""Non-blocking start which prints worker output to LOG_PATH, and passes the process"""
		cmd = [self.worker.exec_path, self.wr_name, "idle"]
		if args:
			cmd.extend(args)
		return subprocess.Popen(
			cmd, stdout=open(self.worker.log_path, "w"), stderr=subprocess.STDOUT
		)

	def recover(self, o_setting: str, run_id: Optional[str] = None):
		"""Recover from a run failure, by loading the last run from Temp"""
		av_set = ['sWatch', 'sMonitor']
		
		if o_setting not in av_set:
			raise ValueError(f"Setting: {o_setting} is not one of available settings: {', '.join(av_set)}")	
  
		info = retrieve_Data(self.temp_path)
		if run_id:
			info['wandb']['run_id'] = run_id

		info['admin']['recover'] = True

		if not info:
			raise ValueError(f"Tried to read info from {self.temp_path} but found nothing")
			
		self.que.to_run.insert(0, info)		
		self.que.save_state()
		self.print_v(f"Recovering in mode: {o_setting}\n")
    
		if o_setting == 'sWatch':
			self.start_n_watch()
		elif o_setting == 'sMonitor':
			self.start_n_monitor()
		
			

		

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
		run_path: str = RUN_PATH,
		verbose: bool = True,
		auto_save: bool = True,
	) -> None:
		super().__init__()
		self.que = que(run_path, verbose)
		self.tmux_man = tmux_manager(wr_name=wr_name, dn_name=dn_name, sesh_name=sesh_name, exec_path=exec_path)
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
			self.que.print_v("Changes saved")
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
		self.que.save_state()

	def do_load(self, arg):  # happens automatically anyway
		"""Load state of que from queRuns.json"""
		self.que.load_state()

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

		self.que.list_runs(parsed_args.location, disp=True)

	def do_remove(self, arg):
		"""Remove a run from the past or future"""
		parsed_args = self._parse_args_or_cancel("remove", arg)
		if parsed_args is None:
			return

		self.que.remove_run(parsed_args.location, parsed_args.index)

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
			parsed_args.o_location, parsed_args.n_location, parsed_args.oi_index, parsed_args.of_index
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
			arg_dict, tags, project, entity = maybe_args
		else:
			print("Create cancelled (by user)")
			return

		self.que.create_run(arg_dict, tags, project, entity)

	# process based functions

	#tmux
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
		if parsed_args.setting == 'recover':
			add_args.append(f' -os {parsed_args.o_setting}')
			if parsed_args.run_id:
				add_args.append(f' -ri {parsed_args.run_id}')
  		
		ext_args = None if len(add_args) == 0 else add_args
    
		self.tmux_man.start(self.dn_name, parsed_args.setting, ext_args=ext_args)
  
	def do_worker(self, arg):
		"""Start the worker with the given setting"""
		parsed_args = self._parse_args_or_cancel("worker", arg)
		if parsed_args is None:
			return 
		
		self.tmux_man.start(self.wr_name, parsed_args.setting)

	# helper functions

	def _apply_synonyms(self, parsed_args):
		"""Apply synonyms to location arguments"""
		
		if hasattr(parsed_args, 'o_location'):
			parsed_args.o_location = SYNONYMS.get(
				parsed_args.o_location.lower(), 
				parsed_args.o_location
			)
		
		if hasattr(parsed_args, 'n_location'):
			parsed_args.n_location = SYNONYMS.get(
				parsed_args.n_location.lower(), 
				parsed_args.n_location
			)
		
		if hasattr(parsed_args, 'location'):
			parsed_args.location = SYNONYMS.get(
				parsed_args.location.lower(), 
				parsed_args.location
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
			"create": lambda: configs.take_args(
				return_parser_only=True,
				prog="create",
				desc="Create a new training run",
			),
			"remove": self._get_remove_parser,
			"clear": self._get_clear_parser,
			"list": self._get_list_parser,
			"quit": self._get_quit_parser,
			"shuffle": self._get_shuffle_parser,
			"move": self._get_move_parser,
			"attach": self._get_attach_parser,
			"daemon": self._get_daemon_parser,
			"worker": self._get_worker_parser
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
			'setting',
			choices=['sWatch', 'sMonitor','monitorO', 'idle', 'idle_log'],
			help='Operation of daemon:  worker here, worker in seperate window, tail log file, worker idle here, worker idle and log'
		)
		parser.add_argument(
			'-re',
			'--recover',
			action='store_true',
			help='Recover from run failure'
		)
		parser.add_argument(
			'-ri',
			'--run_id',
			type=str,
			help='The run id, if needed. Otherwise keeps the run id written to Temp',
			default=None
		)
		return parser
	
	def _get_worker_parser(self) -> argparse.ArgumentParser:
		"""Get parser for worker command"""
		parser = argparse.ArgumentParser(
			description="Start the que worker with a given setting", prog="worker"
		)
		parser.add_argument(
			'setting',
			choices=['work', 'idle', 'idle_log', 'idle_gpu'],
			help='Operation of worker: do its main job, idle here, idle in log, idle on GPU'
		)
		return parser
	
	def _get_attach_parser(self) -> argparse.ArgumentParser:
		"""Get parser for attach command"""
		parser = argparse.ArgumentParser(
			description="Attach to the daemon or worker tmux session"
		)
		parser.add_argument(
			'window',
			choices=['worker', 'daemon'],
			help='Tmux window to attach to'
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
			type=int, help="Final original index if specifying a range",
			required=False, default=None
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
			description="Remove a run from future or past runs", prog="remove"
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


if __name__ == "__main__":
	# try:
	# 	_ = join_session(WR_NAME, SESH_NAME)
	# except subprocess.CalledProcessError as e:
	# 	print("Daemon ran into an error when spawning the worker process: ")
	# 	print(e.stderr)
	# place lock on queRuns.json

	lock = FileLock(f"{RUN_PATH}.lock", timeout=1)
	try:
		with lock:
			# print(f"Acquired lock on {RUN_PATH}")
			queShell().cmdloop()
	except TimeoutError:
		print(f"Error: {RUN_PATH} is currently in used by another user")
		sys.exit(1)
	
		# print(f"Released lock on {RUN_PATH}")
