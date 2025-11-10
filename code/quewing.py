#!/home/luke/miniconda3/envs/wlasl/bin/python
from pathlib import Path
import argparse
import json
from typing import (
	Optional,
	List,
	Literal,
	TypeAlias,
	Tuple,
	Dict,
	Any,
	TypedDict,
	cast,
)

import subprocess
import time
import cmd as cmdLib
import shlex
from filelock import FileLock, Timeout
import sys
import wandb
import torch

# locals
import configs
from configs import ExpInfo, WandbInfo, AdminInfo, RunInfo, _exp_to_run_info
import utils
from utils import gpu_manager
from training import train_loop, _setup_wandb
from testing import test_run

# constants
SESH_NAME = "que_training"
DN_NAME = "daemon"
WR_NAME = "worker"
MR_NAME = "monitor"
QUE_DIR = "./que/"
WR_PATH = "./quefeather.py"
RUN_PATH = QUE_DIR + "Runs.json"
LOG_PATH = QUE_DIR + "Logs.log"
TO_RUN = "to_run"  # havent run yet
CUR_RUN = "cur_run"  # busy running
OLD_RUNS = "old_runs"  # run already
FAIL_RUNS = "fail_runs"  # runs that crashed
# List for argparse choices
QUE_LOCATIONS = [TO_RUN, CUR_RUN, OLD_RUNS]
SYNONYMS = {
	"new": "to_run",
	"tr": "to_run",
	"cur": "cur_run",
	"busy": "cur_run",
	"old": "old_runs",
	"or": "old_runs",
	"fail": "fail_runs",
	"fr": "fail_runs",
}
QueLocation: TypeAlias = Literal["to_run", "cur_run", "old_runs", "fail_runs"]


class FailedExp(ExpInfo):
	error: str


class AllRuns(TypedDict):
	old_runs: List[ExpInfo]
	cur_run: List[ExpInfo]
	to_run: List[ExpInfo]
	fail_runs: List[FailedExp]


def retrieve_Data(path: Path) -> Any:
	"""Retrieves data from a given path."""
	with open(path, "r") as file:
		data = json.load(file)
	return data


def store_Data(path: Path, data: Any):
	"""Stores data to a given path."""
	with open(path, "w") as file:
		json.dump(data, file, indent=4)


class QueEmpty(Exception):
	"""No runs available"""

	def __init__(self, message: str = "No runs available"):
		super().__init__(message)


class QueIdxOOR(IndexError):
	"""Index is out of range for a given location"""

	def __init__(self, loc: QueLocation, idx: int, leng: int):
		"""Create an index error for the specified location

		Args:
						loc (QueLocation): location key
						idx (int): out of bounds index
						leng (int): length of location
		"""
		super().__init__(
			f"Index {idx} is out of range for the que location: {loc} with length: {leng}"
		)


class QueBusy(Exception):
	"""There is already a run (likely for cur_run)"""

	def __init__(self, message: str = "There is already a run in cur_run"):
		super().__init__(message)


class que:
	def __init__(
		self, runs_path: str | Path, verbose: bool = True, auto_save: bool = False
	) -> None:
		self.runs_path: Path = Path(runs_path)
		self.lock_file: Path = Path(f"{runs_path}.lock")
		self.lock: FileLock = FileLock(self.lock_file, timeout=30)
		self.imp_splits: List[str] = configs.get_avail_splits()
		self.verbose: bool = verbose
		self.old_runs: List[ExpInfo] = []
		self.cur_run: List[ExpInfo] = []
		self.to_run: List[ExpInfo] = []
		self.fail_runs: List[FailedExp] = []
		self.auto_save: bool = auto_save
		self.load_state()

	def print_v(self, message: str) -> None:
		"""Prints a message if verbose is True."""
		if self.verbose:
			print(message)

	def fetch_state(self, loc: QueLocation) -> List[ExpInfo]:
		"""Return reference to the specified list"""
		if loc == TO_RUN:
			return self.to_run
		elif loc == CUR_RUN:
			return self.cur_run
		else:
			return self.old_runs

	def _load_Que(self):
		"""Read que from file"""
		try:
			with open(self.runs_path, "r") as f:
				data = json.load(f)
			self.to_run = data.get(TO_RUN, [])
			self.cur_run = data.get(CUR_RUN, [])
			self.old_runs = data.get(OLD_RUNS, [])
			self.fail_runs = data.get(FAIL_RUNS, [])
		except FileNotFoundError:
			self.print_v(
				f"No existing state found at {self.runs_path}. Starting fresh."
			)
			self.to_run = []
			self.cur_run = []
			self.old_runs = []
			self.fail_runs = []

	def _save_Que(self):
		"""Write que to file"""
		with open(self.runs_path, "w") as f:
			all_runs = {
				TO_RUN: self.to_run,
				CUR_RUN: self.cur_run,
				OLD_RUNS: self.old_runs,
				FAIL_RUNS: self.fail_runs,
			}
			json.dump(all_runs, f, indent=4)

	def _get_run(self, loc: QueLocation, idx: int) -> ExpInfo:
		"""Get the run at the given location with the provided index

		Args:
						loc (QueLocation): to_run, cur_run or old_runs
						idx (int): Index of the run

		Raises:
						QueEmpty: len(loc) == 0
						QueIdxOOR: abs(idx) >= len(loc)

		Returns:
						ExpInfo: The specified run
		"""
		to_get = self.fetch_state(loc)
		if len(to_get) == 0:
			raise QueEmpty()
		elif abs(idx) >= len(to_get):
			raise QueIdxOOR(loc, idx, len(to_get))
		return to_get.pop(idx)

	def _set_run(self, loc: QueLocation, idx: int, run: ExpInfo) -> None:
		"""Set a run at a specified location and index

		Args:
						loc (QueLocation): to_run, cur_run or old_runs
						idx (int): New index, must be within [-len(loc), len(loc)]
						run (ExpInfo): Experiment info to add to loc


		Raises:
						QueIdxOOR: The provied index is out of range: [-len(loc), len(loc)]
		"""
		to_set = self.fetch_state(loc)
		if len(to_set) < abs(idx):
			raise QueIdxOOR(loc, idx, len(to_set))
		to_set.insert(idx, run)

	def save_state(self):
		"""Saves state to Runs.jso, with filelock"""
		with self.lock:
			self._save_Que()

	def load_state(self) -> None:
		"""Loads state from Runs.json, with filelock"""
		with self.lock:
			self._load_Que()

	def stash_next_run(self) -> None:
		"""Moves next run from to_run to cur_run. Saves state with lock over both read and write

		Raises:
																																		QueEmpty: If to_run is empty
																																		QueBusy: If cur_run is full
																																		Timeout: If cannot acquire file lock
		"""
		self._load_Que()

		# empty to run
		if len(self.to_run) == 0:
			raise QueEmpty(f"Can't get next run, no runs in {TO_RUN}")

		# full cur_run
		if (
			len(self.cur_run) != 0
		):  # NOTE at some point it might be possible to have multiple busy operations
			raise QueBusy(f"Can't stash next run, there is something in {CUR_RUN}")

		next_run = self.to_run.pop(0)
		self.cur_run.append(next_run)
		self.print_v(f"Stashed next run: {self.run_str(self.run_sum(next_run))}")
		self._save_Que()

	def store_fin_run(self):
		"""Moves finished run from cur_run to old_runs. Saves state with lock over both read and write

		Raises:
																																		QueEmpty: If cur_run is empty
																																		Timeout: If cannot acquire file lock
		"""
		# empty cur_run
		if len(self.cur_run) == 0:
			raise QueEmpty(f"Can't move run in {CUR_RUN} because it's empty")

		fin_run = self.cur_run.pop(0)
		self.old_runs.insert(0, fin_run)
		self.print_v(f"Stored finished run: {self.run_str(self.run_sum(fin_run))}")
		self._save_Que()

	def get_cur_run(self) -> ExpInfo:
		"""Get the run stored in cur_run (assumes 1)"""
		return self._get_run("cur_run", 0)

	def set_cur_run(self, run: ExpInfo):
		"""Set the run in cur_run (assumes 1)"""
		self._set_run("cur_run", 0, run)

	def stash_failed_run(self, error: str) -> None:
		"""Move a run to the failed que"""
		run = self._get_run("cur_run", 0)
		failed = cast(FailedExp, run | {"error": error})
		self.fail_runs.append(failed)

	# summarisation

	@classmethod
	def get_config(cls, next_run: ExpInfo) -> str:
		admin = next_run["admin"]
		return admin["config_path"]

	def run_sum(self, run: RunInfo, exc: Optional[List[str]] = None) -> Dict[str, str]:
		"""Extract key details from a run configuration.

		Args:
																																		run: Dictionary containing run configuration with admin details
																																		exc: Optional list of keys to exclude from the summary

		Returns:
																																		Dictionary with model, exp_no, split, and config_path
		"""
		admin = run["admin"]

		dic = {}

		if "wandb" in run:
			run_id = run["wandb"]["run_id"]
			if run_id is None:
				dic["run_id"] = "None"
			else:
				dic["run_id"] = run_id

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
		self, run_confs: List[ExpInfo], head_sum: Optional[Dict[str, str]] = None 
	) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
		"""Get summarised run info, and stats for printing

		Args:
																																																																																																																																		run_confs (List[Dict]): A list of run configs (to_run or old_runs)

		Returns:
																																																																																																																																		Tuple[List[Dict], Dict]: List of summary dictionaries, dictionary of max lengths
		"""
		runs_info = []
		if head_sum is not None:
			runs_info.append(head_sum)
  
		runs_sum = [self.run_sum(run) for run in run_confs]

		runs_info.extend(runs_sum)

		# Calculate column widths
		max_model = max([len(r["model"]) for r in runs_info] + [len("Model")])
		max_exp = max([len(str(r["exp_no"])) for r in runs_info] + [len("Exp")])
		max_split = max([len(r["split"]) for r in runs_info] + [len("Split")])
		max_id = max([len(r["run_id"]) for r in runs_info] + [len("Run ID")])
		stats = {
			"max_model": max_model,
			"max_exp": max_exp,
			"max_split": max_split,
			"max_id": max_id,
		}

		return runs_info, stats

	def run_str0(
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

		if "run_id" in r_info and r_info["run_id"] is not None:
			r_str += f"Run ID: {r_info['run_id']:<{stats['max_id']}}  "

		r_str += (
			f"{r_info['model']:<{stats['max_model']}}  "
			f"Split: {r_info['split']:<{stats['max_split']}}  "
			f"Exp: {r_info['exp_no']:<{stats['max_exp']}}  "
			f"Config: {r_info['config_path']}"
		)

		return r_str

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

		if "run_id" in r_info and r_info["run_id"] is not None:
			r_str += f"{r_info['run_id']:<{stats['max_id']}}  "

		r_str += (
			f"{r_info['model']:<{stats['max_model']}}  "
			f"{r_info['split']:<{stats['max_split']}}  "
			f"{r_info['exp_no']:<{stats['max_exp']}}  "
			f"{r_info['config_path']}"
		)

		return r_str

	# for queShell interface

	def recover_run(self) -> None:
		"""Set the run in cur_run to recover"""
		try:
			run = self._get_run("cur_run", 0)
			run["admin"]["recover"] = True
			self._set_run("cur_run", 0, run)
		except Exception as e:
			self.print_v(str(e))

	def clear_runs(self, loc: QueLocation) -> None:
		"""reset the runs queue"""
		to_clear = self.fetch_state(loc)

		if len(to_clear) > 0:
			to_clear = []
			self.print_v(f"{loc} successfully cleared")
		else:
			self.print_v(f"{loc} already empty")

	def list_runso(self, loc: QueLocation) -> List[str]:
		"""Summarise to a list of runs, in a given location

		Args:
						loc (QueLocation): Location to list
						disp (bool, optional): Print list, with indexes. Defaults to False.

		Returns:
						List[str]: Summarised run info
		"""

		to_disp = self.fetch_state(loc)

		if len(to_disp) == 0:
			self.print_v(" No runs available\n")
			return []

		# Extract run info
		runs_info, stats = self.get_runs_info(to_disp)

		conf_list = []
		for i, info in enumerate(runs_info):
			# Format with padding for alignment
			r_str = f"  [{i:2d}] {self.run_str(info, stats)}"
			conf_list.append(r_str)

		return conf_list

	def list_runs(self, loc: QueLocation) -> List[str]:
		to_disp = self.fetch_state(loc)

		if len(to_disp) == 0:
			self.print_v(" No runs available\n")
			return []

		head_sum = {
			"model" : "Model",
			"split" : "Split", 
			"exp_no": "Exp",
			"run_id" : "Run ID",
			"config_path": "Config"
		}

		# Extract run info
		runs_info, stats = self.get_runs_info(to_disp, head_sum)
  
		conf_list = []
		head = f"  [{'Idx':>3}] {self.run_str(runs_info[0], stats)}"  
		conf_list.append(head)
		# conf_list.append("-" * len(head))
		for i, info in enumerate(runs_info[1:]):
			# Format with padding for alignment
			r_str = f"  [{i:3d}] {self.run_str(info, stats)}"
			conf_list.append(r_str)

		return conf_list

	def disp_runs(self, loc: QueLocation) -> None:
		# Nice header
		loc_display = loc.replace("_", " ").title()
		print(f"\n=== {loc_display} ===")
		print()
		runs = self.list_runs(loc)
		max_len = max(len(r) for r in runs)
		print(runs[0]) #head
		print("-" * max_len)
		for run in runs[1:]:
			print(run)
		print()

	def _is_dup_exp(self, new_run: RunInfo) -> bool:
		"""Check if new_run already exists in to_run or old_runs (ignores run_id and config_path)"""
		exc = ["run_id", "config_path"]
		new_sum = self.run_sum(new_run, exc)

		for run in self.to_run + self.old_runs:
			if self.run_sum(run, exc) == new_sum:
				return True
		return False

	def create_run(
		self,
		arg_dict: AdminInfo,
		wandb_dict: WandbInfo,
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
			self.print_v(f"{arg_dict['config_path']} not found. Create cancelled")
			return

		if self._is_dup_exp(config):
			self.print_v(
				f"Duplicate run detected: {self.run_str(self.run_sum(config))}"
			)
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
			config = cast(ExpInfo, config | {"wandb": wandb_dict})

			self.to_run.append(config)
			self.print_v(f"Added new run: {self.run_str(self.run_sum(config))}")
		else:
			self.print_v("Training cancelled by user")

	def remove_run(self, loc: QueLocation, idx: int) -> None:
		"""Removes a run from the given location safely

		Args:
						loc (QueLocation): to_run, cur_run or old_runs
						idx (int): Index of run
		"""
		try:
			_ = self._get_run(loc, idx)
		except Exception as e:
			self.print_v(str(e))

	def shuffle(self, loc: QueLocation, o_idx: int, n_idx: int) -> None:
		"""Repositions a run from the que
		Args:
						loc: TO_RUN, CUR_RUN or OLD_RUNS
						o_idx: original index of run
						n_idx: new index of run
		"""
		try:
			self._set_run(loc, n_idx, self._get_run(loc, o_idx))
		except Exception as e:
			self.print_v(str(e))

	def move(
		self,
		o_loc: QueLocation,
		n_loc: QueLocation,
		oi_idx: int,
		of_idx: Optional[int] = None,
	) -> None:
		"""Moves a run between locations in que (at beginning)

		Args:
						o_loc (QueLocation): Old location
						n_loc (QueLocation): New location
						oi_idx (int): Old initial index
						of_idx (int): Old final index, if specifying a range.
		"""
		if of_idx is None:
			try:
				self._set_run(n_loc, 0, self._get_run(o_loc, oi_idx))
				self.print_v("Move successful")
			except Exception as e:
				self.print_v(str(e))
		else:
			# Range move
			old_location = self.fetch_state(o_loc)
			new_location = self.fetch_state(n_loc)

			# Validate range
			if abs(oi_idx) >= len(old_location) or abs(of_idx) >= len(old_location):
				self.print_v(
					f"Range: {oi_idx} - {of_idx} is an invalid range. "
					f"Length of {o_loc} is: {len(old_location)}"
				)
				return

			# Extract the runs to move
			tomv = []
			for _ in range(oi_idx, of_idx + 1):
				tomv.append(old_location.pop(oi_idx))

			# Insert into new location (in reverse to maintain order when inserting at 0)
			for run in tomv:
				new_location.insert(0, run)

			self.print_v("multi-move successful")

	def edit_run(
		self,
		loc: QueLocation,
		idx: int,
		key1: str,
		value: Any,
		key2: Optional[str] = None,
	) -> None:
		try:
			run = self._get_run(loc, idx)
			if key2 is not None:
				run[key1][key2] = value
			else:
				run[key1] = value
			self._set_run(loc, idx, run)
			self.print_v("Edit successful")
		except Exception as e:
			self.print_v(str(e))


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


class worker:
	def __init__(
		self,
		exec_path: str = WR_PATH,
		runs_path: str = RUN_PATH,
		log_path: str = LOG_PATH,
		wr_name: str = WR_NAME,
		sesh_name: str = SESH_NAME,
		debug: bool = True,
		verbose: bool = True,
	):
		self.exec_path = exec_path
		self.que = que(runs_path=runs_path, verbose=verbose)
		self.log_path = log_path
		self.wr_name = wr_name
		self.sesh_name = sesh_name
		self.debug = debug
		self.verbose = verbose

	def print_v(self, message: str) -> None:
		if self.verbose:
			print(message)

	def work(self) -> None:
		try:
			gpu_manager.wait_for_completion()

			# get next run
			self.que.load_state()
			info = self.que.get_cur_run()

			wandb_info = info["wandb"]
			admin = info["admin"]
			config = _exp_to_run_info(info)

			# setup wandb run
			run = _setup_wandb(config, wandb_info)
			if run is None:
				return

			# save run_id for recovering
			wandb_info["run_id"] = run.id
			info["wandb"] = wandb_info

			self.print_v("writing my id to temp file")
			self.que.set_cur_run(info)
			self.que.save_state()

			self.print_v(f"Run ID: {run.id}")
			self.print_v(f"Run name: {run.name}")  # Human-readable name
			self.print_v(f"Run path: {run.path}")  # entity/project/run_id format

			train_loop(admin["model"], config, run, recover=admin["recover"])
			run.finish()
		except Exception as e:
			# TODO: handle ctrl+c gracefully
			self.print_v("Training run failed due to an error")
			self.que.stash_failed_run(str(e))
			raise e  # still need to crash so daemon can

	# def test(self) -> None:
	# 	gpu_manager.wait_for_completion()
	# 	self.que.load_state()
	# 	info = self.que.get_cur_run()
	# 	#main test
	# 	res_reg = test_run(
	#   		info, 
	#     	test_val=True, 
	#       	br_graph=True, 	
	#        	cf_matrix=True, 
	#         heatmap=True,
	#     )
	# 	res_shuff = test_run(
	# 		info, 
	# 		shuffle=True,
	# 		re_test=True
	# 	)

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
		exec_path: str = WR_PATH,
		verbose: bool = True,
		wr: Optional[worker] = None,
		q: Optional[que] = None,
		tm: Optional[tmux_manager] = None,
		stp_on_fail: bool = True,  # TODO: add this to parser
	) -> None:
		self.name = name
		self.wr_name = wr_name
		self.sesh = sesh
		self.runs_path = Path(runs_path)
		if wr:
			self.worker = wr
		else:
			self.worker = worker(
				exec_path=exec_path,
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
			self.tmux_man = tmux_manager(
				wr_name=wr_name, dn_name=name, sesh_name=sesh, exec_path=exec_path
			)
		self.verbose = verbose
		self.stp_on_fail = stp_on_fail

	def print_v(self, message: str) -> None:
		"""Prints a message if verbose is True."""
		if self.verbose:
			print(message)

	def seperator(self, run: ExpInfo) -> str:
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
			# prepare next run (move from to_run -> cur_run)
			try:
				self.que.stash_next_run()
			except QueEmpty:
				self.print_v("No more runs to execute")
				break
			except QueBusy:
				self.print_v("Cannot overwrite current run")
				break
			except Timeout:
				self.print_v(
					"Cannot stash next run, file is already held by another process"
				)
				break

			# print seperator
			run = self.que.get_cur_run()
			self.print_v(self.seperator(run))

			# start worker with process output here
			self.worker_here()

			# save finished run (move from cur_run -> old_runs)
			try:
				self.que.store_fin_run()
			except QueEmpty:
				self.print_v("Could not find current run")
			except Timeout:
				self.print_v(
					"Cannot store finished run, file is already held by another process"
				)

	def start_n_monitor(self):
		"""Start process and use existing tmux monitoring"""
		while True:
			# prepare next run (move from to_run -> cur_run)
			try:
				self.que.stash_next_run()
				self.que.save_state()
			except QueEmpty:
				self.print_v("No more runs to execute")
				break
			except QueBusy:
				self.print_v("Cannot overwrite current run")
				break
			except Timeout:
				self.print_v(
					"Cannot stash next run, file is already held by another process"
				)
				break

			run = self.que.get_cur_run()
			self.print_v(self.seperator(run))

			# Start process in background
			proc = self.worker_log()

			# Start monitoring in tmux (non-blocking)
			self.monitor_log()

			# Wait for completion
			return_code = proc.wait()

			if return_code == 0:
				self.print_v("Process completed successfully")
				# save finished run (move from cur_run -> old_runs)
				try:
					self.que.load_state()
					self.que.store_fin_run()
				except QueEmpty:
					self.print_v("Could not find current run")
				except Timeout:
					self.print_v(
						"Cannot store finished run, file is already held by another process"
					)
			else:
				self.print_v(f"Process failed with return code: {return_code}")

				if self.stp_on_fail:
					self.print_v("Stopping exectuion")
					break
				else:
					self.print_v("Continuing with next run")

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

	def recover(self, o_setting: str, run_id: Optional[str] = None):
		"""Recover from a run failure, by loading the last run from Temp"""
		av_set = ["sWatch", "sMonitor"]

		if o_setting not in av_set:
			raise ValueError(
				f"Setting: {o_setting} is not one of available settings: {', '.join(av_set)}"
			)

		self.que.recover_run()
		self.que.save_state()
		self.print_v(f"Recovering in mode: {o_setting}\n")

		if o_setting == "sWatch":
			self.start_n_watch()
		elif o_setting == "sMonitor":
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

		self.que.disp_runs(parsed_args.location)

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
			"create": configs.get_train_parser(
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
			"worker": self._get_worker_parser,
			"edit": self._get_edit_parser,
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
	# try:
	# 	_ = join_session(WR_NAME, SESH_NAME)
	# except subprocess.CalledProcessError as e:
	# 	print("Daemon ran into an error when spawning the worker process: ")
	# 	print(e.stderr)
	# place lock on queRuns.json

	# lock = FileLock(f"{RUN_PATH}.lock", timeout=1)
	# try:
	#     with lock:
	#         # print(f"Acquired lock on {RUN_PATH}")
	#         queShell().cmdloop()
	# except TimeoutError:
	#     print(f"Error: {RUN_PATH} is currently in used by another user")
	#     sys.exit(1)

	# print(f"Released lock on {RUN_PATH}")
	queShell().cmdloop()
