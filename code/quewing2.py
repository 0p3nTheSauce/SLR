from pathlib import Path
import json
from typing import Optional, Callable, Tuple, Any
import subprocess
import time

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
		self.implemented_info = retrieve_Data(implemented_path)
		assert self.implemented_info, "No implemented info found"
		self.verbose = verbose
		self.old_runs = []
		self.to_run = []
		self.load_state()

	@classmethod
	def get_config(cls, next_run: dict) -> str:
		admin = next_run["config"]["admin"]
		return admin['config_path']
		
	def print_v(self, message: str) -> None:
		"""Prints a message if verbose is True."""
		if self.verbose:
			print(message)

	def save_state(self):
		all_runs = {"old_runs": self.old_runs, "to_run": self.to_run}
		store_Data(self.runs_path, all_runs)
		self.print_v(f"Saved state to {self.runs_path}")

	def load_state(self, all_runs: Optional[dict] = None) -> dict:
		"""Loads state from Runs file, or dictionary, returns all_runs"""
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

	def _remove_a_run(self, location: str, idx: int):
		"Removes a run from the run queue"
		all_runs = self.load_state()
		try:
			rem = all_runs[location].pop(idx)
			self.load_state(all_runs)
			return rem
		except KeyError:
			print(f"Key {location} not available in all_runs")
			return {}
		except IndexError:
			print(
				f"Index: {idx} out of range for to run of length {len(all_runs[location])}"
			)
			return {}

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

	def create_run(self):
		"""Add a new entry to to_run"""
		available_splits = self.implemented_info["splits"]
		model_info = self.implemented_info["models"]

		maybe_args = configs.take_args(
			available_splits, model_info.keys()
		)
		if maybe_args:
			arg_dict, tags, output, save_path, project, entity = maybe_args
		else:
			self.print_v("Training cancelled by user")
			return
  

		config = configs.load_config(arg_dict, verbose=True)

		configs.print_config(config)

		model_specifics = model_info[config["admin"]["model"]]

		proceed = utils.ask_nicely(
			message="Confirm: y/n: ",
			requirment=lambda x: x.lower() in ["y", "n"],
			error="y or n: ",
		)
		if proceed.lower() == "y":
			self.print_v("Saving run info")

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

	def remove_run(self, loc: Optional[str] = None, idx: Optional[int] = None):
		"""remove a run"""
		all_runs = self.load_state()

		av_loc = ["to_run", "old_runs", "q"]
		if loc is None or loc not in av_loc:
			loc = utils.ask_nicely(
				message=f"Choose location: {av_loc}",
				requirment=lambda x: x in av_loc,
				error=f"one of {av_loc}: ",
			)
			if loc == "q":
				self.print_v("cancelled")
				return

		if idx is not None:
			self._remove_a_run(loc, idx)
		else:
			for i, run in enumerate(all_runs[loc]):
				print(f"{self.get_config(run)} : {i}")
				
			if len(all_runs["to_run"]) == 0:
				print("runs are finished")
			else:
				ans = utils.ask_nicely(
					message="select index, or q: ",
					requirment=lambda x: x.isdigit() or x == "q",
					error="select index, or q: ",
				)
				if ans == "q":
					self.print_v("cancelled")
					return
				else:
					self._remove_a_run(loc, int(ans))

	def clear_runs(self, past: bool = False, future: bool = False):
		"""reset the runs queue"""
		if past:
			self.old_runs = []
		if future:
			self.to_run = []
		self.print_v("Successfully cleared")

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

	def list_configs(self):
		conf_list = []
		if len(self.to_run) == 0:
			print("runs are finished")
		for run in self.to_run:
			print(self.get_config(run))
			conf_list.append(self.get_config(run))
		return conf_list

	
		

def join_session(
	wndw_name: str,
	sesh_name: str,
) -> subprocess.CompletedProcess[bytes]:
	"""Join a tmux session and optionally target a specific window."""
	tmux_cmd = ["tmux", "attach-session", "-t", f"{sesh_name}:{wndw_name}"]
	return subprocess.run(tmux_cmd, check=True)


def switch_to_window(
	wndw_name: str, sesh_name: str
) -> subprocess.CompletedProcess[bytes]:
	"""Switch to specific window by index."""
	tmux_cmd = ["tmux", "select-window", "-t", f"{sesh_name}:{wndw_name}"]
	return subprocess.run(tmux_cmd, check=True)


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

def seperator(title : str="Next Run"):
	"""This prints out a seperator between training runs"""
	sep = ""
	if title:
		sep += ("\n" * 2) + ( "-" * 10) + ("\n")
		sep += f"{title:^10}"
		sep += ("\n" * 2) + ( "-" * 10) + ("\n")
	else:
		sep += '\n'
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

	def start_here(self, next_run: dict, args: Optional[list[str]]=None) -> None:
		'''Blocking start which prints worker output in daemon terminal'''
    
		store_Data(self.temp_path, next_run)
		cmd = [self.exec_path, self.wr_name]
		if args:
			cmd.extend(args)
		with subprocess.Popen(cmd, 
					  stdout=subprocess.PIPE, 
					  stderr=subprocess.PIPE, 
					  text=True) as proc:
			if proc.stdout:
				for line in proc.stdout:
					print(line.strip())

	def start_log(self, next_run: dict,args: Optional[list[str]]=None) -> subprocess.Popen:
		'''Non-blocking start which prints worker output to temp.json, and passes the process'''
		store_Data(self.temp_path, next_run)
		cmd = [self.exec_path, self.wr_name]
		if args:
			cmd.extend(args)
    
		return subprocess.Popen(cmd, 
                           stdout=open(self.log_path, 'w'),
                           stderr=subprocess.STDOUT)
   
	def monitor_log(self):
		send(f"tail -f {self.log_path}", self.wr_name, self.sesh_name)
   
   
class daemon:
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
	):
		self.name = name
		self.w_name = w_name
		self.sesh = sesh
		self.runs_path = Path(runs_path)
		self.temp_path = Path(temp_path)
		self.que = que(runs_path=runs_path, implemented_path=imp_path, verbose=verbose)
		self.worker = worker(
			exec_path=exec_path,
			temp_path=temp_path,
			wr_name=w_name,
			sesh_name=sesh
		)
		self.verbose = verbose


	def print_v(self, message: str) -> None:
		"""Prints a message if verbose is True."""
		if self.verbose:
			print(message)

 
	def start_n_watch(self):
		'''Start process in this terminal and watch'''
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
		'''Start process and use existing tmux monitoring'''
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
		

if __name__ == "__main__":
	# try:
	# 	_ = join_session(WR_NAME, SESH_NAME)
	# except subprocess.CalledProcessError as e:
	# 	print("Daemon ran into an error when spawning the worker process: ")
	# 	print(e.stderr)
	pass
# class daemon:
#     def __init__(self, verbose:bool=True, ) -> None:
#         pass
