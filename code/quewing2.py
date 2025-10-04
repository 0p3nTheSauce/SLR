from pathlib import Path
import argparse
import json
from typing import Optional, List, Literal, TypeAlias
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
from train import train_loop
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
TO_RUN = "to_run"
OLD_RUNS = "old_runs"
# List for argparse choices
QUE_LOCATIONS = [TO_RUN, OLD_RUNS]
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
		all_runs = {OLD_RUNS: self.old_runs, TO_RUN: self.to_run}
		store_Data(self.runs_path, all_runs)
		self.print_v(f"Saved state to {self.runs_path}")

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
			self.print_v(f"Loaded state from {self.runs_path}")
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

	def get_next_run(self) -> Optional[dict]:
		"""Retrieves the next run from the queue, and moves the run to OLD_RUNS"""
		if self.to_run:
			next_run = self.to_run.pop(0)
			self.old_runs.append(next_run)
			self.print_v(f"Retrieved next run: {next_run}")
			return next_run
		else:
			self.print_v("No runs in the queue.")
			return

	def clear_runs(self, loc: QueLocation):
		"""reset the runs queue"""
		past = loc == OLD_RUNS
		future = loc == TO_RUN
		past, future = loc == "all", loc == "all"
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
		runs_info = [self._run_sum(run) for run in to_disp]

		# Calculate column widths
		max_model = max(len(r["model"]) for r in runs_info)
		max_exp = max(len(str(r["exp_no"])) for r in runs_info)
		max_split = max(len(r["split"]) for r in runs_info)

		conf_list = []
		for i, info in enumerate(runs_info):
			# Format with padding for alignment
			r_str = (
				f"{info['model']:<{max_model}}  "
				f"Exp: {info['exp_no']:<{max_exp}}  "
				f"Split: {info['split']:<{max_split}}  "
				f"Config: {info['config_path']}"
			)

			if disp:
				print(f"  [{i:2d}] {r_str}")
			conf_list.append(r_str)

		if disp:
			print()  # Add spacing after list

		return conf_list

	def _run_sum(self, run: dict) -> dict:
		"""Extract key details from a run configuration.

		Args:
										run: Dictionary containing run configuration with admin details

		Returns:
										Dictionary with model, exp_no, split, and config_path
		"""
		admin = run["config"]["admin"]
		return {
			"model": admin["model"],
			"exp_no": admin["exp_no"],
			"split": admin["split"],
			"config_path": admin["config_path"],
		}

	def create_run(
		self,
		arg_dict: dict,
		tags: list[str],
		output: str,
		save_path: str,
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
		o_idx: int,
	):
		"""Moves a run between locations in que (at beginning)

		Args:
						o_loc (QueLocation): Old location
						n_loc (QueLocation): New location
						o_idx (int): Old index
		"""

		old_location = self.fetch_state(o_loc)

		if abs(o_idx) < len(old_location):
			run = old_location.pop(o_idx)
		else:
			print(
				f"{o_idx} out of range for len({o_loc}) - 1 = {len(old_location) - 1}"
			)
			return

		new_location = self.fetch_state(n_loc)
		new_location.insert(0, run)

		self.print_v("Successfully added\n")

		self.list_runs(n_loc, disp=self.verbose)

	# High level functions taking multistep input


class tmux_manager:
	def __init__(
		self,
		wr_name: str,
		dn_name: str,
		sesh_name: str,
	) -> None:
		self.wr_name = wr_name
		self.dn_name = dn_name
		self.sesh_name = sesh_name

	def join_session(self):
		tmux_cmd = ["tmux", "attach-session", "-t", f"{self.sesh_name}:{self.wr_name}"]
		try:
			_ = subprocess.run(tmux_cmd, check=True)
		except subprocess.CalledProcessError as e:
			print("join_session ran into an error when spawning the worker process: ")
			print(e.stderr)

	def switch_to_window(self):
		tmux_cmd = ["tmux", "select-window", "-t", f"{self.sesh_name}:{self.wr_name}"]
		try:
			_ = subprocess.run(tmux_cmd, check=True)
		except subprocess.CalledProcessError as e:
			print(
				"switch_to_window ran into an error when spawning the worker process: "
			)
			print(e.stderr)

	def send(self, cmd: str):  # use with caution
		tmux_cmd = [
			"tmux",
			"send-keys",
			"-t",
			f"{self.sesh_name}:{self.wr_name}",
			cmd,
			"Enter",
		]
		try:
			subprocess.run(tmux_cmd, check=True)
		except subprocess.CalledProcessError as e:
			print("Send ran into an error when spawning the worker process: ")
			print(e.stderr)

class wandb_manager:
  
	@classmethod
	def get_run_id(cls, 
					run_name,
					entity: str,
				 	project: str,
      				idx: Optional[int] = None) -> Optional[str]:
		api = wandb.Api()

		runs = api.runs(f"{entity}/{project}")
		ids = []
		for run in runs:
			if run.name == run_name:
				ids.append(run.id)
    
		if len(ids) == 0:
			print(f"No runs found with name: {run_name}")
			return None
		elif len(ids) > 1:
			print(f"Multiple runs found with name: {run_name}")
			if isinstance(idx, int) and abs(idx) < len(runs):
				print(f"Returning id for idx: {idx}")
				return ids[idx]
			else:
				print("No idx supplied, returning None")
				return None
		else:
			return ids[0]

	@classmethod
	def list_runs(cls,
				entity:str, 
				project:str,
				disp:bool = False,
    			) -> list[str]:
		api = wandb.Api()
		runs = api.runs(f"{entity}/{project}")

		if disp:
			for run in runs:
				print(f"Run ID: {run.id}")
				print(f"Run name: {run.name}")
				print(f"State: {run.state}")
				print(f"Created: {run.created_at}")
				print("---")
		
		return runs

	@classmethod
	def run_present(cls, run_id:str, runs:List) -> bool:
		return any([run.id == run_id for run in runs])

	@classmethod	
	def validate_runId(cls, run_id:str, entity:str, project:str) -> bool:
		return cls.run_present(run_id, cls.list_runs(entity, project))

class gpu_manager:

	@classmethod
	def get_gpu_memory_usage(cls, gpu_id=0):
		"""Get GPU memory usage across all processes"""
  
		result = subprocess.run(
			['nvidia-smi', f'--id={gpu_id}', 
			'--query-gpu=memory.used,memory.total',
			'--format=csv,noheader,nounits'],
			capture_output=True,
			text=True
		)
		used, total = map(float, result.stdout.strip().split(','))
  
		return used / 1024, total / 1024  # In GB

	@classmethod
	def wait_for_completion(
		cls,
		check_interval: int = 3600,  # 1 hour
		confirm_interval: int = 60,  # 1 minute
		num_checks: int = 5,  # confirm consistency over 5 minutes
		verbose: bool = False,
		gpu_id: int = 0,
		max_util_gb: float = 1.0  # Maximum memory usage in GB
	) -> bool:

		assert torch.cuda.is_available(), 'CUDA is not available'
		
		used, total = cls.get_gpu_memory_usage(gpu_id)		
  
		proceed = used > max_util_gb  # Fixed logic
		
		while proceed:
			if verbose:
				print()
				print(f"Monitoring GPU: {gpu_id}, current memory usage: {used:.2f}/{total:.2f} GB ({used/total*100:.1f}%)")
				print(f"Last checked at {time.strftime('%Y-%m-%d %H:%M:%S')}")
			try:
				time.sleep(check_interval)
				used, _ = cls.get_gpu_memory_usage(gpu_id)		
			except KeyboardInterrupt:
				print()
				print("Monitoring interrupted by user")
				return False

			if used <= max_util_gb and cls._confirm_usage(confirm_interval, num_checks, gpu_id, max_util_gb):
				break

		if verbose:
			print()
			print(f"GPU: {gpu_id} is available, current memory usage: {used:.2f}/{total:.2f} GB ({used/total*100:.1f}%)")
		
		return True

	@classmethod
	def _confirm_usage(
		cls,
		confirm_interval: int = 60,  # 1 minute
		num_checks: int = 5,  # confirm consistency over 5 minutes
		gpu_id: int = 0,
		max_util_gb: float = 1.0
	) -> bool:
		for _ in range(num_checks):
			used, _ = cls.get_gpu_memory_usage(gpu_id)
			if used > max_util_gb:
				return False
			time.sleep(confirm_interval)
		return True
 
class worker:
	def __init__(
		self,
		exec_path: str = WR_PATH,
		temp_path: str = TMP_PATH,
		log_path: str = LOG_PATH,
		wr_name: str = WR_NAME,
		sesh_name: str = SESH_NAME,
		debug : bool = True,
		verbose : bool = True
	):
		self.temp_path = Path(temp_path)
		self.exec_path = exec_path
		self.log_path = log_path
		self.wr_name = wr_name
		self.sesh_name = sesh_name
		self.debug = debug
		self.verbose = verbose

	def print_v(self, message : str) -> None:
		if self.verbose:
			print(message)
  
	def work(self):
		info = retrieve_Data(self.temp_path)

		if not info or 'run_id' in info.keys():
			#empty temp file
			raise ValueError(f'Tried to read next run from {self.temp_path} but it was empty')

		model_specifcs = info['model_info']
		config = info['config']
		entity = info['entity']
		project = info['project']
		tags = info['tags']

		admin = config['admin']
  
		#setup wandb run
		run_name = f"{admin['model']}_{admin['split']}_exp{admin['exp_no']}"
  
		if admin['recover']:
			if 'run_id' in info:
				run_id = info['run_id']
			else:
				run_id = wandb_manager.get_run_id(run_name, entity, project, idx=-1) #probably want the last one
    
			self.print_v(f"Resuming run with ID: {run_id}")
			
			run = wandb.init(
				entity=entity,
				project=project,
				id=run_id,
				resume='must',
				name=run_name,
				tags=tags,
				config=config      
			)		
		else:
			self.print_v(f"Starting new run with name: {run_name}")
			self.print_v(f"Starting new run with name: {run_name}")
			run = wandb.init(
				entity=entity, project=project, name=run_name, tags=tags, config=config
			)
			# write run id to temp, so that daemon waits for it
			run_info = {"run_id": run.id, "run_name": run.name, "run_project": project}
			self.print_v("writing my id to temp file")
			store_Data(self.temp_path, run_info)
   
		self.print_v(f"Run ID: {run.id}")
		self.print_v(f"Run name: {run.name}")  # Human-readable name
		self.print_v(f"Run path: {run.path}")  # entity/project/run_id format
  
		train_loop(model_specifcs, run, recover=admin["recover"])
		run.finish()

	def idle(self, message: str) -> str:
		# testing if blocking
		print(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
		for i in range(10):
			print(f"Idling: {i}")
			print(message)
			time.sleep(10)
		print(f"Finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
		return message

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
		imp_path: str = IMP_PATH,
		exec_path: str = WR_PATH,
		verbose: bool = True,
		wr: Optional[worker] = None,
		q: Optional[que] = None,
		tm: Optional[tmux_manager] = None 
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
				exec_path=exec_path, temp_path=temp_path, wr_name=wr_name, sesh_name=sesh
			)
		if q:
			self.que = q
		else:
			self.que = que(
				runs_path=runs_path, implemented_path=imp_path, verbose=verbose
			)
		if tm:
			self.tmux_man = tm
		else:
			self.tmux_man = tmux_manager(
				wr_name=wr_name,
				dn_name=name,
				sesh_name=sesh
			)
		self.verbose = verbose

	def print_v(self, message: str) -> None:
		"""Prints a message if verbose is True."""
		if self.verbose:
			print(message)

	def seperator(self, run: dict) -> str:
		sep = ""
		r_info = self.que._run_sum(run)
		r_str = f"{r_info['model']} \
				Exp: {r_info['exp_no']} \
				Split: {r_info['split']} \
				Config: {r_info['config_path']}"

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
			self.que.load_state()
			next_run = self.que.get_next_run()
			self.que.save_state()

			if next_run is None:
				self.print_v("No more runs to execute")
				break

			self.print_v(self.seperator(next_run))

			self.start_here(next_run)

	def start_n_monitor(self):
		"""Start process and use existing tmux monitoring"""
		while True:
			self.que.load_state()
			next_run = self.que.get_next_run()
			self.que.save_state()

			if next_run is None:
				self.print_v("No more runs to execute")
				break

			self.print_v(self.seperator(next_run))

			# Start process in background
			proc = self.start_log(next_run)

			# Start monitoring in tmux (non-blocking)
			self.monitor_log()

			# Wait for completion
			return_code = proc.wait()

			if return_code == 0:
				self.print_v("Process completed successfully")
			else:
				self.print_v(f"Process failed with return code: {return_code}")

	def monitor_log(self):
		self.tmux_man.send(f"tail -f {self.worker.log_path}")

	def start_here(self, next_run: dict, args: Optional[list[str]] = None) -> None:
		"""Blocking start which prints worker output in daemon terminal"""

		store_Data(self.temp_path, next_run)
		cmd = [self.worker.exec_path, self.wr_name]
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
		"""Non-blocking start which prints worker output to LOG_PATH, and passes the process"""
		store_Data(self.temp_path, next_run)
		cmd = [self.worker.exec_path, self.wr_name]
		if args:
			cmd.extend(args)

		return subprocess.Popen(
			cmd, stdout=open(self.worker.log_path, "w"), stderr=subprocess.STDOUT
		)

class queShell(cmdLib.Cmd):
	intro = "queShell: Type help or ? to list commands.\n"
	prompt = "(que)$ "

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
		auto_save: bool = True,
	) -> None:
		super().__init__()
		self.que = que(run_path, imp_path, verbose)
		self.daemon = daemon(
			name=dn_name,
			wr_name=wr_name,
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
			parsed_args.o_location, parsed_args.n_location, parsed_args.o_index
		)

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

	# daemon based functions
 
	

	# helper functions

	def _parse_args_or_cancel(self, cmd: str, arg: str) -> Optional[argparse.Namespace]:
		"""Parse arguments or return None if parsing fails/is cancelled"""
		args = shlex.split(arg)
		parser = self._get_parser(cmd)
		# assert isinstance(parser, argparse.ArgumentParser), f"{cmd} cannot use this generic parser"
		if parser:
			try:
				return parser.parse_args(args)
			except (SystemExit, ValueError):
				print(f"{cmd} cancelled")
				return None
		else:
			print(f"{cmd} not found")

	def _get_parser(self, cmd: str) -> Optional[argparse.ArgumentParser]:
		"""Get argument parser for a given command"""
		parsers = {
			"create": lambda: configs.take_args(
				self.que.imp_splits,
				self.que.imp_models,
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
		}

		if cmd in parsers:
			parser = parsers[cmd]()
			# assert isinstance(parser, argparse.ArgumentParser), f"{cmd} parser invalid"
			return parser
		return None

	def _get_move_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(
			description="Moves a run between locations in que", prog="move"
		)
		parser.add_argument(
			"o_location", choices=QUE_LOCATIONS, help="Original location."
		)
		parser.add_argument("n_location", choices=QUE_LOCATIONS, help="New location.")
		parser.add_argument(
			"o_index", type=int, help="Old position of run in original location"
		)
		return parser

	def _get_shuffle_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(
			description="Repositions a run from the que", prog="shuffle"
		)
		parser.add_argument(
			"location", choices=QUE_LOCATIONS, help="Location of the run"
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
			"location", choices=QUE_LOCATIONS, help="Location of the run"
		)
		parser.add_argument("index", type=int, help="Position of run in location")
		return parser

	def _get_clear_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(
			description="Clear future or past runs", prog="clear"
		)
		parser.add_argument(
			"location",
			choices=[TO_RUN, OLD_RUNS, "all"],
			help="Location of the run",
		)
		return parser

	def _get_list_parser(self) -> argparse.ArgumentParser:
		parser = argparse.ArgumentParser(
			description="Summarise to a list of runs, in a given location", prog="list"
		)
		parser.add_argument(
			"location", choices=QUE_LOCATIONS, help="Location of the run"
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

	lock = FileLock(f"{RUN_PATH}.lock", timout=1)
	try:
		with lock:
			print(f"Acquired lock on {RUN_PATH}")
			queShell().cmdloop()
	except TimeoutError:
		print(f"Error: {RUN_PATH} is currently in used by another user")
		sys.exit(1)
	finally:
		print(f"Released lock on {RUN_PATH}")
