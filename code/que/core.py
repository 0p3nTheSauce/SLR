#!/usr/bin/env python
from typing import (
	Optional,
	Callable,
	List,
	Literal,
	TypeAlias,
	Tuple,
	Dict,
	Any,
	TypedDict,
	cast,
	Union,
	TypeGuard,
)
from pathlib import Path
import json
from logging import Logger
#locals
from run_types import (
	ExpInfo, CompExpInfo,
	AdminInfo, WandbInfo, RunInfo, CompRes, Sumarised
)
from testing import full_test, load_comp_res
from configs import print_config, load_config

# constants
# MR_NAME = "monitor"
QUE_DIR = Path(__file__).parent

RUN_PATH = QUE_DIR / "Runs.json"
WR_LOG_PATH = QUE_DIR / "Worker.log"
SR_LOG_PATH = QUE_DIR / "Server.log"
WR_PATH = QUE_DIR / "worker.py"

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


GenExp: TypeAlias = Union[ExpInfo, FailedExp, CompExpInfo]
ExpQue: TypeAlias = Union[List[ExpInfo], List[FailedExp], List[CompExpInfo]]


class AllRuns(TypedDict):
	old_runs: List[CompExpInfo]
	cur_run: List[ExpInfo]
	to_run: List[ExpInfo]
	fail_runs: List[FailedExp]

# class ConfigLoader(Protocol):
#     def load_config(self, admin: AdminInfo) -> RunInfo: ...
#     def print_config(self, config: Dict) -> None: ...
#     def get_avail_splits(self) -> List[str]: ...
	
class QueException(Exception):
    """Base exception for Que-related errors"""
    pass

class QueEmpty(QueException):
    """Raised when no runs are available in the queue"""
    def __init__(self, loc: QueLocation, message: str = "No runs available"):
        super().__init__(f"{message} in {loc}")
        self.loc = loc  # Store for potential programmatic access

class QueIdxOOR(QueException):
    """Raised when index is out of range for a given location"""
    
    def __init__(self, loc: QueLocation, idx: int, leng: int):
        super().__init__(
            f"Index {idx} is out of range for que location {loc} (length: {leng})"
        )
        self.loc = loc
        self.idx = idx
        self.length = leng

class QueBusy(QueException):
    """Raised when attempting to add a run when one already exists"""
    
    def __init__(self, message: str = "Run already exists in cur_run"):
        super().__init__(message)

def retrieve_Data(path: Path) -> Any:
	"""Retrieves data from a given path."""
	with open(path, "r") as file:
		data = json.load(file)
	return data


def store_Data(path: Path, data: Any):
	"""Stores data to a given path."""
	with open(path, "w") as file:
		json.dump(data, file, indent=4)



class que:
	def __init__(
		self,
		logger: Logger,
		runs_path: str | Path = RUN_PATH,
		auto_save: bool = False,
		
	) -> None:
		self.runs_path: Path = Path(runs_path)
		self.old_runs: List[CompExpInfo] = []
		self.cur_run: List[ExpInfo] = []
		self.to_run: List[ExpInfo] = []
		self.fail_runs: List[FailedExp] = []
		self.auto_save: bool = auto_save
		self.load_state()
		self.logger = logger

	def _fetch_state(self, loc: QueLocation) -> ExpQue:
		"""Return reference to the specified list"""
		if loc == TO_RUN:
			return self.to_run
		elif loc == CUR_RUN:
			return self.cur_run
		elif loc == FAIL_RUNS:
			return self.fail_runs
		else:
			return self.old_runs


	def _pop_run(self, loc: QueLocation, idx: int) -> GenExp:
		"""Pop the run at the given location with the provided index

		Args:
			loc (QueLocation): to_run, cur_run or old_runs
			idx (int): Index of the run

		Raises:
			QueEmpty: len(loc) == 0
			QueIdxOOR: abs(idx) >= len(loc)

		Returns:
			ExpInfo: The specified run
		"""
		to_get = self._fetch_state(loc)
		if len(to_get) == 0:
			raise QueEmpty(loc)
		elif abs(idx) >= len(to_get):
			raise QueIdxOOR(loc, idx, len(to_get))
		return to_get.pop(idx)

	def _set_run(self, loc: QueLocation, idx: int, run: GenExp) -> None:
		"""Set a run at a specified location and index

		Args:
			loc (QueLocation): to_run, cur_run or old_runs
			idx (int): New index, must be within [-len(loc), len(loc)]
			run (ExpInfo): Experiment info to add to loc

		Raises:
			QueIdxOOR: The provided index is out of range: [-len(loc), len(loc)]
			TypeError: The run type doesn't match the queue location
		"""
		to_set = self._fetch_state(loc)
		if len(to_set) < abs(idx):
			raise QueIdxOOR(loc, idx, len(to_set))

		# Runtime type checking with type narrowing
		if loc == FAIL_RUNS:
			if not self._is_failed_exp(run):
				raise TypeError("fail_runs requires FailedExp with 'error' field")
			self.fail_runs.insert(idx, run)
		elif loc == OLD_RUNS:
			if not self._is_comp_exp_info(run):
				raise TypeError("old_runs requires CompExpInfo with 'results' field")
			self.old_runs.insert(idx, run)
		elif loc == CUR_RUN:
			if len(self.cur_run) != 0:
				raise QueBusy
		elif loc == TO_RUN:
			self.to_run.insert(idx, run)
		else:  # CUR_RUN
			self.cur_run.insert(idx, run)


	def _is_failed_exp(self, run: RunInfo) -> TypeGuard[FailedExp]:
		"""Check if run is a FailedExp"""
		return isinstance(run, dict) and "error" in run

	def _is_comp_exp_info(self, run: GenExp) -> TypeGuard[CompExpInfo]:
		"""Check if run is a CompExpInfo"""
		return isinstance(run, dict) and "results" in run

	def _run_sum(self, run: RunInfo) -> Sumarised:
		"""Extract key details from a run configuration.

		Args:
						run: Dictionary containing run configuration with admin details
						exc: Optional list of keys to exclude from the summary
		Returns:
						Sumarised: Dictionary with model, exp_no, split, and config_path
		"""
		return Sumarised(
			model=run["admin"]["model"],
			exp_no=run["admin"]["exp_no"],
			dataset=run["admin"]["dataset"],
			split=run["admin"]["split"],
			config_path=run["admin"]["config_path"],
			run_id=run.get("wandb", {}).get("run_id") if "wandb" in run else None,
			best_val_acc=(
				run["results"]["best_val_acc"]
				if "results" in run
				else None
			),
			best_val_loss=(
				run["results"]["best_val_loss"]
				if "results" in run
				else None
			),
			error=run.get("error") if self._is_failed_exp(run) else None,
		)
		
	def _run_to_str(self, run_sum: Sumarised) -> str:
		"""Convert a summarised run to a string for display

		Args:
						run_sum (Sumarised): Summarised run information

		Returns:
						str: Formatted string
		"""
		return (
			f"Model: {run_sum['model']}, Exp No: {run_sum['exp_no']}, "
			f"Dataset: {run_sum['dataset']}, Split: {run_sum['split']}, "
			f"Config Path: {run_sum['config_path']}"
		)
  
	def _is_dup_exp(self, new_run: RunInfo) -> bool:
		"""Check if run is a duplicate experiment"""
		
		new_sum = self._run_sum(new_run)

		for run in self.to_run + self.old_runs + self.cur_run:
			run_sum = self._run_sum(run)
			if new_sum["model"] == run_sum["model"] and \
				new_sum["exp_no"] == run_sum["exp_no"] and \
				new_sum["dataset"] == run_sum["dataset"] and \
				new_sum["split"] == run_sum["split"]:
				return True
		return False

	def _get_print_stats(self, runs: List[Sumarised]) -> Dict[str, int]:
		"""Get statistics for string formatting"""
		stats = {
			"max_model_len": 0,
			"max_exp_no_len": 0,
			"max_run_id_len": 0,
			"max_dataset_len": 0,
			"max_split_len": 0,
			"max_config_path_len": 0,
		}
  
		for run in runs:
			stats["max_model_len"] = max(stats["max_model_len"], len(run["model"]))
			stats["max_exp_no_len"] = max(stats["max_exp_no_len"], len(run["exp_no"]))
			if run["run_id"] is not None:
				stats["max_run_id_len"] = max(stats["max_run_id_len"], len(run["run_id"]))
			stats["max_dataset_len"] = max(stats["max_dataset_len"], len(run["dataset"]))
			stats["max_split_len"] = max(stats["max_split_len"], len(run["split"]))
			stats["max_config_path_len"] = max(
				stats["max_config_path_len"], len(run["config_path"])
			)
		
		if runs[0]["best_val_acc"] is not None:
			stats["max_best_val_acc_len"] = len("Best Val Acc")
			stats["max_best_val_loss_len"] = len("Best Val Loss")
   
		return stats

	def _get_val(self, run: GenExp, keys: List[str]) -> Any:
		"""Unpack the value in a run using a list of keys

		Args:
						run (GenExp): Provided general run
						keys (List[str]): Keys to unpack dictionary

		Returns:
						Any: The value
		"""
		unpack = cast(Dict[str, Any], run)
		for k in keys:
			unpack = unpack[k]
		return unpack

	def _find_runs(
		self, to_search: ExpQue, keys: List[str], criterion: Callable[[Any], bool]
	) -> Tuple[List[int], List[GenExp]]:
		"""Find runs with matching keys, if any

		Args:
						to_search (List[GenExp]): A run list
						keys (List[str]): Run keys
						value (Any): The desired value

		Returns:
						List[Tuple[int, GenExp]]: A List of runs
		"""
		idxs = []
		runs = []
		for i, run in enumerate(to_search):
			if criterion(self._get_val(run, keys)):
				idxs.append(i)
				runs.append(run)
		return idxs, runs

	def load_state(self):
		"""Read que from file"""
		try:
			with open(self.runs_path, "r") as f:
				data = json.load(f)
			self.to_run = data.get(TO_RUN, [])
			self.cur_run = data.get(CUR_RUN, [])
			self.old_runs = data.get(OLD_RUNS, [])
			self.fail_runs = data.get(FAIL_RUNS, [])
			self.logger.info(f"Loaded que state from {self.runs_path}")
		except FileNotFoundError:
			self.logger.warning(
				f"No existing state found at {self.runs_path}. Starting fresh."
			)
			self.to_run = []
			self.cur_run = []
			self.old_runs = []
			self.fail_runs = []

	def save_state(self):
		"""Write que to file"""
		with open(self.runs_path, "w") as f:
			all_runs = {
				TO_RUN: self.to_run,
				CUR_RUN: self.cur_run,
				OLD_RUNS: self.old_runs,
				FAIL_RUNS: self.fail_runs,
			}
			json.dump(all_runs, f, indent=4)
		self.logger.info(f'Saved que to {self.runs_path}')

	# for worker

	def pop_cur_run(self) -> ExpInfo:
		"""Pops the current run

		Returns:
			ExpInfo: Dictionary of experiment info
		"""
		return self._pop_run(CUR_RUN, 0)

	def set_cur_run(self, run: ExpInfo) -> None:
		"""Sets the current run

		Args:
			run (ExpInfo): Dictionary of experiment info
		"""
		self._set_run(CUR_RUN, 0, run)

	def peak_run(self, loc: QueLocation, idx: int) -> GenExp:
		"""Get the run at the given location with the provided index, but don't remove

		Args:
			loc (QueLocation): to_run, cur_run or old_runs
			idx (int): Index of the run

		Raises:
			QueEmpty: len(loc) == 0
			QueIdxOOR: abs(idx) >= len(loc)

		Returns:
			ExpInfo: The specified run
		"""
		to_get = self._fetch_state(loc)
		if len(to_get) == 0:
			raise QueEmpty(loc)
		elif abs(idx) >= len(to_get):
			raise QueIdxOOR(loc, idx, len(to_get))
		return to_get[idx]
	

	def stash_next_run(self) -> str:
		"""Moves next run from to_run to cur_run. Saves state with lock over both read and write
		"""
		next_run = self._pop_run(TO_RUN, 0)
		sum_str = self._run_to_str(self._run_sum(next_run))
		try:
			self.set_cur_run(next_run)
			self.logger.info(f'Stashed new run: {sum_str}')
		except QueBusy as qb:
			# put back
			self.logger.error(f'Failed to stash new run: {sum_str}')
			self._set_run(TO_RUN, 0, next_run)
			raise qb
		return sum_str

	def store_fin_run(self):
		"""Moves finished run from cur_run to old_runs. Saves state with lock over both read and write

		Raises:
			QueEmpty: If cur_run is empty
		"""
		# fin_run = self._pop_run(CUR_RUN, 0)
		fin_run = self.peak_run(CUR_RUN, 0) #safer incase crash during test
		results = full_test(admin=fin_run["admin"], data=fin_run["data"])
		comp_run = CompExpInfo(
			admin=fin_run["admin"],
			training=fin_run["training"],
			optimizer=fin_run["optimizer"],
			model_params=fin_run["model_params"],
			data=fin_run["data"],
			scheduler=fin_run["scheduler"],
			early_stopping=fin_run["early_stopping"],
			wandb=fin_run["wandb"],
			results=results,
		)
		self._set_run(OLD_RUNS, 0, comp_run)
		_ = self._pop_run(CUR_RUN, 0) #still remove from cur
		self.logger.info("Stored finished run")

	def stash_failed_run(self, error: str) -> None:
		"""Move a run to the failed que"""
		run = self.pop_cur_run()
		failed = cast(FailedExp, run | {"error": error})
		self._set_run(FAIL_RUNS, 0, failed)

	# summarisation

	@classmethod
	def get_config(cls, next_run: RunInfo) -> str:
		admin = next_run["admin"]
		return admin["config_path"]

	def run_str(self, loc: QueLocation, idx: int) -> str:
		"""Method to """
		return self._run_to_str(self._run_sum(self.peak_run(loc, idx)))

	def list_runs(self, loc: QueLocation) -> List[Sumarised]:
		"""List runs at a given location in summarised format

		Args:
						loc (QueLocation): to_run, cur_run or old_runs

		Returns:
						List[List[str]]: Summarised runs
		"""
		return [self._run_sum(run) for run in self._fetch_state(loc)]

	def disp_runs(self, loc: QueLocation, exc: Optional[List[str]] = None) -> None:
		print(f"{loc} runs".title())
		runs = self.list_runs(loc)
  
		if len(runs) == 0:
			print("  No runs available")
			return
		stats = self._get_print_stats(runs)
		header_parts = [
			"Idx".ljust(5),
			"Run ID".ljust(stats.get('max_run_id_len', len('Run Id')) + 2),
			"Model".ljust(stats["max_model_len"] + 2),
			"Exp No".ljust(stats["max_exp_no_len"] + 2),
			"Dataset".ljust(stats["max_dataset_len"] + 2),
			"Split".ljust(stats["max_split_len"] + 2),
		]
  
		if 'best_val_acc' in runs[0]:
			header_parts.append("Best Val Acc".ljust(stats["max_best_val_acc_len"] + 2))
			header_parts.append("Best Val Loss".ljust(stats["max_best_val_loss_len"] + 2))
  
		header_parts.append("Config Path".ljust(stats["max_config_path_len"] + 2))
  
		if 'error' in runs[0]:
			header_parts.append("Error")
   
		if exc is not None:
			header_parts = [h for h in header_parts if h.strip().lower() not in exc]
   
		header = " | ".join(header_parts)
		print(header)
		print("-" * len(header))
		for i, run in enumerate(runs):
			row_parts = [
				str(i).ljust(5),
				(run["run_id"] if run["run_id"] is not None else "N/A").ljust(stats.get('max_run_id_len', len('Run Id')) + 2),
				run["model"].ljust(stats["max_model_len"] + 2),
				run["exp_no"].ljust(stats["max_exp_no_len"] + 2),
				run["dataset"].ljust(stats["max_dataset_len"] + 2),
				run["split"].ljust(stats["max_split_len"] + 2),
			]
	
			if 'best_val_acc' in run:
				row_parts.append(
					(f"{run['best_val_acc']:.4f}" if run['best_val_acc'] is not None else "N/A").ljust(stats["max_best_val_acc_len"] + 2)
				)
				row_parts.append(
					(f"{run['best_val_loss']:.4f}" if run['best_val_loss'] is not None else "N/A").ljust(stats["max_best_val_loss_len"] + 2)
				)
	
			row_parts.append(run["config_path"].ljust(stats["max_config_path_len"] + 2))
	
			if 'error' in run:
				row_parts.append(run["error"] if run["error"] is not None else "N/A")
	 
			if exc is not None:
				row_parts = [r for r, h in zip(row_parts, header_parts) if h.strip().lower() not in exc]
  
			row = " | ".join(row_parts)
			print(row)

	def disp_run(self, loc: QueLocation, idx: int) -> None:
		"""Print a run config at a specific location}

		Args:
			loc (QueLocation): Location
			idx (int): Index
		"""
		print_config(self.peak_run(loc, idx))
		

	#for QueShell interface


	

	def recover_run(self, move_to: QueLocation = TO_RUN) -> None:
		"""Set the run in cur_run to recover, and move to to_run or cur_run"""
		run = self.pop_cur_run()
		run["admin"]["recover"] = True
		self._set_run(move_to, 0, run)
		self.logger.info('Recovered run')

	def clear_runs(self, loc: QueLocation) -> None:
		"""reset the runs queue"""
		to_clear = self._fetch_state(loc)

		if len(to_clear) > 0:
			to_clear = []
			self.logger.info(f"{loc} successfully cleared")
		else:
			self.logger.warning(f"{loc} already empty")

	

	def create_run(
		self,
		arg_dict: AdminInfo,
		wandb_dict: WandbInfo,
	) -> None:
		"""Create and add a new training run entry

		Args:
						arg_dict (AdminInfo): Arguments used by training function
						wandb_dict (WandbInfo): Wandb information not included in arg_dict.
						ask (bool, optional): Pre-check run before creation. Defaults to True.
		"""

		try:
			config = load_config(arg_dict)
		except ValueError:
			self.logger.warning(f"{arg_dict['config_path']} not found. Create cancelled")
			return

		if self._is_dup_exp(config):
			self.logger.warning("Duplicate run detected. Create cancelled")
			return

		# print_config(config)
		config = cast(ExpInfo, config | {"wandb": wandb_dict})
		self.to_run.append(config)
		self.logger.info("Added new run")
	

	def add_run(self, arg_dict: AdminInfo, wandb_dict: WandbInfo) -> None:
		"""Add a completed (and full tested) run to the old_runs que for storage

		Args:
						arg_dict (AdminInfo): Basic information to load the config and find the results
						wandb_dict (WandbInfo): Wandb information not included in the run config
		"""
		try:
			config =  load_config(arg_dict)
		except ValueError:
			self.logger.warning(f"{arg_dict['config_path']} not found. Create cancelled")
			return

		if self._is_dup_exp(config):
			self.logger.warning("Duplicate run detected. Create cancelled")
			return

		save_path = Path(arg_dict["save_path"])
		res_dir = save_path.parent / "results"
		res_path = res_dir / "best_val_loss.json"
		try:
			results = load_comp_res(res_path)  
			self.logger.info('Successfully loaded results')
		except FileNotFoundError:
			results = full_test(
				admin=config['admin'],
				data=config['data'],
			)#NOTE: this will print to the terminal
			self.logger.info('Could not find results, running full test')
		comp_run = CompExpInfo(
			admin=config["admin"],
			training=config["training"],
			optimizer=config["optimizer"],
			model_params=config["model_params"],
			data=config["data"],
			scheduler=config["scheduler"],
			early_stopping=config["early_stopping"],
			wandb=wandb_dict,
			results=results,
		)
		self.old_runs.insert(0, comp_run)
		self.logger.info("New complete run added")

	def remove_run(self, loc: QueLocation, idx: int) -> None:
		"""Removes a run from the given location safely

		Args:
			loc (QueLocation): to_run, cur_run or old_runs
			idx (int): Index of run
		"""
		_ = self._pop_run(loc, idx)
		self.logger.info('Successfully removed run')

	def shuffle(self, loc: QueLocation, o_idx: int, n_idx: int) -> None:
		"""Repositions a run from the que
		Args:
																		loc: TO_RUN, CUR_RUN or OLD_RUNS
																		o_idx: original index of run
																		n_idx: new index of run
		"""
		try:
			self._set_run(loc, n_idx, self._pop_run(loc, o_idx))
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
				self._set_run(n_loc, 0, self._pop_run(o_loc, oi_idx))
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
			run = self._pop_run(loc, idx)
			if key2 is not None:
				run[key1][key2] = value
			else:
				run[key1] = value
			self._set_run(loc, idx, run)
			self.print_v("Edit successful")
		except Exception as e:
			self.print_v(str(e))
   
	def find_runs(
		self,
		loc: QueLocation,
		key_set: List[List[str]],
		criterions: List[Callable[[Any], bool]],
	) -> Tuple[List[int], List[GenExp]]:
		"""Find the set of runs which match all of the key list value pairs

		Args:
						loc (QueLocation): Location to search
						key_set (List[List[str]]): A list of keys to unpack a dictionary to get to a particular value. Multiple values can be searched with a list of these sets of keys
						values (List[Any]): The corresponding values for each set of keys

		Returns:
						Tuple[List[int], List[GenExp]]: Indexes, and runs, if found
		"""

		assert len(key_set) == len(criterions), (
			f"Length of key_set: {len(key_set)} does not match length of values: {len(criterions)}"
		)
		runs = [run for run in self._fetch_state(loc)]
		idxs = []
		for k_lst, crit in zip(key_set, criterions):
			idxs, runs = self._find_runs(runs, k_lst, crit)
		return idxs, runs


if __name__ == "__main__":
	q = que()
	q.disp_runs(OLD_RUNS)