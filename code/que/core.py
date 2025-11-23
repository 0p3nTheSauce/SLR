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

#locals
from run_types import (
    ExpInfo, CompExpInfo,
    AdminInfo, WandbInfo, RunInfo, CompRes
)
from testing import full_test, load_comp_res
from configs import print_config, load_config
# import ..utils
from utils import  ask_nicely
# constants
SESH_NAME = "que_training"
DN_NAME = "daemon"
WR_NAME = "worker"
SR_NAME = "server"
# MR_NAME = "monitor"
QUE_DIR = Path(__file__).parent

RUN_PATH = QUE_DIR / "Runs.json"
WR_LOG_PATH = QUE_DIR / "Worker.log"
DN_LOG_PATH = QUE_DIR / "Daemon.log"
SR_LOG_PATH = QUE_DIR / "Server.log"
WR_PATH = "./quefeather.py"


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
    
class QueEmpty(Exception):
    """No runs available"""

    def __init__(self, loc: QueLocation, message: str = "No runs available"):
        super().__init__(f"{message} in {loc}")

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

    def __init__(
        self,
        message: str = "There is already a run in cur_run, and there should only be one",
    ):
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
        # conf_loader: ConfigLoader,
        runs_path: str | Path = RUN_PATH,
        verbose: bool = True,
        auto_save: bool = False,
        
    ) -> None:
        self.runs_path: Path = Path(runs_path)
        self.verbose: bool = verbose
        self.old_runs: List[CompExpInfo] = []
        self.cur_run: List[ExpInfo] = []
        self.to_run: List[ExpInfo] = []
        self.fail_runs: List[FailedExp] = []
        self.auto_save: bool = auto_save
        self.load_state()

    def print_v(self, message: str) -> None:
        """Prints a message if verbose is True."""
        if self.verbose:
            print(message)

    def fetch_state(self, loc: QueLocation) -> ExpQue:
        """Return reference to the specified list"""
        if loc == TO_RUN:
            return self.to_run
        elif loc == CUR_RUN:
            return self.cur_run
        elif loc == FAIL_RUNS:
            return self.fail_runs
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
            self.print_v(f"Loaded que state from {self.runs_path}")
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

    def _is_failed_exp(self, run: GenExp) -> TypeGuard[FailedExp]:
        """Check if run is a FailedExp"""
        return isinstance(run, dict) and "error" in run

    def _is_comp_exp_info(self, run: GenExp) -> TypeGuard[CompExpInfo]:
        """Check if run is a CompExpInfo"""
        return isinstance(run, dict) and "results" in run

    def _peak_run(self, loc: QueLocation, idx: int) -> GenExp:
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
        to_get = self.fetch_state(loc)
        if len(to_get) == 0:
            raise QueEmpty(loc)
        elif abs(idx) >= len(to_get):
            raise QueIdxOOR(loc, idx, len(to_get))
        return to_get[idx]
    
    def peak_run(self, loc: QueLocation, idx: int) -> Optional[GenExp]:
        """Get the run at the given location with the provided index, but don't remove

        Args:
                                                                        loc (QueLocation): to_run, cur_run or old_runs
                                                                        idx (int): Index of the run

        Returns:
                                                                        Optional[ExpInfo]: The specified run, or None if not found
        """
        try:
            return self._peak_run(loc, idx)
        except (QueEmpty, QueIdxOOR) as e:
            self.print_v(str(e))
            return None


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
        to_get = self.fetch_state(loc)
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
        to_set = self.fetch_state(loc)
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
        elif loc == TO_RUN:
            self.to_run.insert(idx, run)
        else:  # CUR_RUN
            self.cur_run.insert(idx, run)

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
        runs = [run for run in self.fetch_state(loc)]
        idxs = []
        for k_lst, crit in zip(key_set, criterions):
            idxs, runs = self._find_runs(runs, k_lst, crit)
        return idxs, runs

    def save_state(self):
        """Saves state to Runs.jso, with filelock"""
        # with self.lock:
        self._save_Que()

    def load_state(self) -> None:
        """Loads state from Runs.json, with filelock"""
        # with self.lock:
        self._load_Que()

    def peak_cur_run(self) -> ExpInfo:
        """Get the run stored in cur_run (assumes 1, dont pop)"""
        return self._peak_run(CUR_RUN, 0)

    def pop_cur_run(self) -> ExpInfo:
        """Pop the run stored in cur_run (assumes 1)"""
        return self._pop_run(CUR_RUN, 0)

    def set_cur_run(self, run: ExpInfo) -> None:
        """Set the run in cur_run (assumes 1)"""
        # full cur_run
        if (
            len(self.cur_run) != 0
        ):  # NOTE at some point it might be possible to have multiple busy operations
            self.print_v("Failed to set cur run")
            raise QueBusy()

        self._set_run(CUR_RUN, 0, run)

    def stash_next_run(self) -> None:
        """Moves next run from to_run to cur_run. Saves state with lock over both read and write

        Raises:
                        QueEmpty: If to_run is empty
                        QueBusy: If cur_run is full
                        Timeout: If cannot acquire file lock
        """
        next_run = self._pop_run(TO_RUN, 0)
        try:
            self.set_cur_run(next_run)
        except QueBusy as qb:
            # put back
            self._set_run(TO_RUN, 0, next_run)
            raise qb
        self.print_v(f"Stashed next run: {self.run_str(self.run_sum(next_run))}")

    def store_fin_run(self):
        """Moves finished run from cur_run to old_runs. Saves state with lock over both read and write

        Raises:
                        QueEmpty: If cur_run is empty
                        Timeout: If cannot acquire file lock
        """

        fin_run = self.pop_cur_run()
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
        self.print_v(f"Stored finished run: {self.run_str(self.run_sum(fin_run))}")

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

        if "wandb" in run:  # ExpInfo
            run_id = run["wandb"]["run_id"]
            if run_id is None:
                dic["run_id"] = "None"
            else:
                dic["run_id"] = run_id

        if "error" in run:  # FaileExp
            dic["error"] = run["error"]

        if "results" in run:  # CompExpInfo
            results = cast(CompRes, run["results"])
            dic["best_val_acc"] = f"{results['best_val_acc']:.4f}"
            dic["best_val_loss"] = f"{results['best_val_loss']:.4f}"

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

    def _sum_runs(
        self, run_confs: ExpQue, head_sum: Optional[Dict[str, str]] = None
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

        max_model = max(len(r["model"]) for r in runs_info)
        max_exp = max(len(str(r["exp_no"])) for r in runs_info)
        max_split = max(len(r["split"]) for r in runs_info)
        max_id = max(len(r["run_id"]) for r in runs_info)
        stats = {
            "max_model": max_model,
            "max_exp": max_exp,
            "max_split": max_split,
            "max_id": max_id,
        }

        # check for extra keys (in 1 so we ignore header)
        if "error" in runs_info[0]:  # FailedExp
            max_error = max(len(r["error"]) for r in runs_info)
            stats["max_error"] = max_error

        if "best_val_acc" in runs_info[0]:  # CompExpInfo
            max_val_loss = max(len(r["best_val_loss"]) for r in runs_info)
            max_val_acc = max(len(r["best_val_acc"]) for r in runs_info)
            stats["max_val_loss"] = max_val_loss
            stats["max_val_acc"] = max_val_acc

        return runs_info, stats

    def _get_head_sum(self, loc: QueLocation) -> Dict[str, str]:
        head_sum = {
            "model": "Model",
            "split": "Split",
            "exp_no": "Exp",
            "run_id": "Run ID",
            "config_path": "Config",
        }

        # check for extra headings
        if loc == FAIL_RUNS:
            head_sum["error"] = "Exception"
        if loc == OLD_RUNS:
            head_sum["best_val_acc"] = "Best Val Acc"
            head_sum["best_val_loss"] = "Best Val Loss"

        return head_sum

    def _get_runs_info(self, loc: QueLocation):
        """Get summarised run info for a particular que location"""
        to_disp = self.fetch_state(loc)
        if len(to_disp) == 0:
            raise QueEmpty(loc)
        head_sum = self._get_head_sum(loc)
        return self._sum_runs(to_disp, head_sum)

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

        def_stats = {
            "max_id": 0,
            "max_model": 0,
            "max_exp": 0,
            "max_split": 0,
            "max_val_acc": 0,
            "max_val_loss": 0,
        }

        if stats is not None:
            def_stats.update(stats)

        r_str = ""

        if "run_id" not in r_info:
            r_info["run_id"] = "None"

        r_str += (
            f"{r_info['run_id']:<{def_stats['max_id']}}  "
            f"{r_info['model']:<{def_stats['max_model']}}  "
            f"{r_info['split']:<{def_stats['max_split']}}  "
            f"{r_info['exp_no']:<{def_stats['max_exp']}}  "
        )

        # check for extra keys
        if "error" in r_info:  # FailedExp
            r_str += f"{r_info['error']:<{def_stats['max_error']}}  "

        if "max_val_loss" in def_stats and 'best_val_acc' in r_info:  # CompExpInfo
            r_str += (
                f"{r_info['best_val_acc']:<{def_stats['max_val_acc']}}  "
                f"{r_info['best_val_loss']:<{def_stats['max_val_loss']}}  "
            )

        r_str += f"{r_info['config_path']}"  # keep config at end

        return r_str

    # for queShell interface

    def list_runs(self, loc: QueLocation) -> List[str]:
        """Summarise all the runs in a particular que into a list of strings

        Args:
                        loc (QueLocation): The location to summarise

        Returns:
                        List[str]: List of summarised run strings
        """
        # Extract run info
        try:
            runs_info, stats = self._get_runs_info(loc)
        except QueEmpty as qe:
            self.print_v(str(qe))
            return []

        conf_list = []
        head = f"   {self.run_str(runs_info[0], stats)}"
        conf_list.append(head)
        # conf_list.append("-" * len(head))
        for info in runs_info[1:]:
            # Format with padding for alignment
            r_str = f"  {self.run_str(info, stats)}"
            conf_list.append(r_str)

        return conf_list

    @classmethod
    def disp_runs(cls, runs: List[str], loc: QueLocation) -> None:
        # Nice header
        loc_display = loc.replace("_", " ").title()
        print(f"\n=== {loc_display} ===")
        print()
        # runs = cls.list_runs(loc)
        if len(runs) == 0:
            return
        max_len = max(len(r) for r in runs)
        print(runs[0])  # head
        print("-" * max_len)
        for run in runs[1:]:
            print(run)
        print()

    def disp_run(self, loc: QueLocation, idx: int) -> None:
        try:
            print_config(self._peak_run(loc, idx))
        except Exception as e:
            print(f"Could not display run {idx} : {loc} due to: {e}")

    def recover_run(self, move_to: QueLocation = TO_RUN) -> None:
        """Set the run in cur_run to recover, and move to to_run or cur_run"""
        try:
            run = self.pop_cur_run()
            run["admin"]["recover"] = True
            self._set_run(move_to, 0, run)
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
                        arg_dict (AdminInfo): Arguments used by training function
                        wandb_dict (WandbInfo): Wandb information not included in arg_dict.
                        ask (bool, optional): Pre-check run before creation. Defaults to True.
        """

        try:
            config = load_config(arg_dict)
        except ValueError:
            self.print_v(f"{arg_dict['config_path']} not found. Create cancelled")
            return

        if self._is_dup_exp(config):
            self.print_v(
                f"Duplicate run detected: {self.run_str(self.run_sum(config))}"
            )
            return

        if ask:
            print_config(config)

        # if ask:
        #     proceed = (
        #         ask_nicely(
        #             message="Confirm: y/n: ",
        #             requirment=lambda x: x.lower() in ["y", "n"],
        #             error="y or n: ",
        #         ).lower()
        #         == "y"
        #     )
        # else:
        #     proceed = True
        proceed = True
        if proceed:
            config = cast(ExpInfo, config | {"wandb": wandb_dict})

            self.to_run.append(config)
            self.print_v(f"Added new run: {self.run_str(self.run_sum(config))}")
        else:
            self.print_v("Training cancelled by user")

    def add_run(self, arg_dict: AdminInfo, wandb_dict: WandbInfo) -> None:
        """Add a completed (and full tested) run to the old_runs que for storage

        Args:
                        arg_dict (AdminInfo): Basic information to load the config and find the results
                        wandb_dict (WandbInfo): Wandb information not included in the run config
        """
        try:
            config =  load_config(arg_dict)
        except ValueError:
            self.print_v(f"{arg_dict['config_path']} not found. Add cancelled")
            return

        if self._is_dup_exp(config):
            self.print_v(
                f"Duplicate run detected: {self.run_str(self.run_sum(config))}. Add cancelled"
            )
            return

        save_path = Path(arg_dict["save_path"])
        res_dir = save_path.parent / "results"
        res_path = res_dir / "best_val_loss.json"
        results = load_comp_res(
            res_path
        )  # TODO: this throws an error if testing has not been run
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
        self.print_v(f"New complete run added: {self.run_str(self.run_sum(comp_run))}")

    def remove_run(self, loc: QueLocation, idx: int) -> None:
        """Removes a run from the given location safely

        Args:
                                                                        loc (QueLocation): to_run, cur_run or old_runs
                                                                        idx (int): Index of run
        """
        try:
            _ = self._pop_run(loc, idx)
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

if __name__ == "__main__":
    q = que()
    q.disp_runs(q.list_runs(OLD_RUNS), OLD_RUNS)