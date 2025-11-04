import configparser
import argparse
import ast
from typing import Dict, Any, List, Optional, Union, TypedDict, Tuple
from utils import enum_dir, ask_nicely
from stopping import EarlyStopper, StopperOn
from pathlib import Path
import torch
import numpy as np
import random
# locals
from models import avail_models

# TODO: make configs the sole source of these constants

# constants
# - wandb
ENTITY = "ljgoodall2001-rhodes-university"
PROJECT = "WLASL-100"
PROJECT_BASE = "WLASL"
# - data
LABELS_PATH = "preprocessed/labels"
LABEL_SUFFIX = "fixed_frange_bboxes_len.json"
CLASSES_PATH = "./info/wlasl_class_list.json"
WLASL_ROOT = "../data/WLASL"
RAW_DIR = "WLASL2000"
SPLIT_DIR = "splits"
# - training/testing
RUNS_PATH = "./runs"
SEED = 42

####################### Typed Dictionaries #############################
class AdminInfo(TypedDict):
	model: str
	dataset: str
	split: str
	exp_no: str
	recover: bool
	config_path: str
	save_path: str

class TrainingInfo(TypedDict):
	batch_size: int
	update_per_step: int
	batch_size_equivalent: int
	max_epoch: int

class OptimizerInfo(TypedDict):
	eps: float
	backbone_init_lr: float
	backbone_weight_decay: float
	classifier_init_lr: float
	classifier_weight_decay: float

class Model_paramsInfo(TypedDict):
	drop_p: float

class SchedulerInfo(TypedDict):
	type: str
	tmax: int
	eta_min: float

class DataInfo(TypedDict):
	num_frames: int
	frame_size: int

class WandbInfo(TypedDict):
	entity: str
	project: str
	tags: List[str]
	run_id: Optional[str]
	
class ExperimentInfo(TypedDict):
	admin: AdminInfo
	training: TrainingInfo
	optimizer: OptimizerInfo
	model_params: Model_paramsInfo
	data: DataInfo
	scheduler: Optional[SchedulerInfo]
	early_stopping: Optional[StopperOn]
 
class CompletedExpInfo(ExperimentInfo):
    wandb: Optional[WandbInfo]

####################### Utility functions ##############################

def set_seed(seed: int=SEED):
	"""Set the random seed across multiple environments, as well as use torch deterministic settings. 

	Args:
		seed (int, optional): The random seed, otherwise use defined constant. Defaults to SEED.
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def get_avail_splits(pre_proc_dir: str = LABELS_PATH) -> List[str]:
	"""Get the available splits from preprocessed labels directory

	Args:
		pre_proc_dir (str, optional): The root directory for preprocessed labels. Defaults to LABELS_PATH.

	Raises:
		ValueError: If preprocessed directory is invalid.

	Returns:
		List[str]: List of available splits.
	"""
	
	ppd = Path(pre_proc_dir)
	if not ppd.exists() or not ppd.is_dir():
		raise ValueError(
			f"Invalied preprocessed directory: {pre_proc_dir}, must exist and be directory"
		)
	return list(map(lambda x: x.name, ppd.iterdir()))

def print_config(config_dict):
	"""
	Print configuration dictionary in a more readable format.

	Args:
					config_dict (dict): Dictionary containing configuration sections

	"""
	for section in config_dict.keys():
		print(f"{section.upper()}:")
		section_data = config_dict[section]

		# Calculate max key length for alignment
		max_key_len = (
			max(len(str(k)) for k in section_data.keys()) if section_data else 0
		)
		if isinstance(section_data, dict):
			for key, value in section_data.items():
				print(f"    {key:<{max_key_len}} : {value}")
		else:
			print(section_data)
		print()

###################### Config generation ###############################

def _schedular_precheck(sched_info: Optional[SchedulerInfo]) -> None:
	"""Fail early if scheduler config is invalid. 

	Args:
		config (Dict[str, Any]): Entire training config dictionary.

	Raises:
		ValueError: If scheduler type invalid
		ValueError: If warmup_epochs negative
		ValueError: If start_factor and end_factor not specified for warmup
		ValueError: If start_factor and end_factor invalid
	"""

	if sched_info is None:
		return
 
	#these have already been implemented
	valid_types = [
		'CosineAnnealingLR',
		'CosineAnnealingWarmRestarts',
	]
	
	if sched_info["type"] not in valid_types:
		raise ValueError(
			f"Invalid scheduler type: {sched_info['type']}. Available types: {valid_types}"
		)
		
	if 'warmup_epochs' in sched_info:
		if sched_info["warmup_epochs"] < 0:
			raise ValueError("warmup_epochs must be non-negative")
		if "start_factor" not in sched_info or "end_factor" not in sched_info:
			raise ValueError("Both start_factor and end_factor must be specified for warmup")
		if not (0 < sched_info["start_factor"] < sched_info["end_factor"] <= 1.0):
			raise ValueError("start_factor must be > 0 and < end_factor <= 1.0")



def _stopper_precheck(config: Optional[StopperOn]) -> None:
	"""Fail early if stopper is invalid

	Args:
		config (Optional[StopperOn]): _description_
	"""
	if config:
		EarlyStopper.config_precheck(config)
		

def _convert_type(value: str) -> Any:
	"""Convert string values to appropriate types"""
	try:
		return ast.literal_eval(value)
	except (ValueError, SyntaxError):
		return value

def parse_ini_config(ini_file: Union[str, Path]) -> Dict[str, Any]:
	"""Parse .ini file for wandb config"""
	config = configparser.ConfigParser()
	config.read(ini_file)

	# Nested structure
	wandb_config = {}
	for section in config.sections():
		wandb_config[section] = {}
		for key, value in config[section].items():
			wandb_config[section][key] = _convert_type(value)

	return wandb_config

def _to_exp_info(config: Dict[str, Any]) -> ExperimentInfo:
	"""Convert raw config to typed ExperimentInfo"""
	
	# Optional sections with None default
	scheduler = None
	if 'scheduler' in config:
		scheduler = SchedulerInfo(
			type=config['scheduler']['type'],
			tmax=config['scheduler']['tmax'],
			eta_min=config['scheduler']['eta_min']
		)
	
	early_stopping = None
	if 'early_stopping' in config:
		early_stopping = StopperOn(
			metric=config['early_stopping']['metric'],  # Convert to list/tuple
			mode=config['early_stopping']['mode'],
			patience=config['early_stopping']['patience'],
			min_delta=config['early_stopping']['min_delta']
		)
	
	return ExperimentInfo(
		admin=AdminInfo(**config['admin']),
		training=TrainingInfo(**config['training']),
		optimizer=OptimizerInfo(**config['optimizer']),  # Shorthand if keys match exactly
		model_params=Model_paramsInfo(**config['model_params']),
		data=DataInfo(**config['data']),
		scheduler=scheduler,
		early_stopping=early_stopping
	)

def _add_eq_bs(conf: Dict[str, Any]) -> Dict[str, Any]:
	"""Add equivalent batch size"""
	conf["training"]["batch_size_equivalent"] = (
		conf["training"]["batch_size"] * conf["training"]["update_per_step"]
	)
	return conf

def load_config(admin: AdminInfo) -> ExperimentInfo:
	"""Load config from flat file and merge with command line args

	Args:
		admin (Dict[str, Any]): Admin args from command line

	Raises:
		ValueError: If config path not found, or there are issues with the config
		KeyError: Various issues with config file

	Returns:
		ExperimentInfo: Dictionary containing all required information for a run
	"""
	
	conf_path = Path(admin["config_path"])
	if not conf_path.exists():
		raise ValueError(f"{conf_path} not found")

	config = parse_ini_config(admin["config_path"])
	config = _add_eq_bs(config)
	ndict = {"admin": admin}
	ndict.update(config)

	e_info = _to_exp_info(ndict)

	_stopper_precheck(e_info.get('early_stopping'))
	_schedular_precheck(e_info.get('scheduler'))
 
	return e_info

def get_train_parser(prog: Optional[str] = None,desc: str = "Train a model") -> argparse.ArgumentParser:
	"""Get parser for a training configuration

	Args:
		prog (Optional[str], optional): Script name, (e.g. train.py). Defaults to None.
		desc (str, optional): Program desctiption. Defaults to "Train a model".

	Returns:
		argparse.ArgumentParser: Parser which takes training arguments
	"""
	models_available = avail_models()
	splits_available = get_avail_splits()

	parser = argparse.ArgumentParser(description=desc, prog=prog)

	# admin
	parser.add_argument(
		"model",
		type=str,
		choices=models_available,
		help=f"Model name from one of the implemented models: {models_available}",
	)
	parser.add_argument(
		"split",
		type=str,
		choices=splits_available,
		help=f"The class split, one of:  {', '.join(splits_available)}",
	)
	parser.add_argument("exp_no", type=int, help="Experiment number (e.g. 10)")
	parser.add_argument(
		'-ds',
		'--dataset',
		type=str,
		choices=['WLASL'],
		help="Not implemented yet",
		default='WLASL'
	)
 
	parser.add_argument(
		"-r", "--recover", action="store_true", help="Recover from last checkpoint"
	)
	parser.add_argument(
		"-ri",
		"--run_id",
		type=str,
		default=None,
		help="The run id to use (especially when also usign recover)",
	)
	parser.add_argument(
		"-p",
		"--project",
		type=str,
		default=PROJECT,
		help=f"wandb project name, if not {PROJECT}",
	)
	parser.add_argument(
		"-et", "--entity", type=str, default=ENTITY, help=f"Entity if not {ENTITY}"
	)
	parser.add_argument(
		"-ee",
		"--enum_exp",
		action="store_true",
		help="enumerate the experiment dir num (for output)",
	)
	parser.add_argument(
		"-ec",
		"--enum_chck",
		action="store_true",
		help="enumerate the checkpoint dir num (for output)",
	)
	parser.add_argument(
		"-t", "--tags", nargs="+", type=str, help="Additional wandb tags"
	)
	parser.add_argument("-c", "--config_path", help="path to config .ini file")

	return parser

#NOTE: Seperate into having seperate function for train parser
def take_args(
	sup_args: Optional[List[str]] = None,
	make_dirs: bool = False,
) -> Optional[Tuple[AdminInfo, WandbInfo]]:
	"""Retrieve arguments for new training run

	Args:
		sup_args (Optional[List[str]], optional): Supply arguments instead of taking from command line. Defaults to None.
		make_dirs (bool, optional): Make output and checkpoint dirs. Defaults to False.

	Raises:
		ValueError: If model or split supplied are not available, or if recovering and save path is invalid.

	Returns:
		Optional[tuple | argparse.ArgumentParser]: Arguments or parser, if successful.
	"""

	models_available = avail_models()
	splits_available = get_avail_splits()

	parser = get_train_parser()

	if sup_args:
		args = parser.parse_args(sup_args)
	else:
		args = parser.parse_args()

	if args.split not in splits_available:
		raise ValueError(
			f"Sorry {args.split} not processed yet.\n\
			Currently available: {splits_available}"
		)
	if args.model not in models_available:
		raise ValueError(
			f"Sorry {args.model} not implemented yet.\n\
			Currently available: {models_available}"
		)

	exp_no = str(int(args.exp_no)).zfill(3)

	if args.project is None:
		args.project = f"{PROJECT_BASE}-{args.split[3:]}"

	args.exp_no = exp_no

	output = Path(f"{RUNS_PATH}/{args.split}/{args.model}_exp{exp_no}")

	# recovering
	if not args.recover and output.exists():  # fresh run
		if not args.enum_exp:
			ans = ask_nicely(
				message=f"{output} already exists, do you want to cancel, proceed, or enumerate (c, p, e): ",
				requirment=lambda x: x.lower() in ["c", "p", "e"],
				error=f"Must choose one of: {['c', 'p', 'e']}",
			)
		else:
			ans = "e"

		if ans.lower() == "e":
			output = enum_dir(output, make_dirs)

		if ans.lower() == "c":
			return

	# saving
	save_path = output / "checkpoints"
	args.save_path = save_path
	# if not args.recover and args.enum_chck:
	if not args.recover and save_path.exists():
		if not args.enum_chck:
			ans = ask_nicely(
				message=f"{save_path} already exists, do you want to cancel, overwrite, or enumerate (c, o, e): ",
				requirment=lambda x: x.lower() in ["c", "o", "e"],
				error=f"Must choose one of: {['c', 'o', 'e']}",
			)
		else:
			ans = "e"

		if ans.lower() == "e":
			args.save_path = enum_dir(save_path, make_dirs)

		if ans.lower() == "c":
			return
	elif args.recover:
		if not save_path.exists() or not save_path.is_dir():
			raise ValueError(
				f"Cannot recover, {save_path} does not exist or is not a directory"
			)
		if len([f for f in save_path.iterdir() if f.is_file()]) == 0:
			raise ValueError(f"Cannot recover, {save_path} is empty")
	else:
		args.save_path = save_path

	args.save_path = str(args.save_path)

	# Set config path
	if args.config_path is None:
		args.config_path = f"./configfiles/{args.split}/{args.model}_{exp_no}.ini"
 
	# Load config
	arg_dict = vars(args)
	clean_dict = {}

	redundants = ["project", "tags", "enum_exp", "enum_chck", "entity"]

	for key, value in arg_dict.items():
		if key in redundants:
			continue
		if value is not None:
			clean_dict[key] = value

	# NOTE: these tags are redundant
	tags = [
		args.split,
		args.model,
		f"exp-{exp_no}",
	]
	if args.recover:
		tags.append("Recovered")
	if args.tags is not None:
		tags.extend(args.tags)

	wandb_info = WandbInfo(entity=args.entity, project=args.project, tags=tags, run_id=args.run_id)
	admin_info = AdminInfo(
		model=args.model,
		dataset=args.dataset,
		split=args.split,
		exp_no=args.exp_no,
		recover=args.recover,
		config_path=args.config_path,
		save_path=args.save_path
	)

	return admin_info, wandb_info

def main():
	try:
		# maybe_args = take_args(available_splits,available_model,
		#                  sup_args=['-x', '5', '-m', 'S3D', '-sp', 'asl100'])
		# maybe_args = take_args(sup_args=["-h"])
		maybe_args = take_args()
	except Exception as e:
		maybe_args = None
		print(f"Parsing failed with error: {e}")

	if isinstance(maybe_args, tuple):
		admin_info, wandb_info = maybe_args
	else:
		return
	# str_dict(arg_dict, disp=True)
	config = load_config(admin_info)
	print_config(config)

	# print_dict(config)


if __name__ == "__main__":
	main()
