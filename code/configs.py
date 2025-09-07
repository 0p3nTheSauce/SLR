import configparser
import argparse
import importlib
import ast
from typing import Dict, Any

from utils import enum_dir, print_dict
import wandb
from stopping import EarlyStopper

def load_config(arg_dict, verbose=False):
	"""Load config from flat file and merge with command line args"""
	config = parse_ini_config(arg_dict['config_path'])
	
	finished_keys = []
	
	if verbose:
		print()
	#update existing sections
	for key, value in arg_dict.items():
		if value is None:
			finished_keys.append(key)
			continue
		# print(f'{key}: {value}')
		for section in config.keys():#config is nested
			if key in config[section]: 
				if verbose:
					print(f'Overide:\n config[{section}][{key}] = {value}')
				config[section][key] = value
				finished_keys.append(key)
				if verbose:
					print(f'Becomes: \n config[{section}][{key}] = {value}')    		
	
	#update unset section
	not_finished = [key for key in arg_dict.keys() if key not in finished_keys]
	admin = {}
	for key in not_finished:
		admin[key] = arg_dict[key]
		if verbose:
			print(f'Added: conf[admin][{key}] = {arg_dict[key]}')
	config['admin'] = admin
	if verbose:	
		print()
	
	#want to add equivalent batch size
	config['training']['batch_size_equivalent'] = config['training']['batch_size'] * config['training']['update_per_step']
 
	config = EarlyStopper.config_precheck(config)
		
	# return post_process(config, verbose)
	return config

def _convert_type(value: str) -> Any:
	"""Convert string values to appropriate types"""
	try:
		return ast.literal_eval(value)
	except (ValueError, SyntaxError):
	 	return value
 
def parse_ini_config(ini_file: str) -> Dict[str, Any]:
	"""Parse .ini file for wandb config """
	config = configparser.ConfigParser()
	config.read(ini_file)

	# Nested structure
	wandb_config = {}
	for section in config.sections():
		wandb_config[section] = {}
		for key, value in config[section].items():
			wandb_config[section][key] = _convert_type(value)
	
	return wandb_config

def print_config_old(config_dict, title='Training'):
	"""
	Print configuration dictionary in a more readable format.
	
	Args:
		config_dict (dict): Dictionary containing configuration sections
		title: 'Testing' or 'Training' 
	"""
	# Extract admin info for header
	admin = config_dict.get('admin', {})
	model = admin.get('model', 'Unknown Model')
	split = admin.get('split', 'unknown')
	exp_no = admin.get('exp_no', '000')
	
	# Print header
	print(f"{title} {model} on split {split}")
	
	# Print admin section in formatted way
	if 'admin' in config_dict:
		print(f"              Experiment no: {admin.get('exp_no', 'N/A')}")
		print(f"              Raw videos at: {admin.get('root', 'N/A')}")
		print(f"              Labels at: {admin.get('labels', 'N/A')}")
		print(f"              Saving files to: {admin.get('save_path', 'N/A')}")
		if title == 'Training':
			print(f"              Recovering: {admin.get('recovering', False)}")
		print(f"              Config: {admin.get('config_path', 'N/A')}")
		print()

	# Print other sections in organized format
	sections_order = ['training', 'optimizer', 'scheduler', 'data', 'model_params']
	
	for section in sections_order:
		if section in config_dict:
			print(f"{section.upper()}:")
			section_data = config_dict[section]
			
			# Calculate max key length for alignment
			max_key_len = max(len(str(k)) for k in section_data.keys()) if section_data else 0
			
			for key, value in section_data.items():
				print(f"    {key:<{max_key_len}} : {value}")
			print()

def print_config(config_dict):
	"""
	Print configuration dictionary in a more readable format.
	
	Args:
		config_dict (dict): Dictionary containing configuration sections
		title: 'Testing' or 'Training' 
	"""

	for section in config_dict.keys():
		print(f"{section.upper()}:")
		section_data = config_dict[section]
		
		# Calculate max key length for alignment
		max_key_len = max(len(str(k)) for k in section_data.keys()) if section_data else 0
		
		for key, value in section_data.items():
			print(f"    {key:<{max_key_len}} : {value}")
		print()

def take_args(splits_available, models_available, make=False, default_project='WLASL-SLR'):
	parser = argparse.ArgumentParser(description='Train a model')
 
	#admin
	parser.add_argument('-ex', '--exp_no',type=int, help='Experiment number (e.g. 10)', required=True)
	parser.add_argument('-r', '--recover', action='store_true', help='Recover from last checkpoint')
	parser.add_argument('-ri', '--run_id', type=str, default=None, help=f'The run id to use (especially when also usign recover)')
	parser.add_argument('-m', '--model', type=str,help=f'One of the implemented models: {models_available}', required=True)
	parser.add_argument('-p', '--project', type=str, default=default_project, help='wandb project name')
	parser.add_argument('-sp', '--split',type=str, help='The class split (e.g. asl100)', required=True)
	parser.add_argument('-ed', 'enum_dir', action='store_true', help='set enumerate directories to on (for output)')
	#TODO: maybe add tags for wandb as parameters
	parser.add_argument('-t', '--tags', nargs='+', type=str,help='Additional wandb tags')

	#overides
	parser.add_argument('-c' , '--config_path', help='path to config .ini file')	
	parser.add_argument('-nf','--num_frames', type=int, help='video length')
	parser.add_argument('-fs', '--frame_size', type=int, help='width, height')
	parser.add_argument('-bs', '--batch_size', type=int,help='data_loader')
	parser.add_argument('-us', '--update_per_step', type=int, help='gradient accumulation')
	parser.add_argument('-ms', '--max_steps', type=int,help='gradient accumulation')
	parser.add_argument('-me', '--max_epoch', type=int,help='mixumum training epoch')
	
 
	args, _ = parser.parse_known_args()
	
	if args.split not in splits_available:
		raise ValueError(f"Sorry {args.split} not processed yet.\n\
			Currently available: {splits_available}")
	if args.model not in models_available:
		raise ValueError(f"Sorry {args.model} not implemented yet.\n\
			Currently available: {models_available}")
	
	exp_no = str(int(args.exp_no)).zfill(3)
	
	# args.model = model
	args.exp_no = exp_no
	args.root = '../data/WLASL/WLASL2000'
	args.labels = f'./preprocessed/labels/{args.split}'
	output = f'runs/{args.split}/{args.model}_exp{exp_no}'
	
	if not args.recover and args.enum_dir: #fresh run
		output = enum_dir(output, make)  	
 
	save_path = f'{output}/checkpoints'
	if not args.recover and args.enum_dir:
		args.save_path = enum_dir(save_path, make) 
	else:
		args.save_path = save_path
	
	# Set config path
	if args.config_path is None:
		args.config_path = f'./configfiles/{args.split}/{args.model}_{exp_no}.ini'
	
	# Load config
	arg_dict = vars(args)
	clean_dict = {}
	for key, value in arg_dict.items():
		if key == 'project' or key == 'tags': #I don't wandb redundant
			continue
		if value is not None:
			clean_dict[key] = value
	 
	tags = [
		args.split,
		args.model,
		f"exp-{exp_no}"
	]
	if args.tags is not None:
		tags.extend(args.tags)
 
	return clean_dict, tags, output, save_path, args.project


# def print_wandb_config(config):
	
	
	
if __name__ == '__main__':
	from models.pytorch_mvit import MViTv2S_basic 
	from models.pytorch_swin3d import Swin3DBig_basic
	# config_path = './configfiles/asl100/MViT_V2_S_000.ini'
	# config = configparser.ConfigParser()
	# config.read(config_path)    
	# for section in config.sections():
	# 	print(section)
	model_info = {
		"MViT_V2_S" : {
			"class" : MViTv2S_basic,
			"mean" : [0.45, 0.45, 0.45],
			"std" : [0.225, 0.225, 0.225]
		},
		"Swin3D_B" : {
			"class" : Swin3DBig_basic,
			"mean" : [0.485, 0.456, 0.406],
			"std" : [0.229, 0.224, 0.225],
		}
	}
	available_model = model_info.keys()
	available_splits = ['asl100', 'asl300']
	arg_dict, tags, output, save_path, project = take_args(available_splits, available_model, make=False)
	print(f'project selected: {project}')
 
	print_dict(arg_dict)
	print()
	config = load_config(arg_dict, verbose=True)
	# print_dict(config)
	print()
	print_config(config)
	print(tags)
	# run = wandb.init(
	# 	entity='ljgoodall2001-rhodes-university',
	# 	project='Testing-configs',
	# 	name=f"{config['admin']['model']}_{config['admin']['split']}_exp{config['admin']['split']}",
	# 	tags=tags,
	# 	config=config      
	# )
	# wconf = run.config
	# print(type(wconf))
 