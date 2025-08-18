
import subprocess
import os
import sys
import json
import argparse
import wandb
import time

#local imports
from train import PROJECT, ENTITY
from configs import load_config, print_config, take_args

#constants
TEMP_PATH = './queTemp.json'
RUNS_PATH = './queRuns.json'
SCRIPT_PATH = './quefeather.py'
RETRY_WAIT_TIME = 300 # 5 minutes

def get_run_id(run_name, entity, project):
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
		for idx, run_id in enumerate(ids):
			print(f"{idx}: {run_id}")
		choice = input("Select run idx to use: ")
		try:
			choice = int(choice)
			if 0 <= choice < len(ids):
				return ids[choice]
			else:
				print("Invalid choice, returning None")
				return None
		except ValueError:
			print("Invalid input, returning None")
	else:
		return ids[0]

def list_runs(entity, project):
	# import wandb

	api = wandb.Api()
	runs = api.runs(f"{entity}/{project}")

	for run in runs:
			print(f"Run ID: {run.id}")
			print(f"Run name: {run.name}")
			print(f"State: {run.state}")
			print(f"Created: {run.created_at}")
			print("---")

def wait_for_run_completion(entity, project, check_interval=300):
	api = wandb.Api()
	# if last:
	#   run_id
	no_runs_dict = {
		'id': None,
		'name': None,
		'state': 'no_runs',
		'created_at': None
	}
	
	runs = api.runs(f"{entity}/{project}")
	if not runs:
		print("No runs found.")
		return no_runs_dict
	
	# Get the most recent run
	run = runs[-1]  # Assuming the last run is the one we want to monitor
	run_info = {
		'id': run.id,
		'name': run.name,
		'state': run.state,
		'created_at': run.created_at,
	}
	
	if run.state in ["finished", "crashed", "failed"]:
		return run_info
	
	print(f"Monitoring run: {run.name} (ID: {run.id})")
	
	# Wait for the run to finish
	
	run = api.run(f"{entity}/{project}/{run.id}")
	try:
		while run.state == 'running':
			print(f"Run state: {run.state}, Last checked at {time.strftime('%Y-%m-%d %H:%M:%S')}")
			time.sleep(check_interval)
			api = wandb.Api()  # Refresh API to get the latest state
			run = api.run(f"{entity}/{project}/{run.id}")
		run_info['state'] = run.state
	except wandb.Error as e:
		print(f"Error while waiting for run completion: {e}")
		run_info['state'] = 'wandb_error'
		run_info['wandb_error'] = str(e)
		return run_info
	except KeyboardInterrupt:
		print()
		print("Monitoring interrupted by user.")
		run_info['state'] = 'key_board_interrupt'
	
	return run_info

def add_new_run(runs_path, info, verbose=False):
	all_runs = {
		'old_runs': [],
		'to_run': []
	}
	if os.path.exists(runs_path):
		with open(runs_path, 'r') as f:
			all_runs = json.load(f)
	all_runs['to_run'].append(info)
	
	with open(runs_path, 'w') as f:
		json.dump(all_runs, f, indent=2)
		
	if verbose:
		print(f"Added new run to {runs_path}:")
		print(json.dumps(info, indent=2))
	
	return all_runs

def get_next_run(runs_path, verbose=False):
	all_runs = {
		'old_runs': [],
		'to_run': []
	}
	if not os.path.exists(runs_path):
		if verbose:
			print(f"No runs file found at {runs_path}. Returning None.")
		return None
	
	with open(runs_path, 'r') as f:
		all_runs = json.load(f)
		
	if not all_runs['to_run']:
		if verbose:
			print(f"No runs to run in {runs_path}. Returning None.")
		return None
	
	# next_run = all_runs['to_run'].pop(0)
	next_run = all_runs['to_run'][0] #lets not remove yet
	
	all_runs['old_runs'].append(next_run)
	
	with open(runs_path, 'w') as f:
		json.dump(all_runs, f, indent=2)
		
	if verbose:
		print(f"Next run to run from {runs_path}:")
		print(json.dumps(next_run, indent=2))
			
	return next_run
	
def remove_old_run(runs_path, verbose=False):
	all_runs = {
		'old_runs': [],
		'to_run': []
	}
	if not os.path.exists(runs_path):
		if verbose:
			print(f"No runs file found at {runs_path}.")
		return None
	
	with open(runs_path, 'r') as f:
		all_runs = json.load(f)
		
	if not all_runs['to_run']:
		if verbose:
			print(f"No runs to run in {runs_path}. Returning None.")
		return None
	
	old_run = all_runs['to_run'].pop(0)
	
	all_runs['old_runs'].append(old_run)
	
	with open(runs_path, 'w') as f:
		json.dump(all_runs, f, indent=2)
		
	if verbose:
		print(f"Next run to run from {runs_path}:")
		print(json.dumps(old_run, indent=2))
			
	return old_run

def start(proc_type, session_name='training', script_path=SCRIPT_PATH):
	'''proc_type can be "train" or "daemon'''
	tmux_cmd = [
		"tmux", "new-window",
		"-t", session_name,
		"-n", f"que_{proc_type}",
		script_path, proc_type 
	]
	subprocess.run(tmux_cmd)

def daemon():
	runs_path = RUNS_PATH
	retries = 0
	max_retries = 5
	while True:

		run_info = wait_for_run_completion(ENTITY, PROJECT, check_interval=300)
		if run_info['state'] == 'finished':
			print(f"Run {run_info['name']} (ID: {run_info['id']}) completed successfully.")
			remove_old_run(runs_path, verbose=True)
		elif run_info['state'] == 'wandb_error':
			print(f"Run {run_info['name']} (ID: {run_info['id']}) encountered a WandB error: {run_info.get('wandb_error', 'Unknown error')}")
			print("Retrying...")
			time.sleep(RETRY_WAIT_TIME)
			retries += 1
			if retries >= max_retries:
				print("Max retries reached. Exiting.")
				break
			continue
		elif run_info['state'] == 'key_board_interrupt':
			print(f'Monitoring interrupted by user for run {run_info["name"]} (ID: {run_info["id"]})')
			print("Exiting without further action.")
			break
		else: #killed, crashed or failed
			print(f"Run {run_info['name']} (ID: {run_info['id']}) did not complete successfully. State: {run_info['state']}")
			proceed = input("Do you want to proceed with the next run? (y/n): ")
			if proceed.lower() != 'y':
				print('removing failed run')
				remove_old_run(runs_path, verbose=True)
			else:
				print("Exiting without further action.")
				break
		
		next_run = get_next_run(runs_path, verbose=True)
		if next_run is None:
			print("No more runs to execute.")
			break
		
		start('train')
	
		retries = 0
print("Closing quewing daemon")
			
def create_run(verbose=True):
	with open('./wlasl_implemented_info.json') as f:
		info = json.load(f)
	available_splits = info['splits']
	model_info = info['models']
	
	arg_dict, tags, output, save_path, project = take_args(available_splits, model_info.keys(),
																												 default_project=PROJECT)
	
	config = load_config(arg_dict, verbose=True)
	
	print_config(config)

	proceed = input("Confirm: y/n: ")
	if proceed.lower() != 'y':
		if verbose:
			print("Saving run info ")
		info = {
			'model_info': model_info,
			'config': config,
			'entity': ENTITY,
			'project': project,
			'tags': tags,
			'output': output,
			'save_path': save_path
		}
		add_new_run(TEMP_PATH, info, verbose=verbose)
		if verbose:
			print(f"Run info saved to {TEMP_PATH}")
		# Start training
		os.makedirs(output, exist_ok=True)
		os.makedirs(save_path, exist_ok=True)
	else:
		if verbose:
			print("Training cancelled by user.")
		return

def main():
	parser = argparse.ArgumentParser(description="Queuing training runs")
	parser.add_argument('-a', '--add', action='store_true', help='add new training command')
	parser.add_argument('-d', '--daemon', action='store_true', help='start the quewing daemon')
	
	args, other = parser.parse_known_args()
	
	if args.add:
		create_run()
	
	if args.daemon:
		start('daemon')