
import subprocess
import os
import sys
import json
import argparse
import wandb
import time

#local imports
# from train import PROJECT, ENTITY
ENTITY= 'ljgoodall2001-rhodes-university'
PROJECT = 'WLASL-SLR'
from configs import load_config, print_config, take_args

#constants
TEMP_PATH = './queTemp.json' #used by the worker process to get info on the next run
CODE_PATH = './code' #directory where other PATH files are stored
RUNS_PATH = './queRuns.json' #used by the daemon process to store future and past que
SCRIPT_PATH = './quefeather.py' #the script to run worker processes
RETRY_WAIT_TIME = 300 # 5 minutes, use for checking wandb status
SESSION = 'que_training'
DAEMON = 'daemon'
WORKER = 'worker' 

def print_v(item : str, verbose : bool) -> None:
	if verbose:
		print(item)

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


def handle_already_running(entity, project, check_interval=300, verbose=False, max_retries=5):
	'''make sure the program does not start training when there is already a run going.
	
		- Uses wandb api to achieve this. Waits for run to complete
	
		- Because subprocess.run is blocking, in theory we only need to do
			this if we already had something running
		
		Returns bool (shoud daemon continue):
			true : continue 
			false : break
	'''
	#NOTE this function might be better off just checking available GPU memory in future
	run_info = wait_for_run_completion(entity, project, check_interval, verbose)
	
	#check for wandb error first so that if it resolves itself, we drop into the rest of 
	#the conditions
	retry = 0
	while run_info['state'] == 'wandb_error' and retry < max_retries:
		print_v(f"Run {run_info['name']} (ID: {run_info['id']}) encountered a WandB error: {run_info.get('wandb_error', 'Unknown error')}", verbose)
		print_v(f"Retrying... ({retry}/{max_retries})", verbose)
		time.sleep(check_interval) #give it a quick break
		run_info = wait_for_run_completion(entity, project, check_interval, verbose)
		retry += 1
	
	if run_info['state'] == 'wandb_error':
		#max_retries exceeded, break
		print_v("Max retries reached. Exiting.", verbose)
		return False 
 
	elif run_info['state'] == 'finished':
		#ideal state, continue
		print_v(f"Run {run_info['name']} (ID: {run_info['id']}) completed successfully.", verbose)
		return True

	elif run_info['state'] == 'keyboard_interrupt':
		#user wants to close program, break
		print_v(f'Monitoring interrupted by user for run {run_info["name"]} (ID: {run_info["id"]})', verbose)
		print_v("Exiting without further action.", verbose)
		return False
	
	elif run_info['state'] in ['killed', 'crashed', 'failed']:
		#previous run didn't complete, could be an error, break
		print_v(f"Run {run_info['name']} (ID: {run_info['id']}) did not complete successfully. State: {run_info['state']}", verbose)
		return False

	else:
		#likely a wandb state i wasn't aware of but safety first, break.
		print_v(f"Run {run_info['name']} (ID: {run_info['id']}) has an unknown state. State: {run_info['state']}", verbose)
		return False

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

def wait_for_run_completion(entity, project, check_interval=300, verbose=False):
	'''Uses wandb Api to check and wait if the last run is still busy
		Returns a dictionary with keys:
			id : run id 
			name : run name
			state : different values:
							- if all was successful, the final state of the run after
			 					completion (finished, crashed, failed, killed)
							- otherwise, error codes based on what happened (
								no_runs -> no wandb runs found,
								wandb_error -> something went wrong with the api,
								keyboard_interrupt -> user intervention interupted monitoring
			created_at : time run was created at
 	'''
 
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
		print_v("No runs found.", verbose)
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
	
	print_v(f"Monitoring run: {run.name} (ID: {run.id})", verbose)
	
	# Wait for the run to finish
	
	run = api.run(f"{entity}/{project}/{run.id}")
	try:
		while run.state == 'running':
			print_v(f"Run state: {run.state}, Last checked at {time.strftime('%Y-%m-%d %H:%M:%S')}", verbose)
			time.sleep(check_interval)
			api = wandb.Api()  # Refresh API to get the latest state
			run = api.run(f"{entity}/{project}/{run.id}")
		run_info['state'] = run.state
	except wandb.Error as e:
		print_v(f"Error while waiting for run completion: {e}", verbose)
		run_info['state'] = 'wandb_error'
		run_info['wandb_error'] = str(e)
		return run_info
	except KeyboardInterrupt:
		print_v('', verbose)
		print_v("Monitoring interrupted by user.", verbose)
		run_info['state'] = 'keyboard_interrupt'
	
	return run_info

def add_new_run(runs_path, info, verbose=False):
	'''Adds a new entry to the runs.json file

		by adding info at the end of the  "to_run" list
	
		Returns all_runs
	'''
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
		
	print_v(f"Added new run to {runs_path}:", verbose)
	print_v(json.dumps(info, indent=2), verbose)
	
	return all_runs

def get_next_run(runs_path, verbose=False):
	'''gets the next entry next in line (FIFO)
 
		by taking the first entry of the "to_run" list

		Returns None if error, else next_run info
	'''

	all_runs = {
		'old_runs': [],
		'to_run': []
	}
	if not os.path.exists(runs_path):
		print_v(f"No runs file found at {runs_path}. Returning None.", verbose)
		return None
	
	with open(runs_path, 'r') as f:
		all_runs = json.load(f)
		
	if not all_runs['to_run']:
		print_v(f"No runs to run in {runs_path}. Returning None.", verbose)
		return None
	
	# next_run = all_runs['to_run'].pop(0)
	next_run = all_runs['to_run'][0] #lets not remove yet
	
	all_runs['old_runs'].append(next_run)
	
	with open(runs_path, 'w') as f:
		json.dump(all_runs, f, indent=2)
		
	
	print_v(f"Next run to run from {runs_path}:", verbose)
	print_v(json.dumps(next_run, indent=2), verbose)
			
	return next_run
	
def remove_old_run(runs_path, verbose=False):
	'''clean up after get_next_run

		by moving the first entry of the "to_run" list,
		to the last entry of the "old_runs" list
	
		Returns None if error, else old_run info
	'''
	all_runs = {
		'old_runs': [],
		'to_run': []
	}
	if not os.path.exists(runs_path):
		print_v(f"No runs file found at {runs_path}.", verbose)
		return None
	
	with open(runs_path, 'r') as f:
		all_runs = json.load(f)
		
	if not all_runs['to_run']:
		print_v(f"No runs to run in {runs_path}. Returning None.", verbose)
		return None
	
	old_run = all_runs['to_run'].pop(0)
	
	all_runs['old_runs'].append(old_run)
	
	with open(runs_path, 'w') as f:
		json.dump(all_runs, f, indent=2)
		
	
	print_v(f"Successfully move old run to  {runs_path}:", verbose)
	print_v(json.dumps(old_run, indent=2), verbose)
			
	return old_run

def store_Temp(temp_path, next_run):
	''''Stores the next run in the temp file for retrieval'''
	with open(temp_path, 'w') as f:
		json.dump(next_run, f, indent=2)

def clean_Temp(temp_path, verbose=False):
	'''cleans the temp file after run'''
	cleaned = {}
	with open(temp_path, 'w') as f:
		json.dump(cleaned, f)
	print_v('Cleaned temp fil', verbose)

def start(mode : str, sesh_name : str, script_path : str, 
					verbose : bool = False) \
       -> subprocess.CompletedProcess[bytes]:
	'''Starts a quefeather worker or daemon subprocess 
 
		Args:
			mode:  				worker or daemon
			sesh_name: 		tmux session name
			script_path: 	path to worker script (quefeather)
			verbose: 			verbose output from task
		Returns:
			result: 			from subprocess.run
		Raises:
			subprocess.CalledProcessError
	'''
	available_modes = ['daemon', 'worker']
	if mode not in available_modes:
		raise ValueError(f'{mode} not one of available modes: {available_modes}')
 
	feather_cmd = f'{script_path} {mode}' #./quefeather.py mode
	if verbose:
		feather_cmd += ' --verbose'
 
	tmux_cmd = [ 
		'tmux', 'send-keys', '-t', f'{sesh_name}:{mode}', 
		feather_cmd, 'Enter'
	]
	return subprocess.run(tmux_cmd, check=True)


def separate(mode: str, sesh_name : str, script_path : str,
					title : str = '',verbose : bool = False) \
	-> subprocess.CompletedProcess[bytes]:
	'''Prints a seperator in the teminal
 
		Args:
			mode:  				worker or daemon
			sesh_name: 		tmux session name
			script_path: 	path to worker script (quefeather)
			verbose: 			verbose output from task
		Returns:
			result: 			from subprocess.run
		Raises:
			subprocess.CalledProcessError
	'''
	feather_cmd = f'{script_path} separator'
	if title:
		feather_cmd += f' -t {title}'
	if verbose:
		feather_cmd += ' --verbose'
	tmux_cmd = [ 
		'tmux', 'send-keys', '-t', f'{sesh_name}:{mode}', 
		feather_cmd, 'Enter'
	]
	
	return subprocess.run(tmux_cmd, check=True)


def setup_tmux_session(sesh_name : str, dWndw_name : str, wWndw_name : str,
											 verbose : bool =False) \
	-> list[subprocess.CompletedProcess[bytes]]:
	'''Initialises a tmux session with a window for the daemon and worker processes

		Args:
			sesh_name: 		tmux session name
			dWndw_name:   daemon window name
			wWndw_name:   worker window name
			verbose: 			verbose output 
		Returns:
			result: 			list outputs from subprocess.run
		Raises:
			subprocess.CalledProcessError
	'''
 
	print_v("Setting up tmux environments", verbose)
 
	create_sesh_cmd = [
		'tmux', 'new-session', '-d', '-s', sesh_name, # -d for detach 
		'-n', f'{dWndw_name}'
 	]
	create_wWndw_cmd = [ #daemon window created in first command
		'tmux', 'new-window', '-t', sesh_name, '-n', wWndw_name 
	]
 
	return [subprocess.run(create_sesh_cmd, check=True),
         subprocess.run(create_wWndw_cmd, check=True)]
 
	# try:
	# 	 #we want errors if this fails
  #    subprocess.run(create_sesh_cmd, check=True)
	# 	print_v('daemon session created successfully', verbose)	
	# except subprocess.CalledProcessError as e:
	# 	print('Failed to create daemon session', verbose)
	# 	return e.stderr
	
	# try:
	# 	subprocess.run(create_wWndw_cmd, check=True)
	# 	print_v('worker session created successfully', verbose)	
	# except subprocess.CalledProcessError as e:
	# 	print('Failed to create daemon session', verbose)
	# 	return e.stderr #otherwise difficult to view errors

	# return 'ok'


	 
def check_tmux_session(sesh_name: str,dWndw_name: str, wWndw_name: str,
											 verbose=False):
	'''Verify that the tmux training session is set up

		Args:
			sesh_name: 		tmux session name
			dWndw_name:   daemon window name
			wWndw_name:   worker window name
			verbose: 			verbose output 
		Returns:
			result: 			list outputs from subprocess.run
		Raises:
			subprocess.CalledProcessError
 	'''
	#one of the only ways to get an error code out of tmux is to attatch to 
 	#something that doesnt exist.
	window_names = [dWndw_name, wWndw_name]
	results = []
	for win_name in window_names:
		tmux_cmd = ['tmux', 'has-session', '-t', f'{sesh_name}:{win_name}']
		results.append(subprocess.run(tmux_cmd, check=True, capture_output=verbose, text=verbose))
	return results

def daemon(verbose=True, proceed_after_fail=False, max_retries=5):
	# with open()
	runs_path = RUNS_PATH
	proceed = False
	outcomes = ['success', 'failure']
	advice = handle_already_running(ENTITY,
																 PROJECT,
																 check_interval = RETRY_WAIT_TIME,
																 verbose=verbose,
																 max_retries=max_retries)
	proceed = advice or proceed_after_fail
	if not advice and proceed_after_fail:
		print_v('Continuing despite errors with previous runs', verbose)
	
	while proceed:
		
		next_run = get_next_run(runs_path, verbose=True)
		if next_run is None:
			print_v("No more runs to execute.", verbose)
			break
		else:
			#stash run for quefeather to find
			store_Temp(TEMP_PATH, next_run)
			
			#move the next_run from "to_run" to "old_runs" 
			remove_old_run(runs_path, verbose=True)
	 	
		#start a process in a detached tmux window. 
		#session: que_worker, name: [num] worker  	
		#if worker exists, increments num 
		try:
			result = start('worker', SESSION, SCRIPT_PATH, verbose)
			print_v('worker started successfully', verbose)
		except subprocess.CalledProcessError as e:
			print("Daemon ran into an error when spawning the worker process: ")
			print(e.stderr)
			break
	 
		#remember to clean up the temp folder when finished
		clean_Temp(TEMP_PATH, verbose=True)
	
		
	print_v("Closing quewing daemon", verbose)
			
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
	if proceed.lower() == 'y':
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
		add_new_run(RUNS_PATH, info, verbose=verbose)
		if verbose:
			print(f"Run info saved to {RUNS_PATH}")
		# Start training
		os.makedirs(output, exist_ok=True)
		os.makedirs(save_path, exist_ok=True)
	else:
		if verbose:
			print("Training cancelled by user.")
		return

def clear_runs(runs_path, verbose=True, past_only=False):
	all_runs = {
		'old_runs': [],
		'to_run': []
	}
	if not os.path.exists(runs_path):
		raise ValueError(f'could not find runs file: {runs_path}')
	
	if past_only:
		with open(runs_path, 'w') as f:
			all_runs = json.load(f)
		all_runs['old_runs'] = []
	
	with open(runs_path, 'w') as f:
		json.dump(all_runs, f, indent=2)
	
	if verbose:
		print(f'Succesfully cleared {runs_path}')
	

def main():
	parser = argparse.ArgumentParser(description="Queuing training runs")
	parser.add_argument('-a', '--add', action='store_true', help='add new training command')
	parser.add_argument('-d', '--daemon', action='store_true', help='start the quewing daemon')
	parser.add_argument('-cl', '--clear', action='store_true', help='clear the runs file')	
	parser.add_argument('-co', '--clear_old', action='store_true', help='clear the old runs only')	
	args, other = parser.parse_known_args()
 
	if args.add:
		create_run(verbose=True)
		print("run added successfully")
	
	if args.daemon:
		start('daemon',SESSION,SCRIPT_PATH)
		print("daemon started successfully")
	
	if args.clear:
		clear_runs(RUNS_PATH, verbose=True)
	elif args.clear_old:
		clear_runs(RUNS_PATH, verbose=True, past_only=True)
	
if __name__ == '__main__':
	main()