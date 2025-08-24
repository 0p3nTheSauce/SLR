
import subprocess
import os
import sys
import json
import argparse
import wandb
import time

from typing import Union, Optional, Any
#local imports
# from train import PROJECT, ENTITY
ENTITY= 'ljgoodall2001-rhodes-university'
PROJECT = 'WLASL - SLR'
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
		
def handle_already_running(entity, project, check_interval=300,
													 verbose=False, max_retries=5, run_id=None):
	'''make sure the program does not start training when there is already a run going. Checks last available run
		unless id is specified. Takes a few seconds to update api, so if in between runs either use id, or force
		to sleep for a bit to give time to check runs

		Args:
			entity: wandb entity
			project: wandb project
			check_interval: how long to sleep between checks
			verbose: verbose output
			max_retries: when receiving a wandb error
			id: run id to monitor (otherwise last)
		
		Returns bool (shoud daemon continue):
			true : continue 
			false : break
	'''
	#NOTE this function might be better off just checking available GPU memory in future
	#NOTE I want to remove the retry functionality
	run_info = wait_for_run_completion(entity, project, check_interval, verbose, run_id)
	
	#check for wandb error first so that if it resolves itself, we drop into the rest of 
	#the conditions
	retry = 0
	while run_info['state'] == 'wandb_error' and retry < max_retries:
		print_v(f"Retrying... ({retry}/{max_retries})", verbose)
		time.sleep(check_interval) #give it a quick break
		run_info = wait_for_run_completion(entity, project, check_interval, verbose, run_id)
		retry += 1
	
	#check for the run not found error next, as there is a chance it needs a second for the wandb api to load
	retry = 0
	while run_info['state'] == 'run_not_found' and retry < max_retries:
		print_v(f"Retrying... ({retry}/{max_retries})", verbose)
		time.sleep(check_interval) #give it a quick break
		run_info = wait_for_run_completion(entity, project, check_interval, verbose, run_id)
		retry += 1
	
	if run_info['state'] == 'no_runs':
	 	# something wrong with entity or project
		return False
	
	elif run_info['state'] == 'wandb_error':
		#max_retries exceeded, break
		print_v("Could not resolve errors with wandb", verbose)
		print_v("Max retries reached. Exiting.", verbose)
		return False
	
	elif run_info['state'] == 'run_not_found':
		#max_retries exceeded, break
		print_v(f"Could not issue with id: {run_id}", verbose)
		print_v("Max retries reached. Exiting.", verbose)
		return False
 
	elif run_info['state'] == 'finished':
		#ideal state, continue
		print_v(f"Run {run_info['name']} (ID: {run_info['id']}) completed successfully.", verbose)
		return True

	elif run_info['state'] == 'keyboard_interrupt':
		#user wants to close program, break
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

def run_present(run_id, runs):
	for run in runs:
		if run.id == run_id:
			return True
	return False

def wait_for_run_completion(entity, project, check_interval=300, verbose=False, run_id=None):
	'''Uses wandb Api to check and wait if the last run is still busy. Checks last available run if id not specified.
		
		Args:
			entity: wandb entity
			project: wandb project
			check_interval: how long to sleep between checks
			verbose: verbose output
			max_retries: when receiving a wandb error
			id: run id to monitor (otherwise last)
	
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
		print_v(f"No runs found for {entity}/{project}", verbose)
		return no_runs_dict
	
	# Get the most run specified
	if run_id is not None:
		if run_present(run_id, runs):
			run = api.run(f"{entity}/{project}/{run_id}")
		else:
			print_v(f'Could not find run with id: {run_id}', verbose)
			no_runs_dict['state'] = 'run_not_found'
			return no_runs_dict
	else:
		run = runs[-1]  # last available run
	
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
	
	try:
		while run.state == 'running':
			print_v(f"Run state: {run.state}, Last checked at {time.strftime('%Y-%m-%d %H:%M:%S')}", verbose)
			time.sleep(check_interval)
			api = wandb.Api()  # Refresh API to get the latest state
			run = api.run(f"{entity}/{project}/{run.id}")
		run_info['state'] = run.state
	except wandb.Error as e:
		print_v(f"Wandb Error while waiting for run completion: {e}", verbose)
		run_info['state'] = 'wandb_error'
		run_info['wandb_error'] = str(e)
		return run_info
	except KeyboardInterrupt:
		print_v('', verbose)
		print_v(f"Monitoring of run: {run.name} (ID: {run.id}) interrupted by user.", verbose)
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
	# print_v(json.dumps(info, indent=2), verbose) too verbosed
	
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
	# print_v(json.dumps(next_run, indent=2), verbose) 				hella verbose
			
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
	# print_v(json.dumps(old_run, indent=2), verbose)									hella verbose
			
	return old_run

def store_Temp(temp_path, next_run):
	''''Stores dictionary in the temp file for retrieval'''
	with open(temp_path, 'w') as f:
		json.dump(next_run, f, indent=2)
	
def retrieve_Temp(temp_path):
	''''retrieves dictionary in the temp file for retrieval'''
	with open(temp_path, 'r') as f:
		data = json.load(f)
	return data 

def clean_Temp(temp_path: str, verbose: bool=False) -> None:
	'''cleans the temp file after run'''
	cleaned = {}
	with open(temp_path, 'w') as f:
		json.dump(cleaned, f)
	print_v('Cleaned temp file', verbose)

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
 
	feather_cmd = f'{script_path}' #./quefeather.py mode
	if verbose:
		feather_cmd += ' -v'
	feather_cmd += f' {mode}'
 
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
	feather_cmd = f'{script_path}'
 
	if verbose:
		feather_cmd += ' -v'
	
	feather_cmd += ' separator'
 
	if title:
		feather_cmd += f' -t "{title}"'
	
	tmux_cmd = [ 
		'tmux', 'send-keys', '-t', f'{sesh_name}:{mode}', 
		feather_cmd, 'Enter'
	]
	
	return subprocess.run(tmux_cmd, check=True)


def setup_tmux_session(sesh_name : str, dWndw_name : str, wWndw_name : str) \
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
 
	create_sesh_cmd = [
		'tmux', 'new-session', '-d', '-s', sesh_name, # -d for detach 
		'-n', f'{dWndw_name}'
 	]
	create_wWndw_cmd = [ #daemon window created in first command
		'tmux', 'new-window', '-t', sesh_name, '-n', wWndw_name 
	]
	#NOTE may need to  add capture_output=True, and Text=True
	return [subprocess.run(create_sesh_cmd, check=True),
				 subprocess.run(create_wWndw_cmd, check=True)]
	 
def check_tmux_session(sesh_name: str,dWndw_name: str, wWndw_name: str):
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
		results.append(subprocess.run(tmux_cmd, check=True, capture_output=True, text=True))
	return results

def daemon(verbose: bool=True, max_retries: int=5) \
	-> None:
	'''Function for the queue daemon process. The function works in a fetch execute repeat
	cycle. The function reads runs from the queRuns.json file, then writes them to the
	queTemp.json file for the worker process to find. The daemon spawns the worker, then 
	waits for it to complete before proceeding. It waits for completed by checking the status of the wandb API.
	
	Args:
		verbose: verbose output
		max_retries: when getting a wandb api error  
	'''
	runs_path = RUNS_PATH
	proceed = handle_already_running(ENTITY, PROJECT,
																 check_interval = RETRY_WAIT_TIME,
																 verbose=verbose,
																 max_retries=max_retries)
	
	while proceed:
		
		next_run = get_next_run(runs_path, verbose=True)
		if next_run is None:
			print_v("No more runs to execute.", verbose)
			break
		else:
			#stash run for quefeather to find
			store_Temp(TEMP_PATH, next_run) #worker cleans

			#move the next_run from "to_run" to "old_runs" 
			remove_old_run(runs_path, verbose=True)
	 	
		#start a process in a detached tmux window. 
		#session: que_worker, name: [num] worker  	
		#if worker exists, increments num 
		try:
			result = start('worker', SESSION, SCRIPT_PATH, verbose) #this actually not blocking 
			print_v('worker started successfully', verbose)
			time.sleep(10) #just give it a sec so that the run_id is written to file
		except subprocess.CalledProcessError as e:
			print("Daemon ran into an error when spawning the worker process: ")
			print(e.stderr)
			break
		
		#worker writes its id to the temp file, get for monitoring
		run_info = retrieve_Temp(TEMP_PATH)
		if 'run_id' in run_info.keys():
			print_v("Waiting for run completion: \n", verbose)
			#because start is non blocking, we need to wait for the run to finish
			proceed = handle_already_running(ENTITY, PROJECT,
																	check_interval = RETRY_WAIT_TIME,
																	verbose=verbose,
																	max_retries=max_retries,
								 run_id=run_info['run_id'])
			if proceed:
				print_v("Run completed successfully: \n", verbose)
			else:
				print_v("Ran into an error waiting for run to complete, exiting", verbose)
				break
		else:
			print_v("After starting new run, could not find its id in Temp", verbose)
			break

		#print a seperator to the output to seperate runs
		try:
			title = 'Starting new worker process'
			result = separate('worker', SESSION, SCRIPT_PATH, title, verbose)
		except subprocess.CalledProcessError as e:
			print("Daemon ran into an error when writing the separator: ")
			print(e.stderr)
			break
		
	print_v("Closing quewing daemon", verbose)
			
def create_run(verbose=True):
	'''Add an entry to the runs file'''
	with open('./wlasl_implemented_info.json') as f:
		info = json.load(f)
	available_splits = info['splits']
	model_info = info['models']
	
	arg_dict, tags, output, save_path, project = take_args(available_splits, model_info.keys(),
																												 default_project=PROJECT)
	
	config = load_config(arg_dict, verbose=True)
	
	print_config(config)

	model_specifics = model_info[config['admin']['model']]
 
	proceed = input("Confirm: y/n: ")
	if proceed.lower() == 'y':
		if verbose:
			print("Saving run info ")
		info = {
			'model_info': model_specifics,
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

def remove_run(runs_path:str, verbose:bool=False, idx:Optional[int]=None) -> dict[str,Any]:
	'''remove a run from rus'''
	all_runs = {
		'old_runs': [],
		'to_run': []
	}

	def _rem(all_runs, idx):
		try:
			rem = all_runs['to_run'].pop(idx)
			return rem
		except IndexError:
			print(f"Index: {idx} out of range for to run of length {len(all_runs['to_run'])}")
			return {}	

	with open(runs_path, 'r') as f:
		all_runs = json.load(f)
	
	rem = {}
 
	if not all_runs:
		print("no runs available")
		return rem

	if idx is not None:
		rem = _rem(all_runs, idx)

	if not rem:
		for i, run in enumerate(all_runs['to_run']):
			admin = run['config']['admin']
			print(f"{admin['config_path']} : {i}")
		if len(all_runs['to_run']) == 0:
			print("runs are finished")
		else:
			idx = int(input("select index: "))
			rem = _rem(all_runs, idx)
	
	if rem:
		with open(runs_path, 'w') as f:
			json.dump(all_runs, f, indent=2)
		print_v(f"run removed: {rem['config']['admin']['config_path']}", verbose)
	else:
		print_v(f"no run removed", verbose)
	
	return rem

def clear_runs(runs_path, verbose=True, past_only=False, future_only=False):
	'''reset the runs file'''
	all_runs = {
		'old_runs': [],
		'to_run': []
	}
	if not os.path.exists(runs_path):
		raise ValueError(f'could not find runs file: {runs_path}')
	
	if past_only:
		print_v("only clearing old runs", verbose)
		with open(runs_path, 'r') as f:
			all_runs = json.load(f)
		all_runs['old_runs'] = []
	
	if future_only:
		print_v("only clearing new runs", verbose)
		with open(runs_path, 'r') as f:
			all_runs = json.load(f)
		all_runs['to_run'] = []
	
	with open(runs_path, 'w') as f:
		json.dump(all_runs, f, indent=2)
	
	print_v(f'Succesfully cleared {runs_path}', verbose)

def return_old(runs_path: str, verbose: bool=False) -> None:
	'''Return the runs in old_runs to to_run. Adds them at the beggining of to_runs.
	 
			Args:
				runs_path: path to runs file
				verbose: verbose output
	'''
	all_runs = {
		'old_runs': [],
		'to_run': []
	}
	
	with open(runs_path, 'r') as f:
		all_runs = json.load(f)
	
	curr_to_run = all_runs['to_run'] if all_runs['to_run'] else [] 
	curr_old_run = all_runs['old_runs'] if all_runs['old_runs'] else [] 
	
	curr_to_run.extend(curr_old_run)
	all_runs['to_run'] = curr_to_run
	all_runs['old_runs'] = []

	with open(runs_path, 'w') as f:
		json.dump(all_runs, f, indent=2)
 
	print_v(f'Successfully moved old_runs to to_run', verbose)
	

def check_err(err, session):
	against = f"can't find session: {session}".strip().split()
	for i,c in enumerate(err.strip().split()):
		if against[i] != c:
			print(f"against: {against[i]}, err: {c}")
			return False
	return True

def list_configs(runs_path:str) -> list[str]:
	
	conf_list = []
	with open(runs_path, 'r') as f:
		all_runs = json.load(f)
	
	if all_runs:
		if len(all_runs['to_run']) == 0:
			print("runs are finished")
		for run in all_runs['to_run']:
			admin = run['config']['admin']
			print(admin['config_path'])
			conf_list.append(admin['config_path'])
	else:
		print("no runs available")
	
	return conf_list

def main():
	#NOTE chose to make verbose true by default when running this script
	#NOTE these arguments have to be compatible with take_args if creating run
	parser = argparse.ArgumentParser(description="quewing.py")
	parser.add_argument('-a', '--add', action='store_true', help='add new training command')
	parser.add_argument('-d', '--daemon', action='store_true', help='start the quewing daemon')
	parser.add_argument('-cr', '--clear', action='store_true', help='clear the runs file')	
	parser.add_argument('-co', '--clear_old', action='store_true', help='clear the old runs only')
	parser.add_argument('-cn', '--clear_new', action='store_true', help='clear the new runs only')
	parser.add_argument('-ct', '--clear_temp', action='store_true', help='clear the temp file')	
	parser.add_argument('-rr', '--remove_run', nargs='?',type=str, help='remove a run from to_run', const='ask_me', default=None)
	parser.add_argument('-sh', '--silent', action='store_true', help='Turn off verbose output')
	parser.add_argument('-sm', '--separate_mode', type=str, help='Send a seperator to the "mode" process')
	parser.add_argument('-ti', '--title', type=str, nargs='?', help="title for seperate_mode used", const="Calling next process", default=None)
	parser.add_argument('-ro', '--return_old', action='store_true', help='return old runs to new')
	parser.add_argument('-lc', '--list_configs', action='store_true', help='list the config files used in runs file')
	args, other = parser.parse_known_args()

	#invert
	verbose = not args.silent
 
	if args.add:
		create_run(verbose=verbose)
		print("run added successfully")
	
	if args.daemon:
		create = False
		#first check that the tmux session is set up
		try:
			results = check_tmux_session(SESSION, DAEMON, WORKER)
			print_v('Tmux session validated', verbose)
		except subprocess.CalledProcessError as e:
			# if e.stderr.strip() != f"can't find session: {SESSION}":
			if check_err(e.stderr, SESSION):
				#hasn't been created yet
				print_v("Setting up tmux environments", verbose)
				create = True
			else:
				print("Ran into an unexpected issue when checking tmux sessions: ")
				print(e.stderr)
				return
		
		#first time use, or after restart
		if create:
			try:
				result = setup_tmux_session(SESSION, DAEMON, WORKER)
			except subprocess.CalledProcessError as e:
				print("Ran into an unexpected issue when creating tmux sessions: ")
				print(e.stderr)
				return

		#run the daemon
		try:
			result = start('daemon', SESSION, SCRIPT_PATH, verbose)
			print_v("daemon started successfully", verbose)
		except subprocess.CalledProcessError as e:
			print("Ran into an unexpected issue when starting the daemon process: ")
			print(e.stderr)
			return
	
	if args.clear:
		clear_runs(RUNS_PATH, verbose=True)
	elif args.clear_old:
		clear_runs(RUNS_PATH, verbose=True, past_only=True)
	elif args.clear_new:
		clear_runs(RUNS_PATH, verbose=True, future_only=True)
	
	if args.clear_temp:
		clean_Temp(TEMP_PATH, verbose)
	
	if args.return_old:
		return_old(RUNS_PATH, verbose)
	
	if args.list_configs:
		list_configs(RUNS_PATH)
	
	if args.separate_mode:
		try:
			result = separate(args.separate_mode, SESSION, SCRIPT_PATH, args.title, verbose)
			print_v(f'Separator: "{args.title}" sent to {args.separate_mode}', verbose)
		except subprocess.CalledProcessError as e:
			print(f"Ran into an unexpected issue when sending separator: {args.title} to {args.separate_mode}")
			print(e.stderr)
			return
	
	if args.remove_run:
		if args.remove_run == 'ask_me':
			remove_run(RUNS_PATH,verbose)
		else:
			remove_run(RUNS_PATH,verbose, int(args.remove_run))


if __name__ == '__main__':
	main()