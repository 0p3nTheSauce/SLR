import subprocess
import os
import json
import argparse
import wandb
import time

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
    run_info['state'] = 'exit_without_training'
  
  return run_info

def add_new_run(runs_path, arguments, verbose=False):
  all_runs = {
    'old_runs': [],
    'to_run': []
  }
  if os.path.exists(runs_path):
    with open(runs_path, 'r') as f:
      all_runs = json.load(f)
  all_runs['to_run'].append(arguments)
  
  with open(runs_path, 'w') as f:
    json.dump(all_runs, f, indent=2)
    
  if verbose:
    print(f"Added new run to {runs_path}:")
    print(json.dumps(arguments, indent=2))
  
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
  
  next_run = all_runs['to_run'].pop(0)
  
  all_runs['old_runs'].append(next_run)
  
  with open(runs_path, 'w') as f:
    json.dump(all_runs, f, indent=2)
    
  if verbose:
    print(f"Next run to run from {runs_path}:")
    print(json.dumps(next_run, indent=2))
      
  return next_run
  