#!/home/luke/miniconda3/envs/wlasl/bin/python

#NOTE might not need shebang, as apparently spawned processes
#inherit the environment of their parents

import json
import wandb
import os
import sys
from train import train_loop
from quewing import get_run_id, daemon, TEMP_PATH

def run_train():
  '''An easy to execute script for quewing'''
  with open(TEMP_PATH) as f:
    info = json.load(f)
  model_info = info['model_info'] #TODO: this takes up space
  #can probably do:
  #model_specifics = info['model_specifics']
  config = info['config']
  entity = info['entity']
  project = info['project']
  tags = info['tags']
  output = info['output']
  save_path = info['save_path']

  admin = config['admin']
  model_specifcs = model_info[admin['model']]
  run_id = None #only used if we are recovering a run
    
  #setup wandb run
  run_name = f"{admin['model']}_{admin['split']}_exp{admin['exp_no']}"
  if admin['recover']:
    if run_id is None: 
      run_id = get_run_id(run_name, entity, project)
    if run_id is None:
      run_id = ("No run found automatically. Enter run ID, or leave blank to cancel: ")
      if run_id == '':
        print("Training cancelled")
        return
    print(f"Resuming run with ID: {run_id}")
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
    print(f"Starting new run with name: {run_name}")
    run = wandb.init(
      entity=entity,
      project=project,
      name=run_name,
      tags=tags,
      config=config      
    )
  print(f"Run ID: {run.id}")
  print(f"Run name: {run.name}")  # Human-readable name
  print(f"Run path: {run.path}")  # entity/project/run_id format
  
  # Start training
  os.makedirs(output, exist_ok=True)
  os.makedirs(save_path, exist_ok=True)
  train_loop(model_specifcs, run, recover=admin['recover'])
  run.finish()



if __name__ == '__main__':
  if len(sys.argv) == 2:
    if sys.argv[1] == 'daemon':
      daemon()
    elif sys.argv[1] == 'train':
      run_train()
    else:
      print("Usage: python quefeather.py [daemon|train]")
      print("  daemon: Run the queuing daemon")
      print("  train: Start a training run")
      sys.exit(1)
  else:
    print("Usage: python quefeather.py [daemon|train]")
    print("  daemon: Run the queuing daemon")
    print("  train: Start a training run")
    sys.exit(1)
  sys.exit(0)
  