#!/home/luke/miniconda3/envs/wlasl/bin/python

import json
import wandb
import os
import sys
from train import train_loop
from quewing import get_run_id, daemon, TEMP_PATH, print_v
import argparse

def run_train(verbose=False):
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
      run_id = input("No run found automatically. Enter run ID, or leave blank to cancel: ")
      if run_id == '':
        print("Training cancelled")
        return
    print_v(f"Resuming run with ID: {run_id}", verbose)
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
    print_v(f"Starting new run with name: {run_name}", verbose)
    run = wandb.init(
      entity=entity,
      project=project,
      name=run_name,
      tags=tags,
      config=config      
    )
  print_v(f"Run ID: {run.id}", verbose)
  print_v(f"Run name: {run.name}", verbose)  # Human-readable name
  print_v(f"Run path: {run.path}", verbose)  # entity/project/run_id format
  
  # Start training
  os.makedirs(output, exist_ok=True)
  os.makedirs(save_path, exist_ok=True)
  train_loop(model_specifcs, run, recover=admin['recover'])
  run.finish()

def print_seperator(title='', verbose=True):
  '''This prints out a seperator between training runs
  
    Dual use to check if the tmux session is working,
    as a kind of dummy command 
  '''
  #i see this function possibly being used to probe tmux sessions, hence verbose option
  #is this the best way to do things? idk. Change it if you want, make title optional, idgaf
  if verbose:
    print("\n"*2,"-"*10,"\n")
    print(f"{title:^10}")
    print("\n","-"*10,"\n"*2)
  else:
    # print() #just send smt to the terminal
    pass

if __name__ == '__main__':

  parser = argparse.ArgumentParser(prog='quefeather.py')
  subparsers = parser.add_subparsers(dest='mode', help='Operation mode', required=True)

  # Daemon subcommand
  daemon_parser = subparsers.add_parser('daemon', help='Run as daemon')

  # Worker subcommand  
  worker_parser = subparsers.add_parser('worker', help='Run as worker')

  # Separator subcommand (optional title)
  separator_parser = subparsers.add_parser('separator', help='Run as separator')
  separator_parser.add_argument('-t', '--title',type=str, help='Title for separator', default='')

  parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose output')
  args = parser.parse_args()
    

  if args.mode == 'daemon':
    daemon(args.verbose)
  elif args.mode == 'worker':
    run_train(args.verbose)
  elif args.mode == 'separator':
    print_seperator(args.title, args.verbose)
  else:
    print('htf did you get here?')

  