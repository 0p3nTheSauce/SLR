#!/home/luke/miniconda3/envs/wlasl/bin/python

import json
import wandb
import os
import sys
from train import train_loop
from quewing import daemon, TEMP_PATH, print_v, clean_Temp, store_Temp, retrieve_Temp, get_run_id
import argparse
import time

def run_train(verbose=False):
  '''An easy to execute script for quewing'''
  info = retrieve_Temp(TEMP_PATH)
  
  if not info or 'run_id' in info.keys():
    #empty temp file
    raise ValueError(f'Tried to read next run from {TEMP_PATH} but it was empty')
    
  model_specifcs = info['model_info']
  config = info['config']
  entity = info['entity']
  project = info['project']
  tags = info['tags']
  output = info['output']
  save_path = info['save_path']

  admin = config['admin']
    
  #setup wandb run
  run_name = f"{admin['model']}_{admin['split']}_exp{admin['exp_no']}"

  if admin['recover']:
    if 'run_id' in config['admin']:
      run_id = config['admin']['run_id']
    else: 
      run_id = get_run_id(run_name, entity, project)
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
    #write run id to temp, so that daemon waits for it
    run_info = {'run_id': run.id, 'run_name': run.name, 'run_project': project}
    print_v("writing my id to temp file", verbose)
    store_Temp(TEMP_PATH, run_info)
    
  print_v(f"Run ID: {run.id}", verbose)
  print_v(f"Run name: {run.name}", verbose)  # Human-readable name
  print_v(f"Run path: {run.path}", verbose)  # entity/project/run_id format
  
  # Start training
  # os.makedirs(output, exist_ok=True)
  # os.makedirs(save_path, exist_ok=True)
  
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
    print(title) #just send smt to the terminal

def idle():
  #testing if blocking
  print(f'Starting at {time.strftime("%Y-%m-%d %H:%M:%S")}')
  for i in range(10):
    print(f'Idling: {i}')
    time.sleep(10)
  print(f'Finishign at {time.strftime("%Y-%m-%d %H:%M:%S")}')


def main():
  parser = argparse.ArgumentParser(prog='quefeather.py')
  subparsers = parser.add_subparsers(dest='mode', help='Operation mode', required=True)

  # Daemon subcommand
  daemon_parser = subparsers.add_parser('daemon', help='Run as daemon')

  # Worker subcommand  
  worker_parser = subparsers.add_parser('worker', help='Run as worker')

  # Separator subcommand (optional title)
  separator_parser = subparsers.add_parser('separator', help='Run as separator')
  separator_parser.add_argument('-t', '--title',type=str, nargs='?', help='Title for separator',const='',  default=None)

  parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose output')
  parser.add_argument('-poff', '--proceed_onFF', action='store_true', help='proceed on first fail')
  parser.add_argument('-dp', '--daemon_project', type=str, help='overide default project')
  args = parser.parse_args()
    

  if args.mode == 'daemon':
    daemon(args.verbose, proceed_onFF=args.proceed_onFF, project=args.daemon_project)
  elif args.mode == 'worker':
    run_train(args.verbose)
  elif args.mode == 'separator':
    print_seperator(args.title, args.verbose)
  else:
    print('htf did you get here?')
    
if __name__ == '__main__':
  main()
  # idle()
  

  