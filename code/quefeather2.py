import sys
#!/home/luke/miniconda3/envs/wlasl/bin/python

import json
import wandb
import os
import sys
from train import train_loop
from quewing import daemon, TEMP_PATH, print_v, clean_Temp, store_Temp, retrieve_Temp, get_run_id
import argparse
import time
from quewing2 import worker
#TODO: would be great to be able to run scheduled tests as well
# from test import on_the_fly

  
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

def idle(message:str):
    #testing if blocking
    print(f'Starting at {time.strftime("%Y-%m-%d %H:%M:%S")}')
    for i in range(10):
        print(f'Idling: {i}')
        print(message)
        time.sleep(10)
    print(f'Finishign at {time.strftime("%Y-%m-%d %H:%M:%S")}')


def main():
    parser = argparse.ArgumentParser(prog='quefeather.py')
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode', required=True)

    # Daemon subcommand
    daemon_parser = subparsers.add_parser('daemon', help='Run as daemon')

    # Worker subcommand  
    worker_parser = subparsers.add_parser('worker', help='Run as worker')



    args, kwargs = parser.parse_known_args()

              
    # run_train(args.verbose)




  