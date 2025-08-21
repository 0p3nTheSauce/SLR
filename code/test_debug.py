from stopping import EarlyStopper
import wandb
import matplotlib.pyplot as plt
# import pytest
import random
# @pytest.fixture

from quewing import wait_for_run_completion
from utils import print_dict
import time
import subprocess
import argparse

def plot_simulated_training(x_range, f):
  x = x_range
  y = [f(i) for i in x]
  plt.plot(x, y)
  plt.xlabel('Epochs')
  plt.ylabel('Metric Value')
  plt.title('Simulated Training Progress')
  plt.savefig('./media/plot.png')
  # plt.show()

def sim_loss(x):
  # noise = random.uniform(-0.2, 0.2)
  # y = -x^2 + 2x + 1 
  noise = random.uniform(-0.002, 0.002)
  # Simulate a quadratic function with noise
  # return (-x**2 + 2*x + 1) + noise
  # Simulate a quadratic function with noise
  #
  if x < 200:
    return 1/(x**0.3) + noise 
  else:
    return 1/(200**0.3) + noise
  # return 1/(x**0.3)

def sim_acc(x):
  # Simulate a linear function with noise
  # return 0.5 * x + 1 + noise
  noise = random.uniform(-0.002, 0.002)
  if x < 200:
    return (x**0.3) + noise 
  else:
    return (200**0.3) + noise
  # return (x**0.3)

def test_early_stopping(mode='min'):
  # test with wandb run
  metric = ('val','loss')
  # mode = 'min'
  patience = 20
  min_delta = 0.01 #for fictional data, this is a large value
  # run = wandb.init(
  #   entity='ljgoodall2001-rhodes-university',
  #   project='Debugging',
  #   name=f"test_early_stopping",
  #   config={
  #     'metric': metric,
  #     'mode': mode,
  #     'patience': patience,
  #     'min_delta': min_delta
  #   }
  # )
  arg_dict = {
    'metric': metric,
    'mode': mode,
    'patience': patience,
    'min_delta': min_delta
  }
    
  
  # stopper = EarlyStopper(metric=metric,
  #                         mode=mode,
  #                         patience=patience,
  #                         min_delta=min_delta,
  #                         wandb_run=None)
  
  stopper = EarlyStopper(arg_dict=arg_dict, wandb_run=None)
  x_range = []
  x = 0
  f = lambda x: sim_loss(x)
  if mode == 'max':
    f = lambda x: sim_acc(x)
  max_epoch = 300
  score=0
  while not stopper.stop and x < max_epoch:
    x += 1
    x_range.append(x)
    score = f(x)
    print(f"Epoch: {x}, Score: {score}")
    if stopper.stop:
      print("Early stopping triggered.")
    stopper.step(score)
  
  print(f"Early stopping at epoch {x} with score {score}")
  print(f"Best score: {stopper.best_score}, Best epoch: {stopper.best_epoch}")
    
  plot_simulated_training(x_range, f)
  
  test_stopper_save_and_load(stopper, arg_dict)
  # wandb.finish()


def sim_train_script():
    
  arg_parser = argparse.ArgumentParser(description="Simulate a wandb run for waiting testing.")
  arg_parser.add_argument('-p', '--project', type=str, default='test_quewing',
                          help='Wandb project name')
  arg_parser.add_argument('-n', '--name', type=str, default='test_wait_for_run_completion',
                          help='Wandb run name')
  arg_parser.add_argument('-e', '--entity', type=str, default='ljgoodall2001-rhodes-university',
                          help='Wandb entity name')
  arg_parser.add_argument('-s', '--sleep', type=int, default=10,
                          help='Sleep time between epochs in seconds')
  arg_parser.add_argument('--early_stop', action='store_true',
                          help='Enable early stopping')
  arg_parser.add_argument('--patience', type=int, default=20,
                          help='Number of epochs to wait for improvement before stopping')
  arg_parser.add_argument('--min_delta', type=float, default=0.01,
                          help='Minimum change in the monitored metric to qualify as an improvement')

  args = arg_parser.parse_args()

  print("Starting simulation with the following parameters:")
  print(f"Project: {args.project}, Name: {args.name}, Entity: {args.entity}, Sleep: {args.sleep}, Early Stop: {args.early_stop}")
  
  simulate_wandb_run(
      mode='min',
      entity=args.entity,
      project=args.project,
      name=args.name,
      sleep=args.sleep,
      early_stop=args.early_stop,
      patience=args.patience,
      min_delta=args.min_delta
  )
  
  print("Simulation completed.")

def simulate_wandb_run(mode='min',entity='ljgoodall2001-rhodes-university',
                       project='Debugging',
                       name='test_early_stopping',
                       sleep=10, early_stop=False, patience=20, min_delta=0.01):
  # test with wandb run
  metric = ('val','loss')
  # mode = 'min'
  run = wandb.init(
    entity=entity,
    project=project,
    name=name,
    config={
      'metric': metric,
      'mode': mode,
      'patience': patience,
      'min_delta': min_delta
    }
  )
  arg_dict = {
    'on': early_stop,
    'metric': metric,
    'mode': mode,
    'patience': patience,
    'min_delta': min_delta
  }
    
  stopper = EarlyStopper(arg_dict=arg_dict, wandb_run=run)
  x_range = []
  x = 0
  f = lambda x: sim_loss(x)
  if mode == 'max':
    f = lambda x: sim_acc(x)
  max_epoch = 300
  score=0
  while not stopper.stop and x < max_epoch:
    x += 1
    x_range.append(x)
    score = f(x)
    print(f"Epoch: {x}, Score: {score}")
    run.log({f"{metric[0]}_{metric[1]}": score})
    if stopper.stop:
      print("Early stopping triggered.")
    stopper.step(score)
    time.sleep(sleep)  # Simulate time taken for training
  
  print(f"Early stopping at epoch {x} with score {score}")
  print(f"Best score: {stopper.best_score}, Best epoch: {stopper.best_epoch}")
    
    
def test_wait_for_run_completion():
  # Test the wait_for_run_completion function
  # This is a mock test, as we cannot run actual wandb runs here
  entity = 'ljgoodall2001-rhodes-university'
  project = 'test_quewing'
  
  # Simulate a wandb run
  
  
  run_info = wait_for_run_completion(entity, project, check_interval=5)
  if run_info is None:
    print("No run found or run is still in progress.")
    return
  print("Run information:")
  print_dict(run_info)



def test_stopper_save_and_load(stopper, arg_dict):
  # Test the EarlyStopper save and load functionality
  
  # stopper = EarlyStopper(arg_dict=arg_dict, wandb_run=None)
    
  # Save the stopper state
  stopper_dict = stopper.state_dict()
  stopper = None
  stopper = EarlyStopper(arg_dict=arg_dict, wandb_run=None)
  # Load the stopper state
  stopper.load_state_dict(stopper_dict)
  
  # Check if the state is restored correctly
  assert stopper.best_score is not None, "Best score should not be None after loading state"
  print(f"Restored best score: {stopper.best_score}, Best epoch: {stopper.best_epoch}, Counter: {stopper.counter}")

def tmux_session():
  from quewing import setup_tmux_session, check_tmux_session, separate
  # result = check_tmux_session('test', 'd', 'w', True)
  # if result != 'ok':
  #   setup_tmux_session('test', 'd', 'w', True)
  session = 'test'
  try:
    result = check_tmux_session('test', 'd', 'w')
  except subprocess.CalledProcessError as e:
    # print(e.stderr)
    if e.stderr.strip() != f"can't find session: {session}":
      print("'",e.stderr,"'")
    else:
      print("check completed successfully")
    setup_tmux_session('test', 'd', 'w')
  print(separate('d', 'test', './quefeather.py','Testing', True)  )


if __name__ == "__main__":
  # test_early_stopping(mode='min')
  # test_wait_for_run_completion()
  # pytest.main([__file__])
  # test_stopper_save_and_load()
  tmux_session()
  
