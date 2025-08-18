from stopping import EarlyStopper
import wandb
import matplotlib.pyplot as plt
# import pytest
import random
# @pytest.fixture

from train import wait_for_run_completion
from utils import print_dict

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
  
  #simulate training
  # x_range = range(10)
  # x_range = [x*0.1 for x in x_range][1:]
  x_range = []
  x = 0
  # f = lambda x: 1 / x if x != 0 else 1
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
    
def test_wait_for_run_completion():
  # Test the wait_for_run_completion function
  # This is a mock test, as we cannot run actual wandb runs here
  entity = 'ljgoodall2001-rhodes-university'
  project = 'WLASL-SLR'
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


if __name__ == "__main__":
  test_early_stopping(mode='min')
  # test_wait_for_run_completion()
  # pytest.main([__file__])
  # test_stopper_save_and_load()
  