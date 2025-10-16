from stopping import EarlyStopper
import wandb
import matplotlib.pyplot as plt
# import pytest
import random
# @pytest.fixture

from utils import print_dict
import time
import subprocess
import argparse
from pathlib import Path
import re  
import json
import test
from collections import Counter, defaultdict
from typing import Optional



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



def test_blocking():
	#have a feeling the run.subprocess is 'blocking' in the sences
	print(f'Starting at {time.strftime("%Y-%m-%d %H:%M:%S")}')
	# subprocess.run(['tmux', 'send-keys', '-t', 'test:q', './quefeather.py', 'Enter'], check=True) #non-blocking
	try:
		subprocess.run(['tmux', 'respawn-window', '-t', 'test:q', '-k', './quefeather.py', 'Enter'], check=True) #non-blocking
	except subprocess.CalledProcessError as e:
		print(e.stderr)
	print(f'Finishign at {time.strftime("%Y-%m-%d %H:%M:%S")}')

def test_dict_mod():
	dic = {"a":[1,2],"b":[2,3,4]}
	rem = dic["b"].pop(2)  
	print(rem)
	print_dict(dic)

def test_arg_defaults():
	parser = argparse.ArgumentParser(description='test_debug.py')
	parser.add_argument('-rr', '--remove_run',nargs='?', type=str,const='ask_me', default=None,  help='remove a run from to_run')
	parser.add_argument('-rl', '--recover_last', nargs='?',type=str,const='last', default=None, help='recover premature run termination')
	args, other = parser.parse_known_args()
	print(args.remove_run)
	print(args.recover_last)
	
def test_basename_extraction():

	r = Path('./')
	fs = sorted([x for x in r.iterdir() if x.is_file()])
	ns = []
	for f in fs:
		if f.name.endswith('.json'):
			ns.append(f.name.replace('.json', ''))
		elif f.name.endswith('.py'):
			ns.append(f.name.replace('.py', ''))
	print(ns)

def reformat_results(runs_dict):
	for split in runs_dict.keys():
		for arch in runs_dict[split].keys():
			for i, exp_no in enumerate(runs_dict[split][arch]):
				output = Path(f'runs/{split}/{arch}_exp{exp_no}')
				res_fs = [x for x in output.iterdir() if x.name.endswith('.txt')]
				for p in res_fs:
					test.parse_run_info_to_json(p) #type: ignore 

def ask_nicely(message, requirment, error):
	passed=False
	ans = 'none'
	while not passed:
		passed=True
		ans = input(message)
		if not requirment(ans):
			print(error)
			passed=False
	return ans


def get_gloss_stats(instances):
	stats_managers = {
				"train": {'examples': 0, 'seen_signers': {}},
				"val": {'examples': 0, 'seen_signers': {}},
				"test": {'examples': 0, 'seen_signers': {}}
	}
	for inst in instances:
		sid = inst['signer_id']
		stat_man = stats_managers[inst['split']] 
		stat_man['examples'] += 1
		if sid in stat_man['seen_signers']:
			stat_man['seen_signers'][sid] += 1
		else:
			stat_man['seen_signers'][sid] = 1
		stats_managers[inst['split']]  = stat_man
	return stats_managers
			

def get_stats(split_path, output):
	stats = {}
	with open(split_path, 'r') as f:
		data = json.load(f)
	for entry in data:
		instances = entry['instances']
		stats[entry['gloss']] = get_gloss_stats(instances)
	with open(output, 'w') as f:
		json.dump(stats, f, indent=2)
	return stats

def merge_seen(current, incoming):
	return dict(Counter(current) + Counter(incoming))
 
def refine_stats(stat_path,
								 output_refined:Optional[str|Path] = None,
								 output_overlapping:Optional[str|Path] = None):
	#we want to know:
	#- the number of examples in each split
	#- the number of different signers in each split
	#- the average number of examples per class, per split
	#- the average number of signers per class, per split
	#bonus
	#for each class, check if there are overlap between signers in classes
	with open(stat_path, 'r') as f:
		stats = json.load(f)
	num_classes = len(stats)
	refined = {}
	splits = ['train', 'val', 'test']
	
	for split in splits:
		num_ex = 0
		signers_count = 0	
		seen_signers = {}
		for gloss in stats.keys():
			info = stats[gloss][split]
			num_ex += info['examples']
			signers_count += len(info['seen_signers']) #for the average we need to recount
			seen_signers = merge_seen(seen_signers, info['seen_signers'])
			 
		refined[split] = {
			'num_ex': num_ex,
			'num_s': len(seen_signers),
			'mean_ex': round(num_ex / num_classes, 2),
			'mean_s': round(signers_count / num_classes, 2)
		}
	
	overlapping = {}
	#bonus
	for gloss in stats.keys():
		test_signers = list(stats[gloss]['test']['seen_signers'].keys()) # + \
									# list(stats[gloss]['val']['seen_signers'].keys())
		train_signers = list(stats[gloss]['train']['seen_signers'].keys())
		overlapping[gloss] =  any(ts in train_signers for ts in test_signers)
	
	if output_refined:
		with open(output_refined, 'w') as f:
			json.dump(refined, f, indent=2)
	if output_overlapping:
		with open(output_overlapping, 'w') as f:
			json.dump(overlapping, f, indent=2)
 
	return refined, overlapping
 
 
def merge_seen_ex():
	d1 = {
				"118": 1,
				"90": 1,
				"110": 1,
				"113": 1,
				"109": 2,
				"121": 2,
				"49": 6,
				"18": 16,
				"31": 1,
				"36": 1,
				"59": 1,
				"12": 1,
				"14": 3,
				"11": 1,
				"5": 2
			}
	
	
	d2 = {
				"115": 2,
				"94": 2,
				"121": 1,
				"113": 1,
				"109": 3,
				"5": 6,
				"65": 1,
				"36": 1,
				"44": 1,
				"12": 2,
				"52": 1,
				"6": 1,
				"21": 1,
				"20": 3,
				"14": 3,
				"2": 1,
				"11": 2,
				"10": 3
			}
	
	d3 = merge_seen(d1, d2)
	d4 = {}
	print_dict(merge_seen(d3, d4))

def test_use_rudid():
	from experiments.scripts.quewing import wait_for_run_completion
	run_id = input("Enter run ID: ")
	info = wait_for_run_completion(
		entity = 'ljgoodall2001-rhodes-university',
		project = 'WLASL-100',
		check_interval = 5,
	 verbose=True,
	 run_id=run_id,
	 max_retries=5
	)
	
def frame_shuffling():
	import torch

	# Create a 2D tensor
	tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	# tensor = torch.rand(8, 3, 4, 4)
	t = len(tensor)
	# Shuffle rows
	shuffled_indices = torch.randperm(tensor.size(0))
	# shuffled_tensor = tensor[:, :, shuffled_indices, :, :]
	shuffled_tensor = tensor[shuffled_indices]	
	print("Shuffled indices:")
	print(shuffled_indices)
	print("Original:")
	print(tensor)
	print("\nShuffled rows:")
	print(shuffled_tensor)
	print(tensor.shape == shuffled_tensor.shape)
 
	og_idxs = torch.arange(tensor.size(0))
	inv_permutation = shuffled_indices
	print("Inverse permutation indices:")
	print(inv_permutation)
	# reversed_tensor = shuffled_tensor[:, :, inv_permutation, :, :]
	reversed_tensor = shuffled_tensor[inv_permutation]
 
	print("\nReversed to original:")
	print(reversed_tensor)
	print(tensor.shape == reversed_tensor.shape)
	print(torch.all(tensor == reversed_tensor))


def frame_shuffling2():
	import utils
	from video_transforms import Shuffle
	import torch
	import torchvision.transforms.v2 as ts
	vid = './media/69241.mp4'
	num_frames = 4
	frames = utils.load_rgb_frames_from_video(vid, 4, 4+num_frames)
	# print(frames.shape)
	# utils.visualise_frames(frames, num_frames)

	# shuffled_indices = torch.randperm(frames.size(0))
	shuffled_indices = Shuffle.create_permutation(frames.size(0), seed=42)
	print("Shuffled indices:")
	print(shuffled_indices)
	t = ts.Compose([
		Shuffle(shuffled_indices)
	])

	f2 = t(frames)

	# print(f2.shape)
	# utils.visualise_frames(f2,num_frames)
	unshuffled = t(f2)

	print(unshuffled.shape)
	print("EQUAL: ", torch.allclose(frames, unshuffled))
	
	
def test_shuffle():
		import torch
		from video_transforms import Shuffle
		# from torchvision.transforms import Transform
		# Create a test tensor with 5 frames, 3 channels, 4x4 resolution
		# Each frame has a unique value to easily track shuffling
		num_frames = 5
		test_tensor = torch.zeros(num_frames, 3, 4, 4)
		for i in range(num_frames):
				test_tensor[i] = i  # Each frame gets a unique value
		
		print("Original tensor (first channel of each frame):")
		print(test_tensor[:, 0, 0, 0])  # Show first pixel of each frame
		
		# Test 1: Create a specific permutation and test
		custom_perm = torch.tensor([2, 0, 4, 1, 3])
		shuffle_transform = Shuffle(custom_perm)
		shuffled = shuffle_transform(test_tensor)
		
		print("\nShuffled with custom permutation [2, 0, 4, 1, 3]:")
		print(shuffled[:, 0, 0, 0])
		
		# Verify the shuffle worked correctly
		expected = torch.tensor([2., 0., 4., 1., 3.])  # Expected order after shuffle
		assert torch.allclose(shuffled[:, 0, 0, 0], expected), "Custom permutation test failed"
		
		# Test 2: Test with random permutation (fixed seed for reproducibility)
		random_perm = Shuffle.create_permutation(num_frames, seed=42)
		shuffle_transform_random = Shuffle(random_perm)
		shuffled_random = shuffle_transform_random(test_tensor)
		
		print(f"\nShuffled with random permutation {random_perm.tolist()}:")
		print(shuffled_random[:, 0, 0, 0])
		
		# Test 3: Verify that the original tensor is unchanged (transform should not modify in place)
		print("\nOriginal tensor after shuffling (should be unchanged):")
		print(test_tensor[:, 0, 0, 0])
		
		# Test 4: Test error cases
		try:
				# Wrong dimension
				wrong_dim_tensor = torch.randn(5, 3, 4)  # 3D instead of 4D
				shuffle_transform(wrong_dim_tensor)
				assert False, "Should have raised assertion error for wrong dimension"
		except AssertionError as e:
				assert "Input tensor must be 4D" in str(e)
		
		try:
				# Wrong channel count
				wrong_channel_tensor = torch.randn(5, 1, 4, 4)  # 1 channel instead of 3
				shuffle_transform(wrong_channel_tensor)
				assert False, "Should have raised assertion error for wrong channel count"
		except AssertionError as e:
				assert "Input tensor must have 3 channels" in str(e)
		
		print("\nAll tests passed!")

def test__sum_model():
  from test import _sum_model
  d = {
	  "004": {
		"best": {
		  "test set": {
			"top_k_average_per_class_acc": {
			  "top1": 0.5343333333333333,
			  "top5": 0.82,
			  "top10": 0.895
			},
			"top_k_per_instance_acc": {
			  "top1": 0.5387596899224806,
			  "top5": 0.8217054263565892,
			  "top10": 0.8992248062015504
			}
		  }
		},
		"checkpoint_154": {
		  "test set": {
			"top_k_average_per_class_acc": {
			  "top1": 0.5965,
			  "top5": 0.8516666666666666,
			  "top10": 0.8916666666666666
			},
			"top_k_per_instance_acc": {
			  "top1": 0.5891472868217055,
			  "top5": 0.8488372093023255,
			  "top10": 0.8914728682170543
			}
		  }
		}
	  },
	  "005": {
		"checkpoint_118": {
		  "test set": {
			"top_k_average_per_class_acc": {
			  "top1": 0.48899999999999993,
			  "top5": 0.8083333333333332,
			  "top10": 0.8758333333333332
			},
			"top_k_per_instance_acc": {
			  "top1": 0.49224806201550386,
			  "top5": 0.8062015503875969,
			  "top10": 0.875968992248062
			}
		  }
		},
		"best": {
		  "test set": {
			"top_k_average_per_class_acc": {
			  "top1": 0.45699999999999996,
			  "top5": 0.7583333333333333,
			  "top10": 0.8658333333333332
			},
			"top_k_per_instance_acc": {
			  "top1": 0.46124031007751937,
			  "top5": 0.7596899224806202,
			  "top10": 0.8682170542635659
			}
		  }
		}
	  }
	}
  print(type(d))

def test_sub():
	with subprocess.Popen(['ping', 'google.com'], 
					  stdout=subprocess.PIPE, 
					  stderr=subprocess.PIPE, 
					  text=True) as proc:
		for line in proc.stdout:
			print(f"Received: {line.strip()}")

def test_send():
	from quewing import print_tmux
	print_tmux(
		"hello",
		"worker",
		"que_training"
	)

def test_shelly():
	import cmd

	class MyShell(cmd.Cmd):
		intro = 'Welcome! Type help or ? to list commands.\n'
		prompt = '> '

		def do_greet(self, arg):
			"""Greet someone: greet <name>"""
			print(f'Hello, {arg}!')
		
		def do_quit(self, arg):
			"""Exit the application"""
			return True

	
	MyShell().cmdloop()

def test_shelly2():
	import cmd
	from quewing import que, join_session
	
	class QueShell(cmd.Cmd):
		intro = 'Queue Management Shell. Type help or ? to list commands.\n'
		prompt = '(que) > '
		
		def __init__(self, runs_path: str, implemented_path: str, verbose: bool = True):
			super().__init__()
			self.que = que(runs_path, implemented_path, verbose)
		
		def do_create(self, arg):
			"""Create a new run and add it to the queue"""
			self.que.create_run()
			self.que.save_state()
		
		def do_join_session(self, arg):
			"""Join a tmux session: join_session <wndw_name> <sesh_name>"""
			args = arg.split()
			res = join_session(args[0], args[1])
			if res:
				self.que.print_v("success")
   
		def do_next(self, arg):
			"""Get the next run from the queue"""
			next_run = self.que.get_next_run()
			if next_run:
				print(f"Next run: {self.que.get_config(next_run)}")
				self.que.save_state()
		
		def do_list(self, arg):
			"""List all configs in the queue"""
			self.que.list_configs()
		
		def do_remove(self, arg):
			"""Remove a run from the queue: remove [to_run|old_runs] [index]"""
			args = arg.split()
			loc = args[0] if len(args) > 0 else None
			idx = int(args[1]) if len(args) > 1 else None
			self.que.remove_run(loc, idx)
			self.que.save_state()
		
		def do_return_old(self, arg):
			"""Return old runs back to the queue: return_old [num_from_end]"""
			num = int(arg) if arg.strip() else None
			self.que.return_old(num)
			self.que.save_state()
		
		def do_shuffle(self, arg):
			"""Shuffle/reorder runs in the queue"""
			self.que.shuffle_configs()
			self.que.save_state()
		
		def do_clear(self, arg):
			"""Clear runs: clear [past] [future]"""
			past = 'past' in arg.lower()
			future = 'future' in arg.lower()
			self.que.clear_runs(past, future)
			self.que.save_state()
		
		def do_save(self, arg):
			"""Manually save the current state"""
			self.que.save_state()
		
		def do_quit(self, arg):
			"""Exit the shell"""
			print("Goodbye!")
			return True
		
		def do_exit(self, arg):
			"""Exit the shell"""
			return self.do_quit(arg)



	from quewing import RUN_PATH, IMP_PATH
	QueShell(
		runs_path=RUN_PATH,
		implemented_path=IMP_PATH
	 ).cmdloop()

def test_safe_index():
	l = [1,2,3]
	i = random.randint(-4, 4)
	while len(l) > 0:
		if abs(i) < len(l):
			print(f'{i} in range for len(l) = {len(l)}')
			print(f'l.pop({i}) = {l.pop(i)}')
		else:
			print(f'{i} out of range for len(l) = {len(l)}')
		i = random.randint(-len(l) - 1, len(l) + 1)

def test_wait_for_completion():
    from quewing import gpu_manager
    
    print(gpu_manager.wait_for_completion(verbose=True, check_interval=5, confirm_interval=1))

def test_test_run():
	with open('queRuns.json') as f:
		all_runs = json.load(f)

	old_runs = all_runs['old_runs']

	test.test_run(old_runs[0])


if __name__ == "__main__":
	test_test_run()