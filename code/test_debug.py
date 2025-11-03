from stopping import EarlyStopper
import wandb
import matplotlib.pyplot as plt
import random
from utils import print_dict
import time
import subprocess
import argparse
from pathlib import Path 
import json
import test
from collections import Counter
from typing import Optional, Literal
from functools import partial
import math
import torch

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
	f = partial(sim_loss)
	if mode == 'max':
		f = partial(sim_acc)
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
	f = partial(sim_loss)
	if mode == 'max':
		f = partial(sim_acc)
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

	
def frame_shuffling():
	import torch

	# Create a 2D tensor
	tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	# tensor = torch.rand(8, 3, 4, 4)
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
	shuflr = Shuffle(frames.size(0))
	print("Shuffled indices:")
	print(shuflr.permutation)
	t = ts.Compose([
		shuflr
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
		shuffle_transform = Shuffle(len(custom_perm), perm=custom_perm)
		shuffled = shuffle_transform(test_tensor)
		
		print("\nShuffled with custom permutation [2, 0, 4, 1, 3]:")
		print(shuffled[:, 0, 0, 0])
		
		# Verify the shuffle worked correctly
		expected = torch.tensor([2., 0., 4., 1., 3.])  # Expected order after shuffle
		assert torch.allclose(shuffled[:, 0, 0, 0], expected), "Custom permutation test failed"
		
		# Test 2: Test with random permutation (fixed seed for reproducibility)
		# random_perm = Shuffle.create_permutation(num_frames, seed=42)
		shuffle_transform_random = Shuffle(num_frames)
		shuffled_random = shuffle_transform_random(test_tensor)
		
		print(f"\nShuffled with random permutation {shuffle_transform_random.permutation.tolist()}:")
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

def test_sub():
	with subprocess.Popen(['ping', 'google.com', '-c', '4'],  # Limit to 4 pings
					  stdout=subprocess.PIPE, 
					  stderr=subprocess.PIPE, 
					  text=True) as proc:
		stdout, stderr = proc.communicate()
		if stdout:
			for line in stdout.splitlines():
				print(f"Received: {line.strip()}")



def test_safe_index():
	lst = [1,2,3]
	i = random.randint(-4, 4)
	while len(lst) > 0:
		if abs(i) < len(lst):
			print(f'{i} in range for len(l) = {len(lst)}')
			print(f'l.pop({i}) = {lst.pop(i)}')
		else:
			print(f'{i} out of range for len(l) = {len(lst)}')
		i = random.randint(-len(lst) - 1, len(lst) + 1)

def test_wait_for_completion():
	from quewing import gpu_manager
	
	print(gpu_manager.wait_for_completion(verbose=True, check_interval=5, confirm_interval=1))


def test_get_avail_splits():
	from configs import get_avail_splits
	print(get_avail_splits())

def reformet():
	#reformat queRuns so that entity, project, tags, and run_id, are encaplusated in a wandb dictionary
	with open('queRuns.json', 'r') as f:
		all_runs = json.load(f)
	old_runs = all_runs['old_runs']
	for run in old_runs:
		wandb_info = {
			'entity': run.pop('entity'),
			'project': run.pop('project'),
			'tags': run.pop('tags'),
			'run_id': run.pop('run_id')
		}
		run['wandb'] = wandb_info
	with open('queRuns.json', 'w') as f:
		json.dump(all_runs, f, indent=4)
  
  
def position_entropy(perm: torch.Tensor):
	#this does not work very effectively, because shifting the 
	#permutation to the right by 1, produces a large entropy, 
	#while conceptually, the video likely remains unchanged      
	sorted_perm = torch.argsort(perm)
	print(sorted_perm)
	print(torch.arange(len(perm)))
	displacements = torch.abs(torch.arange(len(perm)) - sorted_perm)
	print(displacements)
	hist = torch.bincount(displacements).float()
	print(hist)
	normed = hist / len(perm)
	normed_no0 = normed[normed > 0]
	
	return -torch.sum(normed_no0 * torch.log2(normed_no0)).item()


def _get_log_func(unit):
	if unit == 'bit':
		return lambda x: math.log(x, 2)
	elif unit == 'nat':
		return lambda x: math.log(x)
	else:
		return lambda x: math.log(x, 10)


def shannon_entropy(perm: torch.Tensor, unit:Literal['bit', 'nat', 'dit'] = 'bit') -> float:
	perml = list(map(int, perm.numpy()))
	
	p_len = len(perml)
	diffs = [0] * p_len  
	for i in range(p_len-1):
		diff = perml[i+1] - perml[i]
		if diff < 0:
			diff += p_len
		diffs[i] = diff

	diff = perml[0] - perml[p_len-1]
	if diff < 0:
		diff += p_len
	
	diffs[p_len-1] = diff
	
	hist = [0] * p_len
	
	for d in diffs:
		hist[d] += 1
				
	print(diffs)
	normed = [d / p_len for d in hist if d > 0]
	print(normed)
	e = 0
		
	log = _get_log_func(unit)            
		
	for n in normed:
		e += -n * log(n)
	print(f"E: {e}")
	return e

def test_shuffle2():
	from video_transforms import Shuffle
	import torch
	# perm = Shuffle.create_permutation(4, 42)
	perm = torch.tensor([3, 0, 1, 2])
	print(f"original: {perm}")
	shannon_entropy(perm)
	print()
	Shuffle.shannon_entropy(perm)
	print()
	print(position_entropy(perm))
	
def test_shuffle3():
	from video_transforms import Shuffle
	from configs import set_seed
	set_seed()
	num_frames = 16
	for i in range(10):
		s1 = Shuffle(num_frames)
		perm1 = map(str, s1.permutation.numpy())
		print(f"Perm {i+1}: [{', '.join(perm1)}]")
		print(f"Shannon entropy: {Shuffle.shannon_entropy(s1.permutation)}")
		print()
  
def test_tmux_man():
	from quewing import tmux_manager
	
	tman = tmux_manager()
	
	print(tman.exec_path)
	
	tman._send("./quefeather.py -h", "worker")
	
	tman.join_session("worker")

def test_loss():
	loss_func = torch.nn.CrossEntropyLoss(reduce=None)
	
	input_t = torch.rand(5,3)
	target = torch.rand(5, 3)
	print(input_t)
	print(target)
	loss = loss_func(input_t, target)
	print(loss)
	print(loss.item())
	
def test_loss2():
	# Create some dummy data
	import torch
	import torch.nn as nn

	# Create some dummy data
	batch_size = 4
	num_classes = 3

	# Predictions and targets
	predictions = torch.randn(batch_size, num_classes)
	targets = torch.tensor([0, 1, 2, 1])

	print("Predictions shape:", predictions.shape)
	print("Targets:", targets)
	print("\n" + "="*60)

	# Test with NO arguments (default behavior)
	print("\n1. Testing with NO arguments (default):")
	criterion_default = nn.CrossEntropyLoss()
	loss_default = criterion_default(predictions, targets)
	print(f"Loss (no args): {loss_default.item()}")

	# Test with explicit reduction='mean'
	print("\n2. Testing with explicit reduction='mean':")
	criterion_mean = nn.CrossEntropyLoss(reduction='mean')
	loss_mean = criterion_mean(predictions, targets)
	print(f"Loss (reduction='mean'): {loss_mean.item()}")

	# Test with reduction='sum'
	print("\n3. Testing with reduction='sum':")
	criterion_sum = nn.CrossEntropyLoss(reduction='sum')
	loss_sum = criterion_sum(predictions, targets)
	print(f"Loss (reduction='sum'): {loss_sum.item()}")

	# Test with reduction='none' (get individual losses)
	print("\n4. Testing with reduction='none':")
	criterion_none = nn.CrossEntropyLoss(reduction='none')
	losses_individual = criterion_none(predictions, targets)
	print(f"Individual losses: {losses_individual}")
	print(f"Manual mean of individual losses: {losses_individual.mean().item()}")
	print(f"Manual sum of individual losses: {losses_individual.sum().item()}")

	print("\n" + "="*60)
	print("\nVERIFICATION:")
	print(f"loss_default (no args) = {loss_default.item():.6f}")
	print(f"loss_mean (explicit) = {loss_mean.item():.6f}")
	print(f"loss_sum / batch_size = {loss_sum.item() / batch_size:.6f}")
	print(f"losses_individual.mean() = {losses_individual.mean().item():.6f}")
	print(f"losses_individual.sum() = {losses_individual.sum().item():.6f}")

	print("\n" + "="*60)
	print("\nCONCLUSION:")
	if abs(loss_default.item() - loss_mean.item()) < 1e-6:
		print("✓ Default (no args) is the SAME as reduction='mean'")
		
	if abs(loss_mean.item() - losses_individual.mean().item()) < 1e-6:
		print("✓ With reduction='mean', loss.item() returns the MEAN loss per sample")
		print(f"✓ This equals: {loss_mean.item():.6f}")
	else:
		print("✗ Something unexpected happened")

	if abs(loss_sum.item() - losses_individual.sum().item()) < 1e-6:
		print("✓ With reduction='sum', loss.item() returns the SUM of all losses")
		print(f"✓ This equals: {loss_sum.item():.6f}")
		
	print("\n" + "="*60)
	print("\nSo when you multiply loss.item() by batch_size:")
	print(f"loss.item() * batch_size = {loss_mean.item():.6f} * {batch_size} = {loss_mean.item() * batch_size:.6f}")
	print(f"This converts the mean back to a sum for accumulation purposes!")

def test_loss3():
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import numpy as np

	print("="*70)
	print("CROSS ENTROPY LOSS: STEP BY STEP")
	print("="*70)

	# Example: 3-class classification problem
	# Suppose we're classifying images as: cat=0, dog=1, bird=2

	logits = torch.tensor([[2.0, 1.0, 0.1]])  # Model's raw outputs (logits)
	target = torch.tensor([0])  # True class is 0 (cat)

	print("\n📊 SETUP:")
	print(f"Logits (raw model output): {logits[0].tolist()}")
	print(f"True class: {target.item()} (cat)")
	print(f"Classes: 0=cat, 1=dog, 2=bird")

	print("\n" + "="*70)
	print("STEP 1: Convert logits to probabilities using Softmax")
	print("="*70)

	# Manual softmax calculation
	exp_logits = torch.exp(logits[0])
	print(f"\nexp(logits) = {exp_logits.tolist()}")

	sum_exp = exp_logits.sum()
	print(f"sum(exp(logits)) = {sum_exp.item():.4f}")

	probabilities = exp_logits / sum_exp
	print(f"\nProbabilities = exp(logits) / sum(exp(logits))")
	print(f"P(cat)  = {probabilities[0].item():.4f}")
	print(f"P(dog)  = {probabilities[1].item():.4f}")
	print(f"P(bird) = {probabilities[2].item():.4f}")
	print(f"Sum = {probabilities.sum().item():.4f} (should be 1.0)")

	# Verify with PyTorch's softmax
	softmax_probs = F.softmax(logits[0], dim=0)
	print(f"\n✓ PyTorch softmax matches: {torch.allclose(probabilities, softmax_probs)}")

	print("\n" + "="*70)
	print("STEP 2: Calculate -log(probability of correct class)")
	print("="*70)

	correct_class_prob = probabilities[target.item()]
	print(f"\nProbability of correct class (cat): {correct_class_prob.item():.4f}")

	manual_loss = -torch.log(correct_class_prob)
	print(f"Loss = -log({correct_class_prob.item():.4f}) = {manual_loss.item():.4f}")

	print("\n" + "="*70)
	print("STEP 3: Verify with PyTorch's CrossEntropyLoss")
	print("="*70)

	criterion = nn.CrossEntropyLoss()
	pytorch_loss = criterion(logits, target)
	print(f"\nPyTorch CrossEntropyLoss: {pytorch_loss.item():.4f}")
	print(f"Our manual calculation:    {manual_loss.item():.4f}")
	print(f"✓ Match: {torch.allclose(manual_loss, pytorch_loss)}")

	print("\n" + "="*70)
	print("INTUITION: What different losses mean")
	print("="*70)

	test_cases = [
		(torch.tensor([[5.0, 0.0, 0.0]]), "Very confident and CORRECT"),
		(torch.tensor([[2.0, 1.0, 0.1]]), "Moderately confident and CORRECT"),
		(torch.tensor([[0.5, 0.4, 0.1]]), "Low confidence but CORRECT"),
		(torch.tensor([[0.0, 5.0, 0.0]]), "Very confident but WRONG"),
		(torch.tensor([[1.0, 1.0, 1.0]]), "Completely uncertain (equal probs)"),
	]

	for logits_test, description in test_cases:
		loss = criterion(logits_test, target)
		probs = F.softmax(logits_test[0], dim=0)
		print(f"\n{description}")
		print(f"  Probs: cat={probs[0].item():.3f}, dog={probs[1].item():.3f}, bird={probs[2].item():.3f}")
		print(f"  Loss: {loss.item():.4f}")

	print("\n" + "="*70)
	print("KEY INSIGHTS")
	print("="*70)
	print("""
	1. Cross entropy = -log(probability of correct class)
	2. Lower loss = model is more confident in correct answer
	3. Loss → 0 as probability → 1 (perfect prediction)
	4. Loss → ∞ as probability → 0 (terrible prediction)
	5. PyTorch's CrossEntropyLoss combines softmax + negative log likelihood
	6. You give it raw logits (not probabilities) and it handles softmax internally
	""")

	print("\n" + "="*70)
	print("BONUS: Multi-sample batch example")
	print("="*70)

	# Batch of 3 samples
	batch_logits = torch.tensor([
		[2.0, 1.0, 0.1],  # Sample 1: predict cat
		[0.5, 3.0, 0.2],  # Sample 2: predict dog
		[0.1, 0.2, 4.0],  # Sample 3: predict bird
	])
	batch_targets = torch.tensor([0, 1, 2])  # All correct!

	batch_loss = criterion(batch_logits, batch_targets)
	print(f"\nBatch logits shape: {batch_logits.shape}")
	print(f"Batch targets: {batch_targets.tolist()}")
	print(f"Mean loss across batch: {batch_loss.item():.4f}")

	# Calculate individual losses
	individual_losses = []
	for i in range(3):
		loss = criterion(batch_logits[i:i+1], batch_targets[i:i+1])
		individual_losses.append(loss.item())
		print(f"  Sample {i+1} loss: {loss.item():.4f}")

	print(f"\nManual mean: {np.mean(individual_losses):.4f}")
	print(f"PyTorch mean: {batch_loss.item():.4f}")
	print(f"✓ Match: {abs(np.mean(individual_losses) - batch_loss.item()) < 1e-6}")


def reformet2():
	with open('queRuns.json', 'r') as f:
		all_runs = json.load(f)
		
	old_runs = all_runs['old_runs']
	# print(len(old_runs))
	# print(all(['run_id' in run['wandb'] for run in old_runs]))
	for i, run in enumerate(old_runs):
		if 'run_id' in run:
			_ = run.pop('run_id')
		old_runs[i] = run

	all_runs['old_runs'] = old_runs
	with open('queRuns.json', 'w') as f:
		json.dump(all_runs, f, indent=4)
			

if __name__ == "__main__":
	# test_get_avail_splits()
	# reformet2()
	# test_shuffle3()
	# test_loss3()
	# test_tmux_man()
	pass