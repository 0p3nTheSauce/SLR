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
	 
if __name__ == "__main__":
	# test_early_stopping(mode='min')
	# test_wait_for_run_completion()
	# pytest.main([__file__])
	# test_stopper_save_and_load()
	# tmux_session()
	# test_blocking()
	# test_dict_mod()
	# test_arg_defaults()
	# test_basename_extraction()
	# parse_run_info_to_json('/home/luke/ExtraStorage/SLR/code/best_val-top-k.txt')
	# with open('./wlasl_runs_done.json', 'r') as f:
	#   reformat_results(json.load(f))
	# messages = ['enter split: ', 'split not found']
	# requirements = [lambda x: x in ['asl100', 'asl300']]
	# ask_nicely(messages, requirements, 'error')
 
	# get_stats(
	#  '/home/luke/ExtraStorage/SLR/data/WLASL/splits/asl100.json',
	#  '/home/luke/ExtraStorage/SLR/code/wlasl_100_stats.json')
	# get_stats(
	#  '/home/luke/ExtraStorage/SLR/data/WLASL/splits/asl300.json',
	#  '/home/luke/ExtraStorage/SLR/code/wlasl_300_stats.json')
 
	refine_stats('/home/luke/ExtraStorage/SLR/code/wlasl_300_stats.json',
							'/home/luke/ExtraStorage/SLR/code/wlasl_300_stats_ref.json',
       				'/home/luke/ExtraStorage/SLR/code/wlasl_300_stats_overlapping.json')
	refine_stats('/home/luke/ExtraStorage/SLR/code/wlasl_100_stats.json',
							'/home/luke/ExtraStorage/SLR/code/wlasl_100_stats_ref.json',
       				'/home/luke/ExtraStorage/SLR/code/wlasl_100_stats_overlapping.json')

	merge_seen_ex()