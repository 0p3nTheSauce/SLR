from typing import  Optional
import time
import torch
from argparse import ArgumentParser
from pathlib import Path
import multiprocessing
import sys

#locals
from .core import WR_LOG_PATH
from training import train_loop, _setup_wandb
from configs import _exp_to_run_info, ExpInfo
from utils import gpu_manager
from .server import connect_que

class worker:
	def __init__(
		self,
		log_path: str | Path = WR_LOG_PATH,
		verbose: bool = True,
	):
		self.que = connect_que()
		self.verbose = verbose
		self.log_path = log_path
  
  
	def print_v(self, message: str) -> None:
		if self.verbose:
			print(message)

	def train(self) ->  Optional[ExpInfo]:
		try:
			gpu_manager.wait_for_completion()
			self.print_v("starting work")

			# get next run
			self.que.load_state()
			info = self.que.pop_cur_run()

			wandb_info = info["wandb"]
			admin = info["admin"]
			config = _exp_to_run_info(info)

			# setup wandb run
			run = _setup_wandb(config, wandb_info)
			if run is None:
				return

			# save run_id for recovering
			wandb_info["run_id"] = run.id
			info["wandb"] = wandb_info

			self.print_v("writing my id to temp file")
			self.que.set_cur_run(info)
			self.que.save_state()

			self.print_v(f"Run ID: {run.id}")
			self.print_v(f"Run name: {run.name}")  # Human-readable name
			self.print_v(f"Run path: {run.path}")  # entity/project/run_id format

			train_loop(admin["model"], config, run, recover=admin["recover"])
			run.finish()
		except Exception as e:
			# TODO: handle ctrl+c gracefully
			self.print_v("Training run failed due to an error")
			self.que.stash_failed_run(str(e))
			self.que.save_state()
			raise e  # still need to crash so daemon can

	def run(self) -> Optional[ExpInfo]:
		with open(self.log_path, 'w') as log_file:
			sys.stdout = log_file
			sys.stderr = log_file
			return self.train()
 
 
 
	def idle(
		self,
		message: str,
		wait: Optional[int] = None,
		cycles: Optional[int] = None,
	) -> str:
		gpu_manager.wait_for_completion(
			check_interval=5,  # seconds,
			confirm_interval=1,  # second
			num_checks=10,  # confirm consistency over 10 seconds
			verbose=self.verbose,
		)
		t = wait if wait else 1
		c = cycles if cycles else 10
		print("\n" * 2)
		print("-" * 10)
		print(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
		for i in range(c):
			print(f"Idling: {i}")
			print(message)
			time.sleep(t)
		print(f"Finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
		return message

	def idle_log(
		self, message: str, wait: Optional[int] = None, cycles: Optional[int] = None
	):
		gpu_manager.wait_for_completion(
			check_interval=5,  # seconds,
			confirm_interval=1,  # second
			num_checks=10,  # confirm consistency over 10 seconds
			verbose=self.verbose,
		)
		t = wait if wait else 1
		c = cycles if cycles else 10
		with open(self.log_path, "a") as f:
			f.write("\n" * 2)
			f.write("-" * 10)
			f.write("\n")
			f.write(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
			print(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
			f.flush()

			for i in range(c):
				f.write(f"Idling: {i}\n")
				print(f"Idling: {i}")
				f.write(message + "\n")
				time.sleep(t)
				f.flush()

			f.write(f"Finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
			print(f"Finishing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
			f.flush()

	def sim_gpu_usage(self):
		print("about to work")
		x = torch.rand(10000, 10000, device="cuda")
		y = torch.rand(10000, 10000, device="cuda")
		z = torch.rand(10000, 10000, device="cuda")
		for i in range(100):
			z = x @ y
			time.sleep(2)
			print(f"Idling: {i}")
		print("bunch of work done")
		return z.cpu()

def get_worker_parser() -> ArgumentParser:
	parser = ArgumentParser(prog='que_worker.py')
	
	parser.add_argument(
		'setting',
		choices=['work', 'idle', 'idle_log', 'idle_gpu'],
		help='Operation of worker'
	)
	
	parser.add_argument(
		'-w',
		'--wait',
		help='Idle time (seconds)',
		type=int,
		default= None
	)
	
	parser.add_argument(
		'-c',
		'--cycles',
		help='Number of idle cycles',
		type=int,
		default= None
	)
	
	return parser




if __name__ == '__main__':
	
	p = multiprocessing.Process(target=worker)
	p.start()
	p.join()