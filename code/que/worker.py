from typing import  Optional
import time
import torch
from argparse import ArgumentParser
from pathlib import Path
import multiprocessing
import sys
import time
#locals
from .core import WR_LOG_PATH, que
from training import train_loop, _setup_wandb
from configs import _exp_to_run_info, ExpInfo
from utils import gpu_manager
from .server import connect_manager

class worker:
	def __init__(
		self,
		que: que,
		log_path: str | Path = WR_LOG_PATH,
		verbose: bool = True,
	):
		self.que = que
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
			print("Training run failed due to an error")
			print(str(e))
			self.que.stash_failed_run(str(e))
			self.que.save_state()
			raise e  # still need to crash so daemon can

	def run(self) -> Optional[ExpInfo]:
		with open(self.log_path, 'w') as log_file:
			sys.stdout = log_file
			sys.stderr = log_file
			return self.train()

	def debug(self) -> Optional[ExpInfo]:
		#simulate work
		try:
			i = 0
			while i < 5:
				print("running...")
				print(f"Run number: {i}")
				time.sleep(2)
				i += 1
			# raise Exception('sim error')
			print("finished")
		except KeyboardInterrupt:
			print("Worker interrupted.")
			
	def debug2(self) -> Optional[ExpInfo]:
		
     
		try:
			i = 0
			while i < 5:
				print("running...")
				print(f"Run number: {i}")
				time.sleep(2)
				i += 1
			# raise Exception('sim error')
			print("finished")
		except KeyboardInterrupt:
			print("Worker interrupted.")
		#test an exception
		# raise RuntimeError("Test exception in worker")
  
if __name__ == '__main__':
	server = connect_manager()
	wr = worker(server.get_que())
	wr.debug()