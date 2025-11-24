from typing import  Optional, Tuple
import subprocess
from pathlib import Path
from argparse import ArgumentParser
import sys
import multiprocessing
import logging
#locals
from .core import (
	RUN_PATH,
	WR_PATH,
	DN_LOG_PATH,
	WR_LOG_PATH,
	QueEmpty,
	QueBusy,
	ExpInfo
)

from .server import connect_manager
from .core import que
from utils import gpu_manager

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	filename=DN_LOG_PATH  # Optional: log to file
)


class daemon:
	"""Class for the queue daemon process. The function works in a fetch execute repeat
	cycle. The function reads runs from the queRuns.json file, then writes them to the
	queTemp.json file for the worker process to find. The daemon spawns the worker, then
	waits for it to complete before proceeding. The daemon runs in it's own tmux session,
	while the worker outputs to a log file.

	Args:
					name: of own tmux window
					wr_name: of worker tmux window (monitor)
					sesh: tmux session name
					runs_path: to queRuns.json
					temp_path: to queTemp.json
					imp_path: to implemented_info.json
					exec_path: to quefeather.py
					verbose: speaks to you
					wr: supply its worker
					q: supply its que
	"""

	def __init__(
		self,
		que: que, 
		runs_path: str | Path = RUN_PATH,
		exec_path: str | Path = WR_PATH,
		dn_log_path : str | Path = DN_LOG_PATH,
		wr_log_path : str | Path = WR_LOG_PATH,
		verbose: bool = True,
		stp_on_fail: bool = False,  
	) -> None:
		self.que = que
		self.runs_path = Path(runs_path)
		self.wr_path = exec_path
		self.dn_log_path = dn_log_path
		self.wr_log_path = wr_log_path
		self.verbose = verbose
		self.stp_on_fail = stp_on_fail
		self.worker = worker(self.que) 
		
		self.logger = logging.getLogger(__name__)
  

	# def print_v(self, message: str) -> None:
	# 	"""Prints a message if verbose is True."""
		
			# print(message)


	def seperator(self, run: ExpInfo) -> str:
		sep = ""
		r_str = self.que.run_str(self.que.run_sum(run))

		if r_str:
			sep += ("\n" * 2) + ("-" * 10) + ("\n")
			sep += f"{r_str:^10}"
			sep += ("\n" * 2) + ("-" * 10) + ("\n")
		else:
			sep += "\n"
		return sep.title()


   
	def start(self):
		while True:
			# prepare next run (move from to_run -> cur_run)
			try:
				self.que.load_state()
				self.que.stash_next_run()
				self.que.save_state()
			except QueEmpty:
				self.logger.info("No more runs to execute")
				break
			except QueBusy:
				self.logger.info("Cannot overwrite current run")
				break

			run = self.que.peak_cur_run()
			self.logger.info(self.seperator(run))

			# Start process in background
			p = multiprocessing.Process(target=self.worker.run)
			p.start()
			p.join()

			if p.exitcode == 0:
				self.logger.info("Process completed successfully")
				# save finished run (move from cur_run -> old_runs)
				self.que.store_fin_run()
				self.que.save_state()
			else:
				self.logger.warning(f"Process failed with exit code: {p.exitcode}")

				if self.stp_on_fail:
					self.logger.warning("Stopping exectuion")
					break
				else:
					self.logger.warning("Continuing with next run")
	 
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

def main():
	# parser = get_daemon_parser()
	# args = parser.parse_args()

	# daem = daemon()

	# setting = args.setting

	# if setting == 'sMonitor':
	# 	daem.start_n_monitor()
	# else:
	# 	print('huh?')
	# 	print(f'You gave me: {setting}')
	# 	print('but i only accept: ["sWatch", "sMonitor"]')
	server = connect_manager()
	daem = daemon(server.get_que())
	daem.debug2()
  
if __name__ == '__main__':
	main()
			
	
	