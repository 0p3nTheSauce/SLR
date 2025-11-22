from typing import  Optional
import subprocess
from pathlib import Path
from argparse import ArgumentParser
import sys
import multiprocessing
#locals
from .core import (
	WR_NAME,
	SESH_NAME,
	RUN_PATH,
	WR_PATH,
	DN_NAME,
	DN_LOG_PATH,
	WR_LOG_PATH,
	QueEmpty,
	QueBusy,
	ExpInfo
)

from .server import connect_que
from .worker import worker

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
		name: str = DN_NAME,
		wr_name: str = WR_NAME,
		sesh: str = SESH_NAME,
		runs_path: str | Path = RUN_PATH,
		exec_path: str | Path = WR_PATH,
		dn_log_path : str | Path = DN_LOG_PATH,
		wr_log_path : str | Path = WR_LOG_PATH,
		verbose: bool = True,
		stp_on_fail: bool = False,  # TODO: add this to parser
	) -> None:
		self.name = name
		self.wr_name = wr_name
		self.sesh = sesh
		self.runs_path = Path(runs_path)
		self.wr_path = exec_path
		self.dn_log_path = dn_log_path
		self.wr_log_path = wr_log_path
		self.que = connect_que()
		self.verbose = verbose
		self.stp_on_fail = stp_on_fail
		self.worker = worker() #TODO: maybe pass args?

	def print_v(self, message: str) -> None:
		"""Prints a message if verbose is True."""
		if self.verbose:
			print(message)

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

	def start_log(self):
		print(self.dn_log_path)
		
		with open(self.dn_log_path, 'w', buffering=1) as log_file:
			sys.stdout = log_file
			sys.stderr = log_file
			print("Starting daemon log...")
			self.run_worker()
			
			# sys.stdout.flush()
   
   
	def run_worker(self):
		while True:
			# prepare next run (move from to_run -> cur_run)
			try:
				self.que.load_state()
				self.que.stash_next_run()
				self.que.save_state()
			except QueEmpty:
				self.print_v("No more runs to execute")
				break
			except QueBusy:
				self.print_v("Cannot overwrite current run")
				break

			run = self.que.peak_cur_run()
			self.print_v(self.seperator(run))

			# Start process in background
			p = multiprocessing.Process(target=self.worker.run)
			p.start()
			p.join()

			if p.exitcode == 0:
				self.print_v("Process completed successfully")
				# save finished run (move from cur_run -> old_runs)
				self.que.store_fin_run()
				self.que.save_state()
			else:
				self.print_v(f"Process failed with exit code: {p.exitcode}")

				if self.stp_on_fail:
					self.print_v("Stopping exectuion")
					break
				else:
					self.print_v("Continuing with next run")
	
	#Subprocess version below - deprecated

	def start_n_monitor(self):
		"""Start process using subprocess.Popen and use existing tmux monitoring"""
		while True:
			# prepare next run (move from to_run -> cur_run)
			try:
				self.que.load_state()
				self.que.stash_next_run()
				self.que.save_state()
			except QueEmpty:
				self.print_v("No more runs to execute")
				break
			except QueBusy:
				self.print_v("Cannot overwrite current run")
				break
 

			run = self.que.peak_cur_run()
			self.print_v(self.seperator(run))

			# Start process in background
			proc = self.worker_log()

			# Wait for completion
			return_code = proc.wait()

			if return_code == 0:
				self.print_v("Process completed successfully")
				# save finished run (move from cur_run -> old_runs)
				try:
					self.que.store_fin_run()
					self.que.save_state()
				except QueEmpty:
					self.print_v("Could not find current run")
					break
			else:
				self.print_v(f"Process failed with return code: {return_code}")

				if self.stp_on_fail:
					self.print_v("Stopping exectuion")
					break
				else:
					self.print_v("Continuing with next run")
 
	def worker_here(self, args: Optional[list[str]] = None) -> None:
		"""Blocking start which prints worker output in daemon terminal"""

		cmd = [self.wr_path, "work"]
		if args:
			cmd.extend(args)
		subprocess.run(cmd, check=True)

	def worker_log(self, args: Optional[list[str]] = None) -> subprocess.Popen:
		"""Non-blocking start which prints worker output to LOG_PATH, and passes the process"""

		cmd = [self.wr_path, "work"]
		if args:
			cmd.extend(args)

		return subprocess.Popen(
			cmd, stdout=open(self.wr_log_path, "w"), stderr=subprocess.STDOUT
		)

	  
def get_daemon_parser() -> ArgumentParser:
	parser = ArgumentParser(description="Run the que daemon process")
	parser.add_argument(
		"setting",
		type=str,
		choices=['sWatch', 'sMonitor'],
		help="Daemon mode to run in",
	)
	parser.add_argument(
		"--recover",
		action="store_true",
		help="Recover from last run in queTemp.json",
	)
	parser.add_argument(
		"--run_id",
		type=str,
		default=None,
		help="WandB run ID to recover (if applicable)",
	)
	return parser

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
	daem = daemon()
	daem.start_log()
  
if __name__ == '__main__':
	main()
			
	
	