from typing import  Optional
import subprocess
from pathlib import Path
from quewing import tmux_manager, WR_NAME, SESH_NAME, RUN_PATH, WR_PATH, DN_NAME, LOG_PATH
from quewing import QueEmpty, QueBusy, ExpInfo
from argparse import ArgumentParser

from que_server import connect_que

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
		runs_path: str = RUN_PATH,
		exec_path: str = WR_PATH,
		log_path : str = LOG_PATH,
		verbose: bool = True,
		stp_on_fail: bool = False,  # TODO: add this to parser
	) -> None:
		self.name = name
		self.wr_name = wr_name
		self.sesh = sesh
		self.runs_path = Path(runs_path)
		self.wr_path = exec_path
		self.log_path = log_path
		self.que = connect_que()
		self.verbose = verbose
		self.stp_on_fail = stp_on_fail

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


	def start_n_monitor(self):
		"""Start process and use existing tmux monitoring"""
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
			# except Timeout:
			# 	self.print_v(
			# 		"Cannot stash next run, file is already held by another process"
			# 	)
			# 	break

			run = self.que.peak_cur_run()
			self.print_v(self.seperator(run))

			# Start process in background
			proc = self.worker_log()

			# # Start monitoring in tmux (non-blocking)
			# self.monitor_log()

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
				# except Timeout:
				# 	self.print_v(
				# 		"Cannot store finished run, file is already held by another process"
				# 	)
				# 	break
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
			cmd, stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT
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
	parser = get_daemon_parser()
	args = parser.parse_args()

	daem = daemon()

	setting = args.setting

	if setting == 'sMonitor':
		daem.start_n_monitor()
	else:
		print('huh?')
		print(f'You gave me: {setting}')
		print('but i only accept: ["sWatch", "sMonitor"]')
  
	
			
	
    