from typing import  Optional
from pathlib import Path

import logging
#locals
from .core import (
	RUN_PATH,
	CUR_RUN,
	WR_LOG_PATH,
	QueEmpty,
	QueBusy,
	QueException,
	ExpInfo
)

from .server import connect_manager
from .core import que
from utils import gpu_manager
from configs import _exp_to_run_info
from training import train_loop, _setup_wandb

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	filename=WR_LOG_PATH  # Optional: log to file
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
		verbose: bool = True,
		stp_on_fail: bool = False,  
	) -> None:
		self.que = que
		self.runs_path = Path(runs_path)
		self.verbose = verbose
		self.stp_on_fail = stp_on_fail
		
		self.logger = logging.getLogger(__name__)

	def seperator(self, r_str: str) -> str:
		sep = ""

		if r_str:
			sep += ("\n" * 2) + ("-" * 10) + ("\n")
			sep += f"{r_str:^10}"
			sep += ("\n" * 2) + ("-" * 10) + ("\n")
		else:
			sep += "\n"
		return sep.title()


   
	def start(self):
		while True: 
			try:
				# prepare next run (move from to_run -> cur_run)
				run_sum = self.que.stash_next_run()
				self.que.save_state()
				# print a seperator between runs
				self.logger.info(self.seperator(
					run_sum
				))
				# start training
				self.train()
				self.logger.info('Training finished successfully')
				# move to old runs (cur_run -> old_runs)
				self.que.store_fin_run()
				self.que.save_state()
			except QueException as Qe:
				self.logger.critical(f'que based error, cannot continue: {Qe}')
				raise Qe
			except KeyboardInterrupt:
				self.logger.warning('Worker killed by user')
				break 
			except Exception as e:
				self.logger.error(f"Training run failed due to an error: {e}")
				self.que.stash_failed_run(str(e))
				self.que.save_state()
				if self.stp_on_fail:
					self.logger.info('Stopping from failure, tor stp_on_fail to True for continuation behaviour')
					break
   
    
    
	def train(self) ->  Optional[ExpInfo]:
		
		gpu_manager.wait_for_completion()
		self.logger.info("starting work")

		# get next run
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

		self.logger.info("saving my id")
		self.que.set_cur_run(info)
		self.que.save_state()

		self.logger.info(f"Run ID: {run.id}")
		self.logger.info(f"Run name: {run.name}")  # Human-readable name
		self.logger.info(f"Run path: {run.path}")  # entity/project/run_id format

		train_loop(admin["model"], config, run, recover=admin["recover"])
		run.finish()
		

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

  
if __name__ == '__main__':
	main()
			
	
	