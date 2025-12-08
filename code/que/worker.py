from typing import Optional
from pathlib import Path

import logging

# locals
from .core import RUN_PATH, WR_LOG_PATH, WR_NAME, QueException, ExpInfo

from .server import connect_manager
from .core import Que
from utils import gpu_manager
from configs import _exp_to_run_info
from training import train_loop, _setup_wandb



class Worker:
    def __init__(
        self,
        que: Que,
        runs_path: str | Path = RUN_PATH,
        verbose: bool = True,
        stp_on_fail: bool = False,
    ) -> None:
        self.que = que
        self.runs_path = Path(runs_path)
        self.verbose = verbose
        self.stp_on_fail = stp_on_fail

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=WR_LOG_PATH,  # Optional: log to file
        )
        
        self.logger = logging.getLogger(WR_NAME)

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
                self.logger.info(self.seperator(run_sum))
                # start training
                self.train()
                self.logger.info("Training finished successfully")
                # move to old runs (cur_run -> old_runs)
                self.que.store_fin_run()
                self.que.save_state()
            except QueException as Qe:
                self.logger.critical(f"que based error, cannot continue: {Qe}")
                raise Qe
            except KeyboardInterrupt:
                self.logger.warning("Worker killed by user")
                break
            except Exception as e:
                self.logger.error(f"Training run failed due to an error: {e}")
                self.que.stash_failed_run(str(e))
                self.que.save_state()
                if self.stp_on_fail:
                    self.logger.info(
                        "Stopping from failure, tor stp_on_fail to True for continuation behaviour"
                    )
                    break

    def train(self) -> Optional[ExpInfo]:
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
    server = connect_manager()
    w = Worker(server.get_Que())
    w.start()


if __name__ == "__main__":
    main()
