from typing import Optional, IO, cast
import time
import torch
import gc
import logging
from logging import Logger
from contextlib import redirect_stdout
import io
import os
from multiprocessing.synchronize import Event as EventClass
# locals
from .core import QueException, ExpInfo, WORKER_NAME, TRAINING_LOG_PATH


from .core import Que, connect_manager, QueException 
from utils import gpu_manager
from configs import _exp_to_run_info
from training import train_loop, _setup_wandb
import wandb


class LoggerWriter(io.TextIOBase):
    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, s: str) -> int:
        if s.strip():
            self.logger.log(self.level, s.strip())
        # File objects must return the number of characters written
        return len(s) 

    def flush(self):
        # We need to explicitly implement flush, but it doesn't need to do anything
        pass

class Worker:
    def __init__(
        self,
        server_logger: Logger,
    ) -> None:
        self.server_logger = server_logger
        self.server_logger.info("Worker initialized")

    def seperator(self, r_str: str) -> str:
        sep = ""

        if r_str:
            sep += ("\n" * 2) + ("-" * 10) + ("\n")
            sep += f"{r_str:^10}"
            sep += ("\n" * 2) + ("-" * 10) + ("\n")
        else:
            sep += "\n"
        return sep.title()

    def cleanup(self):
        self.server_logger.info("Cleaning up GPU memory")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect() # Clears memory shared between processes


    def _work(self, event: EventClass, que: Que) -> Optional[ExpInfo]:
        try:
            self.server_logger.info(f"starting work with pid: {os.getpid()}")

            self.server_logger.info("Checking GPU usage")
            used, total = gpu_manager.get_gpu_memory_usage()
            self.server_logger.info(f"Current GPU usage: {used}/{total} GiB")

            if not gpu_manager.wait_for_completion(
                check_interval=10,
                logger=self.server_logger,
                # max_util_gb=0.5, #debugging
                event=event,
            ):
                self.server_logger.info("GPU not available, exiting")
                return
            else:
                self.server_logger.info("GPU is available")
            
            # prepare next run (move from to_run -> cur_run)
            run_sum = que.stash_next_run()
            que.save_state()

            # self.server_logger.info a seperator between runs
            self.server_logger.info(self.seperator(run_sum))
            # get next run
            info = que.peak_cur_run()

            wandb_info = info["wandb"]
            admin = info["admin"]
            config = _exp_to_run_info(info)

            # setup wandb run
            with redirect_stdout(self.log_adapter):
                run = _setup_wandb(config, wandb_info, run_id_required=True)

            # save run_id for recovering
            wandb_info["run_id"] = run.id
            info["wandb"] = wandb_info

            self.server_logger.info("saving my id")
            _ = que.pop_cur_run()
            que.set_cur_run(info)
            que.save_state()

            self.server_logger.info(f"Run ID: {run.id}")
            self.server_logger.info(f"Run name: {run.name}")  # Human-readable name
            self.server_logger.info(f"Run path: {run.path}")  # entity/project/run_id format

            with redirect_stdout(self.log_adapter):
                train_loop(admin["model"], config, run, recover=admin["recover"], event=event)
            
            if not event.is_set():
            
                self.server_logger.info("Training finished successfully")
                que.store_fin_run()
                que.save_state()
            else:
                self.server_logger.warning("Training was interrupted before completion.")
                que.save_state() #keep current run for recovery
            
            with redirect_stdout(self.log_adapter):
                run.finish(exit_code=0)
        
            self.server_logger.info("_work method completed successfully")  
            return None  
            
        finally:
            self.server_logger.info("Exiting _work method")
            
            
    def work(self, event: EventClass, que: Que) -> Optional[ExpInfo]:
        try:
            self._work(event, que)
        except QueException as Qe:
            self.server_logger.info(f"que based error, cannot continue: {Qe}")
            raise 
        except KeyboardInterrupt:
            self.server_logger.info("Worker killed by user")
        except Exception as e:
            self.server_logger.error(f"Training run failed due to an error: {e}")
            que.stash_failed_run(str(e))
            que.save_state()
            #exit with error
            raise
                
    def start(self, event: EventClass ):
        #this is likely started in a seperate process, so que requirs connecting
        
        manager = connect_manager()
        que = manager.get_que()
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=TRAINING_LOG_PATH,
        )
        self.training_logger = logging.getLogger(WORKER_NAME)
        self.log_adapter: IO[str] = cast(IO[str], LoggerWriter(self.training_logger))
        
        self.work(event, que)
        
    
        





if __name__ == "__main__":
    pass
