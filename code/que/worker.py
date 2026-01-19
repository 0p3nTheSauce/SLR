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
from .core import QueException, ExpInfo


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
        training_logger: Logger
    ) -> None:
        self.logger = server_logger
        self.log_adapter: IO[str] = cast(IO[str], LoggerWriter(training_logger))

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
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect() # Clears memory shared between processes


    def _work(self, event: EventClass, que: Que) -> Optional[ExpInfo]:
        try:
            self.logger.info(f"starting work with pid: {os.getpid()}")

            self.logger.info("Attempting to clean first")
            used, total = gpu_manager.get_gpu_memory_usage()
            self.logger.info(f"Current usage: {used}/{total} GiB")
            self.cleanup()
            used, total = gpu_manager.get_gpu_memory_usage()
            self.logger.info(f"After cleanup: {used}/{total} GiB")

            self.logger.info("Checking GPU usage")

            if not gpu_manager.wait_for_completion(
                check_interval=10,
                logger=self.logger,
                # max_util_gb=0.5, #debugging
                event=event,
            ):
                self.logger.info("GPU not available, exiting")
                return
            else:
                self.logger.info("GPU is available")
            
            # prepare next run (move from to_run -> cur_run)
            run_sum = que.stash_next_run()
            que.save_state()
        
            # self.logger.info a seperator between runs
            self.logger.info(self.seperator(run_sum))
            # get next run
            info = que.peak_cur_run()

            wandb_info = info["wandb"]
            admin = info["admin"]
            config = _exp_to_run_info(info)

            # setup wandb run
            run = _setup_wandb(config, wandb_info, run_id_required=True)

            # save run_id for recovering
            wandb_info["run_id"] = run.id
            info["wandb"] = wandb_info

            self.logger.info("saving my id")
            _ = que.pop_cur_run()
            que.set_cur_run(info)
            que.save_state()

            self.logger.info(f"Run ID: {run.id}")
            self.logger.info(f"Run name: {run.name}")  # Human-readable name
            self.logger.info(f"Run path: {run.path}")  # entity/project/run_id format
        
            with redirect_stdout(self.log_adapter):
                train_loop(admin["model"], config, run, recover=admin["recover"], event=event)
            
            if not event.is_set():
            
                self.logger.info("Training finished successfully")
                que.store_fin_run()
                que.save_state()
            else:
                self.logger.warning("Training was interrupted before completion.")
                que.save_state() #keep current run for recovery
            
            run.finish(exit_code=0, quiet=True)
        
            self.logger.info("_work method completed successfully")  # ADD THIS
            return None  
            
        finally:
            self.logger.info("Exiting _work method")

    def work(self, event: EventClass, que: Que) -> Optional[ExpInfo]:
        try:
            self._work(event, que)
        except QueException as Qe:
            self.logger.info(f"que based error, cannot continue: {Qe}")
            raise 
        except KeyboardInterrupt:
            self.logger.info("Worker killed by user")
        except Exception as e:
            self.logger.error(f"Training run failed due to an error: {e}")
            que.stash_failed_run(str(e))
            que.save_state()
            #exit with error
            raise
        # finally:
        #     self.logger.info("Cleaning up")
        #     self.cleanup()

    def idle(self, event: EventClass, que: Que):
        
        for i in range(100):
            try:
                run_str = que.run_str('cur_run', 0)
            except QueException:
                run_str = "No current run"
            self.logger.info(f"Busy with: \n {run_str}")
            self.logger.info(f'working...{i}')
            time.sleep(1)
            if event.is_set():
                self.logger.info('stop event detected, finishing work early')
                break
        self.logger.info('finished working')
        
    def idle2(self, event: EventClass, que: Que):
        while not event.is_set():
            try:
                run_str = que.run_str('cur_run', 0)
            except QueException:
                run_str = que.stash_next_run()
            self.logger.info(f"Busy with: \n {run_str}")
            for i in range(20):
                self.logger.info(f'working...{i}')
                time.sleep(1)
                if event.is_set():
                    self.logger.info('stop event detected, finishing work early')
                    break
            que.store_fin_run()
            self.logger.info('finished working')
            # que.stash_next_run()
                
    def idle3(self, event: EventClass, que: Que):    
        try:
            run_str = que.run_str('cur_run', 0)
        except QueException:
            run_str = que.stash_next_run()
        self.logger.info(f"Busy with: \n {run_str}")
        for i in range(20):
            self.logger.info(f'working...{i}')
            time.sleep(1)
            if event.is_set():
                self.logger.info('stop event detected, finishing work early')
                break
        que.store_fin_run()
        self.logger.info('finished working')
            # que.stash_next_run()
                
                
    def start(self, event: EventClass ):
        #this is likely started in a seperate process, so que requirs connecting
        
        manager = connect_manager()
        que = manager.get_que()
        # self.idle(event, que) #dummy method to plug actual functionality
        # self.idle3(event, que)
        
        
        self.work(event, que)
        
    
        





if __name__ == "__main__":
    pass
