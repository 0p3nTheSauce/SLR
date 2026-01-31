from multiprocessing import Process
from typing import Literal, Optional, IO, TypeAlias, cast, TypedDict
import time
import torch
import gc
import logging
from logging import Logger
from contextlib import redirect_stdout
import io
import os
from multiprocessing.synchronize import Event as EventClass

import training
# locals
from .core import QueException, ExpInfo, WORKER_NAME, TRAINING_LOG_PATH, WorkerState, Worker_tasks


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
        logger: Logger,
        que: Que,
    ) -> None:
        self.logger = logger
        self.que = que
        self.stop_event: Optional[EventClass] = None
        self.current_task: Worker_tasks = 'training'
        self.current_run_id: Optional[str] = None
        self.subprocess_pid: Optional[int] = None
        self.working_process: Optional[Process] = None
        self.exception: Optional[str] = None
        self.logger.info("Worker initialized")

    def get_state(self) -> WorkerState:
        
        return WorkerState(
            task=self.current_task if self.current_task else 'training',
            current_run_id=self.current_run_id,
            working_pid=self.subprocess_pid,
            exception=self.exception,
        )
        
    def set_state(self, state: WorkerState) -> None:
        self.current_task = state['task']
        self.current_run_id = state['current_run_id']
        self.subprocess_pid = state['working_pid']
        self.exception = state['exception']

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
        self.logger.debug("Cleaning up GPU memory")
        used, total = gpu_manager.get_gpu_memory_usage()
        self.logger.debug(f"Current GPU usage: {used}/{total} GiB")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect() # Clears memory shared between processes
        used, total = gpu_manager.get_gpu_memory_usage()
        self.logger.debug(f"GPU memory after cleanup: {used}/{total} GiB")

    def _train(self) -> None:
        try:
            if not gpu_manager.wait_for_completion(
                check_interval=10,
                logger=self.logger,
                # max_util_gb=0.5, #debugging
                event=self.stop_event,
            ):
                self.logger.info("GPU not available, exiting")
                return
            else:
                self.logger.info("GPU is available")
            
            # prepare next run (move from to_run -> cur_run)
            run_sum = self.que.stash_next_run()
            self.que.save_state()

            # self.logger.info a seperator between runs
            self.logger.info(self.seperator(run_sum))
            # get next run
            info = self.que.peak_cur_run()

            wandb_info = info["wandb"]
            admin = info["admin"]
            config = _exp_to_run_info(info)

            # setup wandb run
            with redirect_stdout(self.log_adapter):
                run = _setup_wandb(config, wandb_info, run_id_required=True)

            # save run_id for recovering
            wandb_info["run_id"] = run.id
            info["wandb"] = wandb_info
            
            #save for server context
            self.current_run_id = run.id

            self.logger.info("saving my id")
            _ = self.que.pop_cur_run()
            self.que.set_cur_run(info)
            self.que.save_state()

            self.logger.info(f"Run ID: {run.id}")
            self.logger.info(f"Run name: {run.name}")  # Human-readable name
            self.logger.info(f"Run path: {run.path}")  # entity/project/run_id format

            with redirect_stdout(self.log_adapter):
                train_loop(admin["model"], config, run, recover=admin["recover"], event=self.stop_event)
                run.finish(exit_code=0)
            
            if self.stop_event is not None and self.stop_event.is_set():
                self.logger.warning("Training was interrupted by stopping event.")
                self.que.save_state() #keep current run for recovery
            else:
                self.logger.info("Training finished successfully")
                self.que.store_fin_run()
                self.que.save_state()
            self.logger.info("_train method completed successfully")  
        finally:
            self.logger.info("Exiting _train method")
            self.cleanup()
            
    def train(self) -> None:
        """
        The train method is the main entry point for training a model.
        Currently implemented to be in a process started by the Daemon.
        
        :param event: Multiprocessing event to signal termination
        :type event: EventClass
        :param que: Shared Que object for managing training runs
        :type que: Que
        :param logger: Logger for logging messages
        :type logger: Logger
        :return: None
        """
        try:
            self.current_task = 'training'
            self._train()
        except QueException as Qe:
            self.logger.info(f"que based error, cannot continue: {Qe}")
            self.exception = str(Qe)
            raise 
        except KeyboardInterrupt:
            self.logger.info("Worker killed by user")
            self.exception = "KeyboardInterrupt"
        except Exception as e:
            self.logger.error(f"Training run failed due to an error: {e}")
            self.exception = str(e)
            self.que.stash_failed_run(str(e))
            self.que.save_state()
            #exit with error
            raise
        finally:
            self.current_task = 'inactive'
            
        
    def work(self, event: EventClass):
        #this is likely started in a seperate process, so que requirs connecting
        self.stop_event = event
        self.stop_event.clear()
        
        manager = connect_manager()
        que = manager.get_que()
        
        
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=TRAINING_LOG_PATH,
        )
        self.logger = logging.getLogger(WORKER_NAME)
        self.log_adapter: IO[str] = cast(IO[str], LoggerWriter(self.logger))
        
        self.train()
                
    def start(self, event: EventClass):
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
        
        self.train(event, que, self.training_logger)
    
    
        





if __name__ == "__main__":
    pass
