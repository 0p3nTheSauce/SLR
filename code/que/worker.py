from typing import Optional, IO, cast
from pathlib import Path
import time
import logging
from logging import Logger
from contextlib import redirect_stdout
import io
import sys
from multiprocessing.synchronize import Event as EventClass
from typing import Protocol, runtime_checkable
# locals
from .core import RUN_PATH, WR_LOG_PATH, WR_NAME, QueException, ExpInfo


from .core import Que, connect_manager
from utils import gpu_manager
from configs import _exp_to_run_info
from training import train_loop, _setup_wandb



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




# class Worker:
#     def __init__(
#         self,
#         que: Que,
#         logger: Logger
#     ) -> None:
#         self.que = que
#         self.logger = logger
#         self.log_adapter: IO[str] = cast(IO[str], LoggerWriter(logger))

#     def seperator(self, r_str: str) -> str:
#         sep = ""

#         if r_str:
#             sep += ("\n" * 2) + ("-" * 10) + ("\n")
#             sep += f"{r_str:^10}"
#             sep += ("\n" * 2) + ("-" * 10) + ("\n")
#         else:
#             sep += "\n"
#         return sep.title()

#     def _work(self, event: EventClass) -> Optional[ExpInfo]:
#         gpu_manager.wait_for_completion()
#         self.logger.info("starting work")
#         # prepare next run (move from to_run -> cur_run)
#         run_sum = self.que.stash_next_run()
#         self.que.save_state()
        
#         # self.logger.info a seperator between runs
#         self.logger.info(self.seperator(run_sum))
#         # get next run
#         info = self.que.peak_cur_run()

#         wandb_info = info["wandb"]
#         admin = info["admin"]
#         config = _exp_to_run_info(info)

#         # setup wandb run
#         run = _setup_wandb(config, wandb_info)
#         if run is None:
#             return

#         # save run_id for recovering
#         wandb_info["run_id"] = run.id
#         info["wandb"] = wandb_info

#         self.logger.info("saving my id")
#         _ = self.que.pop_cur_run()
#         self.que.set_cur_run(info)
#         self.que.save_state()

#         self.logger.info(f"Run ID: {run.id}")
#         self.logger.info(f"Run name: {run.name}")  # Human-readable name
#         self.logger.info(f"Run path: {run.path}")  # entity/project/run_id format
        
#         with redirect_stdout(self.log_adapter):
#             train_loop(admin["model"], config, run, recover=admin["recover"], event=event)
            
#         if not event.is_set():
            
#             self.logger.info("Training finished successfully")
#             self.que.store_fin_run()
#             self.que.save_state()
#         else:
#             self.logger.warning("Training was interrupted before completion.")
#             self.que.save_state() #keep current run for recovery
#             #pause wandb run without finishing
            
#         run.finish()

#     def work(self, event: EventClass) -> Optional[ExpInfo]:
#         try:
#             self._work(event)
#         except QueException as Qe:
#             self.logger.info(f"que based error, cannot continue: {Qe}")
#             raise 
#         except KeyboardInterrupt:
#             self.logger.info("Worker killed by user")
#         except Exception as e:
#             self.logger.info(f"Training run failed due to an error: {e}")
#             self.que.stash_failed_run(str(e))
#             self.que.save_state()

#     def idle(self, event: EventClass ):
#         self.logger.info(f"Busy with: \n {self.que.run_str('cur_run', 0)}")
#         for i in range(100):
#             self.logger.info(f'working...{i}')
#             time.sleep(1)
#             if event.is_set():
#                 self.logger.info('stop event detected, finishing work early')
#                 break
#         self.logger.info('finished working')
        
#     def start(self, event: EventClass ):
#         self.idle(event) #dummy method to plug actual functionality
#         # self.work(event)
        


class Worker:
    def __init__(
        self,
        logger: Logger
    ) -> None:
        self.logger = logger
        self.log_adapter: IO[str] = cast(IO[str], LoggerWriter(logger))

    def seperator(self, r_str: str) -> str:
        sep = ""

        if r_str:
            sep += ("\n" * 2) + ("-" * 10) + ("\n")
            sep += f"{r_str:^10}"
            sep += ("\n" * 2) + ("-" * 10) + ("\n")
        else:
            sep += "\n"
        return sep.title()

    def _work(self, event: EventClass, que: Que) -> Optional[ExpInfo]:
        gpu_manager.wait_for_completion()
        self.logger.info("starting work")
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
        run = _setup_wandb(config, wandb_info)
        if run is None:
            return

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
            #pause wandb run without finishing
            
        run.finish()

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
            self.logger.info('finished working')
            que.store_fin_run()
            # que.stash_next_run()
                
    def start(self, event: EventClass ):
        #this is likely started in a seperate process, so que requirs connecting
        manager = connect_manager()
        que = manager.get_que()
        # self.idle(event, que) #dummy method to plug actual functionality
        # self.idle2(event, que)
        self.work(event, que)
        
    
        





if __name__ == "__main__":
    pass
