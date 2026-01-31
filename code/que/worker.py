import os
from multiprocessing import Process
from typing import Optional, IO, cast

import torch
import gc
import logging
from logging import Logger
from contextlib import redirect_stdout
import io

from multiprocessing.synchronize import Event as EventClass

# locals
from .core import connect_manager, TRAINING_LOG_PATH, WORKER_NAME, QueException, WorkerState, Worker_tasks


from .core import Que
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


class Worker:
    def __init__(
        self,
        server_logger: Logger,
        training_logger: Logger,
        que: Que,
        stop_event: Optional[EventClass] = None
    ) -> None:
        self.server_logger = server_logger
        self.training_logger = training_logger
        self.que = que
        self.log_adapter: IO[str] = cast(IO[str], LoggerWriter(self.training_logger))
        self.stop_event: Optional[EventClass] = stop_event
        self.current_task: Worker_tasks = 'inactive'
        self.current_run_id: Optional[str] = None
        self.working_pid: Optional[int] = None
        self.exception: Optional[str] = None

        self.server_logger.info("Worker initialized")

    def get_state(self) -> WorkerState:
        return WorkerState(
            task=self.current_task if self.current_task else "training",
            current_run_id=self.current_run_id,
            working_pid=self.working_pid,
            exception=self.exception,
        )

    def set_state(self, state: WorkerState) -> None:
        self.current_task = state["task"]
        self.current_run_id = state["current_run_id"]
        self.working_pid = state["working_pid"]
        self.exception = state["exception"]

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
        self.server_logger.debug("Cleaning up GPU memory")
        used, total = gpu_manager.get_gpu_memory_usage()
        self.server_logger.debug(f"Current GPU usage: {used}/{total} GiB")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()  # Clears memory shared between processes
        used, total = gpu_manager.get_gpu_memory_usage()
        self.server_logger.debug(f"GPU memory after cleanup: {used}/{total} GiB")

    def _train(self) -> None:
        try:
            if not gpu_manager.wait_for_completion(
                check_interval=10,
                logger=self.server_logger,
                # max_util_gb=0.5, #debugging
                event=self.stop_event,
            ):
                self.server_logger.info("GPU not available, exiting")
                return
            else:
                self.server_logger.info("GPU is available")

            # prepare next run (move from to_run -> cur_run)
            run_sum = self.que.stash_next_run()
            self.que.save_state()

            # self.logger.info a seperator between runs
            self.server_logger.info(self.seperator(run_sum))
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

            # save for server context
            self.current_run_id = run.id

            self.server_logger.info("saving my id")
            _ = self.que.pop_cur_run()
            self.que.set_cur_run(info)
            self.que.save_state()

            self.server_logger.info(f"Run ID: {run.id}")
            self.server_logger.info(f"Run name: {run.name}")  # Human-readable name
            self.server_logger.info(
                f"Run path: {run.path}"
            )  # entity/project/run_id format

            with redirect_stdout(self.log_adapter):
                train_loop(
                    admin["model"],
                    config,
                    run,
                    recover=admin["recover"],
                    event=self.stop_event,
                )
                run.finish(exit_code=0)

            if self.stop_event is not None and self.stop_event.is_set():
                self.server_logger.warning(
                    "Training was interrupted by stopping event."
                )
                self.que.save_state()  # keep current run for recovery
            else:
                self.server_logger.info("Training finished successfully")
                self.que.store_fin_run()
                self.que.save_state()
            self.server_logger.info("_train method completed successfully")
        finally:
            self.server_logger.info("Exiting _train method")
            self.cleanup()

    def train(self) -> None:
        """
        The train method is the main entry point for training a model.
        Currently implemented to be in a process started by the Daemon.
        """
        try:
            self.current_task = "training"
            self._train()
        except QueException as Qe:
            self.server_logger.info(f"que based error, cannot continue: {Qe}")
            self.exception = str(Qe)
            raise
        except KeyboardInterrupt:
            self.server_logger.info("Worker killed by user")
            self.exception = "KeyboardInterrupt"
        except Exception as e:
            self.server_logger.error(f"Training run failed due to an error: {e}")
            self.exception = str(e)
            self.que.stash_failed_run(str(e))
            self.que.save_state()
            # exit with error
            raise
        finally:
            self.current_task = "inactive"

    def start(self, event: EventClass):
        #this is likely started in a seperate process, so que requirs connecting

        self.stop_event = event
        self.working_pid = os.getpid()
        
        manager = connect_manager()
        self.que = manager.get_que()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=TRAINING_LOG_PATH,
        )
        self.training_logger = logging.getLogger(WORKER_NAME)
        self.log_adapter: IO[str] = cast(IO[str], LoggerWriter(self.training_logger))
        self.logger = self.training_logger

        self.train()



if __name__ == "__main__":
    pass
