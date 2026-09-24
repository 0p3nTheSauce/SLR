import gc
import io
import json
import logging
import os
import traceback
from contextlib import redirect_stdout
from logging import Logger
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from typing import IO, cast

import torch
from pydantic import ValidationError

import wandb
from src.que.core import (
    CUR_RUN,
    SERVER_LOG_PATH,
    TRAINING_LOG_PATH,
    TRAINING_NAME,
    WORKER_NAME,
    CompExpInfo,
    Que,
    QueException,
    ServerContextProtocol,
    SweepInfo,  # now also carries model/split/dataset -- see note below
    SweepProgressDict,
    WorkerStateDict,
    connect_manager,
    sweep_info_validate,
)
from src.run_types import RunInfo, WandbInfo
from src.sweeping import create_sweep_run

# locals
from src.testing import full_test
from src.training import _setup_wandb, train_loop
from src.utils import gpu_manager


def _is_wandb_injected_stop(exc: Exception) -> bool:
    """Heuristic for wandb's Hyperband/early-terminate stop signal.

    wandb kills an in-process sweep trial (no subprocess to terminate) via
    ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, Exception) -- see
    wandb/agents/pyagent.py's _terminate_thread. That raises the bare `Exception`
    class with no args wherever the thread happens to be executing. A base
    `Exception` with an empty message is not otherwise raised anywhere in this
    codebase (see src/que/todo), so this is a reliable-in-practice signature,
    not a certainty.
    """
    return type(exc) is Exception and not exc.args


class SweepTrialFailed(Exception):
    """A sweep trial crashed inside wandb.agent's callback.

    wandb's agent swallows any exception from the callback (see Worker._sweep_train), so the
    original is recorded there and re-raised as this from Worker.sweep() once wandb.agent returns
    -- making the worker process exit non-zero so the Daemon's stop_on_fail applies, the same as
    a crash in Worker.train(). The original exception is chained as `__cause__`.
    """

    def __init__(self, sweep_id: str):
        super().__init__(f"Sweep {sweep_id} trial failed")


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
        que: Que,
        state: WorkerStateDict,
        stop_event: EventClass | None = None,
        do_traceback: bool = True
    ) -> None:
        self.server_logger = server_logger
        self.que = que
        self.stop_event: EventClass | None = stop_event
        self.state = state
        self.do_traceback = do_traceback
        self.sweep_info: SweepInfo | None = None
        self._trial_error: Exception | None = None
        self.live_sweep: SweepInfo | dict | None = None
        self.sweep_progress: SweepProgressDict | None = None
        self.server_context: ServerContextProtocol | None = None
        self.server_logger.info("Worker initialized")


    def build_exception_info(self, e: Exception) -> str:
        if self.do_traceback:
            return f"{e!s}\n{traceback.format_exc()}"
        else:
            return str(e)

    def _fail(self, exc: Exception, label: str, clear_pid: bool = True) -> None:
        """Shared failure-path state update for train/sweep/test except blocks."""
        self.server_logger.info(f"{label}: {exc}")
        self.state['exception'] = self.build_exception_info(exc)
        if clear_pid:
            self.state['working_pid'] = None

    def get_state(self) -> WorkerStateDict:
        return self.state

    def set_state(self, state: WorkerStateDict) -> None:
        self.state = state

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
        #TODO: gets stuck if the GPU was actually busy, whole server process needs to be restarted
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

        # print a seperator between runs
        self.server_logger.info(self.seperator(run_sum))
        # get next run
        info = self.que.peak_cur_run()

        wandb_info = info.wandb
        admin = info.admin

        config = RunInfo(**info.model_dump())

        # setup wandb run
        with redirect_stdout(self.log_adapter):
            run = _setup_wandb(config, wandb_info, run_id_required=True)

        # save run_id for recovering
        wandb_info.run_id = run.id
        info.wandb = wandb_info

        # save for server context
        self.state['current_run_id'] = run.id

        self.server_logger.info("saving my id")
        _ = self.que.pop_cur_run()
        self.que.set_cur_run(info)
        self.que.save_state()

        self.server_logger.info(f"Run ID: {run.id}")
        self.server_logger.info(f"Run name: {run.name}")  # Human-readable name
        self.server_logger.info(f"Run path: {run.path}")  # entity/project/run_id format

        # NOTE (see src/que/todo, Server section): print() output during train_loop
        # only reaches this redirect target, never wandb's own console capture --
        # confirmed empirically, not just a block-scoping issue. wandb patches the
        # *write method* of whichever object is sys.stdout at first `import wandb`
        # (wandb/sdk/lib/console_capture.py), once, permanently, before this
        # redirect ever runs. Reassigning sys.stdout to self.log_adapter here means
        # print() calls sys.stdout.write() on our object instead, which never
        # touches wandb's patched object -- no amount of restructuring these
        # `with` blocks changes that. A real fix has to either stop reassigning
        # sys.stdout during training, or additionally register self.log_adapter's
        # write via wandb.sdk.lib.console_capture.capture_stdout(...) directly.
        with redirect_stdout(self.log_adapter):
            train_loop(
                admin.model,
                config,
                run,
                recover=admin.recover,
                event=self.stop_event,
            )
            run.finish(exit_code=0)

        self.server_logger.info("_train method completed successfully")

    def _test(self) -> None:
        """Tests the run in cur_runs and moves to old_runs"""
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

        fin_run = self.que.peak_cur_run()
        with redirect_stdout(self.log_adapter):
            results = full_test(admin=fin_run.admin, data=fin_run.data, re_test=True)
        comp_run = CompExpInfo(
            admin=fin_run.admin,
            training=fin_run.training,
            optimizer=fin_run.optimizer,
            model_params=fin_run.model_params,
            data=fin_run.data,
            scheduler=fin_run.scheduler,
            stopping=fin_run.stopping,
            wandb=fin_run.wandb,
            results=results,
        )
        _ = self.que.pop_cur_run()
        self.que.set_cur_run(comp_run)
        self.que.store_fin_run()
        self.server_logger.info("Exiting _test method")

    def _inject_sweep_config(self, config: RunInfo, run_id: str) -> None:
        """Injects the sweep config into the Que"""
        assert self.sweep_info is not None, "_inject_sweep_config called without sweep_info set"
        wandb_i = WandbInfo(
            entity=self.sweep_info["sweep_entity"],
            project=self.sweep_info["sweep_project"],
            run_id=run_id,
            sweep_id=self.sweep_info["sweep_id"],
            tags=["sweep"],
        )
        self.server_logger.debug(f"Injecting run: {json.dumps(config.model_dump())}")
        self.que.add_new_run(config, wandb_i, loc=CUR_RUN) # inject directly into cur_run, and skip the move from to_run -> cur_run step in _train, since the sweep controller already sampled the hyperparameters for this trial
        self.que.save_state()

    def _sweep_train(self) -> None:
        """Callback passed to wandb.agent(function=...). Runs in-process, in
        the thread the agent invokes it on -- NOT a separate process, so when
        wandb's backend decides to early-stop this trial (e.g. hyperband), it
        has no subprocess to kill and instead injects a bare exception into
        this thread (see _is_wandb_injected_stop). That case is detected and
        logged below rather than fixed at the root -- running each trial in
        an actual subprocess would sidestep the injected-exception mechanism
        entirely, but that's a bigger change than this handles.

        wandb's own agent thread (pyagent.py's _run_job) swallows *any*
        exception raised here unconditionally and never re-raises, so every
        other failure -- in create_sweep_run, the que injection, or training --
        is handled here: recorded in worker state, stashed to fail_runs if the
        run already reached cur_run (which also makes Worker.start() skip
        testing), and saved to `_trial_error` for Worker.sweep() to re-raise
        once wandb.agent returns.
        """
        if not gpu_manager.wait_for_completion(
            check_interval=10,
            logger=self.server_logger,
            event=self.stop_event,
        ):
            # only returns False when stopped (stop event / Ctrl+C), so not a failure
            self.server_logger.info("GPU not available, exiting")
            return
        else:
            self.server_logger.info("GPU is available")

        try:
            self._run_sweep_trial()
        except Exception as e:  # noqa: BLE001
            if _is_wandb_injected_stop(e):
                self.server_logger.info(
                    f"Sweep trial {self.state['current_run_id']} was stopped early by wandb (e.g. hyperband "
                    "early-terminate); continuing to testing with the last saved checkpoint"
                )
                return
            self._fail(e, "Sweep trial failed due to an error")
            if self.que.len_loc(CUR_RUN) == 1:
                self.que.stash_failed_run(self.build_exception_info(e))
                self.que.save_state()
            self._trial_error = e
            return

        self.server_logger.info("_sweep_train method completed successfully")

    def _run_sweep_trial(self) -> None:
        """Build, register and train one sweep trial (the body of _sweep_train).

        Unlike _train, there's no que injection step beforehand: create_sweep_run
        builds the RunInfo directly from this trial's sweep-sampled hyperparameters,
        so there's nothing sitting in TO_RUN for this run.
        """
        assert self.sweep_info is not None, "_run_sweep_trial called without sweep_info set"

        with redirect_stdout(self.log_adapter):
            config, run = create_sweep_run(
                model=self.sweep_info["model"],
                split=self.sweep_info["split"],
                config_path=Path(self.sweep_info['base_config']),
                dataset=self.sweep_info["dataset"],
            )

        #inject the sweep config into the Que for recovery/state tracking
        self._inject_sweep_config(config, run.id)

        self.state['current_run_id'] = run.id
        self.server_logger.info(f"Run ID: {run.id}")
        self.server_logger.info(f"Run name: {run.name}")
        self.server_logger.info(f"Run path: {run.path}")

        # NOTE (see src/que/todo, Server section): this redirect means wandb's own
        # console capture never sees train_loop's print() output -- see _train's
        # matching comment for the actual mechanism (it's not a block-scoping bug).
        with redirect_stdout(self.log_adapter):
            train_loop(
                config.admin.model,
                config,
                run,
                recover=False,
                event=self.stop_event,
            )
        run.finish(exit_code=0)

    def _reset_state(self):
        self.set_state(
            WorkerStateDict(
                task="inactive", current_run_id=None, working_pid=None, exception=None, 
            )
        )

    def _reattach_server_logger(self):
        """Re-attach the server log file handler in a spawned child process."""
        logger = logging.getLogger(WORKER_NAME)
        if not logger.handlers:  # avoid duplicate handlers on repeated calls
            handler = logging.FileHandler(SERVER_LOG_PATH)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
        self.server_logger = logger

    def _attach_training_loggers(self):
        """Attach the training log file hanlder in a spawned child process"""
        self.training_logger = logging.getLogger(TRAINING_NAME)
        if not self.training_logger.handlers:
            handler = logging.FileHandler(TRAINING_LOG_PATH)
            handler.setLevel(logging.INFO)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.training_logger.addHandler(handler)
            self.training_logger.setLevel(logging.INFO)
            self.training_logger.propagate = (
                False  # match what _setup_training_logger does
            )
        self.log_adapter: IO[str] = cast(IO[str], LoggerWriter(self.training_logger))

    def train(self) -> None:
        """
        The train method is the main entry point for training a model.
        Currently implemented to be in a process started by the Daemon.
        """
        try:
            self.training_logger.info("Starting training run")
            self.state['task'] = "training"
            self._train()
        except QueException as e:
            self._fail(e, "que based error, cannot continue")
            raise
        except KeyboardInterrupt:
            self.server_logger.info("Worker killed by user")
            self.state['exception'] = "KeyboardInterrupt"
            self.state['working_pid'] = None
            raise
        except Exception as e:
            self._fail(e, "Training run failed due to an error")
            self.que.stash_failed_run(str(e))
            self.que.save_state()
            raise
        finally:
            self.cleanup()
            self.state['task'] = "inactive"

    def _register_sweep_trial_completion(self, sweep_info: SweepInfo) -> None:
        """Increment the shared completed-trial counter and stop the sweep if max_runs is
        reached. Called right after wandb.agent(..., count=1) returns, only for trials that
        finished (naturally or via a wandb hyperband early-stop) -- a failed trial makes
        Worker.sweep() raise SweepTrialFailed instead, so it isn't counted.

        The cap is read from the live shared sweep rather than `sweep_info` (the snapshot this
        trial started with), so `daemon set_max_runs` also applies to the trial in flight. A
        trial whose sweep was cleared or replaced while it ran isn't counted.
        """
        assert (
            self.sweep_progress is not None
            and self.server_context is not None
            and self.live_sweep is not None
        )
        sweep_id = sweep_info["sweep_id"]
        if self.live_sweep.get("sweep_id") != sweep_id:
            self.training_logger.info(
                f"Sweep {sweep_id} was cleared or replaced during this trial; not counting it."
            )
            return

        completed = self.sweep_progress['completed_runs'] + 1
        self.sweep_progress['completed_runs'] = completed

        max_runs = self.live_sweep["max_runs"]
        if max_runs is not None and completed >= max_runs:
            self.training_logger.info(
                f"Sweep {sweep_id} reached max_runs={max_runs} "
                f"({completed} trials completed); clearing sweep."
            )
            self.server_context.set_sweep({})

    def sweep(self, sweep_info: SweepInfo) -> None:
        """Run one sweep trial via wandb.agent.

        Raises:
            SweepTrialFailed: If the trial failed inside wandb.agent's callback (which wandb
                itself swallows -- see _sweep_train), so the worker exits non-zero.
        """
        self.sweep_info = sweep_info
        self._trial_error = None
        try:
            sweep_info_validate(sweep_info)
            self.training_logger.info(f"Starting sweep trial: {sweep_info['sweep_id']}")
            self.state['task'] = "training"
            wandb.agent(
                sweep_info["sweep_id"],
                entity=sweep_info["sweep_entity"],
                project=sweep_info["sweep_project"],
                function=self._sweep_train,
                count=1,
            )
            if self._trial_error is None:
                self._register_sweep_trial_completion(sweep_info)
        except (QueException, ValidationError) as e:
            self._fail(e, f"{type(e).__name__} — cannot continue")
            raise
        except KeyboardInterrupt:
            self.server_logger.info("Worker killed by user")
            self.state['exception'] = "KeyboardInterrupt"
            self.state['working_pid'] = None
            raise
        except Exception as e:
            # Only for things raised outside wandb's callback thread (e.g. wandb.agent
            # itself); failures inside it are handled by _sweep_train and re-raised below.
            self._fail(e, "Sweep trial failed due to an error")
            if self.que.len_loc(CUR_RUN) == 1:
                self.que.stash_failed_run(str(e))
                self.que.save_state()
            raise
        finally:
            self.cleanup()
            self.state['task'] = "inactive"

        # outside the try, so the handlers above don't re-record an already-recorded failure
        if self._trial_error is not None:
            raise SweepTrialFailed(sweep_info["sweep_id"]) from self._trial_error

    def test(self) -> None:
        try:
            self.state['task'] = "testing"
            self._test()
        except QueException as e:
            self._fail(e, "que based error, cannot continue", clear_pid=False)
            raise
        except KeyboardInterrupt:
            self.server_logger.info("Worker killed by user")
            self.state['exception'] = "KeyboardInterrupt"
            raise
        except Exception as e:
            err_str = self.build_exception_info(e)
            self._fail(e, "Testing run failed due to an error")
            self.que.stash_failed_run(err_str)
            self.que.save_state()
            raise
        finally:
            self.cleanup()
            self.state['current_run_id'] = None
            self.state['task'] = "inactive"


    def start(self, sweep_info: SweepInfo | None = None) -> None:
        """this is likely started in a seperate process, so que requires connecting"""

        #get state handlers
        manager = connect_manager()
        self.que = manager.get_que()
        self.state = manager.get_worker_state()
        self.sweep_progress = manager.get_sweep_progress()
        self.live_sweep = manager.get_sweep()
        self.server_context = manager.get_server_context()

        #update state
        self.state['working_pid'] = os.getpid()
        self.state['exception'] = None

        self._attach_training_loggers()
        self._reattach_server_logger()

        if sweep_info is not None and self.que.len_loc('to_run') == 0: #give preference to runs on Que
            self.sweep(sweep_info)
        else:
            self.train()

        if self.stop_event is not None and self.stop_event.is_set():
            self.server_logger.warning("Training was interrupted by stopping event.")
        elif self.que.len_loc('cur_run') == 1:
            self.server_logger.info("Training finished successfully. Running tests")
            self.test()
        else:
            self.server_logger.warning("No run in cur_run. Skipping tests.")
        
        self.que.save_state()  # save state


if __name__ == "__main__":
    pass