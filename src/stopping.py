from typing import Optional
from wandb.sdk.wandb_run import Run
from multiprocessing.synchronize import Event as EventClass
#local
from src.run_types import StopperInfo, StopperState, EarlyStopperInfo, StopperConfig


    
class Stopper:
    """Stopping utility for training processes.
    
    Stops if max epochs reached, and bundles other optional early stopping utilitise
    including config based, and multiprocessing event class. 
    
    For config based setup: Monitors a specified metric and stops training if no improvement is seen 
    for a defined number of epochs (patience). Supports both minimization and 
    maximization modes.

    Methods:
        step(score): Updates the early stopping state.
        state_dict(): Returns the current state as a StopperState model.
        load_state_dict(state_dict): Loads state from a StopperState model.

    """

    def __init__(
        self,
        arg_dict: StopperConfig,
        wandb_run: Optional[Run] = None,
        event: Optional[EventClass] = None, # if in a multiprocessing context, can pass an Event to signal stopping
    ):
        """Initialise the stopper

        Args:
            arg_dict (StopperConfig): Stopping information, at least max_epoch.
            wandb_run (Optional[Run], optional): For logging patience. Defaults to None.
            event (Optional[EventClass], optional): Can pass an Event to signal stopping. Defaults to None.
        """
    
        self.max_epoch = arg_dict.max_epoch
        
        if isinstance(arg_dict, EarlyStopperInfo):
            self.on = True
            self.metric = arg_dict.metric
            self.phase = arg_dict.phase
            self.mode = arg_dict.mode
            self.patience = arg_dict.patience
            self.min_delta = arg_dict.min_delta
            
        self.curr_epoch = 0
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.wandb_run = wandb_run
        self.stop = False
        self.event = event
        self.stopped_by_event = False

    def step(self, score) -> None:
        """Update early stopping state based on current score.
        
        Args:
            score: The current metric value to evaluate.
        """
        
        if self.curr_epoch >= self.max_epoch:
            self.stop = True
            print("Maximum epochs reached")
            return
        
        if self.event is not None and self.event.is_set():
            self.stop = True
            self.stopped_by_event = True
            return
        
        if not self.on:
            self.curr_epoch += 1
            return 
        
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == "min":
            if score < self.best_score - self.min_delta:
                improved = True
        else:  # 'max'
            if score > self.best_score + self.min_delta:
                improved = True
                                
        if improved:
            self.best_score = score
            self.best_epoch = self.curr_epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(
                f"Early stopping triggered after {self.patience} epochs without improvement."
            )
            print(
                f"Best {self.phase} {self.metric}: {self.best_score:.4f} at epoch {self.best_epoch}"
            )
            self.stop = True

        if self.wandb_run:
            self.wandb_run.log({"Patience count": self.counter})
            
            # Poll wandb for early stop signal
            # if self.wandb_run.should_stop():
            #     print("Hyperband requested early stop")
            #     self.stop = True
            
            
        self.curr_epoch += 1

    def state_dict(self) -> StopperState:
        """Return the current state as a StopperState model.
        
        Returns:
            StopperState model containing the current state.
        """
        
        return StopperState(
            on=self.on,
            max_epoch=self.max_epoch,
            phase=self.phase,
            metric=self.metric,
            mode=self.mode,
            patience=self.patience,
            min_delta=self.min_delta,
            curr_epoch=self.curr_epoch,
            best_score=self.best_score,
            best_epoch=self.best_epoch,
            counter=self.counter,
            stop=self.stop,
            stopped_by_event=self.stopped_by_event
        )

    def load_state_dict(self, state_dict: StopperState) -> None:
        """Load state from a StopperState model.
        
        Args:
            state_dict: StopperState model containing state to restore.
        """
        for key, value in state_dict.model_dump().items():
            setattr(self, key, value)
