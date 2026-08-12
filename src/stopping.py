from wandb.sdk.wandb_run import Run

#local
from src.run_types import EarlyStopperInfo, StopperState


#TODO: add propper logging
class EarlyStopper:
    """Early stopping utility for training processes.
    
    Monitors a specified metric and stops training if no improvement is seen 
    for a defined number of epochs (patience). Supports both minimization and 
    maximization modes.

    Methods:
        step(score): Updates the early stopping state.
        state_dict(): Returns the current state as a StopperState model.
        load_state_dict(state_dict): Loads state from a StopperState model.

    """

    def __init__(
        self,
        config: EarlyStopperInfo,
        wandb_run: Run | None = None,
    ):
        """Initialise the stopper

        Args:
            config (StopperConfig): Stopping information, at least max_epoch.
            wandb_run (Optional[Run], optional): For logging patience. Defaults to None.
        """
        
        self.state = StopperState.model_validate(config, from_attributes=True)
        self.wandb_run = wandb_run
    
    
    def step(self, phase: str, metrics: dict[str, float], epoch: int) -> None:
        """Update early stopping state based on current score.
        
        Args:
            score: The current metric value to evaluate.
            epoch: The current epoch number.
        """
        if self.should_stop():
            return
        if phase != self.state.phase:
            return
        score = metrics[self.state.metric]

        improved = False
        if self.state.best_score is None:
            improved = True
        elif self.state.mode == "min":
            if score < self.state.best_score - self.state.min_delta:
                improved = True
        else:  # 'max'
            if score > self.state.best_score + self.state.min_delta:
                improved = True

        if improved:
            self.state.best_score = score
            self.state.best_epoch = epoch
            self.state.counter = 0
        else:
            self.state.counter += 1

        if self.state.counter >= self.state.patience:
            print(
                f"Early stopping triggered after {self.state.patience} epochs without improvement."
            )
            print(
                f"Best {self.state.phase} {self.state.metric}: {self.state.best_score:.4f} at epoch {self.state.best_epoch}"
            )
            self.state.stop = True

        if self.wandb_run:
            self.wandb_run.log({"Patience count": self.state.counter})
            
    def state_dict(self) -> StopperState:
        """Return the current state as a StopperState model.
        
        Returns:
            StopperState model containing the current state.
        """
        
        return self.state.model_copy() #return a copy not a reference

    def load_state_dict(self, state_dict: StopperState) -> None:
        """Load state from a StopperState model.
        
        Args:
            state_dict: StopperState model containing state to restore.
        """
        self.state = state_dict

    def should_stop(self) -> bool:
        """Check if early stopping has been triggered.
        
        Returns:
            True if early stopping is triggered, False otherwise.
        """
        return self.state.stop

class NullEarlyStopper:
    """No-op stand-in when early stopping is disabled."""
    
    

    def step(self, phase: str, metrics: dict[str, float], epoch: int) -> None:
        pass

    def state_dict(self) -> None:
        return None

    def load_state_dict(self, state_dict) -> None:
        pass
    
    def should_stop(self) -> bool:
        return False


def build_early_stopper(
    config: EarlyStopperInfo | None, wandb_run: Run | None = None
) -> EarlyStopper | NullEarlyStopper:
    if config is None:
        return NullEarlyStopper()
    return EarlyStopper(config, wandb_run)

MaybeStopper = EarlyStopper | NullEarlyStopper