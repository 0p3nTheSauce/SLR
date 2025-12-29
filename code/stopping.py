from typing import  List, Tuple, Union , Optional
from wandb.sdk.wandb_run import Run
from run_types import StopperOn, StopperState
from multiprocessing.synchronize import Event as EventClass

    
class EarlyStopper:
    """Early stopping utility for training processes.
    
    Monitors a specified metric and stops training if no improvement is seen 
    for a defined number of epochs (patience). Supports both minimization and 
    maximization modes.

    Args:
        arg_dict: Dictionary containing early stopping parameters. Overrides 
            default values if provided.
        metric: Tuple specifying the phase and metric to monitor, 
            e.g., ('val', 'loss').
        mode: 'min' for minimization or 'max' for maximization.
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum change in the monitored metric to qualify as an 
            improvement.
        on: Whether early stopping is enabled. Defaults to True.
        wandb_run: Optional wandb run object for logging.

    Raises:
        ValueError: If metric is not in available_metrics.
        ValueError: If mode is not in available_modes.
        ValueError: If patience is not a positive integer.
        ValueError: If min_delta is negative.

    Methods:
        step(score): Updates the early stopping state based on the current score.
        state_dict(): Returns the current state as a dictionary.
        load_state_dict(state_dict): Loads state from a dictionary.

    Static Methods:
        config_precheck(config): Validates the early stopping configuration.
    """

    available_metrics = [
        ("val", "loss"),
        ("val", "acc"),
        ("train", "loss"),
        ("train", "acc"),
    ]
    available_modes = ["min", "max"]
    required_keys = ["metric", "mode", "patience", "min_delta"]

    def __init__(
        self,
        arg_dict: Optional[StopperOn]=None,
        metric: Union[Tuple[str, str], List[str]]=("val", "loss"),
        mode: str ="min",
        patience: int =20,
        min_delta: float=0.01,
        on: bool=True,
        wandb_run: Optional[Run]=None,
        event: Optional[EventClass]=None, #if in a multiprocessing context, can pass an Event to signal stopping
    ):
        """Initialize the EarlyStopper."""
        
        if arg_dict:  # coming straight from configs.py
            # self.on = arg_dict.get("on", True)
            self.on = True
            metric = arg_dict.get("metric", metric)
            mode = arg_dict.get("mode", mode)
            patience = arg_dict.get("patience", patience)
            min_delta = arg_dict.get("min_delta", min_delta)
        else:
            self.on = on
            
        if isinstance(metric, list):
            metric = (metric[0], metric[1])

        check_dict: StopperOn = {
            'metric': metric,
            'mode': mode,
            'patience': patience,
            'min_delta': min_delta
        }
        self.config_precheck(check_dict)
        
        self.phase = metric[0]
        self.metric = metric[1]
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.curr_epoch = 0
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.wandb_run = wandb_run
        self.stop = False
        self.event = event

    def step(self, score) -> None:
        """Update early stopping state based on current score.
        
        Args:
            score: The current metric value to evaluate.
        """
        if self.event is not None and self.event.is_set():
            self.stop = True
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
        self.curr_epoch += 1

    @staticmethod
    def config_precheck(config: StopperOn) -> None:
        
        if config["metric"] not in EarlyStopper.available_metrics:
            raise ValueError(
                f"Invalid metric: {config['metric']}. Available metrics: {EarlyStopper.available_metrics}"
            )
        if config['mode'] not in EarlyStopper.available_modes:
            raise ValueError(
                f"Invalid mode: {config['mode']}. Available modes: {EarlyStopper.available_modes}"
            )
        if config["patience"] <= 0:
            raise ValueError(
                f"Patience must be a positive integer, got {config['patience']}"
            )
        if config["min_delta"] < 0:
            raise ValueError(
                f"Min delta must be non-negative, got {config['min_delta']}"
            )

    def state_dict(self) -> StopperState:
        """Return the current state as a dictionary.
        
        Returns:
            Dictionary containing the current state.
        """
        
        return {
            "on": self.on,
            "phase": self.phase,
            "metric": self.metric,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "curr_epoch": self.curr_epoch,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "stop": self.stop,
        }

    def load_state_dict(self, state_dict: StopperState) -> None:
        """Load state from a dictionary.
        
        Args:
            state_dict: Dictionary containing state to restore.
        """
        for key, value in state_dict.items():
            setattr(self, key, value)


