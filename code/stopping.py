from typing import Dict, Any, TypedDict, List, Tuple, Union ,Literal, Optional
from wandb.sdk.wandb_run import Run
class StopperOff(TypedDict):
    """Dictionary of required parameters for the Stopper to be initialised in 'off' state"""
    on: Literal[False]

class StopperOn(TypedDict):
    """Dictionary of required parameters for the Stopper to be initialised in 'on' state"""
    on: Literal[True]
    metric: Tuple[str, str]
    mode: str
    patience: int
    min_delta: float

"""Early stopping utility for training processes.
    Monitors a specified metric and stops training if no improvement is seen for a defined number of epochs (patience).
    Supports both minimization and maximization modes.

    Args:
      arg_dict (dict): Dictionary containing early stopping parameters. Overides default values if provided.
      metric (tuple): Tuple specifying the phase and metric to monitor, e.g., ('val', 'loss').
      mode (str): 'min' for minimization or 'max' for maximization.
      patience (int): Number of epochs to wait for improvement before stopping.
      min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
      wandb_run: Optional wandb run object for logging.
      on (bool): Whether early stopping is enabled. Defaults to True.

    Methods:
      step(score): Updates the early stopping state based on the current score.

    Static Methods:
      config_precheck(config): Validates and prepares the early stopping configuration from a given config dictionary.
    """

class EarlyStopper:
    """_summary_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
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
        arg_dict: Optional[Union[StopperOn, StopperOff]]=None,
        metric: Tuple[str, str]=("val", "loss"),
        mode: str ="min",
        patience: int =20,
        min_delta: float=0.01,
        on: bool=True,
        wandb_run: Optional[Run]=None,
    ):
        """Early stopping utility for training processes.
        Monitors a specified metric and stops training if no improvement is seen for a defined number of epochs (patience).
        Supports both minimization and maximization modes.


        Args:
            arg_dict (Optional[Union[StopperOn, StopperOff]], optional): Dictionary containing early stopping parameters. Overides default values if provided.. Defaults to None.
            metric (Tuple[str, str], optional):Tuple specifying the phase and metric to monitor, e.g., ('val', 'loss').. Defaults to ("val", "loss").
            mode (str, optional): 'min' for minimization or 'max' for maximization. Defaults to "min".
            patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 20.
            min_delta (float, optional): Minimum change in the monitored metric to qualify as an improvement. Defaults to 0.01.
            on (bool, optional): _description_. Defaults to True.
            wandb_run (Optional[Run], optional):  Optional wandb run object for logging. Defaults to None.

        Raises:
            ValueError: For invalid metric
            ValueError: For invalid mode
            ValueError: For invalid patience
            ValueError: For invalid min delta
        """
        
        if arg_dict:  # coming straight from configs.py
            self.on = arg_dict.get("on", True)
            metric = arg_dict.get("metric", metric)
            mode = arg_dict.get("mode", mode)
            patience = arg_dict.get("patience", patience)
            min_delta = arg_dict.get("min_delta", min_delta)
        else:
            self.on = on

        if metric not in self.available_metrics:
            raise ValueError(
                f"Invalid metric: {metric}. Available metrics: {self.available_metrics}"
            )
        if mode not in self.available_modes:
            raise ValueError(
                f"Invalid mode: {mode}. Available modes: {self.available_modes}"
            )
        if patience <= 0:
                raise ValueError(
                    f"Patience must be a positive integer, got {patience}"
                )
        if min_delta < 0:
            raise ValueError(
                f"Min delta must be non-negative, got {min_delta}"
            )
        
        
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

    def step(self, score):
        if self.on:
            if self.best_score is None:
                self.best_score = score
            if self.mode == "min":
                if score < self.best_score - self.min_delta:
                    self.best_score = score
                    self.best_epoch = self.curr_epoch
                    self.counter = 0
                else:
                    self.counter += 1
            else:  # 'max'
                if score > self.best_score + self.min_delta:
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
    def config_precheck(config: Union[StopperOn, StopperOff]) -> None:
        
        if "metric" not in config:
            return
        else:
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

    def state_dict(self):
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
            "wandb_run": self.wandb_run,
            "stop": self.stop,
        }

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            setattr(self, key, value)


