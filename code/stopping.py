class EarlyStopper:
  '''Early stopping utility for training processes.
  Monitors a specified metric and stops training if no improvement is seen for a defined number of epochs (patience).
  Supports both minimization and maximization modes.
  
  Args:
    arg_dict (dict): Dictionary containing early stopping parameters. Overides default values if provided.
    metric (tuple): Tuple specifying the phase and metric to monitor, e.g., ('val', 'loss').
    mode (str): 'min' for minimization or 'max' for maximization.
    patience (int): Number of epochs to wait for improvement before stopping.
    min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
    wandb_run: Optional wandb run object for logging.
  
  Attributes:
    on (bool): Whether early stopping is enabled.
    phase (str): Phase of the metric being monitored (e.g., 'val').
    metric (str): Name of the metric being monitored (e.g., 'loss').
    mode (str): Mode of operation ('min' or 'max').
    patience (int): Number of epochs to wait for improvement.
    min_delta (float): Minimum change in the metric to consider as an improvement.
    curr_epoch (int): Current epoch number.
    best_score (float): Best score observed so far.
    best_epoch (int): Epoch at which the best score was observed.
    counter (int): Counter for epochs without improvement.
    wandb_run: Optional wandb run object for logging.
    stop (bool): Flag indicating whether to stop training.
    
  Methods:
    step(score): Updates the early stopping state based on the current score.
    
  Static Methods:
    config_precheck(config): Validates and prepares the early stopping configuration from a given config dictionary.
  '''
  
  available_metrics = [('val','loss'), ('val','acc'),
                       ('train','loss'), ('train','acc')]
  available_modes = ['min', 'max']
  required_keys = ['metric', 'mode', 'patience', 'min_delta']
  
  def __init__(self,arg_dict=None,  
               metric=('val','loss'), mode='min', patience=20, min_delta=0.01,
               wandb_run=None):
    
    if arg_dict: #coming straight from configs.py
      self.on = arg_dict.get('on', True)
      metric = arg_dict.get('metric', metric)
      mode = arg_dict.get('mode', mode)
      patience = arg_dict.get('patience', patience)
      min_delta = arg_dict.get('min_delta', min_delta)
    else:
      self.on = True
    
    if not self.on:
      print("Early stopping is switched off")
      return
    
    if isinstance(metric, list):
      metric = tuple(metric)
    
    if metric not in self.available_metrics:
      raise ValueError(f"Invalid metric: {metric}. Available metrics: {self.available_metrics}")
    if mode not in self.available_modes:
      raise ValueError(f"Invalid mode: {mode}. Available modes: {self.available_modes}")
      
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
      if self.mode == 'min':
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
        print(f'Early stopping triggered after {self.patience} epochs without improvement.')
        print(f'Best {self.phase} {self.metric}: {self.best_score:.4f} at epoch {self.best_epoch}')
        self.stop = True
      
      if self.wandb_run:
        self.wandb_run.log({f'Patience count': self.counter})
      self.curr_epoch += 1
  
  #for interface with configs.py
  @staticmethod
  def config_precheck(config):
    if 'early_stopping' not in config['training']:
      config['training']['early_stopping'] = {'on':False} #switch off 
    else:
      es_info = config['training']['early_stopping']
      
      for key in EarlyStopper.required_keys:
        if key not in es_info:
          raise ValueError(f"Missing required key: {key} in early stopping config")
      
      if es_info['metric'] not in EarlyStopper.available_metrics:
        raise ValueError(f"Invalid metric: {es_info['metric']}. Available metrics: {EarlyStopper.available_metrics}")
      if es_info['mode'] not in EarlyStopper.available_modes:
        raise ValueError(f"Invalid mode: {es_info['mode']}. Available modes: {EarlyStopper.available_modes}")
      
      if es_info['patience'] <= 0:
        raise ValueError(f"Patience must be a positive integer, got {es_info['patience']}")
      if es_info['min_delta'] < 0:
        raise ValueError(f"Min delta must be non-negative, got {es_info['min_delta']}")
      
    return config