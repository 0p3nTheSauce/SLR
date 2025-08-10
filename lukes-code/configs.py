import configparser
import importlib
from typing import Dict, Any

def load_config(arg_dict):
	# Load from .ini
	config = parse_ini_config(arg_dict['config_path'], flatten=False)
	
	# Override with command line args if provided
	if arg_dict:
		for key, value in arg_dict.items():
			config[key] = value
	
	return config
 
def parse_ini_config(ini_file: str, flatten: bool = True) -> Dict[str, Any]:
	"""Parse .ini file for wandb config"""
	config = configparser.ConfigParser()
	config.read(ini_file)
	
	if flatten:
			# Flat structure: section_key format
			wandb_config = {}
			for section in config.sections():
					for key, value in config[section].items():
							wandb_config[f"{section}_{key}"] = _convert_type(value)
	else:
			# Nested structure
			wandb_config = {}
			for section in config.sections():
					wandb_config[section] = {}
					for key, value in config[section].items():
							wandb_config[section][key] = _convert_type(value)
	
	return wandb_config

def _convert_type(value: str) -> Any:
	"""Convert string values to appropriate types"""
	# Boolean
	if value.lower() in ['true', 'false']:
		return value.lower() == 'true'
	
	# Integer
	try:
		return int(value)
	except ValueError:
		pass
	
	# Float
	try:
		return float(value)
	except ValueError:
		pass
	
	# String
	return value

############################################old config style#########################
	
class Config:
	def __init__(self, config_path):
		config = configparser.ConfigParser()
		config.read(config_path)

		# Training
		train_config = config['TRAIN'] 
		self.batch_size = int(train_config['BATCH_SIZE'])
		self.max_steps = int(train_config['MAX_STEPS'])
		self.update_per_step = int(train_config['UPDATE_PER_STEP'])
		self.drop_p = float(train_config['DROP_P'])
		
		# Optimizer
		opt_config = config['OPTIMIZER']
		self.init_lr = float(opt_config['INIT_LR']) #deprecated
		self.adam_eps = float(opt_config['ADAM_EPS'])  #deprecated
		self.adam_weight_decay = float(opt_config['ADAM_WEIGHT_DECAY']) #deprecated
		if 'BACKBONE_INIT_LR' in opt_config: #backwards compatibility
			self.backbone_init_lr = float(opt_config['BACKBONE_INIT_LR'])
		if 'BACKBONE_WEIGHT_DECAY' in opt_config:
			self.backbone_weight_decay = float(opt_config['BACKBONE_WEIGHT_DECAY'])
		if 'CLASSIFIER_INIT_LR' in opt_config:
			self.classifier_init_lr = float(opt_config['CLASSIFIER_INIT_LR'])
		if 'CLASSIFIER_WEIGHT_DECAY' in opt_config:
			self.classifier_weight_decay = float(opt_config['CLASSIFIER_WEIGHT_DECAY'])
		
		# Scheduler
		sched_config = config['SCHEDULER']
		self.t_max = int(sched_config['T_MAX'])
		self.eta_min = float(sched_config['ETA_MIN'])

		# Model
		model_config = config['MODEL']
		self.model_wrapper = self._import_from_string(model_config['WRAPPER'])
		self.transform_method = model_config['TRANSFORMS_METHOD']
		self.weights_path = model_config.get('WEIGHTS_PATH', None)
		if 'BACKBONE_WEIGHTS' in model_config:
			self.backbone_weights_path = model_config['BACKBONE_WEIGHTS']
		if 'FROZEN' in model_config:
			self.frozen = model_config['FROZEN'].split() if model_config['FROZEN'] else [] 
		if 'NUM_CLASSES' in model_config:
			self.num_classes = int(model_config['NUM_CLASSES'])
		
		# Transforms specific parameters
		#TODO


		# Optional attention parameters
		if 'IN_LINEAR' in model_config:
			self.in_linear = int(model_config.get('IN_LINEAR', 1))
		if 'N_ATTENTION' in model_config:
			self.n_attention = int(model_config.get('N_ATTENTION', 5))
		
		# Dataset 
		dataset_config = config['DATASET']
		self.frame_size = int(dataset_config['FRAME_SIZE'])
		self.num_frames = int(dataset_config['NUM_FRAMES'])

	def create_model(self):
		"""Create model using the wrapper's from_config method"""
		return self.model_wrapper.from_config(self)
	
	def get_transforms(self):
		'''Create train and test transforms using wrapper's get_transforms method'''
		return getattr(self.model_wrapper, self.transform_method)(self.frame_size)

	def __str__(self):
		if hasattr(self, 'frozen'):
			return f"""Config:
			Model: {self.model_wrapper}
			Weights Path: {self.weights_path}
			Frozen layers: {self.frozen}
			Scheduler: t_max={self.t_max}, eta_min={self.eta_min}
			Training: bs={self.batch_size}, steps={self.max_steps}, ups={self.update_per_step}
			Optimizer: lr={self.init_lr}, eps={self.adam_eps}, wd={self.adam_weight_decay}
			Backbone: lr={self.backbone_init_lr}, wd={self.backbone_weight_decay}
			Classifier: lr={self.classifier_init_lr}, wd={self.classifier_weight_decay}"""
		else:
			return f"""Config:
			Training: bs={self.batch_size}, steps={self.max_steps}, ups={self.update_per_step}
			Optimizer: lr={self.init_lr}, eps={self.adam_eps}, wd={self.adam_weight_decay}"""

	def _import_from_string(self, import_string):
		"""Import a class/function from a module string like 'models.pytorch_r3d.Resnet3D18_basic'"""
		module_path, class_name = import_string.rsplit('.', 1)
		module = importlib.import_module(module_path)
		return getattr(module, class_name)
 
 
if __name__ == '__main__':
	config_path = '/home/dxli/workspace/nslt/code/VGG-GRU/configs/test.ini'
	print(str(Config(config_path)))