import configparser
import importlib
from typing import Dict, Any

import configparser
import importlib
from typing import Dict, Any

def load_config(arg_dict, verbose=False):
	"""Load config from flat file and merge with command line args"""
	config = parse_ini_config(arg_dict['config_path'])
	
	finished_keys = []
	
	if verbose:
		print()
	#update existing sections
	for key, value in arg_dict.items():
		if value is None:
			finished_keys.append(key)
			continue
		# print(f'{key}: {value}')
		for section in config.keys():#config is nested
			if key in section: 
				if verbose:
					print(f'Overide:\n config[{section}][{key}] = {config[section][key]}')
				config[section][key] = value
				finished_keys.append(key)
				if verbose:
					print(f'Becomes: \n config[{section}][{key}] = {value}')    		
	
	#update unset section
	not_finished = [key for key in arg_dict.keys() if key not in finished_keys]
	admin = {}
	for key in not_finished:
		admin[key] = arg_dict[key]
		if verbose:
			print(f'Added: conf[admin][{key}] = {arg_dict[key]}')
	config['admin'] = admin
	if verbose:	
		print()
	return post_process(config)

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

def parse_ini_config(ini_file: str) -> Dict[str, Any]:
	"""Parse .ini file for wandb config """
	config = configparser.ConfigParser()
	config.read(ini_file)

	# Nested structure
	wandb_config = {}
	for section in config.sections():
		wandb_config[section] = {}
		for key, value in config[section].items():
			wandb_config[section][key] = _convert_type(value)
	
	return wandb_config

def post_process(conf):
	#remove unused keys
	to_remove = [
		'experiment',
		'recover'
	]
	for key in to_remove:
		for section in conf: #nested
			conf[section].pop(key, None)
	return conf

def print_config(config_dict, title='Training'):
	"""
	Print configuration dictionary in a more readable format.
	
	Args:
		config_dict (dict): Dictionary containing configuration sections
		title: 'Testing' or 'Training' 
	"""
	# Extract admin info for header
	admin = config_dict.get('admin', {})
	model = admin.get('model', 'Unknown Model')
	split = admin.get('split', 'unknown')
	exp_no = admin.get('exp_no', '000')
	
	# Print header
	print(f"{title} {model} on split {split}")
	
	# Print admin section in formatted way
	if 'admin' in config_dict:
		print(f"              Experiment no: {admin.get('exp_no', 'N/A')}")
		print(f"              Raw videos at: {admin.get('root', 'N/A')}")
		print(f"              Labels at: {admin.get('labels', 'N/A')}")
		print(f"              Saving files to: {admin.get('save_path', 'N/A')}")
		if title == 'Training':
			print(f"              Recovering: {admin.get('recovering', False)}")
		print(f"              Config: {admin.get('config_path', 'N/A')}")
		print()

	# Print other sections in organized format
	sections_order = ['training', 'optimizer', 'scheduler', 'data']
	
	for section in sections_order:
		if section in config_dict:
			print(f"{section.upper()}:")
			section_data = config_dict[section]
			
			# Calculate max key length for alignment
			max_key_len = max(len(str(k)) for k in section_data.keys()) if section_data else 0
			
			for key, value in section_data.items():
				print(f"    {key:<{max_key_len}} : {value}")
			print()

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
	config_path = './configfiles/asl100/MViT_V2_S_000.ini'
	# config = configparser.ConfigParser()
	# config.read(config_path)    
	# for section in config.sections():
	# 	print(section)
	
	wb_config = parse_ini_config(config_path)
	print(wb_config)

	
 