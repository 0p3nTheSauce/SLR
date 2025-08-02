import configparser
import importlib

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
		self.init_lr = float(opt_config['INIT_LR'])
		self.adam_eps = float(opt_config['ADAM_EPS']) 
		self.adam_weight_decay = float(opt_config['ADAM_WEIGHT_DECAY'])
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
		
		# Always treat WEIGHTS as a file path
		self.weights_path = model_config.get('WEIGHTS_PATH', None)
		# If WEIGHTS is specified, use it as weights_path for backward compatibility
		if 'WEIGHTS' in model_config:
			self.weights_path = model_config['WEIGHTS']
		if 'BACKBONE_WEIGHTS' in model_config:
			self.backbone_weights_path = model_config['BACKBONE_WEIGHTS']
		if 'FROZEN' in model_config:
			self.frozen = model_config['FROZEN'].split() if model_config['FROZEN'] else [] 
		if 'NUM_CLASSES' in model_config:
			self.num_classes = int(model_config['NUM_CLASSES'])
		
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