from typing import TypedDict, Literal, Optional, TypeAlias, Union, List, Tuple

########################## EArly stopping #############################

class StopperOn(TypedDict):
    """Dictionary of required parameters for the Stopper to be initialised in 'on' state"""
    metric: Union[Tuple[str, str], List[str]]
    mode: str
    patience: int
    min_delta: float

class StopperState(TypedDict):
    """Dictionary representing the serializable state of an EarlyStopper instance"""
    on: bool
    phase: str
    metric: str
    mode: str
    patience: int
    min_delta: float
    curr_epoch: int
    best_score: Optional[float]
    best_epoch: int
    counter: int
    stop: bool


####################### Typed Dictionaries #############################

class MinInfo(TypedDict):
	model: str
	dataset: str
	split: str
	save_path: str
 
class AdminInfo(MinInfo):
	exp_no: str
	recover: bool
	config_path: str

class TrainingInfo(TypedDict):
	batch_size: int
	update_per_step: int
	batch_size_equivalent: int
	max_epoch: int

class OptimizerInfo(TypedDict):
	eps: float
	backbone_init_lr: float
	backbone_weight_decay: float
	classifier_init_lr: float
	classifier_weight_decay: float

class Model_paramsInfo(TypedDict):
	drop_p: float

class WarmUpSched(TypedDict):
	start_factor: float
	end_factor: float
	warmup_epochs: int
	
class SchedBase(TypedDict):
	warm_up: Optional[WarmUpSched]

class WarmOnly(SchedBase):
	type: Literal['WarmOnly']

class CosAnealInfo(SchedBase):
	type: Literal['CosineAnnealingLR']
	tmax: int
	eta_min: float
 
class WarmRestartInfo(SchedBase):
	type: Literal['CosineAnnealingWarmRestarts']  # Make type specific
	t0: int
	tmult: int
	eta_min: float
 
SchedInfo : TypeAlias = Union[CosAnealInfo, WarmRestartInfo, WarmOnly]

class DataInfo(TypedDict):
	num_frames: int
	frame_size: int

class WandbInfo(TypedDict):
	entity: str
	project: str
	tags: List[str]
	run_id: Optional[str]

#Results
class TopKRes(TypedDict):
	top1: float
	top5: float
	top10: float

class BaseRes(TypedDict):
	top_k_average_per_class_acc: TopKRes
	top_k_per_instance_acc: TopKRes
	average_loss: float
	
class ShuffRes(BaseRes):
	perm: List[int]
	shannon_entropy: float

class CompRes(TypedDict):
	check_name: str
	best_val_acc: float
	best_val_loss: float
	test: BaseRes
	val: BaseRes
	test_shuff: ShuffRes

#Runs

class RunInfo(TypedDict):
	"""Base run information needed for training
	"""
	admin: AdminInfo
	training: TrainingInfo
	optimizer: OptimizerInfo
	model_params: Model_paramsInfo
	data: DataInfo
	scheduler: Optional[SchedInfo]
	early_stopping: Optional[StopperOn]

class ExpInfo(RunInfo):
	"""Inherits from RunInfo, adds wandb"""
	wandb: WandbInfo #NOTE: make wandb optional?
 
class CompExpInfo(ExpInfo):
	"""Inherits from ExpInfo, adds results"""
	results: CompRes