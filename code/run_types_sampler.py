from typing import (
	TypedDict,
	Literal,
	Optional,
	TypeAlias,
	Union,
	List,
	Tuple,
	Dict,
	Any,
 	Annotated)
from pydantic import BaseModel, Field, model_validator
from models import NormDict


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
	stopped_by_event: bool


####################### Data loading and augmentation #############################
# 1. Define base properties shared by all samplers
class BaseSampler(BaseModel):
	target_length: Optional[int]

 
class OG_Sampler(BaseSampler):
	"""Directs to correct_num_frames"""
	method: Literal["og"]
	randomise: bool = False

class UniformSampler(BaseSampler):
	"""Uniformly sampled"""
	method: Literal["uniform"]
	step: Optional[int]  = None
 
	@model_validator(mode='after')
	def check_step_or_target_length(self) -> 'UniformSampler':
		if self.step is None and self.target_length is None:
			raise ValueError("Either step or target_length must be provided for UniformSampler.")
		return self

class ChunkedSampler(BaseSampler):
	"""Random frames in chunks"""
	target_length: Annotated[Optional[int], Field(..., description="Required for chunked")]
	method: Literal["chunked"]

class WobbledSampler(BaseSampler):
	method: Literal["wobbled"]
	max_wobble: int  = 4

class FocalNormalSampler(BaseSampler):
	target_length: Annotated[Optional[int], Field(..., description="Required for NormalSampler")]
	method: Literal["focal_normal"]
	mean: float = 0.5
	std: float = 0.25

class FocalLaplaceSampler(BaseSampler):
	target_length: Annotated[Optional[int], Field(..., description="Required for FocalLaplaceSampler")]
	method: Literal["focal_laplace"]
	mean: float = 0.5
	diversity: float = 0.175
 
class FocalBetaSampler(BaseSampler):
	target_length: Annotated[Optional[int], Field(..., description="Required for FocalBetaSampler")]
	method: Literal["focal_beta"]
	alpha: float = 4.0
	beta: float = 4.0


class SpeedSampler(BaseSampler):
	target_length: Annotated[Optional[int], Field(..., description="Required for SpeedSampler")]
	method: Literal["speed"]
	speed_min: float 
	speed_max: float

	@model_validator(mode='after')
	def check_speeds(self) -> 'SpeedSampler':
		if self.speed_min > self.speed_max:
			raise ValueError("speed_min cannot be > speed_max")
		return self




# ... define others (chunked, focal_laplace, focal_beta) ...
	
# 3. Create the final type using a Discriminated Union
SamplerConfig = Annotated[
	Union[
		UniformSampler, 
		WobbledSampler, 
		SpeedSampler, 
		FocalNormalSampler,
		FocalLaplaceSampler,
		FocalBetaSampler,
		ChunkedSampler,
		OG_Sampler
	],
	Field(discriminator='method')
]
	


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
	weight_path: Optional[str]

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
	drop_p: Optional[float]

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
 
class ReduceLROnPlateau(SchedBase):
	type: Literal['ReduceLROnPlateau']
	mode: Literal['min', 'max']
	factor: float
	patience: int
	threshold: float
	threshold_mode: Literal['rel', 'abs'] 
	cooldown: int
	min_lr: Union[List[float], float]
	eps: float
 
SchedInfo : TypeAlias = Union[CosAnealInfo, WarmRestartInfo, WarmOnly, ReduceLROnPlateau]


# Define our new augmentation configuration options
SpatialStrategy: TypeAlias = Literal["IMAGENET", "CIFAR10", "SVHN", "Horizontal_flip"]  # Add more strategies as needed
TemporalStrategy: TypeAlias = Literal["Shuffle", "Reverse"]
FrameSize_Strategy: TypeAlias = Literal["Centre_crop", "Random_crop", "Scale_and_pad"]


class AugInfo(TypedDict):
	norm_dict: Optional[NormDict]
	frame_sampler: Optional[SamplerConfig]
	frame_size_strategy: List[FrameSize_Strategy]
	temporal_aug: List[TemporalStrategy]
	spatial_aug: List[SpatialStrategy]


class DataInfo(TypedDict):
	num_frames: int
	frame_size: int
	train_augs: Optional[AugInfo]
	test_augs: Optional[AugInfo]

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

class SumRes(TypedDict):
	"""Sumarised results for quick view"""
	check_name: str
	best_val_acc: float
	best_val_loss: float
	test: BaseRes
	val: BaseRes
	test_shuff: BaseRes


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


class SumarisedNew(TypedDict):
	run_id: Optional[str]
	model: str
	exp_no: str
	dataset: str
	split: str
	config_path: str
 
class Sumarised(TypedDict):
	run_id: Optional[str]
	model: str
	exp_no: str
	dataset: str
	split: str
	config_path: str
	best_val_acc: Optional[float]
	best_val_loss: Optional[float]

class SummarisedRes(Sumarised):
	test_top1_acc: Optional[float]
	test_av_loss: Optional[float]



class SummarisedError(Sumarised):
	error: str
 
class CleverDict(Dict):
	def __init__(self, dict: Dict[Any, Any]):
		self.dict = dict
		
	def __getitem__(self, keys: List[Any]) -> Any:
		d = self.dict.copy()
		for key in keys:
			d = d[key]
		return d
	
	def __setitem__(self, keys: List[Any], val: Any):
		self.dict = self._set_inplace(self.dict, keys[0], keys[1:], val)

	def _set_inplace(self, d:Dict[Any, Any], k:Any,ks:List[Any], val:Any) -> Dict[Any, Any]:
		
		if hasattr(d, '__setitem__'):
			if len(ks) == 0:
				d[k] = val    
			else:
				next_key = ks.pop(0)
				d[k] = self._set_inplace(d[k],next_key, ks, val)
		else:
			if len(ks) == 0:
				d = {k:val} 
			else:
				next_key = ks.pop(0)
				d = {k: self._set_inplace({},next_key, ks, val)}

		return d          
		
	
	def _create_inplace(self, d:Dict[Any, Any] | Any, k:Any,ks:List[Any], val:Any) -> Dict[Any, Any]:
		if len(ks) == 0:
			if hasattr(d, '__setitem__'):
				d[k] = val
			else:
				d = {k:val}
			return d
		else:
			next_key = ks.pop(0)
			d[k] = self._set_inplace(d[k],next_key, ks, val)
			return d   
	
	def to_dict(self) -> Dict[Any, Any]:
		return self.dict.copy()
	
	
	def __str__(self) -> str:
		return str(self.dict)
	
	def __delitem__(self, key):
		raise NotImplementedError