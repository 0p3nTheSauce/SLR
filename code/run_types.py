from __future__ import annotations
from typing import (
    Literal,
    Optional,
    Union,
    List,
    Tuple,
    Annotated,
    Any,
    Dict,
)
from pydantic import BaseModel, Field, model_validator, computed_field

from models import NormDict


########################## Early stopping #############################


class StopperOn(BaseModel):
    metric: Union[Tuple[str, str], List[str]]
    mode: str
    patience: int
    min_delta: float


class StopperState(BaseModel):
    on: bool
    phase: str
    metric: str
    mode: str
    patience: int
    min_delta: float
    curr_epoch: int
    best_score: Optional[float] = None
    best_epoch: int
    counter: int
    stop: bool
    stopped_by_event: bool


####################### Models #############################


class MinInfo(BaseModel):
    model: str
    dataset: str
    split: str
    save_path: str


class AdminInfo(MinInfo):
    exp_no: str
    recover: bool
    config_path: str
    weight_path: Optional[str] = None


class TrainingInfo(BaseModel):
    batch_size: int
    update_per_step: int
    max_epoch: int

    @computed_field  # type: ignore[misc]
    @property
    def batch_size_equivalent(self) -> int:
        return self.batch_size * self.update_per_step


class OptimizerInfo(BaseModel):
    eps: float
    backbone_init_lr: float
    backbone_weight_decay: float
    classifier_init_lr: float
    classifier_weight_decay: float


class ModelParamsInfo(BaseModel):
    drop_p: Optional[float] = None


class WarmUpSched(BaseModel):
    start_factor: float
    end_factor: float
    warmup_epochs: int

    @model_validator(mode="after")
    def _check_factors(self) -> WarmUpSched:
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if not (0 < self.start_factor < self.end_factor <= 1.0):
            raise ValueError("start_factor must be > 0 and < end_factor <= 1.0")
        return self


class SchedBase(BaseModel):
    warm_up: Optional[WarmUpSched] = None


class WarmOnly(SchedBase):
    type: Literal["WarmOnly"]


class CosAnealInfo(SchedBase):
    type: Literal["CosineAnnealingLR"]
    tmax: int
    eta_min: float


class WarmRestartInfo(SchedBase):
    type: Literal["CosineAnnealingWarmRestarts"]
    t0: int
    tmult: int
    eta_min: float


class ReduceLROnPlateau(SchedBase):
    type: Literal["ReduceLROnPlateau"]
    mode: Literal["min", "max"]
    factor: float
    patience: int
    threshold: float
    threshold_mode: Literal["rel", "abs"]
    cooldown: int
    min_lr: Union[List[float], float]
    eps: float


# Discriminated union: pydantic dispatches on the 'type' field automatically
SchedInfo = Annotated[
    Union[WarmOnly, CosAnealInfo, WarmRestartInfo, ReduceLROnPlateau],
    Field(discriminator="type"),
]

# Augmentation strategy literals
SpatialStrategy = Literal["IMAGENET", "CIFAR10", "SVHN", "Horizontal_flip"]
TemporalStrategy = Literal["Shuffle", "Reverse"]
FrameSize_Strategy = Literal["Centre_crop", "Random_crop", "Scale_and_pad"]


class AugInfo(BaseModel):
    norm_dict: Optional[NormDict] = None
    frame_size_strategy: List[FrameSize_Strategy] = []
    temporal_aug: List[TemporalStrategy] = []
    spatial_aug: List[SpatialStrategy] = []


class DataInfo(BaseModel):
    num_frames: int
    frame_size: int
    train_augs: Optional[AugInfo] = None
    test_augs: Optional[AugInfo] = None


class WandbInfo(BaseModel):
    entity: str
    project: str
    tags: List[str]
    run_id: Optional[str] = None


# Results

class TopKRes(BaseModel):
    top1: float
    top5: float
    top10: float


class BaseRes(BaseModel):
    top_k_average_per_class_acc: TopKRes
    top_k_per_instance_acc: TopKRes
    average_loss: float


class ShuffRes(BaseRes):
    perm: List[int]
    shannon_entropy: float


class CompRes(BaseModel):
    check_name: str
    best_val_acc: float
    best_val_loss: float
    test: BaseRes
    val: BaseRes
    test_shuff: ShuffRes


class SumRes(BaseModel):
    check_name: str
    best_val_acc: float
    best_val_loss: float
    test: BaseRes
    val: BaseRes
    test_shuff: BaseRes


# Runs

class RunInfo(BaseModel):
    admin: AdminInfo
    training: TrainingInfo
    optimizer: OptimizerInfo
    model_params: ModelParamsInfo = Field(default_factory=ModelParamsInfo)
    data: DataInfo
    scheduler: Optional[SchedInfo] = None
    early_stopping: Optional[StopperOn] = None


class ExpInfo(RunInfo):
    wandb: WandbInfo


class CompExpInfo(ExpInfo):
    results: CompRes

class FailedExp(ExpInfo):
    error: str

class SumarisedNew(BaseModel):
    run_id: Optional[str] = None
    model: str
    exp_no: str
    dataset: str
    split: str
    config_path: str


class Sumarised(SumarisedNew):
    best_val_acc: Optional[float] = None
    best_val_loss: Optional[float] = None


class SummarisedRes(Sumarised):
    test_top1_acc: Optional[float] = None
    test_av_loss: Optional[float] = None


class SummarisedError(Sumarised):
    error: str


# CleverDict is unrelated to config/typing — kept as-is
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

    def _set_inplace(self, d: Dict[Any, Any], k: Any, ks: List[Any], val: Any) -> Dict[Any, Any]:
        if hasattr(d, "__setitem__"):
            if len(ks) == 0:
                d[k] = val
            else:
                next_key = ks.pop(0)
                d[k] = self._set_inplace(d[k], next_key, ks, val)
        else:
            if len(ks) == 0:
                d = {k: val}
            else:
                next_key = ks.pop(0)
                d = {k: self._set_inplace({}, next_key, ks, val)}
        return d

    def to_dict(self) -> Dict[Any, Any]:
        return self.dict.copy()

    def __str__(self) -> str:
        return str(self.dict)

    def __delitem__(self, key):
        raise NotImplementedError