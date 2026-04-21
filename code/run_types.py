from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Optional, Union, List, Tuple, Annotated, Any, Dict
from pydantic import BaseModel, Field, model_validator, computed_field

if TYPE_CHECKING:
    from torchvision.transforms.functional import InterpolationMode
    import torchvision.transforms.v2 as v2
    



class NormDict(BaseModel):
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]

####################### Data loading and augmentation #############################

### Samplers


class BaseSampler(BaseModel):
    """required target frames"""

    # target_length: int #NOTE: leave this then pass f(Tensor, num_frames) -> Tensor
    max_wobble: int = 0


class OG_Sampler(BaseSampler):
    """Directs to correct_num_frames"""

    method: Literal["og"] = "og"
    randomise: bool = False


class UniformSampler(BaseSampler):
    """Uniformly sampled"""

    method: Literal["uniform"] = "uniform"


class ChunkedSampler(BaseSampler):
    """Random frames in chunks"""

    method: Literal["chunked"] = "chunked"


class WobbledSampler(BaseSampler):
    method: Literal["wobbled"] = "wobbled"
    max_wobble: int = 4


class FocalNormalSampler(BaseSampler):
    method: Literal["focal_normal"] = "focal_normal"
    mean: float = 0.5
    std: float = 0.25


class FocalLaplaceSampler(BaseSampler):
    method: Literal["focal_laplace"] = "focal_laplace"
    mean: float = 0.5
    diversity: float = 0.175


class FocalBetaSampler(BaseSampler):
    method: Literal["focal_beta"] = "focal_beta"
    alpha: float = 4.0
    beta: float = 4.0


class SpeedSampler(BaseSampler):
    method: Literal["speed"] = "speed"
    speed_min: float = 0.8
    speed_max: float = 1.2

    @model_validator(mode="after")
    def check_speeds(self) -> "SpeedSampler":
        if self.speed_min > self.speed_max:
            raise ValueError("speed_min cannot be > speed_max")
        return self


SamplerConfig = Annotated[
    Union[
        UniformSampler,
        WobbledSampler,
        SpeedSampler,
        FocalNormalSampler,
        FocalLaplaceSampler,
        FocalBetaSampler,
        ChunkedSampler,
        OG_Sampler,
    ],
    Field(discriminator="method"),
]

### Temporal augs - currently not used

TemporalStrategy = Literal["Shuffle", "Reverse"]


class ShuffleT(BaseModel):
    """Represent the shuffle transform"""

    num_frames: int
    # perm optional tensor


class ReverseT(BaseModel):
    """Represents Reverse Frames"""

    probability: float = 0.5


TemporalTransforms = Annotated[Union[ShuffleT, ReverseT], Field(discriminator="type")]


### Spatial augs

Frame_Size_Strategy = Literal[
    "Centre_crop", "Random_crop", "Random_Resized_crop", "Scale_and_pad"
]


class HorizontalFlipConfig(BaseModel):
    type: Literal["HORIZONTAL_FLIP"] = "HORIZONTAL_FLIP"
    p: float = 0.5

    def build(self) -> v2.Transform:
        import torchvision.transforms.v2 as v2
        return v2.RandomHorizontalFlip(p=self.p)


class RandomGrayscaleConfig(BaseModel):
    type: Literal["RANDOM_GRAYSCALE"]
    p: float = 0.1

    def build(self) -> v2.Transform:
        import torchvision.transforms.v2 as v2
        return v2.RandomGrayscale(p=self.p)


class GaussianBlurConfig(BaseModel):
    type: Literal["GAUSSIAN_BLUR"]
    kernel_size: int = 3
    sigma: tuple[float, float] = (0.1, 2.0)

    def build(self) -> v2.Transform:
        import torchvision.transforms.v2 as v2
        return v2.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)


def _interp_modes() -> dict[str, InterpolationMode]:
    from torchvision.transforms.functional import InterpolationMode
    return {
        "BILINEAR": InterpolationMode.BILINEAR,
        "BICUBIC": InterpolationMode.BICUBIC,
        "NEAREST": InterpolationMode.NEAREST,
    }



class AutoAugmentConfig(BaseModel):
    type: Literal["IMAGENET", "CIFAR10", "SVHN"]

    def build(self) -> v2.Transform:
        import torchvision.transforms.v2 as v2
        match self.type:
            case "IMAGENET":
                return v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET)
            case "CIFAR10":
                return v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)
            case "SVHN":
                return v2.AutoAugment(v2.AutoAugmentPolicy.SVHN)

class RandAugConfig(BaseModel):
    type: Literal["RANDAUGMENT"]
    num_ops: int = 2
    magnitude: int = 9
    num_magnitude_bins: int = 31
    interpolation: Literal["BILINEAR", "BICUBIC", "NEAREST"] = "BILINEAR"

    def build(self) -> v2.RandAugment:
        import torchvision.transforms.v2 as v2
        return v2.RandAugment(
            num_ops=self.num_ops,
            magnitude=self.magnitude,
            num_magnitude_bins=self.num_magnitude_bins,
            interpolation=_interp_modes()[self.interpolation],
        )

SpatialAugConfig = Annotated[
    Union[AutoAugmentConfig, RandAugConfig, HorizontalFlipConfig, RandomGrayscaleConfig, GaussianBlurConfig],
    Field(discriminator="type")
]


class AugInfo(BaseModel):
    normalise: bool = False
    norm_dict: Optional[NormDict] = None
    frame_size_strategy: Frame_Size_Strategy
    frame_sampler: SamplerConfig = OG_Sampler()
    temporal_aug: List[TemporalStrategy] = []
    spatial_aug: List[SpatialAugConfig] = []


class DataInfo(BaseModel):
    num_frames: int
    frame_size: int
    train_augs: Optional[AugInfo] = None
    test_augs: Optional[AugInfo] = None

    @model_validator(mode="after")
    def check_frame_strat(self) -> "DataInfo":
        if self.train_augs is None and self.test_augs is None:
            raise ValueError(
                "At least one aug info must be provided for sampler and croping"
            )
        return self


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

    @model_validator(mode="after")
    def _resolve_norms(self) -> "RunInfo":
        """Substitute norm_dict based on model name when norm=True."""
        from models import norm_vals
        for aug_info in (self.data.train_augs, self.data.test_augs):
            if aug_info is not None and aug_info.normalise:
                aug_info.norm_dict = norm_vals(self.admin.model)
        return self


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

    def _set_inplace(
        self, d: Dict[Any, Any], k: Any, ks: List[Any], val: Any
    ) -> Dict[Any, Any]:
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
