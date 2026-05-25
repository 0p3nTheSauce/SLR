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
    TypeAlias,
    TypeVar,
    Type,
    get_origin,
    get_args,
)
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    computed_field,
    field_validator,
    ConfigDict,
)
from pydantic_core import PydanticUndefined
from pathlib import Path
# constants

ENTITY = "ljgoodall2001-rhodes-university"
PROJECT_BASE = "WLASL"
LABEL_SUFFIX = "instances_fixed_frange_bboxes_len.json"
NUM_INSTANCES_SUFFIX = "num_instances.json"
WORST_INSTANCES_SUFFIX = "f1-score_MViTv2_B_32x3_asl2000_004.json"
# LABEL_INSTANCES_SUFFIX = "instances_fixed_frange_bboxes_len.json"
SLR_ROOT = Path.home() / "Code/SLR"
SRC_ROOT = SLR_ROOT / "src"
CLASSES_PATH = SRC_ROOT / "info/wlasl_class_list.json"
WLASL_ROOT = SLR_ROOT / "data/WLASL"
LABELS_PATH = WLASL_ROOT  / "preprocessed/labels"
RAW_DIR = "WLASL2000"
SPLIT_DIR = "splits"
RUNS_PATH = "runs"
CONFIGS_PATH = "configfiles"
ZFILL = 3
CONFIG_FILETYPE = ".toml"
SEED = 42


### for model normalisation


class NormDict(BaseModel):
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


####################### Data loading and augmentation #############################

AVAIL_SETS = Literal["train", "val", "test"]
ORIGINAL_SPLITS = Literal["asl100", "asl300", "asl1000", "asl2000"]
AVAIL_SPLITS = Union[Literal["asl100_bottom", "asl100_worst"], ORIGINAL_SPLITS]

### Samplers


class BaseSampler(BaseModel):
    """required target frames"""

    # f(Tensor, num_frames) -> Tensor
    target_length: int
    max_wobble: int = 0  # NOTE: this is probably redundant


class OG_Sampler(BaseSampler):
    """Directs to correct_num_frames"""

    type: Literal["og"] = "og"
    randomise: bool = False


class PadFramesT(BaseSampler):
    type: Literal["pad"] = "pad"


class UniformSampler(BaseSampler):
    """Uniformly sampled"""

    type: Literal["uniform"] = "uniform"


class ChunkedSampler(BaseSampler):
    """Random frames in chunks"""

    type: Literal["chunked"] = "chunked"


class WobbledSampler(BaseSampler):
    type: Literal["wobbled"] = "wobbled"
    max_wobble: int = 4


class FocalNormalSampler(BaseSampler):
    type: Literal["focal_normal"] = "focal_normal"
    mean: float = 0.5
    std: float = 0.25


class FocalLaplaceSampler(BaseSampler):
    type: Literal["focal_laplace"] = "focal_laplace"
    mean: float = 0.5
    diversity: float = 0.175


class FocalBetaSampler(BaseSampler):
    type: Literal["focal_beta"] = "focal_beta"
    alpha: float = 4.0
    beta: float = 4.0


class SpeedSampler(BaseSampler):
    type: Literal["speed"] = "speed"
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
        PadFramesT,
        OG_Sampler,
    ],
    Field(discriminator="type"),
]

### Temporal augs


class ShuffleT(BaseModel):
    type: Literal["shuffle"] = "shuffle"
    num_frames: Optional[int] = None


class ReverseT(BaseModel):
    type: Literal["reverse"] = "reverse"
    probability: float = 0.5


TemporalTransforms = Annotated[Union[ShuffleT, ReverseT], Field(discriminator="type")]

TemporalAugs = Annotated[
    Union[TemporalTransforms, SamplerConfig], Field(discriminator="type")
]
### Spatial augs


## Cropping
class CropConfig(BaseModel):
    frame_size: int


class CentreCropConfig(CropConfig):
    type: Literal["Centre_crop"] = "Centre_crop"


class RandomCropConfig(CropConfig):
    type: Literal["Random_crop"] = "Random_crop"


class ScaleAndPadConfig(CropConfig):
    type: Literal["Scale_and_pad"] = "Scale_and_pad"


class RandomResizedConfig(CropConfig):
    type: Literal["Random_Resized_crop"] = "Random_Resized_crop"


CropTransforms = Annotated[
    Union[CentreCropConfig, RandomCropConfig, ScaleAndPadConfig, RandomResizedConfig],
    Field(discriminator="type"),
]


class HorizontalFlipConfig(BaseModel):
    type: Literal["HORIZONTAL_FLIP"] = "HORIZONTAL_FLIP"
    p: float = 0.5


class RandomGrayscaleConfig(BaseModel):
    type: Literal["RANDOM_GRAYSCALE"]
    p: float = 0.1


class GaussianBlurConfig(BaseModel):
    type: Literal["GAUSSIAN_BLUR"]
    kernel_size: int = 3
    sigma: tuple[float, float] = (0.1, 2.0)


InterpMode: TypeAlias = Literal[
    "nearest", "nearest-exact", "bilinear", "bicubic", "box", "hamming", "lanczos"
]


class AutoAugmentConfig(BaseModel):
    type: Literal["IMAGENET", "CIFAR10", "SVHN"]
    interpolation: InterpMode = "nearest"


class RandAugConfig(BaseModel):
    type: Literal["RANDAUGMENT"]
    num_ops: int = 2
    magnitude: int = 9
    num_magnitude_bins: int = 31
    interpolation: InterpMode = "nearest"


SpatialTransforms = Annotated[
    Union[
        AutoAugmentConfig,
        RandAugConfig,
        HorizontalFlipConfig,
        RandomGrayscaleConfig,
        GaussianBlurConfig,
    ],
    Field(discriminator="type"),
]


SpatialAugs = Annotated[
    Union[CropTransforms, SpatialTransforms],
    Field(discriminator="type"),
]


class AugInfo(BaseModel):
    """Augmentation info for a video

    Attributes:
        normalise (bool): Flag to fetch norm values during config parsing. Default False.
        norm_dict (Optional[NormDict]): Supplied Normalisation values. Default None.
        temporal_aug (List[TemporalAugs]): Temporal augmentations to be applied in order. Default [].
        spatial_aug (List[SpatialAugs]): Spatial augmentations to be applied in order. Default [].
        strict_size (bool): Validate that at least one frame sampler and crop strategy is defined. Default True.
    """

    normalise: bool = False
    norm_dict: Optional[NormDict] = None
    temporal_aug: List[TemporalAugs] = []
    spatial_aug: List[SpatialAugs] = []
    strict_size: bool = True
    target_length: Optional[int] = None
    frame_size: Optional[int] = None

    @model_validator(mode="after")
    def _validate_augs(self) -> "AugInfo":
        if not self.strict_size:
            return self

        samplers = [augT for augT in self.temporal_aug if isinstance(augT, BaseSampler)]
        if len(samplers) == 0:
            raise ValueError("At least one temporal aug must be a sampler")
        last_sampler = samplers[-1]
        self.target_length = last_sampler.target_length

        crops = [augS for augS in self.spatial_aug if isinstance(augS, CropConfig)]
        if len(crops) == 0:
            raise ValueError("At least one spatial aug must be a crop")
        last_crop = crops[-1]
        self.frame_size = last_crop.frame_size

        return self


class DataInfo(BaseModel):
    train_augs: Optional[AugInfo] = None
    test_augs: Optional[AugInfo] = None
    strict_size: bool = True  # from config
    target_length: Optional[int] = None
    frame_size: Optional[int] = None

    @model_validator(mode="after")
    def check_frame_strat(self) -> "DataInfo":
        if not self.strict_size:
            return self

        if self.train_augs is None or self.test_augs is None:
            raise ValueError("Aug info cannot be None if strict_size enabled")

        self.target_length = self.train_augs.target_length
        self.frame_size = self.train_augs.frame_size

        assert self.train_augs.target_length == self.test_augs.target_length, (
            f"Train/test target_length mismatch: "
            f"{self.train_augs.target_length} vs {self.test_augs.target_length}"
        )

        assert self.train_augs.frame_size == self.test_augs.frame_size, (
            f"Train/test target_length mismatch: "
            f"{self.train_augs.frame_size} vs {self.test_augs.frame_size}"
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
    split: AVAIL_SPLITS
    save_path: str
    seed: int = SEED


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


TRAIN_TYPE: TypeAlias = Literal["supervised", "unsupervised"]


class SupervisedInfo(BaseModel):
    drop_p: Optional[float] = None
    type: Literal["supervised"] = "supervised"


class MVirTedInfo(BaseModel):
    type: Literal["mvir_ted"] = "mvir_ted"
    drop_p: Optional[float] = None
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    max_frames: int = 64
    mvit_out_dim: int = 768


class UnsupervisedInfo(BaseModel):
    type: Literal["unsupervised"] = "unsupervised"


class MVirTedMaeInfo(BaseModel):
    type: Literal["mvir_ted_mae"] = "mvir_ted_mae"
    encoder_info: MVirTedInfo = MVirTedInfo()
    mask_ratio: float = 0.5
    embed_dim: int = 512


ModelInfo = Annotated[
    Union[SupervisedInfo, MVirTedInfo, UnsupervisedInfo, MVirTedMaeInfo],
    Field(discriminator="type"),
]


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
    model_params: ModelInfo = Field(default_factory=SupervisedInfo)
    data: DataInfo
    scheduler: Optional[SchedInfo] = None
    early_stopping: Optional[StopperOn] = None

    @field_validator("model_params", mode="before")
    @classmethod
    def _default_model_type(cls, v: Any) -> Any:
        if isinstance(v, dict) and "type" not in v:
            v = {"type": "supervised", **v}
        return v

    @model_validator(mode="after")
    def _resolve_norms(self) -> "RunInfo":
        """Substitute norm_dict based on model name when norm=True."""
        from models import norm_vals

        for aug_info in (self.data.train_augs, self.data.test_augs):
            if aug_info is not None and aug_info.normalise:
                aug_info.norm_dict = norm_vals(self.admin.model)  # type: ignore  doesnt liek the 2
        return self


class ExpInfo(RunInfo):
    model_config = ConfigDict(extra="forbid")
    wandb: WandbInfo


class CompExpInfo(ExpInfo):
    results: CompRes


# class GenInfo(BaseModel):
#     training: Optional[TrainingInfo] = None
#     optimizer: Optional[OptimizerInfo] = None
#     model_params: Optional[ModelParamsInfo] = None
#     data: Optional[DataInfo] = None
#     scheduler: Optional[SchedInfo] = None
#     early_stopping: Optional[StopperOn] = None

GenInfo: TypeAlias = Dict[str, Any]


class ResSet(BaseModel):
    spec: GenInfo
    results: List[RunRes]


class RunRes(BaseModel):
    admin: AdminInfo
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
                old_val = d.get(k, {})
                d[k] = self._set_inplace(old_val, next_key, ks, val)
        else:
            if len(ks) == 0:
                d = {k: val}
            else:
                next_key = ks.pop(0)
                d = {k: self._set_inplace({}, next_key, ks, val)}
        return d

    def pop(self, keys: List[Any], default=None) -> Any:
        if len(keys) == 1:
            return self.dict.pop(keys[0], default)

        # Navigate to the parent of the target key
        parent = self.dict
        for key in keys[:-1]:
            parent = parent[key]

        return parent.pop(keys[-1], default)

    def to_dict(self) -> Dict[Any, Any]:
        return self.dict.copy()

    def __str__(self) -> str:
        return str(self.dict)

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        yield from self._iter_leaves(self.dict, [])

    def _iter_leaves(self, d: Any, path: List[Any]):
        if isinstance(d, dict):
            for key, val in d.items():
                yield from self._iter_leaves(val, path + [key])
        else:
            yield path, d


# not ignoring extra keys overrides: Claudes baby


T = TypeVar("T", bound=BaseModel)


def _replace_in_annotation(annotation, old_cls, new_cls):
    """Replace old_cls with new_cls inside an annotation, preserving Optional/Union wrappers."""
    if annotation is old_cls:
        return new_cls
    origin = get_origin(annotation)
    if origin is Union:
        new_args = tuple(
            _replace_in_annotation(arg, old_cls, new_cls)
            for arg in get_args(annotation)
        )
        return Union[new_args]
    return annotation


def make_strict(model_cls: Type[BaseModel]) -> Type[BaseModel]:
    namespace: dict = {"model_config": ConfigDict(extra="forbid")}
    annotations = {}

    for name, field_info in model_cls.model_fields.items():
        annotation = field_info.annotation
        inner = _unwrap_annotation(annotation)
        if inner is not None and issubclass(inner, BaseModel):
            strict_inner = make_strict(inner)
            # Preserve Optional[...] wrapper rather than just using the raw strict class
            annotations[name] = _replace_in_annotation(annotation, inner, strict_inner)
            # Preserve default so Optional fields don't become required
            if field_info.default is not PydanticUndefined:
                namespace[name] = field_info.default
            elif field_info.default_factory is not None:
                namespace[name] = Field(default_factory=field_info.default_factory)

    if annotations:
        namespace["__annotations__"] = annotations

    return type(model_cls.__name__, (model_cls,), namespace)

def _unwrap_annotation(annotation) -> Type | None:
    origin = get_origin(annotation)
    if origin is Union:
        for arg in get_args(annotation):
            result = _unwrap_annotation(arg)
            if result is not None:
                return result
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _strip_computed(model_cls: Type[BaseModel], data: dict) -> dict:
    """Recursively remove computed field keys from a data dict before strict validation."""
    computed_keys = set(model_cls.model_computed_fields.keys())
    result = {}

    for k, v in data.items():
        if k in computed_keys:
            continue
        field_info = model_cls.model_fields.get(k)
        if field_info and isinstance(v, dict):
            inner = _unwrap_annotation(field_info.annotation)
            if inner is not None and issubclass(inner, BaseModel):
                v = _strip_computed(inner, v)
        result[k] = v

    return result


def strict_validate(model_cls: Type[T], data: dict) -> T:
    strict_cls = make_strict(model_cls)
    strict_cls.model_validate(_strip_computed(model_cls, data))
    return model_cls.model_validate(data)
