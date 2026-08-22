from __future__ import annotations

from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticUndefined

# constants
#wandb
ENTITY = "ljgoodall2001-rhodes-university"
PROJECT_BASE = "WLASL"

#Files
LABEL_SUFFIX = "fixed_frange_bboxes.json"

NUM_INSTANCES_SUFFIX = "num_instances.json"
WORST_INSTANCES_SUFFIX = "f1-score_MViTv2_B_32x3_asl2000_004.json"
ZFILL = 3
CONFIG_FILETYPE = ".toml"
#Directories
CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parent
SLR_ROOT = SRC_ROOT.parent
CLASSES_PATH = SRC_ROOT / "info/wlasl_class_list.json"
RUNS_PATH = SRC_ROOT / "runs"
CONFIGS_PATH = SRC_ROOT / "configfiles"
WLASL_ROOT = SLR_ROOT / "data/WLASL"
LABELS_PATH = WLASL_ROOT  / "preprocessed/labels"
RAW_DIR = WLASL_ROOT / "WLASL2000"
SPLIT_DIR = WLASL_ROOT / "splits"
RESULTS_DIR = SRC_ROOT / 'results'
# Misc
SEED = 42



### for model normalisation


class NormDict(BaseModel):
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


####################### Data loading and augmentation #############################

AVAIL_SETS : TypeAlias = Literal["train", "val", "test"]
ORIGINAL_SPLITS : TypeAlias = Literal["asl100", "asl300", "asl1000", "asl2000"]
#Splits with different frame cuttoff
CUTOFF_SPLITS : TypeAlias = Literal["asl100_cutoff_9", "asl300_cutoff_9", "asl1000_cutoff_9", "asl2000_cutoff_9"]
CUTOFF_9_NAMES : list[CUTOFF_SPLITS] = ["asl100_cutoff_9", "asl300_cutoff_9", "asl1000_cutoff_9", "asl2000_cutoff_9"]
#Splits reconstructed from the worst and fewest classes
WORST_SPLITS : TypeAlias = Literal["asl100_bottom", "asl100_worst"]

AVAIL_SPLITS : TypeAlias =  ORIGINAL_SPLITS | CUTOFF_SPLITS | WORST_SPLITS

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
    def check_speeds(self) -> SpeedSampler:
        if self.speed_min > self.speed_max:
            raise ValueError("speed_min cannot be > speed_max")
        return self

SAMPLER_TYPES = {"og", "pad", "uniform", "chunked", "wobbled", "focal_normal", "focal_laplace", "focal_beta", "speed"}
def is_sampler_config(config: TemporalAugs) -> TypeGuard[SamplerConfig]:
    return config.type in SAMPLER_TYPES

SamplerConfig = Annotated[
    UniformSampler | WobbledSampler | SpeedSampler | FocalNormalSampler | FocalLaplaceSampler | FocalBetaSampler | ChunkedSampler | PadFramesT | OG_Sampler,
    Field(discriminator="type"),
]

### Temporal augs


class ShuffleT(BaseModel):
    type: Literal["shuffle"] = "shuffle"
    num_frames: int | None = None


class ReverseT(BaseModel):
    type: Literal["reverse"] = "reverse"
    probability: float = 0.5


TemporalTransforms = Annotated[ShuffleT |  ReverseT, Field(discriminator="type")]

TEMPORAL_TYPES = {"shuffle", "reverse"}
def is_temporal_config(config: TemporalAugs) -> TypeGuard[TemporalTransforms]:
    return config.type in TEMPORAL_TYPES

TemporalAugs = Annotated[
    TemporalTransforms | SamplerConfig, Field(discriminator="type")
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
    CentreCropConfig | RandomCropConfig | ScaleAndPadConfig | RandomResizedConfig,
    Field(discriminator="type"),
]

CROP_TYPES = {"Centre_crop", "Random_crop", "Scale_and_pad", "Random_Resized_crop"}
def is_crop_config(config: SpatialAugs) -> TypeGuard[CropTransforms]:
    return config.type in CROP_TYPES

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
    AutoAugmentConfig | RandAugConfig | HorizontalFlipConfig | RandomGrayscaleConfig | GaussianBlurConfig,
    Field(discriminator="type"),
]

SPATIAL_TYPES = {"HORIZONTAL_FLIP", "RANDOM_GRAYSCALE", "GAUSSIAN_BLUR", "IMAGENET", "CIFAR10", "SVHN", "RANDAUGMENT"}

def is_spatial_transform_config(config: SpatialAugs) -> TypeGuard[SpatialTransforms]:
    return config.type in SPATIAL_TYPES

SpatialAugs = Annotated[
    CropTransforms | SpatialTransforms,
    Field(discriminator="type"),
]


class AugInfo(BaseModel):
    """Augmentation info for a video

    Attributes:
        normalise (bool): Flag to fetch norm values during config parsing. Default False.
        norm_dict (Optional[NormDict]): Supplied Normalisation values. Default None.
        temporal_aug (list[TemporalAugs]): Temporal augmentations to be applied in order. Default [].
        spatial_aug (list[SpatialAugs]): Spatial augmentations to be applied in order. Default [].
        strict_size (bool): Validate that at least one frame sampler and crop strategy is defined. Default True.
    """

    normalise: bool = False
    norm_dict: NormDict | None = None
    temporal_aug: list[TemporalAugs] = []
    spatial_aug: list[SpatialAugs] = []
    strict_size: bool = True
    target_length: int | None = None
    frame_size: int | None = None

    @model_validator(mode="after")
    def _validate_augs(self) -> AugInfo:
        if not self.strict_size:
            return self

        samplers = [augT for augT in self.temporal_aug if is_sampler_config(augT)]
        crops = [augS for augS in self.spatial_aug if is_crop_config(augS)]
        
        if len(samplers) == 0:
            raise ValueError("At least one temporal aug must be a sampler")
        last_sampler = samplers[-1]
        self.target_length = last_sampler.target_length

        if len(crops) == 0:
            raise ValueError("At least one spatial aug must be a crop")
        last_crop = crops[-1]
        self.frame_size = last_crop.frame_size

        return self


class DataInfo(BaseModel):
    train_augs: AugInfo | None = None
    test_augs: AugInfo | None = None
    strict_size: bool = True  # from config
    target_length: int | None = None
    frame_size: int | None = None

    @model_validator(mode="after")
    def check_frame_strat(self) -> DataInfo:
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
StoppingMetrics: TypeAlias = Literal["loss", "acc"]
StoppingPhases: TypeAlias = Literal['val', 'train']
StoppingModes: TypeAlias = Literal["min", "max"]


class EarlyStopperInfo(BaseModel):
    type: Literal['early_stopper'] = 'early_stopper'
    metric: StoppingMetrics
    phase: StoppingPhases = 'val'
    mode: StoppingModes
    patience: int
    min_delta: float

    @model_validator(mode="after")
    def _config_precheck(self) -> EarlyStopperInfo:
        if self.patience <= 0:
            raise ValueError(
                f"Patience must be a positive integer, got {self.patience}"
            )
        if self.min_delta < 0:
            raise ValueError(
                f"Min delta must be non-negative, got {self.min_delta}"
            )
        return self

class StopperState(EarlyStopperInfo):
    best_score: float | None = None
    best_epoch: int = 0
    counter: int = 0
    stop: bool = False

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
    weight_path: str | None = None


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
    drop_p: float | None = None
    type: Literal["supervised"] = "supervised"


class MVirTedInfo(BaseModel):
    type: Literal["mvir_ted"] = "mvir_ted"
    drop_p: float | None = None
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

SUPERVISED_TYPES = {"supervised"}
PRETRAIN_TYPES = {"mvir_ted_mae"}

def is_supervised_config(config: ModelInfo) -> TypeGuard[SupervisedInfo]:
    return config.type in SUPERVISED_TYPES

def is_pretrain_config(config: ModelInfo) -> TypeGuard[MVirTedMaeInfo]:
    return config.type in PRETRAIN_TYPES

ModelInfo = Annotated[
    SupervisedInfo | MVirTedInfo | MVirTedMaeInfo,
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
        if not (0 <= self.start_factor < self.end_factor <= 1.0):
            raise ValueError(f"start_factor must be >= 0 and < end_factor <= 1.0, but got: start {self.start_factor} end {self.end_factor}")
        return self


class SchedBase(BaseModel):
    warm_up: WarmUpSched | None = None


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
    min_lr: list[float] | float
    eps: float


# Discriminated union: pydantic dispatches on the 'type' field automatically
SchedInfo = Annotated[
    WarmOnly | CosAnealInfo | WarmRestartInfo | ReduceLROnPlateau,
    Field(discriminator="type"),
]


class WandbInfo(BaseModel):
    entity: str
    project: str
    tags: list[str] = []
    run_id: str | None = None
    sweep_id: str | None = None

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
    perm: list[int]
    shannon_entropy: float


class ClassReport(BaseModel):
    cls_report: dict[str, dict[str, float]]
    all_targets: list[int]
    all_preds: list[int]

class CompRes(BaseModel):
    check_name: str
    best_val_acc: float
    best_val_loss: float
    test: BaseRes
    val: BaseRes
    # test_shuff: ShuffRes

class VerboseRes(BaseModel):
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
    scheduler: SchedInfo | None = None
    stopping: EarlyStopperInfo | None = None

    @field_validator("model_params", mode="before")
    @classmethod
    def _default_model_type(cls, v: Any) -> Any:
        if isinstance(v, dict) and "type" not in v:
            v = {"type": "supervised", **v}
        return v

    @model_validator(mode="after")
    def _resolve_norms(self) -> RunInfo:
        """Substitute norm_dict based on model name when norm=True."""
        from src.models import norm_vals

        for aug_info in (self.data.train_augs, self.data.test_augs):
            if aug_info is not None and aug_info.normalise:
                aug_info.norm_dict = norm_vals(self.admin.model)  # type: ignore  doesnt liek the 2
        return self


class ExpInfo(RunInfo):
    model_config = ConfigDict(extra="forbid")
    wandb: WandbInfo


class CompExpInfo(ExpInfo):
    results: CompRes



GenInfo: TypeAlias = dict[str, Any]


class ResSet(BaseModel):
    spec: GenInfo
    results: list[RunRes]


class RunRes(BaseModel):
    admin: AdminInfo
    wandb: WandbInfo
    results: CompRes


class FailedExp(ExpInfo):
    error: str


class SumarisedNew(BaseModel):
    run_id: str | None = None
    model: str
    exp_no: str
    dataset: str
    split: str
    config_path: str


class Sumarised(SumarisedNew):
    best_val_acc: float | None = None
    best_val_loss: float | None = None


class SummarisedRes(Sumarised):
    test_top1_acc: float | None = None
    test_av_loss: float | None = None


class SummarisedError(Sumarised):
    error: str


class CleverDict(dict):
    def __init__(self, dict: dict[Any, Any]):
        self.dict = dict

    def __getitem__(self, keys: list[Any]) -> Any:
        d = self.dict.copy()
        for key in keys:
            d = d[key]
        return d

    def __setitem__(self, keys: list[Any], val: Any):
        self.dict = self._set_inplace(self.dict, keys[0], keys[1:], val)

    def _set_inplace(
        self, d: dict[Any, Any], k: Any, ks: list[Any], val: Any
    ) -> dict[Any, Any]:
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

    def pop(self, keys: list[Any], default=None) -> Any:
        if len(keys) == 1:
            return self.dict.pop(keys[0], default)

        # Navigate to the parent of the target key
        parent = self.dict
        for key in keys[:-1]:
            parent = parent[key]

        return parent.pop(keys[-1], default)

    def to_dict(self) -> dict[Any, Any]:
        return self.dict.copy()

    def __str__(self) -> str:
        return str(self.dict)

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        yield from self._iter_leaves(self.dict, [])

    def _iter_leaves(self, d: Any, path: list[Any]):
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
        return new_args
    return annotation


def make_strict(model_cls: type[BaseModel]) -> type[BaseModel]:
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

def _unwrap_annotation(annotation) -> type | None:
    origin = get_origin(annotation)
    if origin is Union:
        for arg in get_args(annotation):
            result = _unwrap_annotation(arg)
            if result is not None:
                return result
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _strip_computed(model_cls: type[BaseModel], data: dict) -> dict:
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


def strict_validate(model_cls: type[T], data: dict) -> T:
    strict_cls = make_strict(model_cls)
    strict_cls.model_validate(_strip_computed(model_cls, data))
    return model_cls.model_validate(data)
