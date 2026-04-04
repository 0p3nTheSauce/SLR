import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms.v2 import Transform
import torchvision.transforms.v2 as v2
from functools import partial
from typing import Callable, List, Tuple,  Optional
import random
import utils
import time
import gc
import statistics
import numpy as np

# locals
from models import NormDict
from preprocess import Instance
from run_types import (
    SpatialStrategy,
    TemporalStrategy,
    Frame_Size_Strategy,
    SamplerConfig,
    OG_Sampler,
    UniformSampler,
    ChunkedSampler,
    WobbledSampler,
    FocalNormalSampler,
    FocalLaplaceSampler,
    FocalBetaSampler,
    SpeedSampler
)


#   --- Bench marking ---


def bench_mark_enhanced(
    data,
    transform: Callable,
    iters: int = 100,
    warmup: int = 10,
    device: Optional[str] = None,
):
    """
    Enhanced benchmarking with warmup, memory tracking, and statistics
    """
    # Move data to specified device if provided
    if device and hasattr(data, "to"):
        data = data.to(device)

    # Warmup runs (not timed)
    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        _ = transform(data)

    # Clear cache and collect garbage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Track memory if on GPU
    memory_before = None
    memory_after = None
    if device and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated()

    # Actual benchmark runs
    times = []
    for i in range(iters):
        start = time.perf_counter()
        result = transform(data)
        end = time.perf_counter()
        times.append(end - start)

        # Clear result to prevent memory buildup
        del result

    # Memory tracking
    if device and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated()

    # Calculate statistics
    total_time = sum(times)
    avg_time = statistics.mean(times)
    median_time = statistics.median(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)

    # Format results
    result = f"""
Benchmark Results:
=================
Transform: {transform}
Data shape: {data.shape if hasattr(data, "shape") else "N/A"}
Device: {device or "CPU"}
Warmup iterations: {warmup}
Benchmark iterations: {iters}

Timing Statistics:
  Total time: {total_time:.4f} seconds
  Average time: {avg_time:.6f} seconds
  Median time: {median_time:.6f} seconds
  Std deviation: {std_time:.6f} seconds
  Min time: {min_time:.6f} seconds
  Max time: {max_time:.6f} seconds
  
Throughput: {iters / total_time:.2f} transforms/second"""

    if memory_before is not None and memory_after is not None:
        memory_used = memory_after - memory_before
        result += f"""
		
Memory Usage:
  Before: {memory_before / 1024**2:.2f} MB
  After: {memory_after / 1024**2:.2f} MB
  Used: {memory_used / 1024**2:.2f} MB"""

    return result


#   --- Frame sampling strategies ---


def sample(frames: Tensor, target_length: int, randomise: bool = False) -> Tensor:
    """Original sampler, uses uniform sampling by taking every nth frame, where n is the step size calculated as total frames divided by target length. If randomise is True, it samples randomly from each chunk of frames instead of taking the first frame of each chunk.

    Args:
        frames (Tensor): Input frames tensor of shape (T, C, H, W).
        target_length (int): Desired number of frames after sampling. It probably wont return this consistently so requires padding.
        randomise (bool, optional): Whether to randomize the sampling (chunk-wise). Defaults to False.

    Returns:
        Tensor: Sampled frames tensor of shape (target_length, C, H, W).
    """
    step = frames.shape[0] // target_length
    if not randomise:
        return frames[::step]
    cnt = 0
    chunk = []
    sampled_frames = []
    for frame in frames:
        if cnt < step:
            chunk.append(frame)
            cnt += 1
        else:
            choice = random.choice(chunk)
            sampled_frames.append(choice)
            chunk = []
            cnt = 0
    return torch.stack(sampled_frames, dim=0)


def correct_num_frames(
    frames: Tensor, target_length: int, randomise: bool = False
):
    """Original function to compensate for sample. Corrects the number of frames to match the target length.
    Args:
        frames (Tensor): The input frames tensor. (T x C x H x W)
        target_length (int): The target length for the number of frames.
        randomise (bool, optional): Whether to randomize the sampling (chunk-wise). Defaults to False.
    Returns:
        Tensor: The corrected frames tensor with the specified target length.
    """
    if frames.shape[0] == 0:
        raise ValueError("Input frames tensor is empty")
    if target_length <= 0:
        raise ValueError("Target length must be a positive integer.")
    if frames.shape[0] == target_length:
        return frames
    if frames.shape[0] < target_length:
        # Pad with zeros if the number of frames is less than the target length
        padding = torch.zeros(
            target_length - frames.shape[0],
            frames.shape[1],
            frames.shape[2],
            frames.shape[3],
            device=frames.device,
        )
        return torch.cat((frames, padding), dim=0)
    else:
        sampled_frames = sample(frames, target_length, randomise=randomise)
        diff = target_length - len(sampled_frames)
        if diff > 0:
            padding = torch.zeros(
                diff,
                frames.shape[1],
                frames.shape[2],
                frames.shape[3],
                device=frames.device,
            )
            return torch.cat((sampled_frames, padding), dim=0)
        elif diff < 0:
            return sampled_frames[:target_length]
        else:
            return sampled_frames


def pad_frames(frames: Tensor, target_length: int) -> Tensor:
    """Original. Pads and trimms to get frames to target length

    Args:
        frames (Tensor): Input frames tensor of shape (T, C, H, W).
        target_length (int): Desired number of frames after padding.

    Returns:
        Tensor: Padded frames tensor of shape (target_length, C, H, W).
    """
    num_frames = frames.shape[0]
    if num_frames == target_length:
        return frames
    elif num_frames < target_length:
        # Pad with zeros if the number of frames is less than the target length
        padding = torch.zeros(
            target_length - num_frames,
            frames.shape[1],
            frames.shape[2],
            frames.shape[3],
            device=frames.device,
        )
        return torch.cat((frames, padding), dim=0)
    else:
        # Trim the frames if the number of frames is greater than the target length
        return frames[:target_length, :, :, :]


def sample_uniform(
    frames: Tensor,
    target_length: int
) -> Tensor:
    """Uniformly sample frames from a video tensor with either target_length or step

    Args:
        frames (Tensor): The input frames tensor. (T x C x H x W)
        target_length (int): The desired number of frames after sampling. 

    Returns:
        Tensor: The sampled frames tensor.
    """
    T = frames.shape[0]
    indices = torch.linspace(0, T - 1, target_length).long()
    return frames[indices]

def sample_chunked(frames: Tensor, target_length: int) -> Tensor:
    """Sample frames randomly from evenly divided chunks

    Args:
        frames (Tensor): The input frames tensor. (T x C x H x W)
        target_length (int): The desired number of frames after sampling.

    Returns:
        Tensor: The sampled frames tensor.
    """
    T = frames.shape[0]
    chunk_size = T / target_length
    indices = [
        int(i * chunk_size + random.uniform(0, chunk_size))
        for i in range(target_length)
    ]
    indices = [min(i, T - 1) for i in indices]
    return frames[Tensor(indices)]

def wobble(frames: Tensor, max_wobble: int = 0) -> Tensor:
    """Add a random wobble to the start and end indeces

    Args:
        frames (Tensor): TxCxHxW
        max_wobble (int): Random wobble. Defaults to 4.

    Returns:
        Tensor: Wobbled Tensor
    """
    T = frames.shape[0]
    start = random.randint(-max_wobble, max_wobble)
    end = (T - 1) + random.randint(-max_wobble, max_wobble)
    indices = torch.linspace(start, end, T).long()

    # repeats last frame if index goes out of bounds, which can happen due to wobble
    indices = indices.clamp(0, T - 1)
    return frames[indices]
    
def sample_wobbled(
    frames: Tensor,
    target_length: Optional[int] = None,
    max_wobble: int = 4,
) -> Tensor:
    """Sample frames uniformly with a random "wobble" added to the start and end indices, creating a jittered sampling effect.
    Optionally leave out the target_length argument to just add wobble to but keep all frames.

    Args:
        frames (Tensor): The input frames tensor. (T x C x H x W)
        target_length (Optional[int], optional): The desired number of frames after sampling. Defaults to None.
        max_wobble (int, optional): The maximum amount of wobble to add to the start and end indices. Defaults to 4.

    Returns:
        Tensor: The sampled frames tensor.
    """
    T = frames.shape[0]
    if target_length is None:
        target_length = T

    start = random.randint(-max_wobble, max_wobble)
    end = (T - 1) + random.randint(-max_wobble, max_wobble)

    indices = torch.linspace(start, end, target_length).long()

    # repeats last frame if index goes out of bounds, which can happen due to wobble
    indices = indices.clamp(0, T - 1)
    return frames[indices]


def sample_focal_normal(
    frames: Tensor,
    target_length: int,
    mean: float = 0.5,
    std: float = 0.25,
) -> Tensor:
    """Sample frames with density weighted by a Normal distribution.

    Frames near the centre of the video are sampled more frequently,
    with the spread controlled by std. Out-of-range samples are clamped
    to valid frame indices, so boundary frames may appear more than once
    at very small std values.

    Args:
        frames (Tensor): Input tensor of shape (T, C, H, W).
        target_length (int): Number of frames to sample.
        mean (float): Centre of the distribution as a fraction of video length. Default 0.5.
        std (float): Standard deviation as a fraction of video length. Default 0.25.

    Returns:
        Tensor: Sampled frames of shape (target_length, C, H, W).
    """
    T = frames.shape[0]
    dist = torch.distributions.Normal(mean * (T - 1), std * (T - 1))
    indices = dist.sample((target_length,)).clamp(0, T - 1).long()
    indices, _ = indices.sort()
    return frames[indices]


def sample_focal_laplace(
    frames: Tensor,
    target_length: int,
    mean: float = 0.5,
    diversity: float = 0.175,
) -> Tensor:
    """Sample frames with density weighted by a Laplace distribution.

    Produces a sharper peak than Normal with heavier tails, concentrating
    more samples at the centre while still covering the boundaries more
    than a tight Normal would. Out-of-range samples are clamped.

    Args:
        frames (Tensor): Input tensor of shape (T, C, H, W).
        target_length (int): Number of frames to sample.
        mean (float): Centre of the distribution as a fraction of video length. Default 0.5.
        diversity (float): Scale parameter (b) of the Laplace distribution as a fraction
            of video length. Smaller values give a sharper peak. Default 0.175.

    Returns:
        Tensor: Sampled frames of shape (target_length, C, H, W).
    """
    T = frames.shape[0]
    dist = torch.distributions.Laplace(mean * (T - 1), diversity * (T - 1))
    indices = dist.sample((target_length,)).clamp(0, T - 1).long()
    indices, _ = indices.sort()
    return frames[indices]


def sample_focal_beta(
    frames: Tensor,
    target_length: int,
    alpha: float = 4.0,
    beta: float = 4.0,
) -> Tensor:
    """Sample frames with density weighted by a Beta distribution.

    The Beta distribution is naturally bounded to [0, 1], so sampled indices
    are always valid with no clamping required. When alpha == beta, the
    distribution is symmetric about the centre. alpha=beta=1 recovers uniform
    sampling; increasing both concentrates samples toward the centre.
    Setting alpha != beta shifts the peak toward the start (alpha < beta)
    or end (alpha > beta) of the video.

    Args:
        frames (Tensor): Input tensor of shape (T, C, H, W).
        target_length (int): Number of frames to sample.
        alpha (float): First shape parameter. Default 4.0.
        beta (float): Second shape parameter. Default 4.0.

    Returns:
        Tensor: Sampled frames of shape (target_length, C, H, W).
    """
    T = frames.shape[0]
    dist = torch.distributions.Beta(alpha, beta)
    indices = (dist.sample((target_length,)) * (T - 1)).long()
    indices, _ = indices.sort()
    return frames[indices]


def sample_speed_perturbed(
    frames: Tensor,
    target_length: int,
    speed_range: tuple[float, float] = (0.8, 1.2),
) -> Tensor:
    T = frames.shape[0]
    speed = random.uniform(*speed_range)
    effective_length = min(T, int(target_length * speed))
    start = random.randint(0, T - effective_length)
    indices = torch.linspace(start, start + effective_length - 1, target_length).long()
    return frames[indices]


def _apply_wobble_then_sample(frames: Tensor, target_length: int, *, fn: Callable, max_wobble: int) -> Tensor:
    return fn(wobble(frames, max_wobble), target_length)

def get_sampler(sampler_config: SamplerConfig) -> Callable[[Tensor, int], Tensor]:
    """
    Returns a specialized sampling function based on the provided SamplerConfig.
    The returned function can be described by: f(frames, target_length) -> sampled_frames.
    """
    match sampler_config:
        case OG_Sampler():
            sampler =  partial(
                correct_num_frames,
                randomise=sampler_config.randomise,
            )

        case UniformSampler():
            sampler = partial(
                sample_uniform,
            )

        case ChunkedSampler():
            # Note: target_length is required for chunked in your Pydantic model
            sampler = partial(sample_chunked,)

        case WobbledSampler():
            return partial(
                sample_wobbled,
                max_wobble=sampler_config.max_wobble,
            )

        case FocalNormalSampler():
            sampler = partial(
                sample_focal_normal,
                mean=sampler_config.mean,
                std=sampler_config.std,
            )

        case FocalLaplaceSampler():
            sampler = partial(
                sample_focal_laplace,
                mean=sampler_config.mean,
                diversity=sampler_config.diversity,
            )

        case FocalBetaSampler():
            sampler = partial(
                sample_focal_beta,
                alpha=sampler_config.alpha,
                beta=sampler_config.beta,
            )

        case SpeedSampler():
            sampler = partial(
                sample_speed_perturbed,
                speed_range=(sampler_config.speed_min, sampler_config.speed_max),
            )

        case _:
            raise ValueError(f"Unknown sampler method: {sampler_config.method}")

    if sampler_config.max_wobble > 0:
        return partial(_apply_wobble_then_sample, fn=sampler, max_wobble=sampler_config.max_wobble)
    else:
        return sampler

# --- Temporal augmentations ---


class Shuffle(Transform):
    """Shuffle frames using the configs defined seed"""

    def __init__(self, num_frames: int, perm: Optional[Tensor] = None):
        super().__init__()
        self.num_frames = num_frames
        if perm is None:
            self.permutation = self.create_permutation()
        else:
            self.permutation = perm

    def _transform(self, inpt, params):
        dim = 0
        assert inpt.dim() == 4, "Input tensor must be 4D (T, C, H, W)"
        assert inpt.size(0) == self.num_frames, (
            "Permutation length must match number of frames"
        )
        assert inpt.size(1) == 3, "Input tensor must have 3 channels (C=3)"
        return inpt.index_select(dim, self.permutation)

    def create_permutation(self) -> Tensor:
        """Create a random permutation for given number of frames"""
        return torch.randperm(self.num_frames)

    @staticmethod
    def shannon_entropy(perm: Tensor) -> float:
        """Compute the Shannon entropy (bits) between differences of consecutive indeces

        Algorythm from:
                first answer of:
                        https://stats.stackexchange.com/questions/78591/correlation-between-two-decks-of-cards
                which was referenced by:
                        https://mikesmathpage.wordpress.com/2017/04/23/card-shuffling-and-shannon-entropy/
        Args:
                perm (Tensor): Permutation generated by shuffle.
        Returns:
                float: The Shannon entropy in bits
        """
        plen = perm.shape[0]
        diffs = torch.zeros(plen, dtype=torch.long)
        # print(diffs)
        for i in range(-1, plen - 1):
            diff = perm[i + 1] - perm[i]
            if diff < 0:
                diff += plen
            diffs[i] = diff

        hist = torch.bincount(diffs, minlength=plen).float()

        normed = hist / plen
        normed_no0 = normed[normed > 0]
        entropy = -torch.sum(normed_no0 * torch.log2(normed_no0))
        return entropy.item()


class ReverseFrames(Transform):
    """Randomly reverse the temporal order of the video sequence."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def _transform(self, inpt: Tensor, params: dict) -> Tensor:
        if random.random() < self.p:
            # Assuming input is (T, C, H, W), flipping along the time dimension
            return inpt.flip(0)
        return inpt

#currently not used
def get_temporal_transform(
    name: str, num_frames: Optional[int] = None, reverse_prob: float = 0.5
) -> Tuple[Transform, Optional[Tuple]]:
    if name == "Shuffle":
        assert num_frames is not None, "num_frames must be specified if Shuffle is used"
        shuffle_t = Shuffle(num_frames)
        perm_tensor = shuffle_t.permutation

        sh_e = Shuffle.shannon_entropy(perm_tensor)
        perm = list(map(int, perm_tensor.numpy()))

        return shuffle_t, (perm, sh_e)

    elif name == "Reverse":
        return ReverseFrames(reverse_prob), None
    else:
        raise ValueError(f"Unknown temporal transform: {name}")

# --- Frame cropping and resizing ---


def crop_frames(frames: Tensor, bbox: List[int]):
    """
    Crop the frames using the bounding box

    :param frames: frames hase shape (num_frames, channels, height, width)
    :type frames: Tensor
    :param bbox: bbox is a list of [x1, y1, x2, y2] coordinates
    :type bbox: List[int]
    """
    x1, y1, x2, y2 = bbox
    return frames[:, :, y1:y2, x1:x2]


def _crop_frames(frames: Tensor, item: Instance):
    """Crop out the bounding box from the frames"""
    return crop_frames(frames, item.bbox)


def resize_by_diag(frames: Tensor, bbox: list[int], target_diag: int):
    """
    Resize frame so person bounding box diagonal equals target_diagonal

    In the WLASL paper, they first resize the frames so that
    the person bounding box diagnol length is 256 pixels
    this doesnt work for us
    TODO: investigate this

    Args:
            frame: input video frame
            bbox: (x1, y1, x2, y2) of person bounding box
            target_diagonal: desired diagonal size in pixels
    """
    x1, y1, x2, y2 = bbox

    orig_width = x2 - x1
    orig_height = y2 - y1

    curr_diag = np.sqrt(orig_width**2 + orig_height**2)

    scale_factor = target_diag / curr_diag

    # resize the tensor
    new_width = int(frames.shape[2] * scale_factor)
    new_height = int(frames.shape[3] * scale_factor)

    transform = v2.Resize((new_height, new_width))

    return transform(frames)


def _resize_by_diagonal(frames, item):
    """Resize the target diagonal to 256 before random cropping as per wlasl"""
    return resize_by_diag(frames, item["bbox"], target_diag=256)


def scale_and_pad(frames: Tensor, size: int) -> Tensor:
    """Scale the larger side to frame size, then pad boundaries. Assumes T C H W"""
    T, C, H, W = frames.shape

    # Scale so the LARGER side hits `size`, preserving aspect ratio
    if H >= W:
        new_H, new_W = size, round(W * size / H)
    else:
        new_H, new_W = round(H * size / W), size

    # torchvision resize handles T C H W directly
    resiz = v2.Resize(
        [new_H, new_W], interpolation=v2.InterpolationMode.BICUBIC, antialias=True
    )

    # Distribute leftover pixels evenly; odd remainder goes to bottom/right
    pad_H = size - new_H
    pad_W = size - new_W
    top, bottom = pad_H // 2, pad_H - pad_H // 2
    left, right = pad_W // 2, pad_W - pad_W // 2

    return F.pad(resiz(frames), (left, right, top, bottom))  # F.pad: W then H


class ScaleAndPad(Transform):
    """Scale the larger side to frame size, then pad boundaries. Assumes T C H W"""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def _transform(self, inpt: Tensor, params: dict) -> Tensor:
        return scale_and_pad(inpt, self.size)


def get_frame_size_strategy(name: Frame_Size_Strategy, frame_size: int) -> Callable[[Tensor], Tensor]:
    match name:
        case "Centre_crop":
            return v2.CenterCrop(frame_size)
        case "Random_crop":
            return v2.RandomCrop(frame_size)
        case "Scale_and_pad":
            return ScaleAndPad(frame_size)
        case "Random_Resized_crop":
            return v2.RandomResizedCrop(frame_size)


# --- Spatial Transforms ---

def get_spatial_transform(name: SpatialStrategy) -> Transform:
    match name:
        case "IMAGENET":
            policy =  v2.AutoAugmentPolicy.IMAGENET
        case "CIFAR10":
            policy = v2.AutoAugmentPolicy.CIFAR10
        case "SVHN":
            policy = v2.AutoAugmentPolicy.SVHN
        case "Horizontal_flip":
            return v2.RandomHorizontalFlip()
    return v2.AutoAugment(policy=policy)


# --- Misc Named transforms and helper functions ---


def _identity_transform(x):
    """Identity transform - returns input unchanged"""
    return x


def _normalize_to_float(x):
    """Convert tensor to float and normalize to [0, 1]"""
    return x.float() / 255.0


def _permute_time_channel(x):
    """Permute tensor from (C, T, H, W) to (T, C, H, W)"""
    return x.permute(1, 0, 2, 3)


# --- Main transform construction function ---

def get_transform(
    num_frames: Optional[int] = None,
    frame_size: Optional[int] = None,
    norm_dict: Optional[NormDict] = None,
    frame_size_strategy: Optional[Frame_Size_Strategy] = None,
    temporal_aug: List[TemporalStrategy] = [],
    spatial_aug: List[SpatialStrategy] = [],
) -> Tuple[Callable[[Tensor], Tensor], Optional[List[int]], Optional[float]]:
    """Construct transform

    Args:
        num_frames (Optional[int], optional): number of frames. Defaults to None.
        frame_size (Optional[int], optional): Frame dimensions (square). Defaults to None.
        norm_dict (Optional[NormDict], optional): Normalisation values. Defaults to None.
        frame_size_strategy (List[Frame_Size_Strategy], optional): How to get frames the right size. Defaults to [].
        temporal_aug (List[TemporalStrategy], optional): Temporal augmentation strategy. Defaults to [].
        spatial_augment (List[SpatialStrategy], optional): Spatial augmentation strategy. Defaults to [].

    Raises:
        ValueError: If frame_size_strategy is specified without frame_size, or if an unexpected strategy is provided for frame sizing, temporal augmentation, or auto augmentation.
        ValueError: If auto_augment is not one of the expected strategies.

    Returns:
        Tuple[Callable[[Tensor], Tensor], Optional[List[int]], Optional[float]]: A tuple containing the composed transform, the permutation of frame indices if shuffling is applied, and the Shannon entropy of the shuffle. The permutation and entropy are None if no shuffling is applied.
    """

    transforms_list = []
    perm = None
    sh_e = None

    # --- 1. Temporal Augmentations (Operates on uint8 T, C, H, W) ---
    if "Shuffle" in temporal_aug:
        assert num_frames is not None, "num_frames must be specified if Shuffle is used"
        shuffle_t = Shuffle(num_frames)
        perm_tensor = shuffle_t.permutation
        transforms_list.append(shuffle_t)

        sh_e = Shuffle.shannon_entropy(perm_tensor)
        perm = list(map(int, perm_tensor.numpy()))

    if "Reverse" in temporal_aug:
        transforms_list.append(ReverseFrames(p=0.5))

    # --- 3. Spatial Cropping / Flipping ---
    if frame_size is not None:
        assert frame_size_strategy is not None,"Frame size strategy must be specified" 
        transforms_list.append(get_frame_size_strategy(frame_size_strategy, frame_size))

    # --- 4. Spatial Augment ---
    for aug in spatial_aug:
        transforms_list.append(get_spatial_transform(aug))

    transforms_list.append(v2.Lambda(_normalize_to_float))

    if norm_dict is not None:
        transforms_list.append(
            v2.Normalize(mean=norm_dict.mean, std=norm_dict.std)
        )

    transforms_list.append(v2.Lambda(_permute_time_channel))

    return v2.Compose(transforms_list), perm, sh_e



if __name__ == "__main__":
    vid = "./media/00333.mp4"
    frames = utils.load_rgb_frames_from_video(vid, 0, 0, all=True)
    print(frames.shape)
    utils.watch_video(frames, title="Original")

    # shuffled_indices = torch.randperm(frames.size(0))

    t = v2.Compose(
        [
            v2.Lambda(lambda x: x.permute(0, 2, 1, 3, 4)),  # T C H W -> C T H W
            Shuffle(frames.size(0)),
        ]
    )

    f2 = t(frames.unsqueeze(0)).squeeze(0)
    print(f2.shape)
    f2 = f2.permute(1, 0, 2, 3)
    print(f2.shape)
    utils.watch_video(f2, title="Shuffled")

    # rand = torch.rand(100, 3, 224, 224)
    # print(rand.shape)
    # t1 = ts.Compose([
    #   ts.Lambda(lambda x: x.unsqueeze(0)), #add batch dim
    #   ConsistentFrameShuffle(seed=42),
    #   ts.Lambda(lambda x: x.squeeze(0)) #remove batch dim
    # ])
    # f3 = t1(rand)
    # print(f3.shape)
    # # utils.watch_video(f4)
