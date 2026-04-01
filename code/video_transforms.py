import torch
from torchvision import tv_tensors
import torch.nn.functional as F
from torchvision.transforms.v2 import Transform
import torchvision.transforms.v2 as v2
from typing import Callable, List, TypeAlias, Literal, Tuple
import random
import utils
import time
import gc
from typing import Optional
import statistics
import numpy as np

# locals
from models import NormDict
from preprocess import InstanceDict
from run_types import AutoAugmentStrategy, TemporalStrategy, FrameSize_Strategy

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


def sample(frames, target_length, randomise=False):
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


def crop_frames(frames: torch.Tensor, bbox: List[int]):
    """
    Crop the frames using the bounding box

    :param frames: frames hase shape (num_frames, channels, height, width)
    :type frames: torch.Tensor
    :param bbox: bbox is a list of [x1, y1, x2, y2] coordinates
    :type bbox: List[int]
    """
    x1, y1, x2, y2 = bbox
    return frames[:, :, y1:y2, x1:x2]


def correct_num_frames(
    frames: torch.Tensor, target_length: int = 64, randomise: bool = False
):
    """Corrects the number of frames to match the target length.
    Args:
      frames (torch.Tensor): The input frames tensor. (T x C x H x W)
      target_length (int): The target length for the number of frames.
    Returns:
      torch.Tensor: The corrected frames tensor with the specified target length.
    """
    if frames is None or frames.shape[0] == 0:
        raise ValueError("Input frames tensor is empty or None.")
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


def pad_frames(frames, target_length):
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


# in the WLASL paper, they first resize the frames so that
# the person bounding box diagnol length is 256 pixels
# this douesnt work for us
# TODO: investigate this


def resize_by_diag(frames: torch.Tensor, bbox: list[int], target_diag: int):
    """
    Resize frame so person bounding box diagonal equals target_diagonal

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


class Shuffle(Transform):
    """Shuffle frames using the configs defined seed"""

    def __init__(self, num_frames: int, perm: Optional[torch.Tensor] = None):
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

    def create_permutation(self) -> torch.Tensor:
        """Create a random permutation for given number of frames"""
        return torch.randperm(self.num_frames)

    @staticmethod
    def shannon_entropy(perm: torch.Tensor) -> float:
        """Compute the Shannon entropy (bits) between differences of consecutive indeces

        Algorythm from:
                first answer of:
                        https://stats.stackexchange.com/questions/78591/correlation-between-two-decks-of-cards
                which was referenced by:
                        https://mikesmathpage.wordpress.com/2017/04/23/card-shuffling-and-shannon-entropy/
        Args:
                perm (torch.Tensor): Permutation generated by shuffle.
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


def _identity_transform(x):
    """Identity transform - returns input unchanged"""
    return x


def _normalize_to_float(x):
    """Convert tensor to float and normalize to [0, 1]"""
    return x.float() / 255.0


def _permute_time_channel(x):
    """Permute tensor from (C, T, H, W) to (T, C, H, W)"""
    return x.permute(1, 0, 2, 3)


def _resize_by_diagonal(frames, item):
    """Resize the target diagonal to 256 before random cropping as per wlasl"""
    return resize_by_diag(frames, item["bbox"], target_diag=256)


def _crop_frames(frames: torch.Tensor, item: InstanceDict):
    """Crop out the bounding box from the frames"""
    return crop_frames(frames, item["bbox"])


def scale_and_pad(frames: torch.Tensor, size: int) -> torch.Tensor:
    """Scale the larger side to frame size, then pad boundaries. Assumes T C H W"""
    T, C, H, W = frames.shape

    # Scale so the LARGER side hits `size`, preserving aspect ratio
    if H >= W:
        new_H, new_W = size, round(W * size / H)
    else:
        new_H, new_W = round(H * size / W), size

    # torchvision resize handles T C H W directly
    resiz = v2.Resize( [new_H, new_W],
                    interpolation=v2.InterpolationMode.BICUBIC, antialias=True)

    # Distribute leftover pixels evenly; odd remainder goes to bottom/right
    pad_H = size - new_H
    pad_W = size - new_W
    top,  bottom = pad_H // 2, pad_H - pad_H // 2
    left, right  = pad_W // 2, pad_W - pad_W // 2

    return F.pad(resiz(frames), (left, right, top, bottom))  # F.pad: W then H





class ReverseFrames(Transform):
    """Randomly reverse the temporal order of the video sequence."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        if random.random() < self.p:
            # Assuming input is (T, C, H, W), flipping along the time dimension
            return inpt.flip(0)
        return inpt

class ToVideoTensor(Transform):
    """Wraps tensor to guarantee temporally consistent spatial augmentations in v2."""
    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        # inpt is (T, C, H, W). tv_tensors.Video expects (C, T, H, W)
        return tv_tensors.Video(inpt.permute(1, 0, 2, 3))

class FromVideoTensor(Transform):
    """Unwraps back to standard (T, C, H, W) tensor."""
    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        # Revert from (C, T, H, W) back to (T, C, H, W)
        return inpt.permute(1, 0, 2, 3).as_subclass(torch.Tensor)
    
class ScaleAndPad(Transform):
    """Scale the larger side to frame size, then pad boundaries. Assumes T C H W"""
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def _transform(self, inpt: torch.Tensor, params: dict) -> torch.Tensor:
        return scale_and_pad(inpt, self.size)
    
def get_transform(
    num_frames: Optional[int] = None,
    frame_size: Optional[int] = None,
    norm_dict: Optional[NormDict] = None,
    frame_size_strategy: Optional[FrameSize_Strategy] = None,
    temporal_aug: Optional[TemporalStrategy] = None,
    auto_augment: Optional[AutoAugmentStrategy] = None
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Optional[List[int]], Optional[float]]:
    """Construct transform

    Args:
        num_frames (Optional[int], optional): number of frames. Defaults to None.
        frame_size (Optional[int], optional): Frame dimensions (square). Defaults to None.
        norm_dict (Optional[NormDict], optional): Normalisation values. Defaults to None.
        frame_size_strategy (Optional[FrameSize_Strategy], optional): How to get frames the right size. Defaults to None.
        temporal_aug (Optional[TemporalStrategy], optional): Temporal augmentation strategy. Defaults to None.
        auto_augment (Optional[AutoAugmentStrategy], optional): Spatial auto-augmentation strategy. Defaults to None.

    Raises:
        ValueError: If frame_size_strategy is specified without frame_size, or if an unexpected strategy is provided for frame sizing, temporal augmentation, or auto augmentation.
        ValueError: If auto_augment is not one of the expected strategies.

    Returns:
        Tuple[Callable[[torch.Tensor], torch.Tensor], Optional[List[int]], Optional[float]]: A tuple containing the composed transform, the permutation of frame indices if shuffling is applied, and the Shannon entropy of the shuffle. The permutation and entropy are None if no shuffling is applied.
    """
    
    transforms_list = []
    perm = None
    sh_e = None

    # --- 1. Temporal Augmentations (Operates on uint8 T, C, H, W) ---
    if temporal_aug in ["Shuffle", "Both"]:
        assert num_frames is not None, "num_frames must be specified if Shuffle is used"
        shuffle_t = Shuffle(num_frames)
        perm_tensor = shuffle_t.permutation
        transforms_list.append(shuffle_t)
        
        sh_e = Shuffle.shannon_entropy(perm_tensor)
        perm = list(map(int, perm_tensor.numpy()))

    if temporal_aug in ["Reverse", "Both"]:
        transforms_list.append(ReverseFrames(p=0.5))

    # --- 2. Enforce Temporal Consistency for Spatial Augs ---
    # AutoAugment expects uint8 tensors and consistent batching. 
    # Wrapping it natively forces the v2 API to treat the sequence as a single Video.
    # transforms_list.append(ToVideoTensor())

    # --- 3. Spatial Cropping / Flipping ---
    if frame_size is not None:
        assert frame_size_strategy is not None, f'Specify Frame sizing strategy, one of {FrameSize_Strategy}'
        
        if frame_size_strategy == "Random_crop":
            transforms_list.append(v2.RandomCrop(frame_size))
            transforms_list.append(v2.RandomHorizontalFlip()) #TODO: seperate out so can apply seperately
        elif frame_size_strategy == "Centre_crop":
            transforms_list.append(v2.CenterCrop(frame_size))
        elif frame_size_strategy == "Scale_and_pad":
            transforms_list.append(ScaleAndPad(frame_size))
        else:
            raise ValueError(f"Unexpected frame resize strategy: {frame_size_strategy}")
            
    # --- 4. Spatial AutoAugment ---
    if auto_augment is not None:
        policy_dict = {
            "IMAGENET": v2.AutoAugmentPolicy.IMAGENET,
            "CIFAR10": v2.AutoAugmentPolicy.CIFAR10,
            "SVHN": v2.AutoAugmentPolicy.SVHN
        }
        if auto_augment not in policy_dict:
            raise ValueError(f"auto_augment must be one of {list(policy_dict.keys())}")
        
        transforms_list.append(v2.AutoAugment(policy=policy_dict[auto_augment]))

    # --- 5. Return to Standard Tensors and Normalize ---
    # transforms_list.append(FromVideoTensor())     # Back to (T, C, H, W)
    transforms_list.append(v2.Lambda(_normalize_to_float))
    
    if norm_dict is not None:
        # v2.Normalize naturally broadcasts over the channel dimension (*, C, H, W)
        transforms_list.append(v2.Normalize(mean=norm_dict["mean"], std=norm_dict["std"]))
        
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
