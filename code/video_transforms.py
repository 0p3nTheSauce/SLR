import torch
from typing import Callable
import random
import utils
import time
import gc
from typing import Optional
import statistics
from torchvision.transforms.v2 import Transform
import torchvision.transforms.v2 as ts
import math

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


def correct_num_frames(frames, target_length=64, randomise=False):
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


class Shuffle(Transform):
    """Shuffle frames with a random seed each time"""

    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def _transform(self, inpt, params):
        dim = 0
        assert inpt.dim() == 4, "Input tensor must be 4D (T, C, H, W)"
        assert inpt.size(0) == len(self.permutation), (
            "Permutation length must match number of frames"
        )
        assert inpt.size(1) == 3, "Input tensor must have 3 channels (C=3)"
        return inpt.index_select(dim, self.permutation)

    @staticmethod
    def create_permutation(num_frames: int, seed: Optional[int] = None) -> torch.Tensor:
        """Create a random permutation for given number of frames"""
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randperm(num_frames)
    
    @staticmethod
    def shannon_entropy(perm: torch.Tensor) -> float:
        perml = list(map(int, perm.numpy()))
        
        p_len = len(perml)
        diffs = [0] * p_len  
        for i in range(p_len-1):
            diff = perml[i+1] - perml[i]
            if diff < 0:
                diff += p_len-1
            diffs[i] = diff

        diffs[p_len-1] = perml[0] - perml[p_len-1]
        
        hist = [0] * p_len
        
        for i in range(p_len):
            for d in diffs:
                if d == i:
                    hist[i] += 1
                    
        print(diffs)
        normed = [d / p_len for d in hist if d > 0]
        print(normed)
        e = 0
        for n in normed:
            e += -n * (math.log(n))
        print(f"E: {e}")
        return e

    @staticmethod
    def shannon_entropy2(perm: torch.Tensor) -> float:
        """
        Calculate the Shannon entropy of a permutation tensor.
        
        Args:
            perm: A tensor representing a permutation or probability distribution
            
        Returns:
            Shannon entropy value (in bits if using log2, nats if using log)
        """
        # Flatten the tensor to 1D if needed
        perm_flat = perm.flatten()
        
        
        # print(leng)
        
        # Normalize to get probability distribution (if not already normalized)
        probs = perm_flat / perm_flat.sum()
        
        # Remove zero probabilities to avoid log(0)
        probs = probs[probs > 0]
        
        # Calculate Shannon entropy: H = -sum(p * log(p))
        # Using log2 gives entropy in bits, log gives nats
        entropy = -torch.sum(probs * torch.log(probs))
        
        return entropy.item()

if __name__ == "__main__":
    vid = "./media/00333.mp4"
    frames = utils.load_rgb_frames_from_video(vid, 0, 0, all=True)
    print(frames.shape)
    utils.watch_video(frames, title="Original")

    # shuffled_indices = torch.randperm(frames.size(0))
    shuffled_indices = Shuffle.create_permutation(frames.size(0), seed=42)
    t = ts.Compose(
        [
            ts.Lambda(lambda x: x.permute(0, 2, 1, 3, 4)),  # T C H W -> C T H W
            Shuffle(shuffled_indices),
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
