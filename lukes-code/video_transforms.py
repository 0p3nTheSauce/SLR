import torch
import torchvision.transforms as ts
from typing import Callable
import torch.nn.functional as F
import random
import utils
from torchvision.transforms import v2
import time

import time
import torch
import gc
from typing import Callable, Any, Optional
import statistics

def bench_mark(data, transform, iters=100):
  start = time.perf_counter()
  for _ in range(iters):
    _ = transform(data)
  end = time.perf_counter()
  time_taken = end- start
  return f'''
    Num iters: {iters},
    Transform: {transform},
    total time: {time_taken:.4f} seconds,
    Average exec time: {(time_taken / iters):.4f} seconds
  '''

def bench_mark_enhanced(data, transform: Callable, iters: int = 100, 
                       warmup: int = 10, device: Optional[str] = None):
    """
    Enhanced benchmarking with warmup, memory tracking, and statistics
    """
    # Move data to specified device if provided
    if device and hasattr(data, 'to'):
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
    if device and device.startswith('cuda') and torch.cuda.is_available():
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
    if device and device.startswith('cuda') and torch.cuda.is_available():
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
    result = f'''
Benchmark Results:
=================
Transform: {transform}
Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}
Device: {device or 'CPU'}
Warmup iterations: {warmup}
Benchmark iterations: {iters}

Timing Statistics:
  Total time: {total_time:.4f} seconds
  Average time: {avg_time:.6f} seconds
  Median time: {median_time:.6f} seconds
  Std deviation: {std_time:.6f} seconds
  Min time: {min_time:.6f} seconds
  Max time: {max_time:.6f} seconds
  
Throughput: {iters/total_time:.2f} transforms/second'''

    if memory_before is not None and memory_after is not None:
        memory_used = memory_after - memory_before
        result += f'''
        
Memory Usage:
  Before: {memory_before / 1024**2:.2f} MB
  After: {memory_after / 1024**2:.2f} MB
  Used: {memory_used / 1024**2:.2f} MB'''
    
    return result


class RandomColourSwap: #actually fast, but also perhaps available through built in methods
  def __init__(self, p=0.5, swap_type='random'):
    self.p = p
    self.swap_type = swap_type
  def __call__(self, frames):
    #expecting data in T C H W Tensor
    if random.random() < self.p:
      frames = frames.permute(1, 0, 2, 3) #will be easier if C T H W
      
      if self.swap_type == 'random':
        og_channels = [frames[i].clone() for i in range(3)]
        for i in range(3):
          frames[i] = random.choice(og_channels)
      
      elif self.swap_type == 'permute': #tbh doesn't do much
        idxs = [0,1,2]
        random.shuffle(idxs)
        frames = frames[idxs]
      
      else:
        raise ValueError(f'swap_type can only be random or permute,\
          not {self.swap_type}')  
        
      frames = frames.permute(1, 0, 2, 3)
    return frames
  def __repr__(self):
    return f"{self.__class__.__name__}(p={self.p})"


def get_base(numframes=16, size=(112, 112)) -> \
    Callable[[torch.Tensor], torch.Tensor]:
  '''Args:
      x : torch.Tensor (T C H W) RGB torch.uint8
    Return:
      base_transform : x -> torch.Tensor (T C H W) RGB torch.float [0, 1]'''
  return ts.Compose([
    ts.Lambda(lambda x: correct_num_frames(x, numframes)),
    ts.Lambda(lambda x: x.float() / 255.0),
    ts.Lambda(lambda x: F.interpolate(x, size=size, 
                    mode='bilinear', align_corners=False)),
  ])

def get_norm(mean,std) -> \
    Callable[[torch.Tensor], torch.Tensor]:
  return lambda x: normalise(x, mean, std)

def get_swap_ct() -> Callable[[torch.Tensor], torch.Tensor]:
  return lambda x: x.permute(1,0,2,3)

def get_rand_norm_aug(base_mean, base_std, mean_var=0.05, std_var=0.03,
    prob=0.5) -> Callable[[torch.Tensor], torch.Tensor]:
  return lambda x: RandomNormalizationAugmentation(x,
    base_mean=base_mean,
    base_std=base_std,
    mean_var=mean_var,
    std_var=std_var,
    prob=prob,                                 
  )
  
def get_adapt_norm(base_mean, base_std, adaption_strength=0.3, prob=0.5) -> \
    Callable[[torch.Tensor], torch.Tensor]:
  return lambda x: AdaptiveNormalisation(x,
    base_mean=base_mean,
    base_std=base_std,
    adaption_strength=adaption_strength,
    prob=prob                                       
  )

def get_base_norm(mean, std, numframes=16, size=(112, 112), norm_func=None)-> \
    Callable[[torch.Tensor], torch.Tensor]:
  '''Args:
      x : torch.Tensor (T C H W) RGB torch.uint8
    Return:
      base_transform : x -> normalised torch.Tensor (T C H W) RGB torch.float'''
  if norm_func is None:
    norm_func = lambda x: normalise(x, mean=mean, std=std)
  base_transform = get_base(numframes=numframes, size=size)
  return ts.Compose([
    base_transform,
    ts.Lambda(norm_func)
  ])




def RandomNormalizationAugmentation(frames, base_mean, base_std,
                                    mean_var=0.1, std_var=0.1,
                                    prob=0.5):
  """
  Randomly adjusts normalization parameters as augmentation
  Args:
    base_mean: Original normalization mean [R, G, B]
    base_std: Original normalization std [R, G, B]
    mean_var: Variance for mean adjustment
    std_var: Variance for std adjustment
    prob: Probability of applying augmentation
  """
  if random.random() > prob:
    return normalise(frames, base_mean, base_std)
  
  base_mean = torch.Tensor(base_mean)
  base_std = torch.Tensor(base_std)
  
  # Randomly adjust mean and std
  mean_noise = torch.randn(3) * mean_var
  std_noise = torch.randn(3) * std_var
  
  new_mean = base_mean + mean_noise
  new_std = torch.clamp(base_std + std_noise, min=0.01)  # Prevent zero std
  
  return normalise(frames, new_mean, new_std)

def AdaptiveNormalisation(frames, base_mean, base_std, adaption_strength=0.3, prob=0.5):
  '''Adapts normalization based on video statistics'''
  base_mean = torch.Tensor(base_mean)
  base_std = torch.Tensor(base_std)
  if random.random() > prob:
    return normalise(frames, base_mean, base_std)
  video_mean = frames.mean(dim=[0,2,3])
  video_std = frames.std(dim=[0,2,3])
  adpt_mean = (1- adaption_strength) * base_mean + \
    adaption_strength * video_mean
  adpt_std = (1-adaption_strength) * base_std + \
    adaption_strength * video_std
  return normalise(frames, adpt_mean, adpt_std)

def ColorJitterNormalisation(frames, base_mean, base_std, brightness=0.2, contrast=0.2,
                             saturation=0.2, hue=0.1, prob=0.5):
  '''Combines color jittering with normalization for more aggressive augmentation'''
  
  if random.random() > prob:
    return normalise(frames, base_mean, base_std)
  frames.permute(0,1,2,3)  # to (C, T, H, W)
  
  # Apply color transformations
  if brightness > 0:
    brightness_factor = random.uniform(1-brightness, 1+brightness)
    frames = frames * brightness_factor
  
  if contrast > 0:
    contrast_factor = random.uniform(1-contrast, 1+contrast)
    mean = frames.mean(dim=[2, 3], keepdim=True)
    frames = (frames - mean) * contrast_factor + mean
  
  if saturation > 0:
    saturation_factor = random.uniform(1-saturation, 1+saturation)
    gray = frames.mean(dim=0, keepdim=True)  # Average across RGB
    frames = gray + (frames - gray) * saturation_factor
  
  frames = torch.clamp(frames, 0, 1)
  return normalise(frames, base_mean, base_std)

def sample(frames, target_length,randomise=False):
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
  '''Corrects the number of frames to match the target length.
  Args:
    frames (torch.Tensor): The input frames tensor. (T x C x H x W)
    target_length (int): The target length for the number of frames.
  Returns:
    torch.Tensor: The corrected frames tensor with the specified target length.
  '''
  if frames is None or frames.shape[0] == 0:
    raise ValueError("Input frames tensor is empty or None.")
  if target_length <= 0:
    raise ValueError("Target length must be a positive integer.")
  if frames.shape[0] == target_length:
    return frames
  if frames.shape[0] < target_length:
    # Pad with zeros if the number of frames is less than the target length
    padding = torch.zeros(target_length - frames.shape[0], frames.shape[1], frames.shape[2], frames.shape[3], device=frames.device)
    return torch.cat((frames, padding), dim=0)
  else:
    step = frames.shape[0] // target_length
    sampled_frames = sample(frames, target_length, randomise=randomise)
    diff = target_length - len(sampled_frames) 
    if diff > 0:
      padding = torch.zeros(diff, frames.shape[1], frames.shape[2], frames.shape[3], device=frames.device)
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
    padding = torch.zeros(target_length - num_frames, frames.shape[1], frames.shape[2], frames.shape[3], device=frames.device)
    return torch.cat((frames, padding), dim=0)
  else:
    # Trim the frames if the number of frames is greater than the target length
    return frames[:target_length, :, :, :]    
    
def normalise(frames, mean, std): #thanks cluade
    """
    Args: 
        frames: (T, C, H, W) tensor
        mean: list or tensor [R, G, B]  
        std: list or tensor [R, G, B]
    """
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=frames.dtype, device=frames.device)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=frames.dtype, device=frames.device)
    
    # Reshape for broadcasting
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)
    
    # In-place normalization (more memory efficient)
    frames.sub_(mean).div_(std)
    return frames

def colour_jitter(frames, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
  '''Applies torchvision colour jitter transform to 4D tensor'''
  jitter = ts.ColorJitter(
    brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
  return torch.stack([jitter(frame) for frame in frames], dim=0)

def min_transform_rI3d(frames):
  '''Prepares videos for rI3d'''
  return F.interpolate(
    correct_num_frames(frames) / 255.0,
    size=(244,244),
    mode='bilinear').permute(1,0,2,3) #r3id expects (C, T, H, W)




if __name__ =='__main__':
  vid = './media/00333.mp4'
  frames = utils.load_rgb_frames_from_video(vid, 0, 0, all=True)
  utils.watch_video(frames, title='Unswapped')
  # t = RandomColourSwap(swap_type='random')
  # # swapped_frames = t(frames)
  # # utils.watch_video(frames, title='Swapped')
  # t2 = v2.RandomChannelPermutation()
  # permed_frames = t2(frames)
  # utils.watch_video(frames)
  # # print(bench_mark(frames, t))
  # # print(bench_mark(frames, t2))
  
  
  # t3 = v2.RandomPhotometricDistort()
  # warped_frames = t3(frames)
  # utils.watch_video(warped_frames)
  b = (0.4, 2)
  c = (0.4, 2)
  s = (0.4, 2)
  h = (0.5, 0.5)
  t4 = v2.ColorJitter(hue=h)
  f4 = t4(frames)
  utils.watch_video(f4)