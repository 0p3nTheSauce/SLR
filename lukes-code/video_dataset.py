import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import json
import random

#local imports
from utils import load_rgb_frames_from_video 

########################## Transforms ##########################

def crop_frames(frames, bbox):
  #frames hase shape (num_frames, channels, height, width)
  #bbox is a list of [x1, y1, x2, y2]
  x1, y1, x2, y2 = bbox
  return frames[:, :, y1:y2, x1:x2]  # Crop the frames using the bounding box

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
    
def normalise(frames, mean, std):
  '''Applies torch vision transform to 4D tensor ''' 
  return torch.stack([
    transforms.Normalize(mean=mean, std=std)(frame) for frame in frames
  ], dim=0)

def colour_jitter(frames, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
  '''Applies torchvision colour jitter transform to 4D tensor'''
  jitter = transforms.ColorJitter(
    brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
  return torch.stack([jitter(frame) for frame in frames], dim=0)

def min_transform_rI3d(frames):
  '''Prepares videos for rI3d'''
  return F.interpolate(
    correct_num_frames(frames) / 255.0,
    size=(244,244),
    mode='bilinear').permute(1,0,2,3) #r3id expects (C, T, H, W)
        
############################ Dataset Class ############################

class VideoDataset(Dataset):
  def __init__(self, root, instances_path, classes_path,crop=True, transform=None, preprocess_strat="off", cache_name='data_cache'):
    '''root is the path to the root directory where the video files are located.'''
    if os.path.exists(root) is False:
      raise FileNotFoundError(f"Root directory {root} does not exist.")
    else:
      self.root = root
    self.cache = os.path.join(self.root,cache_name)
    # self.split = split # this might not do anything
    self.transform = transform
    self.crop = crop
    with open(instances_path, 'r') as f:
      self.data = json.load(f) #created by preprocess_info
      if self.data is None:
        raise ValueError(f"No data found in {instances_path}. Please check the file.")
    with open(classes_path, 'r') as f:
      self.classes = json.load(f) 
    if preprocess_strat == "on":
      self.load_func = self.__load_preprocessed__
    elif preprocess_strat == "off":
      self.load_func = self.__manual_load__
    
  def __preprocess__(self):
    # This method can be used to preprocess the data if needed
    if not os.path.exists(self.cache):
      os.makedirs(self.cache)
    for item in self.data:
      video_id = item['video_id']
      # label_num, video_id = item['label_num'], item['video_id']
      # frame_start, frame_end = item['frame_start'], item['frame_end']
      # bbox = item['bbox']
      fname = os.path.join(self.cache, f"{video_id}.pt")
      if os.path.exists(fname):
        continue
      # torch.save(self.__manual_load__(item), fname)  
      frames, label = self.__manual_load__(item)
      torch.save({"frames" : frames, "label_num" : label}, fname)
      
  
  def __load_preprocessed__(self,item):
    info =  torch.load(os.path.join(self.cache, f"{item['video_id']}.pt"))
    return info['frames'], info['label_num']
  
  
  def __manual_load__(self,item):
    video_path = os.path.join(self.root,item['video_id']+'.mp4')
    if os.path.exists(video_path) is False:
      raise FileNotFoundError(f"Video file {video_path} does not exist.")
    
    frames = load_rgb_frames_from_video(video_path=video_path, start=item['frame_start'],
                                        end=item['frame_end']) 
    # frames = load_rgb_frames_from_video_ioversion(video_path=video_path, start=item['frame_start'],
    #                                     end=item['frame_end']) 
    if self.crop:
      frames = crop_frames(frames, item['bbox'])
    
    if self.transform:
      frames = self.transform(frames)
    # return {"frames" : frames, "label_num" : item['label_num']}
    return frames, item['label_num']
  
  def __getitem__(self, idx):
    item = self.data[idx]
    return self.load_func(item)
  
  def __len__(self):
    return len(self.data)
  #TODO: Make a ca
  
if __name__ == "__main__":

  # test_crop()
  # prep_train() #--run to preprocess the training data
  # prep_test()  #--run to preprocess the test data
  # prep_val() #--run to preprocess the validation data
  # fix_bad_frame_range("./preprocessed_labels/asl100/train_instances.json",
  #                     "../data/WLASL2000/") #--run to fix bad frame ranges in the training instances
  # fix_bad_frame_range("./preprocessed_labels/asl100/test_instances.json",
  #                   "../data/WLASL2000/") #--run to fix bad frame ranges in the test instances
  # fix_bad_frame_range("./preprocessed_labels/asl100/val_instances.json",
  #                   "../data/WLASL2000/") #--run to fix bad frame ranges in the validation instances
  # fix_bad_bboxes("./preprocessed_labels/asl100/train_instances.json",
  #               "../data/WLASL2000/", output='./output') #--run to fix bad bounding boxes in the training instances
  # fix_bad_bboxes("./preprocessed_labels/asl100/test_instances.json",
  #               "../data/WLASL2000/", output='./output')
  # fix_bad_bboxes("./preprocessed_labels/asl100/val_instances.json",
  #               "../data/WLASL2000/", output='./output')
  # remove_short_samples('./output/train_instances_fixed_bboxes.json',
  #                      output='./preprocessed_labels/asl100')
  # remove_short_samples('./output/test_instances_fixed_bboxes.json',
  #                      output='./preprocessed_labels/asl100')
  # remove_short_samples('./output/val_instances_fixed_bboxes.json',
  #                      output='./preprocessed_labels/asl100')
  pass