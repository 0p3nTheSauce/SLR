from torch.utils.data import Dataset
import os
import json
import torch

#local imports
from utils import load_rgb_frames_from_video, crop_frames
from video_transforms import get_base, get_swap_ct,  correct_num_frames
import torchvision.transforms.v2 as transforms_v2

import numpy as np

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
  
  #resize the tensor 
  new_width = int(frames.shape[2] * scale_factor)
  new_height = int(frames.shape[3] * scale_factor)
  
  transform = transforms_v2.Resize((new_height, new_width))
  
  return transform(frames)
  

############################ Dataset Classes ############################

class VideoDataset(Dataset):
  def __init__(self, root, instances_path, classes_path,
               crop=False, num_frames=64, transforms=None, include_meta=False, resize=False):
    '''root is the path to the root directory where the video files are located.'''
    if os.path.exists(root) is False:
      raise FileNotFoundError(f"Root directory {root} does not exist.")
    else:
      self.root = root
    self.transforms = transforms
    self.crop = crop
    self.resize = resize
    self.num_frames = num_frames
    self.include_meta = include_meta
    with open(instances_path, 'r') as f:
      self.data = json.load(f) #created by preprocess.py
      if self.data is None:
        raise ValueError(f"No data found in {instances_path}. Please check the file.")
    with open(classes_path, 'r') as f:
      self.classes = json.load(f) 

  def __manual_load__(self,item):
    video_path = os.path.join(self.root,item['video_id']+'.mp4')
    if os.path.exists(video_path) is False:
      raise FileNotFoundError(f"Video file {video_path} does not exist.")
    
    frames = load_rgb_frames_from_video(video_path=video_path, start=item['frame_start'],
                                        end=item['frame_end'])
    sampled_frames = correct_num_frames(frames, self.num_frames) 

    if self.crop:
      sampled_frames = crop_frames(frames, item['bbox'])
  
    return sampled_frames.to(torch.uint8)
  
  def __getitem__(self, idx):
    item = self.data[idx]
    frames = self.__manual_load__(item)
    
    #in the WLASL paper, they first resize the frames so that 
    #the person bounding box diagnol length is 256 pixels
    #this douesnt work for us
    if self.resize:
      frames = resize_by_diag(frames, item['bbox'], 256)
    
    
    if self.transforms is not None:
      frames = self.transforms(frames)
      
    result = {
      'frames': frames,
      'label_num': item['label_num']
    }
    if self.include_meta:
      result.update(item)
    return result
  
  def __len__(self):
    return len(self.data)
    
  


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