from torch.utils.data import Dataset
import os
import json
import torch

#local imports
from utils import load_rgb_frames_from_video, crop_frames
from video_transforms import get_base, get_swap_ct,  correct_num_frames
import torchvision.transforms as ts

############################ Dataset Classes ############################

class VideoDataset(Dataset):
  def __init__(self, root, instances_path, classes_path,
               crop=False, num_frames=64, transforms=None, include_meta=False):
    '''root is the path to the root directory where the video files are located.'''
    if os.path.exists(root) is False:
      raise FileNotFoundError(f"Root directory {root} does not exist.")
    else:
      self.root = root
    self.transforms = transforms
    self.crop = crop
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
  
    return sampled_frames
  
  def __getitem__(self, idx):
    item = self.data[idx]
    frames = self.__manual_load__(item)
    if self.include_meta:
      return frames, item
    else:
      return frames, item['label_num']
  
  def __len__(self):
    return len(self.data)
    
  
class ContrastiveVideoDataset(VideoDataset):
  def __init__(self, *args, augmentation, **kwargs):
    # Remove transform from kwargs to prevent parent from using it
    super().__init__(*args, **kwargs)
    #Assume that self.transform will be the base_norm_fin, ie the standard
    #But transform should atleast have base_transform, otherwise shape issues when 
    # loading
    #By extension, augmementation must have at least base_tansform
    #But may just be that (augmentation through lack of normalisation)
    self.augmentation = augmentation

  def __getitem__(self, idx):
    item = self.data[idx]
    frames, _ = self.__manual_load__(item)  # Get raw frames, ignore label
    
    # Apply two different augmentations
    view1 = self.transforms(frames) if self.transforms else frames
    view2 = self.augmentation(frames) if self.augmentation else frames
    #leaving like this for now, but likely to cause errors if same base_transform is not used
    
    
    return (view1, view2)

class SemiContrastiveVideoDataset(VideoDataset):
  def __init__(self, *args, augmentation, **kwargs):
    # Remove transform from kwargs to prevent parent from using it
    super().__init__(*args, **kwargs)
    #Assume that self.transform will be the base_norm_fin, ie the standard
    #But transform should atleast have base_transform, otherwise shape issues when 
    # loading
    #By extension, augmementation must have at least base_tansform
    #But may just be that (augmentation through lack of normalisation)
    self.augmentation = augmentation

  def __getitem__(self, idx):
    item = self.data[idx]
    frames, label = self.__manual_load__(item)  # Get raw frames, ignore label
    
    # Apply two different augmentations
    view1 = self.transforms(frames) if self.transforms else frames
    view2 = self.augmentation(frames) if self.augmentation else frames
    #leaving like this for now, but likely to cause errors if same base_transform is not used
    
    
    return ((view1, view2), label)

def contrastive_collate_fn(batch):
  """Custom collate function for contrastive learning dataset"""
  view1_list = []
  view2_list = []
  
  for view1, view2 in batch:
    # Ensure tensors are detached and on CPU for collation
    if hasattr(view1, 'detach'):
      view1 = view1.detach().cpu()
    if hasattr(view2, 'detach'):
      view2 = view2.detach().cpu()
        
    view1_list.append(view1)
    view2_list.append(view2)
  
  try:
    # Stack the views
    view1_batch = torch.stack(view1_list)
    view2_batch = torch.stack(view2_list)
  except RuntimeError as e:
    print(f"Error stacking tensors: {e}")
    print(f"View1 shapes: {[v.shape for v in view1_list[:3]]}")  # Print first 3 shapes
    print(f"View2 shapes: {[v.shape for v in view2_list[:3]]}")
    raise
  
  return view1_batch, view2_batch

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