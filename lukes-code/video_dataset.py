from torch.utils.data import Dataset
import os
import json
import torch

#local imports
from utils import load_rgb_frames_from_video, crop_frames
        

############################ Dataset Classes ############################

class VideoDataset(Dataset):
  def __init__(self, root, instances_path, classes_path,crop=True, transform=None):
    '''root is the path to the root directory where the video files are located.'''
    if os.path.exists(root) is False:
      raise FileNotFoundError(f"Root directory {root} does not exist.")
    else:
      self.root = root
    self.transform = transform
    self.crop = crop
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

    if self.crop:
      frames = crop_frames(frames, item['bbox'])
  
    return frames, item['label_num']
  
  def __getitem__(self, idx):
    item = self.data[idx]
    frames, target= self.__manual_load__(item)
    if self.transform is not None:
      frames = self.transform(frames)
    return frames, target
  
  def __len__(self):
    return len(self.data)
  #TODO: Make a ca
  
class ContrastiveVideoDataset(VideoDataset):
  def __init__(self, *args, transform1=None, transform2=None, **kwargs):
    # Remove transform from kwargs to prevent parent from using it
    kwargs.pop('transform', None)
    super().__init__(*args, **kwargs)
    self.transform1 = transform1
    self.transform2 = transform2

  def __getitem__(self, idx):
    item = self.data[idx]
    frames, _ = self.__manual_load__(item)  # Get raw frames, ignore label
    
    # Apply two different augmentations
    view1 = self.transform1(frames) if self.transform1 else frames
    view2 = self.transform2(frames) if self.transform2 else frames
    
    return (view1, view2)


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