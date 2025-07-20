import torch
from torch.utils.data import Dataset
import os
import json

#local imports
from utils import load_rgb_frames_from_video, crop_frames
        
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