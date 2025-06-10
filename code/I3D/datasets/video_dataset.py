import torch
from torch.utils.data import Dataset
import os
from torchcodec.decoders import VideoDecoder
import json
from torch.utils.data import Dataset
import cv2
import torch.nn.functional as F


def load_rgb_frames_from_video(root, vid, start, num):
  video_path = os.path.join(root,vid+'.mp4')
  # device = "cuda" if torch.cuda.is_available() else "cpu"
  device = "cpu"
  decoder = VideoDecoder(video_path, device=device)
  frames = decoder.get_frames_in_range(start, start+num)
  frames = frames.data
  #has shape: TCHW
  #need shape CTHW
  # print(frames.shape)
  frames =  frames.permute(1, 0, 2, 3)  # Convert to CTHW format
  frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)  # (C, T, H, W)
  return frames
  
def get_num_class(split_file):
    classes = set()

    content = json.load(open(split_file))

    for vid in content.keys():
        class_id = content[vid]['action'][0]
        classes.add(class_id)

    return len(classes)

def make_dataset(split_file, split,root, num_classes):
  dataset = []
  with open(split_file, 'r') as f:
    data = json.load(f)
  for vid in data.keys():
    if data[vid]['subset'] != split:
      continue
    video_path = os.path.join(root, vid + '.mp4')
    num_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    if not os.path.exists(video_path):
      continue
    dataset.append((vid, data[vid]['action'][0], 0, num_frames, "{}".format(vid)))#vid, label, start_f, start_e, output_name
  return dataset

class WLASL_dataset(Dataset):
  def __init__(self, split_file, split, root, mode, transforms=None):
    self.num_classes = get_num_class(split_file)  
    self.data = make_dataset(split_file, split, root, self.num_classes)
    self.split_file = split_file
    self.transforms = transforms
    self.mode = mode
    self.root = root
    
  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    vid, label, start_f, start_e, output_name = self.data[index]

    if self.mode == 'rgb':
      frames = load_rgb_frames_from_video(self.root, vid, start_f, start_e)
      
      if self.transforms:
        frames = self.transforms(frames)
      return frames, label, vid
    else:
      raise NotImplementedError("Only RGB mode is implemented.")
    
  def __len__(self):
    return len(self.data)