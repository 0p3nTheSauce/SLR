import torch
from torch.utils.data import Dataset
import os
from torchcodec.decoders import VideoDecoder
import json
from torch.utils.data import Dataset
import cv2

import random

def load_rgb_frames_from_video(video_path, start, end):
  '''Loads RGB frames from a video file.
  Args:
    root (str): The root directory where the video file is located.
    vid (str): The video file name (without extension).
    start (int): The starting frame index (inclusive).
    end (int): The ending frame index (exclusive).
  Returns:
    torch.Tensor: A tensor containing the RGB frames in the shape (num_frames, channels,
  '''
  
  device = "cuda" if torch.cuda.is_available() else "cpu"
  decoder = VideoDecoder(video_path, device=device)
  num_frames = decoder._num_frames
  if start < 0 or end > num_frames or end <= start:
    # raise ValueError(f"Invalid frame range: start={start}, end={end}, num_frames={num_frames}")
    print(f"Invalid frame range: start={start}, end={end}, num_frames={num_frames}. Adjusting to valid range.")
    print(f"Using start=0 and end={num_frames}")
    start = 0
    end = num_frames
  return decoder.get_frames_in_range(start, end).data

def crop_frames(frames, bbox):
  #frames hase shape (num_frames, channels, height, width)
  #bbox is a list of [x1, y1, x2, y2]
  x1, y1, x2, y2 = bbox
  return frames[:, :, y1:y2, x1:x2]  # Crop the frames using the bounding box


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

def get_split(lst_gloss_dicts, split):
  mod_instances = []
  class_names = []
  for i, gloss_dict in enumerate(lst_gloss_dicts):
    label_text = gloss_dict['gloss']
    instances = gloss_dict['instances']
    for inst in instances:
      if inst['split'] == split:
        mod_instances.append({
          'label_num': i,
          # 'bbox': inst['bbox'],             bad data, do not use
          'frame_end': inst['frame_end'],
          'frame_start': inst['frame_start'],
          'video_id': inst['video_id'],
        }) 
        class_names.append(label_text) 
  return mod_instances, class_names

def preprocess_info(json_path, split, output_path):
  with open(json_path, 'r') as f:
    asl_num = json.load(f)
    instances, classes = get_split(asl_num, split)
  base_name = os.path.basename(json_path).replace('.json', '')
  output_path = os.path.join(output_path, base_name.replace('.json', ''))
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  with open(os.path.join(output_path, f'{split}_instances.json'), 'w') as f:
    json.dump(instances, f, indent=4)
  with open(os.path.join(output_path, f'{split}_classes.json'), 'w') as f:
    json.dump(classes, f, indent=4)
    


class VideoDataset(Dataset):
  def __init__(self, root, split, instances_path, classes_path, transform=None, preprocess_strat="off", cache_name='data_cache'):
    '''root is the path to the root directory where the video files are located.'''
    if os.path.exists(root) is False:
      raise FileNotFoundError(f"Root directory {root} does not exist.")
    else:
      self.root = root
    self.cache = os.path.join(self.root,cache_name)
    self.split = split
    self.transform = transform
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
    # frames = crop_frames(
    #   load_rgb_frames_from_video(self.root, item['video_id'], item['frame_start'],
    #                              item['frame_end'])
    #   , item['bbox'])
    video_path = os.path.join(self.root,item['video_id']+'.mp4')
    if os.path.exists(video_path) is False:
      raise FileNotFoundError(f"Video file {video_path} does not exist.")
    
    frames = load_rgb_frames_from_video(video_path=video_path, start=item['frame_start'],
                                        end=item['frame_end']) 
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
###################################################################################################################################################################  

def test_video():
  video_path = 'video'
  start = 0
  num = 10
  out = 'output'
  if not os.path.exists(out):
    os.makedirs(out)
  data = load_rgb_frames_from_video(video_path, start, num)
  #type 
  
  print()
  print(data.shape)
  print()
  lent = data.shape[0]
  print("Number of frames:", lent)
  # for i in range(lent):
  #   img = data[i]
  #   cv2.imshow('Frame', img.permute(1, 2, 0).cpu().numpy())
  #   cv2.waitKey(100)  # Display each frame for 100 ms
  # cv2.destroyAllWindows()
  # img = data[0]
  # disp_image(img)
  for i in range(lent-1, -1, -1):
    img = data[i].permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(f"{out}/frame_{i:04d}.jpg", img)
  print("Image displayed successfully.")  
  
def test_crop():
  output='./output/'
  
  root = '../data/WLASL2000/'
  info = './preprocessed_labels/asl100/train_instances.json'
  with open(info, 'r') as f:
    items = json.load(f)
  rand_idx = random.randint(0, len(items) - 1)
  item = items[rand_idx]  # Get the first item for testing
  path = os.path.join(root, item['video_id'] + '.mp4')
  frames = load_rgb_frames_from_video(path, item['frame_start'],
                               item['frame_end'])
  
  
  show_bbox(frames, item['bbox'])
  
  corrected_bbox = correct_bbox(item['bbox'], frames.shape)
  print("Corrected bounding box coordinates:", corrected_bbox)
  show_bbox(frames, corrected_bbox)
  
  cropped_frames = crop_frames(frames, corrected_bbox)
  frames = cropped_frames
  
  print("Original frames shape:", frames.shape)
  print("Cropped frames shape:", cropped_frames.shape)
  
  
  frames = frames.permute(0, 2, 3, 1).cpu().numpy()  # Change to (num_frames, height, width, channels)
  # display(frames, output)  # Display or save the cropped frames
  display(frames)

def show_bbox(frames, bbox):
  #frames has shape (num_frames, channels, height, width)
  #bbox is a list of [x1, y1, x2, y2]
  x1, y1, x2, y2 = bbox
  print("Bounding box coordinates:", bbox)
  for i in range(frames.shape[0]):
    frame = frames[i].permute(1, 2, 0).cpu().numpy()  # Change to (height, width, channels)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw the bounding box
    cv2.imshow('Frame with Bounding Box', frame)
    cv2.waitKey(0)  # Display each frame for 100 ms
  cv2.destroyAllWindows()
  
def correct_bbox(bbox, frame_shape):
  # bbox is a list of [x1, y1, x2, y2]
  # on a hunch, the boundign box seems shifted by:
  # 0.5 * width (of bbox) to the right
  x1, y1, x2, y2 = bbox
  width_bbox = x2 - x1
  width_frame = frame_shape[2]  # Assuming frame_shape is (num_frames, channels, height, width)
  x1 = int(max(0, x1 - 0.5 * width_bbox))
  x2 = int(min(width_frame, x2 - 0.5 * width_bbox))
  return [x1, y1, x2, y2]
  
def display(frames,output=None):
  if output is None:
    for i, frame in enumerate(frames):
      cv2.imshow('Cropped Frames', frame)  # Display the first frame
      cv2.waitKey(100)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
  else:
    if not os.path.exists(output):
      os.makedirs(output)
    for i, img in enumerate(frames):
      cv2.imwrite(f"{output}/frame_{i:04d}.jpg", img)  # Save each frame as an image
    print(f"Cropped frames saved to {output}.")
  


def prep_train():
  json_path = '../data/splits/asl100.json'
  split = 'train'
  output_root = './preprocessed_labels/'
  if not os.path.exists(output_root):
    os.makedirs(output_root)
  preprocess_info(json_path, split, output_root)
  
def prep_test():
  json_path = '../data/splits/asl100.json'
  split = 'test'
  output_root = './preprocessed_labels/'
  if not os.path.exists(output_root):
    os.makedirs(output_root)
  preprocess_info(json_path, split, output_root)
  
def prep_val():
  json_path = '../data/splits/asl100.json'
  split = 'val'
  output_root = './preprocessed_labels/'
  if not os.path.exists(output_root):
    os.makedirs(output_root)
  preprocess_info(json_path, split, output_root)
  
if __name__ == "__main__":
  # test_video()
  # test_crop()
  # prep_train()
  # prep_test()
  # prep_val()
  pass