import torch
from torch.utils.data import Dataset
import os
from torchcodec.decoders import VideoDecoder
import json
from torch.utils.data import Dataset
import cv2

def load_rgb_frames_from_video(root, vid, start, end):
  video_path = os.path.join(root,vid+'.mp4')
  device = "cuda" if torch.cuda.is_available() else "cpu"
  decoder = VideoDecoder(video_path, device=device)
  return decoder.get_frames_in_range(start, end).data

def crop_frames(frames, bbox):
  return frames[:, :, bbox[1]:bbox[3], bbox[0]:bbox[2]]
   

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
          'bbox': inst['bbox'],
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
    self.root = root
    self.cache = os.path.join(self.root,cache_name)
    self.split = split
    self.transform = transform
    with open(instances_path, 'r') as f:
      self.data = json.load(f) #created by preprocess_info
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
      label_num, video_id = item['label_num'], item['video_id']
      frame_start, frame_end = item['frame_start'], item['frame_end']
      bbox = item['bbox']
      fname = os.path.join(self.cache, f"{video_id}.pt")
      if os.path.exists(fname):
        continue
      torch.save(self.__manual_load__(item), fname)  
  
  def __load_preprocessed__(self,item):
    return torch.load(os.path.join(self.cache, f"{item['video_id']}.pt"))

  def __manual_load__(self,item):
    frames = crop_frames(
      load_rgb_frames_from_video(self.root, item['video_id'], item['frame_start'],
                                 item['frame_end'])
      , item['bbox']) 
    if self.transform:
      frames = self.transform(frames)
    return {"frames" : frames, "label_num" : item['label_num']}
  
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
  data = load_rgb_frames_from_video('.', video_path, start, num)
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
  video_path = 'video'
  start = 0
  num = 10
  out = 'output'
  if not os.path.exists(out):
    os.makedirs(out)
  data = load_rgb_frames_from_video('.', video_path, start, num)
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
  bbox = [137,16,492,480]
  cropped_data = crop_frames(data, bbox)
  for i, cropped_frame in enumerate(cropped_data):
    img = cropped_frame.permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(f"{out}/cropped_frame_{i:04d}.jpg", img)
    
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
  prep_train()
  prep_test()
  prep_val()