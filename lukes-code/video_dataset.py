import torch
from torch.utils.data import Dataset
import os
from torchcodec.decoders import VideoDecoder
import json
from torch.utils.data import Dataset
import cv2

import random
import tqdm
import matplotlib.pyplot as plt
import numpy

from ultralytics import YOLO
def load_rgb_frames_from_video(video_path, start, end, device='cpu',all=False):
  '''Loads RGB frames from a video file.
  Args:
    root (str): The root directory where the video file is located.
    vid (str): The video file name (without extension).
    start (int): The starting frame index (inclusive).
    end (int): The ending frame index (exclusive).
    device (string): cpu or cuda. 
    all (bool): All frames are passed
  Returns:
    torch.Tensor: A tensor containing the RGB frames in the shape:
    (num_frames, channels, height, width), where channels=3 (RGB).
  '''
  if device =='cuda' and not torch.cuda.is_available():
    device = 'cpu'
    print("Warning: cuda not available so using cpu")
  decoder = VideoDecoder(video_path, device=device)
  num_frames = decoder._num_frames
  if all:
    start = 0
    end = num_frames
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

def correct_num_frames(frames, target_length=64):
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
    #uniformly sample frames
    step = frames.shape[0] // target_length
    sampled_frames = frames[::step]
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
    
def visualise_frames(frames,num):
  # permute and convert to numpy 
  '''Args:
    frames : torch.Tensor (T, C, H, W)
    num : int, to be visualised'''
  if num < 1:
    raise ValueError("num must be >= 1")
  num_frames = len(frames)
  if num_frames <= num:
    step = 1
  else:
    step = num_frames // num
  for frame in frames[::step]:
    np_frame = frame.permute(1,2,0).cpu().numpy()
    plt.imshow(np_frame)
    plt.axis('off')
    plt.show()
  
class VideoDataset(Dataset):
  def __init__(self, root, split, instances_path, classes_path,crop=True, transform=None, device='cpu', preprocess_strat="off", cache_name='data_cache'):
    '''root is the path to the root directory where the video files are located.'''
    if os.path.exists(root) is False:
      raise FileNotFoundError(f"Root directory {root} does not exist.")
    else:
      self.root = root
    self.cache = os.path.join(self.root,cache_name)
    self.split = split # this might not do anything
    self.transform = transform
    self.crop = crop
    self.device = device #use cpu if num_workers > 0 in Dataloader
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
                                        end=item['frame_end'], device=self.device) 
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
  
def fix_bad_frame_range(instance_path, raw_path, log='./output/bad_frames.txt'):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  bad_frames = []
  with open(instance_path, 'r') as f:
    instances = json.load(f)
  for instance in tqdm.tqdm(instances, desc="Fixing frame ranges"):
    vid_path = os.path.join(raw_path, instance['video_id'] + '.mp4')
    decoder = VideoDecoder(vid_path, device=device)
    num_frames = decoder._num_frames
    start = instance['frame_start']
    end = instance['frame_end']
    if start < 0 or start >= num_frames:
      bad_frames.append(f"Invalid start frame {start} for video {instance['video_id']}. Setting to 0." +
                        f" Total frames: {num_frames}")
      start = 0
    if end <= start or end > num_frames:
      bad_frames.append(f"Invalid end frame {end} for video {instance['video_id']}. Setting to {num_frames}." +
                        f" Total frames: {num_frames}")
      end = num_frames
    instance['frame_start'] = start
    instance['frame_end'] = end
  with open(instance_path, 'w') as f:
    json.dump(instances, f, indent=4)
  if bad_frames:
    with open(log, 'a') as log_file:
      for line in bad_frames:
        log_file.write(line + '\n')
    print(f"Bad frame ranges logged to {log}.")
    print(f"Updated instances in {instance_path} with valid frame ranges.")
  else:
    print("No bad frame ranges found. No changes made to the instance file.")
    
def get_largest_bbox(bboxes):
  if not bboxes:
    return None
  x_min, y_min, x_max, y_max = bboxes[0]
  for box in bboxes:
    x1, y1, x2, y2 = box
    if x1 < x_min:
      x_min = x1
    if y1 < y_min:
      y_min = y1
    if x2 > x_max:
      x_max = x2
    if y2 > y_max:
      y_max = y2
  return [x_min, y_min, x_max, y_max]

def fix_bad_bboxes(instance_path, raw_path, output='./output'):
  model = YOLO('yolov8n.pt')  # Load a pre-trained YOLO model
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # model.to(device)
  new_instences = []
  
  bad_bboxes = []
  with open(instance_path, 'r') as f:
    instances = json.load(f)
  for instance in tqdm.tqdm(instances, desc="Fixing bounding boxes"):
    vid_path = os.path.join(raw_path, instance['video_id'] + '.mp4')
    frames = load_rgb_frames_from_video(vid_path, instance['frame_start'], instance['frame_end'], all=True)
    frames = frames.float() / 255.0  # Convert to float and normalize to [0, 1] range
    results = model(frames, device=device, verbose=False)  
    bboxes = []
    for result in results:
      person_bboxes = result.boxes.xyxy[result.boxes.cls == 0]  
      if len(person_bboxes) > 0:
        bboxes.extend(person_bboxes.tolist())
    if not bboxes:
      bad_bboxes.append(f"No bounding boxes found for video {instance['video_id']}. Using default bbox.")
      bboxes = [[0, 0, frames.shape[3], frames.shape[2]]]  # Default bbox covering the whole frame
    largest_bbox = get_largest_bbox(bboxes)
    if largest_bbox is None:
      bad_bboxes.append(f"No bounding boxes found for video {instance['video_id']}. Using default bbox.")
      largest_bbox = [0, 0, frames.shape[3], frames.shape[2]]
    largest_bbox = [round(coord) for coord in largest_bbox]  # Round the coordinates to integers
    new_instences.append({
      'label_num': instance['label_num'],
      'frame_end': instance['frame_end'],
      'frame_start': instance['frame_start'],
      'video_id': instance['video_id'],
      'bbox': largest_bbox
    })
  log_path = os.path.join(output, 'bad_bboxes.txt')
  if bad_bboxes:
    with open(log_path, 'a') as log_file:
      for line in bad_bboxes:
        log_file.write(line + '\n')
    print(f"Bad bounding boxes logged to {log_path}.")
  base_name = os.path.basename(instance_path)
  mod_instances_path = os.path.join(output, base_name.replace('.json', '_fixed_bboxes.json'))
  with open(mod_instances_path, 'w') as f:
    json.dump(new_instences, f, indent=4)
  print(f"Updated instances with fixed bounding boxes saved to {mod_instances_path}.")
  
def remove_short_samples(instances_path, cutoff = 9, output='./output'):
  '''Preprocessing function which removes data with num frames less 
  than provided integer from the instances path. Assums the instances 
  have already been modified by preprocess_info, fix_bad_bboxes, and 
  fix_bad_frame range'''
  with open(instances_path, "r") as f:
    instances = json.load(f)
  mod_instances = [instance for instance in instances
    if (instance['frame_end'] - instance['frame_start'])
    > cutoff]
  base_name = os.path.basename(instances_path)
  out_path = os.path.join(output,
    base_name.replace('.json', '_short.json'))
  #chose this output path because file will end up with 
  # the extension *_fixed_bboxes_short.json, so fixed bounding
  #boxes, and fixed short clips
  with open(out_path, "w") as f:
    json.dump(mod_instances, f, indent=4)
  
if __name__ == "__main__":
  # test_video()
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