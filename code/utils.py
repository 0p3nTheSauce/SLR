import torch
import torch.nn.functional as F
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import gzip
# from torchcodec.decoders import VideoDecoder
import re
from pathlib import Path

############# pretty printing ##############

def print_dict(dict):
	print(string_nested_dict(dict))
	
def string_nested_dict(dict):
	ans = ""
	if type(dict) == type({}):
		ans += "{\n"
		for key, value in dict.items():
			ans += f'{key} : {string_nested_dict(value)}\n'
		ans += "}\n"
	else:
		ans += str(dict)
	return ans



################# Loading #####################

  
def load_rgb_frames_from_video(video_path : str, start : int, end : int
                              , all : bool =False) -> torch.Tensor:
  return cv_to_torch(cv_load(video_path, start, end, all))

def cv_load(video_path, start, end, all=False):
  if not os.path.exists(video_path):
    raise FileNotFoundError(f'File {video_path} does not exist')
  cap = cv2.VideoCapture(video_path)
  frame_count = 0
  frames = []
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    frames.append(frame)
  if len(frames) > 0:
    if all:
      return np.asarray(frames) 
    else:
      return np.asarray(frames[start:end])
  else:
    raise ValueError(f"No frames were loaded for file {video_path}")

def load_tensorboard_json(filepath):
  try:
    # Try regular JSON first
    with open(filepath, 'r', encoding='utf-8') as f:
      return json.load(f)
  except UnicodeDecodeError:
    try:
      # Try gzipped JSON
      with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        return json.load(f)
    except:
        # Try with different encoding
        try:
          with open(filepath, 'r', encoding='latin-1') as f:
            return json.load(f)
        except:
          # Last resort - read as binary and try to decode
          with open(filepath, 'rb') as f:
            content = f.read()
            # Check if it's gzipped
            if content.startswith(b'\x1f\x8b'):
              content = gzip.decompress(content)
              return json.loads(content.decode('utf-8', errors='ignore'))

################## Saving #####################


def save_video(frames, path, fps=30):
  '''Arguments:
  frames : numpy tensor (T H W C) BGR (cv format)
  path : output path (string) 
  fps : int'''
  
  if len(frames.shape) != 4:
    raise ValueError(f"Expected 4D tensor (T,H,W,C), got shape {frames.shape}")
  
  # Ensure frames are uint8
  if frames.dtype != np.uint8:
    if frames.max() <= 1.0:
      frames = (frames * 255).astype(np.uint8)
    else:
      frames = frames.astype(np.uint8)
  
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
  width = frames.shape[2]
  height = frames.shape[1]
  out = cv2.VideoWriter(path, fourcc, fps, (width, height))
  if not out.isOpened():
    raise RuntimeError(f"Could not open video writer for {path}")
  
  for frame in frames:
    out.write(frame)
  out.release()
  
################## Displaying  #####################
  
  ################  CV based ####################
  
def watch_video(frames=None, path='',wait=33, title='Video'):
  if frames is None and not path:
    raise ValueError('pass either a tensor or path')
  elif frames is not None:
    if not (type(frames) is torch.Tensor or type(frames) is np.ndarray):
      raise ValueError('frames must be torch.Tensor or np.ndarray')
    if frames.dtype == torch.uint8:
      frames = torch_to_cv(frames) #type: ignore
    elif frames.dtype != np.uint8:
      raise ValueError('frames must be torch.uint8 (T C H W) RGB OR np.uint8 (T H W C) BGR')
    for img in frames:
      cv2.imshow(f'{title} from tensor', img) #type: ignore
      key = cv2.waitKey(wait) & 0xFF
      if key == ord('q') or key == 27:
        break
  else:
    cap = cv2.VideoCapture(path)
    beg = True
    while cap.isOpened():
      ret, img = cap.read()
      if not ret:
        print('no frames')
        if beg:
          continue
        else:
          print('finished') 
          break #works for files, not well for webcam
      cv2.imshow(f'{title} from file', img)
      beg = False
      key = cv2.waitKey(wait) & 0xFF
      if key == ord('q') or key == 27:
        break
  cv2.destroyAllWindows()
      
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

def cv_display_or_save(frames,output=None):
  if output is None:
    for i, frame in enumerate(frames):
      cv2.imshow('Cropped Frames', frame)  # Display the first frame
      cv2.waitKey(100)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
  else:
    Path(output).mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(frames):
      cv2.imwrite(f"{output}/frame_{i:04d}.jpg", img)  # Save each frame as an image
    print(f"Cropped frames saved to {output}.")

  ###############   PLOT Based       #################

def visualise_frames(frames,num, size=(5,5), adapt=False):
  # permute and convert to numpy 
  '''Args:
    frames : torch.Tensor (T, C, H, W)
    num : int, to be visualised'''
  if adapt:
    #256 ~ 5
    factor = 5 / 256
    w, h = frames.shape[2], frames.shape[3]
    size = (w * factor, h * factor)
    
  if num < 1:
    raise ValueError("num must be >= 1")
  num_frames = len(frames)
  if num_frames <= num:
    step = 1
  else:
    step = num_frames // num
  for frame in frames[::step]:
    np_frame = frame.permute(1,2,0).cpu().numpy()
    plt.figure(figsize=size)
    plt.imshow(np_frame)
    plt.axis('off')
    plt.show()
  

################### Conversions #####################

    
def torch_to_cv(frames: torch.Tensor) -> np.ndarray:
  '''convert 4D torch tensor (T C H W) uint8 to opencv format'''
  frames = frames.permute(0, 2, 3, 1) # change to T H W C
  np_frames = frames.numpy()
  np_frames_col = np.asarray([
    cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
    for np_frame in np_frames
    ])
  return np_frames_col
  
def torch_to_mediapipe(frames : torch.Tensor) -> np.ndarray:
  '''convert 4D torch.uint8 (T C H W) to mediapipe format'''
  frames = frames.permute(0, 2, 3, 1) # convert to (T H W C)
  np_array = frames.numpy()
  return np_array

def cv_to_torch(frames):
  '''convert 4D opencvformat to torch Tensor (T C H W)'''
  frames_rgb = np.asarray([
    cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
    for cv_frame in frames])
  torch_frames = torch.from_numpy(frames_rgb)
  torch_frames = torch_frames.permute(0,3,1,2)
  return torch_frames
  

####################     Plotting utilities  ####################


def plot_from_lists(train_loss, val_loss=None, 
                          title='Training Loss Curve',
                          xlabel='Epochs',
                          ylabel='Loss',
                          save_path=None,
                          show=True):
  # Extract the three components from training loss
  # wall_times = [point[0] for point in train_loss]
  # steps = [point[1] for point in train_loss]
  # values = [point[2] for point in train_loss]
  steps = list(range(len(train_loss)))
  plt.figure(figsize=(10, 6))
  
  # Plot training loss
  plt.plot(steps, train_loss, color='orange', linewidth=2, label='Training Loss')
  
  # Plot validation loss if provided
  if val_loss:
    # val_steps = [point[1] for point in val_loss]
    # val_values = [point[2] for point in val_loss]
    
    plt.plot(steps, val_loss, color='blue', linewidth=2, label='Validation Loss')
  
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.grid(True, alpha=0.3)
  plt.legend()
  
  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
  
  if show:
    plt.show()



##################### Misc ###################################

def extract_num(fname):
  num_substrs = re.findall(r'\d+', fname)
  if len(num_substrs) > 1:
    num_str_concat = ' '.join(sub for sub in num_substrs)
    print(f'fname has multuple number substrings: ')
    print(num_str_concat)
    idx = -1
    while True:
      ans = input('zfill which substring? [-1]')
      if ans == '':
        break
      elif ans.isdigit():
        idx = int(ans)
        if -len(num_substrs) <= idx < len(num_substrs):
          break
        else:
          print(f'invalid index: {idx} for {len(num_substrs)} substrings')
      else:
        print(f'{ans} is not a digit')
    return num_substrs[idx]
  elif len(num_substrs) == 0:
    raise ValueError(f'No valid number substrings found in {fname}')
  else:
    return num_substrs[0]
     
   


def clean_checkpoints(paths, ask=False, add_zfill=True, decimals=3, rem_empty=False):
  for path in paths:
    path_obj = Path(path)
        
    # Find checkpoint directories
    check_point_dirs = [item.name for item in path_obj.iterdir() 
                        if item.is_dir() and 'checkpoint' in item.name]
    
    if len(check_point_dirs) == 0:
      if rem_empty:
        try:
          path_obj.rmdir()
        except OSError as e:
          print(f"Ran into an error when removing {path}")
          print(e)
          continue
      else:
        print(f'Warning, no checkpoints found in {path}') 
      continue
    
    
    for check in check_point_dirs:
      to_empty = path_obj / check
      
      dirty = sorted([item.name for item in to_empty.iterdir() if item.is_file()])
      files = [file for file in dirty 
               if file.endswith('.pth')
               and 'best' not in file]
      
      if add_zfill:
        for i, f in enumerate(files):
          num = extract_num(f)
          files[i] = f.replace(num, num.zfill(decimals))
          
      if len(files) <= 2:
        continue 
      
      #leave best.pth and the last checkpoint
      to_remove = files[:-1]  # not great safety wise, assumes files sort correctly
      if ask:
        ans = 'none'
        while ans != 'y' and ans != '' and ans != 'n':
          print(f'Only keep checkpoint {files[-1]} in {to_empty} (excluding best.pth and non pth files)?')
          ans = input('[y]/n: ')
        
        if ans == 'n':
          continue

      for file in to_remove:
        file_path = to_empty / file
        file_path.unlink()
        print(f'removed {file} in {to_empty}')

def is_empty(path):
  return not any(Path(path).iterdir())

    
def clean_experiments(path, ask=False, rem_empty=False):
  path_obj = Path(path)
  
  if not path_obj.exists():
    raise FileNotFoundError(f'Experiments path: {path} was not found')
  
  sub_paths = [item for item in path_obj.iterdir() if item.is_dir()]
  
  return clean_checkpoints(sub_paths, ask=ask,rem_empty=rem_empty)
    
def clean_runs(path, ask=False, rem_empty=False):
  path_obj = Path(path)
  
  if not path_obj.exists():
    raise FileNotFoundError(f'Runs directory: {path} was not found')
  sub_paths = [item for item in path_obj.iterdir() if item.is_dir()]
  
  return clean_experiments(sub_paths, ask, rem_empty)
  
def crop_frames(frames, bbox):
  #frames hase shape (num_frames, channels, height, width)
  #bbox is a list of [x1, y1, x2, y2]
  x1, y1, x2, y2 = bbox
  return frames[:, :, y1:y2, x1:x2]  # Crop the frames using the bounding box

# def enum_dir(path, make=False):
#   if os.path.exists(path):
#     if not path[-1].isdigit():
#       path += '0'
#     while os.path.exists(path):
#       path = path[:-1] + str(int(path[-1]) + 1)
#   if make:
#     os.makedirs(path, exist_ok=True)
#   return path

import os
def enum_dir(path, make=False, decimals=3):
  '''Enumerate filenames'''
  if os.path.exists(path):
    if not path[-1].isdigit():
      path += '0'.zfill(decimals)
    while os.path.exists(path):
      num = int(path[-decimals:]) 
      path = path[:-decimals] + str(num + 1).zfill(decimals)
  if make:
    os.makedirs(path, exist_ok=True)
  return path

##################### once offs / Testing ######################



def test_save():
  instances = './preprocessed_labels/asl100/train_instances_fixed_bboxes_short.json'
  with open(instances, 'r') as f:
    all_inst = json.load(f)
  inst0 = all_inst[0]
  vid = f"{inst0['video_id']}.mp4"
  vid_path = os.path.join('../data/WLASL2000',vid)
  
  cap = cv2.VideoCapture(vid_path)
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  print(f'FPS: {fps}')
  print(f'width: {width}')
  print(f'height: {height}')  
  frames = []
  frame_count = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    frames.append(frame)
    frame_count += 1
  print(f'processed {frame_count} frames')
  frames = np.asarray(frames)
  print(frames.shape)
  save_video(frames, vid, fps=30)
  # save_video(np_frames_col, f'col_{vid}', fps=15)

def test_save2():
  instances = './preprocessed_labels/asl100/train_instances_fixed_bboxes_short.json'
  with open(instances, 'r') as f:
    all_inst = json.load(f)
  inst0 = all_inst[0]
  vid = f"{inst0['video_id']}.mp4"
  vid_path = os.path.join('../data/WLASL2000',vid)
  frames = cv_load(vid_path,0,0,True)
 
  print(frames.shape)
  print(frames.dtype)
  cv_frames = torch_to_cv(frames) #type: ignore
  save_video(cv_frames, vid, fps=24)



def main():
  # test_save()
  # test_save2()
  #  torch_to_mediapipe()
  # watch_video(path='69241.mp4')
  # name_mapping = {'Loss/Train_Epoch': 'Loss/Train',
  #                 'Loss/Test_Epoch' : 'Loss/Val',
  #                 'Accuracy/Train_Epoch' : 'Accuracy/Train',
  #                 'Accuracy/Test_Epoch' : 'Accuracy/Val'}
  
  # rename_scalars_in_eventfile('./runs/asl100/r3d18_exp004/logs',
  #                             './runs/asl100/r3d18_exp004/logs_fixed',
  #                             name_mapping)
  clean_runs('runs', ask=True)
  # test_vid = '../data/WLASL2000/07393.mp4'
  # torch_frames = load_rgb_frames_from_video(test_vid, 0, 10, True)
  # print(torch_frames.shape)

if __name__ == '__main__':
  main()