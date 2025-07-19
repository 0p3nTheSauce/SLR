import torch
import torch.nn.functional as F
import cv2
import numpy as np
import video_dataset as tools
import json
import os
import matplotlib.pyplot as plt
import gzip

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
  
def watch_video(frames=None, path='',wait=33):
  if not frames and not path:
    raise ValueError('pass either a tensor or path')
  elif frames:
    if frames.dtype == torch.uint8:
      frames = torch_to_cv(frames)
    elif frames.dtype != np.uint8:
      raise ValueError('frames must be torch.uint8 (T C H W) RGB OR np.uint8 (T H W C) BGR')
    for img in frames:
      cv2.imshow('Tensor as video', img)
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
      cv2.imshow('File as video', img)
      beg = False
      key = cv2.waitKey(wait) & 0xFF
      if key == ord('q') or key == 27:
        break
  cv2.destroyAllWindows()
      
      
      
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
    if all:
      frames.append(frame)
    elif start <= frame_count < end:
      frames.append(frame)
  if len(frames) > 0:
    return np.asarray(frames) 
  else:
    raise ValueError("No frames were loaded")
    
def torch_to_cv(frames):
  '''convert 4D torch tensor (T C H W) to opencv format'''
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
  
  
def plot_from_simple_list(train_loss, val_loss=None, 
                          title='Training Loss Curve',
                          xlabel='Epochs',
                          ylabel='Loss',
                          save_path=None):
  # Extract the three components from training loss
  wall_times = [point[0] for point in train_loss]
  steps = [point[1] for point in train_loss]
  values = [point[2] for point in train_loss]
  
  plt.figure(figsize=(10, 6))
  
  # Plot training loss
  plt.plot(steps, values, color='orange', linewidth=2, label='Training Loss')
  
  # Plot validation loss if provided
  if val_loss:
    val_steps = [point[1] for point in val_loss]
    val_values = [point[2] for point in val_loss]
    plt.plot(val_steps, val_values, color='blue', linewidth=2, label='Validation Loss')
  
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.grid(True, alpha=0.3)
  plt.legend()
  
  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
  
  plt.show()

  if save_path:
    plt.savefig(save_path, bbox_inches='tight')
    print(f'Saved plot to {save_path}')

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
#####################once offs######################
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
  cv_frames = torch_to_cv(frames)
  save_video(cv_frames, vid, fps=24)

def main():
  # test_save()
  # test_save2()
  #  torch_to_mediapipe()
  watch_video(path='69241.mp4')
  

if __name__ == '__main__':
  main()