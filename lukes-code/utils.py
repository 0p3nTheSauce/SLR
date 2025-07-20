import torch
import torch.nn.functional as F
import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import gzip
# from torchcodec.decoders import VideoDecoder

################# Loading #####################
def load_rgb_frames_from_video_codec(video_path : str, start : int, end : int
                               ,device : str ='cpu' , all : bool =False) -> torch.Tensor: 
  # '''Loads RGB frames from a video file as a PyTorch Tensor
  # Args:
  #   video_path: Path to video file
  #   start: Start frame index
  #   end: End frame index
  #   device: Device to load tensor on ('cpu' or 'cuda')
  #   all: If True, load all frames (ignores start/end)
  # Returns:
  #   torch.Tensor: RGB frames of shape (T, H, W, C) with dtype uint8
  # '''
  # # if device =='cuda' and not torch.cuda.is_available():
  # #   device = 'cpu'
  # #   print("Warning: cuda not available so using cpu")
  # decoder = VideoDecoder(video_path, device=device)
  # num_frames = decoder._num_frames
  # if all:
  #   start = 0
  #   end = num_frames
  # if start < 0 or end > num_frames or end <= start:
  #   # raise ValueError(f"Invalid frame range: start={start}, end={end}, num_frames={num_frames}")
  #   print(f"Invalid frame range: start={start}, end={end}, num_frames={num_frames}. Adjusting to valid range.")
  #   print(f"Using start=0 and end={num_frames}")
  #   start = 0
  #   end = num_frames
  # return decoder.get_frames_in_range(start, end).data
  
  #stinking torchcodec is borked
  return torch.rand(2)

def load_rgb_frames_from_video(video_path : str, start : int, end : int
                              , all : bool =False, recover=True) -> torch.Tensor:
  if recover:
    try:
      frames = cv_load(video_path, start, end, all)
    except ValueError:
      frames = cv_load(video_path, start, end, all=True)
    return cv_to_torch(frames)
  else:
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

def cv_to_torch(frames):
  '''convert 4D opencvformat to torch Tensor (T C H W)'''
  frames_rgb = np.asarray([
    cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
    for cv_frame in frames])
  torch_frames = torch.from_numpy(frames_rgb)
  torch_frames = torch_frames.permute(0,3,1,2)
  return torch_frames
  

####################     Plotting utilities  ####################


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


##################### Misc ###################################
def clean_checkpoints(paths):
  for path in paths:
    to_empty = os.path.join(path, 'checkpoints')
    files = sorted(os.listdir(to_empty))
    to_remove = files[1:-1] #leave best.pth and the last checkpoint
    for file in to_remove:
      os.remove(os.path.join(to_empty, file))
      print(f'removed {file} in {to_empty}')
      

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
  cv_frames = torch_to_cv(frames)
  save_video(cv_frames, vid, fps=24)



def main():
  # test_save()
  # test_save2()
  #  torch_to_mediapipe()
  # watch_video(path='69241.mp4')
  
  paths = ['r3d18_exp0', 'r3d18_exp1', 'r3d18_exp2', 'r3d18_exp3']
  paths = [os.path.join('runs/asl100', path) for path in paths]
  clean_checkpoints(['runs/asl300/r3d18_exp0'])
  
  # test_vid = '../data/WLASL2000/07393.mp4'
  # torch_frames = load_rgb_frames_from_video(test_vid, 0, 10, True)
  # print(torch_frames.shape)

if __name__ == '__main__':
  main()