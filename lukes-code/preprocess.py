import json
import os
import torch
import tqdm
from ultralytics import YOLO
from torchcodec.decoders import VideoDecoder

#local imports
from utils import load_rgb_frames_from_video 

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
  return instances, classes
    
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
    # frames = load_rgb_frames_from_video_ioversion(vid_path, instance['frame_start'], instance['frame_end'], all=True)
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
    
def preprocess_split(split_path, output_path='preprocessed/labels'):
  preprocess_info(split_path, 'train', output_path)
  preprocess_info(split_path, 'test', output_path)
  preprocess_info(split_path, 'val', output_path)
  