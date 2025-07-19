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
  
def fix_bad_frame_range(instance_path, raw_path, log='./output',
                        instances=None, output=None):
  # device = 'cuda' if torch.cuda.is_available() else 'cpu'
  bad_frames = []
  if instances is None:
    with open(instance_path, 'r') as f:
      instances = json.load(f)
  for instance in tqdm.tqdm(instances, desc="fixing frame ranges"):
    vid_path = os.path.join(raw_path, instance['video_id'] + '.mp4')
    decoder = VideoDecoder(vid_path)
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
  
  log_path = os.path.join(log, 'bad_frame_ranges.txt')
  if bad_frames:
    with open(log_path, 'a') as log_file:
      log_file.write('\n New bad frames \n ')
      for line in bad_frames:
        log_file.write(line + '\n')
    print(f"Bad frame ranges logged to {log}.")
  else:
    print("No bad frame ranges found")
  
  if output is not None:
    os.makedirs(output, exist_ok=True)
    base_name = os.basename(instance_path).replace('.json', '')
    fname = os.path.join(output, f'{base_name}_frange.json' )
    with open(fname, 'w') as f:
      json.dump(instances, f, indent=4)  
    print(f'fixed frame ranges saved to {fname}')
  
  return instances
    
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

def fix_bad_bboxes(instance_path, raw_path, log='./output',
                   instances=None, output=None):
  model = YOLO('yolov8n.pt')  # Load a pre-trained YOLO model
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # model.to(device)
  new_instances = []
  
  bad_bboxes = []
  if instances is None:
    with open(instance_path, 'r') as f:
      instances = json.load(f)
  for instance in tqdm.tqdm(instances, desc='Fixing bounding boxes'):
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
    new_instances.append({
      'label_num': instance['label_num'],
      'frame_end': instance['frame_end'],
      'frame_start': instance['frame_start'],
      'video_id': instance['video_id'],
      'bbox': largest_bbox
    })
  
  log_path = os.path.join(log, 'bad_bboxes.txt')
  
  if bad_bboxes:
    with open(log_path, 'a') as log_file:
      log_file.write('\n New bad bboxes \n ')
      for line in bad_bboxes:
        log_file.write(line + '\n')
    print(f"Bad bounding boxes logged to {log_path}.")
  else:
    print("No bad bounding boxes")
    
  if output is not None:
    os.makedirs(output, exist_ok=True)
    base_name = os.basename(instance_path).replace('.json', '')
    fname = os.path.join(output, f'{base_name}_bboxes.json' )
    with open(fname, 'w') as f:
      json.dump(new_instances, f, indent=4)  
    print(f'fixed bboxes saved to {fname}')
    
  return new_instances
    
def remove_short_samples(instances_path, classes_path,
                         cutoff = 9, log='./output',
                         instances=None, classes=None, output=None):
  '''Preprocessing function which removes data with num frames less 
  than provided integer from the instances path. Assums the instances 
  have already been modified by preprocess_info, fix_bad_bboxes, and 
  fix_bad_frame range'''
  if instances is None:
    with open(instances_path, "r") as f:
      instances = json.load(f)
  if classes is None:
    with open(classes_path, 'r') as f:
      classes = json.load(f)
  mod_instances = []
  mod_classes = []
  short_samples = []
  for i, inst in enumerate(instances):
    num_frame = inst['frame_end'] - inst['frame_start']
    if  num_frame > cutoff:
      mod_instances.append(instances[i])
      mod_classes.append(classes[i])
    else:
      short_samples.append(
        f"bad number of frames {num_frame} for video {inst['video_id']}, removing.")
      
  log_path = os.path.join(log, 'removed_short_samples.txt')
  
  if short_samples:
    with open(log_path, 'a') as log_file:
      log_file.write('\n New short samples \n ')
      for line in short_samples:
        log_file.write(line + '\n')
    print(f'short samples logged to {log_path}')
  else:
    print("no short samples")
    
  if output is not None:
    os.makedirs(output, exist_ok=True)
    base_name = os.basename(instances_path).replace('.json', '')
    fname = os.path.join(output, f'{base_name}_grtr{cutoff}.json' )
    with open(fname, 'w') as f:
      json.dump(instances, f, indent=4)  
    print(f'fixed short samples saved to {fname}')
    
  return mod_instances, mod_classes
    
def preprocess_split(split_path, raw_path='../data/WLASL2000', output_path='preprocessed/labels'):
  
  try:
    with open(split_path, 'r') as f:
      asl_num = json.load(f)
  except FileNotFoundError:
    print(f'Split file: {split_path} not found, stopping')
    return
  
  if not os.path.exists(raw_path):
    print(f"raw folder {raw_path} not found, stopping")
    return
  
  #create train, test, val splits
  train_instances, train_classes = get_split(asl_num, 'train')
  test_instances, test_classes = get_split(asl_num, 'test')
  val_instances, val_classes = get_split(asl_num, 'val')
  
  #setup storage
  base_name = os.path.basename(split_path).replace('.json', '')
  output_path = os.path.join(output_path, base_name)
  os.makedirs(output_path, exist_ok=True)
  f_exstension = '_fixed_frange_bboxes_len.json'
  
  print(f"Processing {base_name}")
  for split, instances, classes in [('train', train_instances, train_classes),
                                    ('test', test_instances, test_classes),
                                    ('val', val_instances, val_classes)]:
    print(f'For split: {split}')
    #fix badly labeled frame ranges
    print('Fixing frame ranges')
    instances = fix_bad_frame_range('', raw_path, instances=instances, log=output_path)
    
    #fix badly labeled bounding boxes
    print('Fixing bounding boxes')
    instances = fix_bad_bboxes('', raw_path, instances=instances, log=output_path)
    
    #finally, remove short samples
    print('Removing small samples')
    instances, classes = remove_short_samples('', '', instances=instances, classes=classes, log=output_path)
   
    #save 
    print('Saving results')
    inst_path = os.path.join(output_path, f'{split}_instances{f_exstension}')
    clss_path = os.path.join(output_path, f'{split}_classes{f_exstension}')
    with open(inst_path, 'w') as f:
      json.dump(instances)
    with open(clss_path, 'w') as f:
      json.dump(classes)
      
  
  print()
  print("------------------------- finished preprocessing ---------------")
  print()
  
if __name__ == '__main__':
  split = '../data/splits/asl300.json'
  preprocess_split(split)