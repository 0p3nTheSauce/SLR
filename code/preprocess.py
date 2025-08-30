import json
# import os
import torch
import tqdm
from ultralytics import YOLO # type: ignore (have a feeling this will bork the system)
# from torchcodec.decoders import VideoDecoder #this thing causes too many problems
import cv2

from argparse import ArgumentParser
from pathlib import Path
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
  json_path = Path(json_path)
  output_path = Path(output_path)
  with open(json_path, 'r') as f:
    asl_num = json.load(f)
    instances, classes = get_split(asl_num, split)
  base_name = json_path.name.replace('.json','')
  output_path = output_path / base_name
  output_path.mkdir(parents=True, exist_ok=True)
  with open(output_path / f'{split}_instances.json', 'w') as f:
    json.dump(instances, f, indent=4)
  with open(output_path / f'{split}_classes.json', 'w') as f:
    json.dump(classes, f, indent=4)
  return instances, classes
    
def prep_train():
  json_path = '../data/splits/asl100.json'
  split = 'train'
  output_root = './preprocessed_labels/'
  output_root = Path(output_root)
  output_root.mkdir(parents=True, exist_ok=True)
  preprocess_info(json_path, split, output_root)
  
def prep_test():
  json_path = '../data/splits/asl100.json'
  split = 'test'
  output_root = './preprocessed_labels/'
  output_root = Path(output_root)
  output_root.mkdir(parents=True, exist_ok=True)
  preprocess_info(json_path, split, output_root)
  
def prep_val():
  json_path = '../data/splits/asl100.json'
  split = 'val'
  output_root = './preprocessed_labels/'
  output_root = Path(output_root)
  output_root.mkdir(parents=True, exist_ok=True)
  preprocess_info(json_path, split, output_root)
  
def fix_bad_frame_range(instance_path, raw_path, log,
                        instances=None, output=None):
  # device = 'cuda' if torch.cuda.is_available() else 'cpu'
  instance_path = Path(instance_path)
  raw_path = Path(raw_path)
  log = Path(log)
  bad_frames = []
  if instances is None:
    with open(instance_path, 'r') as f:
      instances = json.load(f)
  for instance in tqdm.tqdm(instances, desc="fixing frame ranges"):
    vid_path = raw_path / instance['video_id'] + '.mp4'
    # decoder = VideoDecoder(vid_path)
    
    num_frames = 0
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
      print("Error: Could not open video.")
    else:
      num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      
    if num_frames == 0:  
      bad_frames.append(f"No frames detected for video {instance['video_id']}. Skipping")
      instance['frame_start'] = 0
      instance['frame_end'] = 0 # a different function handles short samples anyway
      continue
    
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
  
  log_path = log / 'bad_frame_ranges.txt'
  if bad_frames:
    with open(log_path, 'a') as log_file:
      log_file.write('\n New bad frames \n ')
      for line in bad_frames:
        log_file.write(line + '\n')
    print(f"Bad frame ranges logged to {log}.")
  else:
    print("No bad frame ranges found")
  
  if output is not None:
    output.mkdir(parents=True, exist_ok=True)
    base_name = instance_path.name.replace('.json', '')
    fname = output / f'{base_name}_frange.json' 
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

def fix_bad_bboxes(instance_path, raw_path, log,
                   instances=None, output=None):
  instance_path = Path(instance_path)
  raw_path = Path(raw_path)
  log = Path(log)
  # TODO: could be a bit faster
  model = YOLO('yolov8n.pt')  # Load a pre-trained YOLO model
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # model.to(device)
  new_instances = []
  
  bad_bboxes = []
  if instances is None:
    with open(instance_path, 'r') as f:
      instances = json.load(f)
  for instance in tqdm.tqdm(instances, desc='Fixing bounding boxes'):
    vid_path = raw_path / instance['video_id'] + '.mp4'
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
  
  log_path = log / 'bad_bboxes.txt'
  
  if bad_bboxes:
    with open(log_path, 'a') as log_file:
      log_file.write('\n New bad bboxes \n ')
      for line in bad_bboxes:
        log_file.write(line + '\n')
    print(f"Bad bounding boxes logged to {log_path}.")
  else:
    print("No bad bounding boxes")
    
  if output is not None:
    output.mkdir(parents=True, exist_ok=True)
    base_name = instance_path.name.replace('.json', '')
    fname = output / f'{base_name}_bboxes.json' 
    with open(fname, 'w') as f:
      json.dump(new_instances, f, indent=4)  
    print(f'fixed bboxes saved to {fname}')
    
  return new_instances
    
def remove_short_samples(instances_path, classes_path,
                          log,cutoff = 9,
                         instances=None, classes=None, output=None):
  instance_path = Path(instances_path)
  classes_path = Path(classes_path)
  log = Path(log)
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
      
  log_path = log / 'removed_short_samples.txt'
  
  if short_samples:
    with open(log_path, 'a') as log_file:
      log_file.write('\n New short samples \n ')
      for line in short_samples:
        log_file.write(line + '\n')
    print(f'short samples logged to {log_path}')
  else:
    print("no short samples")
    
  if output is not None:
    output.mkdir(parents=True, exist_ok=True)
    base_name = instances_path.name.replace('.json', '')
    fname = output / f'{base_name}_grtr{cutoff}.json' 
    with open(fname, 'w') as f:
      json.dump(instances, f, indent=4)  
    print(f'fixed short samples saved to {fname}')
    
  return mod_instances, mod_classes
    
def preprocess_split(split_path:Path, raw_path:Path, output_base:Path, verbose:bool=False) -> None:
  
  if not check_paths(split_path, raw_path, output_base, verbose):
    return
  
  with open(split_path, 'r') as f:
    asl_num = json.load(f)
  
  if not asl_num:
    print(f'no data found in {split_path}')
    return
  
  #create train, test, val splits
  train_instances, train_classes = get_split(asl_num, 'train')
  test_instances, test_classes = get_split(asl_num, 'test')
  val_instances, val_classes = get_split(asl_num, 'val')
  
  #setup storage
  # base_name = os.path.basename(split_path).replace('.json', '')
  # output_path = os.path.join(output_path, base_name)
  base_name = split_path.name.replace('.json', '')
  output_dir = output_base / base_name
  # os.makedirs(output_path, exist_ok=True)
  output_dir.mkdir(parents=True, exist_ok=True)
  f_exstension = '_fixed_frange_bboxes_len.json'
  
  print_v(f"Processing {base_name}", verbose)
  for split, instances, classes in [('train', train_instances, train_classes),
                                    ('test', test_instances, test_classes),
                                    ('val', val_instances, val_classes)]:
    print_v(f'For split: {split}', verbose)
    #fix badly labeled frame ranges
    print_v(f'Fixing frame ranges', verbose)
    instances = fix_bad_frame_range('', raw_path, instances=instances, log=output_dir)
    
    #fix badly labeled bounding boxes
    print_v('Fixing bounding boxes', verbose)
    instances = fix_bad_bboxes('', raw_path, instances=instances, log=output_dir)
    
    #finally, remove short samples
    print_v('Removing small samples', verbose)
    instances, classes = remove_short_samples('', '', instances=instances, classes=classes, log=output_dir)
   
    #save 
    print_v('Saving results', verbose)
    # inst_path = os.path.join(output_dir, f'{split}_instances{f_exstension}')
    inst_path = output_dir / f'{split}_instances{f_exstension}'
    # clss_path = os.path.join(output_path, f'{split}_classes{f_exstension}')
    clss_path = output_dir / f'{split}_classes{f_exstension}'
    with open(inst_path, 'w') as f:
      json.dump(instances, f, indent=2)
    with open(clss_path, 'w') as f:
      json.dump(classes, f, indent=2)
      
  
  print()
  print("------------------------- finished preprocessing ---------------")
  print()
  
def print_v(s:str, y:bool)->None:
  if y:
    print(s)

def check_paths(split_path:Path, raw_path:Path, output_path:Path, verbose:bool)->bool:
  if split_path.exists() and split_path.is_file():
    print_v(f'split path: {split_path}, found', verbose) 
  else:
    print(f'split path: {split_path}, not found')
    return False
  if raw_path.exists() and raw_path.is_dir():
    print_v(f'raw path: {raw_path}, found', verbose)
  else:
    print(f'raw path: {raw_path}, not found')    
    return False
  if output_path.exists() and output_path.is_dir():
    print_v(f'output path: {output_path}, found', verbose)    
  else:
    print(f'output path: {output_path}, not found')    
    return False
  return True
  
if __name__ == '__main__':
  def_root = '../data/WLASL'
  def_split_dir = 'splits'
  def_raw_dir = 'WLASL2000'
  def_output_dir = 'preprocessed/labels'
  parser = ArgumentParser(description='preprocess.py')
  parser.add_argument('-as', '--asl_split', type=str, required=True, 
                      help='<asl100|asl300|asl1000|asl2000')
  parser.add_argument('-rt', '--root',type=str,
                      help=f'WLASL root if not {def_root}', default=def_root)
  parser.add_argument('-sd', '--split_dir', type=str,
                      help=f'Split directory if not {def_split_dir}', default=def_split_dir)  
  parser.add_argument('-rd', '--raw_dir', type=str,
                      help=f'Video directory if not {def_raw_dir}', default=def_raw_dir)
  parser.add_argument('-od', '--output_dir', type=str,
                      help=f'Output directory if not {def_output_dir}', default=def_output_dir)
  parser.add_argument('-ve', '--verbose', action='store_true', help='verbose output')
  args = parser.parse_args()
  
  root = Path(args.root)
  split_path = root / args.split_dir / f'{args.asl_split}.json'
  raw_dir = root / args.raw_dir
  output_dir = Path(args.output_dir)
  
  