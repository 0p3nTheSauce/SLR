from pathlib import Path
import json
from preprocess import instance_dict
from utils import cv_load
from configs import WLASL_ROOT, RAW_DIR
from typing import Tuple


def _retrieve_frame_shape(instance: instance_dict, raw_path: Path = Path(WLASL_ROOT) / RAW_DIR) -> Tuple[int, int]:
	"""Get the frame shape of a video.

	Args:
			raw_path (Path): Path to raw videos directory
			instance (instance_dict): Dictionary containing key: video_id

	Raises:
		FileNotFoundError: If the video file does not exist
		ValueError: If no frames were loaded for the video file

	Returns:
			Tuple[int, int]: Height, Width
	"""
	vid_path = raw_path / (instance["video_id"] + ".mp4")
	frames = cv_load(vid_path, 0, 1)
	return frames.shape[1], frames.shape[2]

def add_frame_shape():
    label_dir = Path('./preprocessed/labels_new')
    assert label_dir.exists()
    for split in label_dir.iterdir():
        for file in split.iterdir():
            if file.name.startswith('train') or file.name.startswith('test') or file.name.startswith('val'):
                with open(file, 'r') as f:
                    data = json.load(f)
                
                for i, instance in enumerate(data):
                    height, width = _retrieve_frame_shape(instance)
                    instance['frame_width'] = width
                    instance['frame_height'] = height
                    data[i] = instance
                    
                with open(file, 'w') as f:
                    json.dump(data, f, indent=4)
                
                print(f'updated file: {file}')    
            else:
                print(f'left file: {file}')
                    
def dostuff():
    label_dir = Path('./preprocessed/labels_new')
    assert label_dir.exists()
    for split in label_dir.iterdir():
        for file in split.iterdir():
            if file.name.startswith('train') or file.name.startswith('test') or file.name.startswith('val'):
                with open(file, 'r') as f:
                    data = json.load(f)
                
                for i, instance in enumerate(data):
                    
                    assert instance['frame_width'] == instance['frame_height']  == 256
                    
                
                print(f'also left file: {file}')    
            else:
                # print(f'left file: {file}')
                pass                
            
            
add_frame_shape()        
dostuff()