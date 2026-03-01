from pathlib import Path
import json
from preprocess import _retrieve_frame_shape



def add_frame_shape():
    label_dir = Path('./preprocessed/labels_new')
    assert label_dir.exists()
    for split in label_dir.iterdir():
        for file in split.iterdir():
            if file.name.startswith('train') or file.name.startswith('test') or file.name.startswith('val'):
                with open(file, 'r') as f:
                    data = json.load(f)
                
                # for i, instance in enumerate(data):
                #     height, width = _retrieve_frame_shape(instance)
                #     instance['frame_width'] = width
                #     instance['frame_height'] = height
                #     data[i] = instance
                    
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
                
dostuff()