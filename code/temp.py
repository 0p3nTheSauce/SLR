from pathlib import Path

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
            
            # Get all files in checkpoint directory
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
            
            # Leave best.pth and the last checkpoint
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
                
                
def clean_experiments(path, ask=False, rem_empty=False):
  if not os.path.exists(path):
    raise FileNotFoundError(f'Experiments path: {path} was not found')
  sub_paths = os.listdir(path)#experiments
  
  return clean_checkpoints([os.path.join(path, sub_path) \
    for sub_path in sub_paths], ask)                
  
  from pathlib import Path

def clean_experiments(path, ask=False, rem_empty=False):
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f'Experiments path: {path} was not found')
    
    # Get all subdirectories in the experiments path
    sub_paths = [item for item in path_obj.iterdir() if item.is_dir()]
    
    return clean_checkpoints(sub_paths, ask=ask, rem_empty=rem_empty)