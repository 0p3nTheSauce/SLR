from pathlib import Path

label_dir = Path('./preprocessed/labels/')

if label_dir.exists():
    print('exists')
    
else:
    print('failed')
    raise FileNotFoundError

for split in label_dir.iterdir():
    for file_p in split.iterdir():
        if file_p.name.startswith('removed_'):
            new_file_p = file_p.parent / f"cutoff_9_{file_p.name}"
            print(f'changing: {file_p} to {new_file_p}')
            file_p.rename(new_file_p)