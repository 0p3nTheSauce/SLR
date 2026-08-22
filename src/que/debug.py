import json
from pathlib import Path
from typing import Literal, Optional

# from .server import connect_manager
# from que.shell import QueShell
from src.que.core import CUR_RUN, FAIL_RUNS, OLD_RUNS, TO_RUN, Que, _get_basic_logger
from src.run_types import (
    AVAIL_SPLITS,
    CompExpInfo,
    ExpInfo,
    FailedExp,
)

KEYS = [TO_RUN, CUR_RUN, OLD_RUNS, FAIL_RUNS]


def rem_updated(name:str) -> str:
    return name.replace('_updated', '')
      

def update_runs13():
    
    q = Que()
    
    key_set = ['admin', 'config_path']
    q.update_runs(key_set, rem_updated)
    q.save_state('/home/luke/Code/SLR/src/que/Runs_no_updated.json')
    
def test_read_server_state():
    from src.que.core import read_server_state
    
    s = read_server_state()
    print(s.model_dump())



def fix_file_path(path: str) -> str:
    """Fix file path by replacing '/home/luke/Code/SLR/src/' with 'src/'"""
    try:
        return str(Path(path).relative_to(Path("/home/luke/Code/SLR/src/")))
    except ValueError:
        print(f"Path {path} is not under '/home/luke/Code/SLR/src/'. Attempting to make it relative to 'src/'...")
        pass
    try:
        return str(Path(path).relative_to(Path("src/")))
    except ValueError:
        print(f"Path {path} is not under 'src/'. Returning original path...")
        return path  # Return the original path if it doesn't match either condition
    
    
def update_runs16():
    q = Que()
    
    key_set = ['admin', 'config_path']
    q.update_runs(key_set, fix_file_path)
    key_set2 = ['admin', 'save_path']
    q.update_runs(key_set2, fix_file_path)
    q.save_state('/home/luke/Code/SLR/src/que/Runs_updated.json')

def update_runs17():
    q = Que()
    
    key_set = ['admin', 'config_path']
    q.update_runs(key_set, fix_file_path)
    key_set2 = ['admin', 'save_path']
    q.update_runs(key_set2, fix_file_path)
    q.save_state('/home/luke/Code/SLR/src/que/Runs_updated.json')


def update_runs18():

    with open("/home/luke/Code/SLR/src/que/Runs.json", "r") as f:
        all_runs = json.load(f)

    for loc in KEYS:
        que_list = all_runs[loc]
        new_quelist = []
        for run in que_list:
            
            stopping = run.pop('stopping', None)
            training = run.pop('training', None)
            assert stopping is not None 
            assert training is not None
            
            training['max_epoch'] = stopping.pop('max_epoch', 200)
            
            run['stopping'] = stopping
            run['training'] = training

            if loc in KEYS[:2]:
                run = ExpInfo.model_validate(run).model_dump()
            elif loc == KEYS[2]:
                run = CompExpInfo.model_validate(run).model_dump()
            else:
                run = FailedExp.model_validate(run).model_dump()

            new_quelist.append(run)

        all_runs[loc] = new_quelist

    with open("/home/luke/Code/SLR/src/que/Runs_fixed.json", "w") as f:
        json.dump(all_runs, f, indent=4)


def _fetch_ids(path: str = '/home/luke/Code/SLR/src/que/wlasl_cuttoff_0.csv') -> list[str]:
    import csv
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        
        return [row['ID'] for row in reader]
        

def fix_split(inst: dict, ids: list[str]) -> dict:
    from src.run_types import CUTOFF_SPLITS

    try:
        run = ExpInfo.model_validate(inst)
    except Exception:
        run = CompExpInfo.model_validate(inst)
    except Exception:
        raise
    
    new_split_names : dict[AVAIL_SPLITS, CUTOFF_SPLITS] = {
        'asl100' : 'asl100_cutoff_9',
        'asl300' : 'asl300_cutoff_9',
        'asl1000' : 'asl1000_cutoff_9',
        'asl2000' : 'asl2000_cutoff_9'
    }
    split = run.admin.split
    if run.wandb.run_id not in ids and split in new_split_names:
        new_split = new_split_names[run.admin.split]
        # print(f'Updating split: {split} -> {new_split}')
        run.admin.split = new_split
    else:
        print(f'Skipping split: {split} with run_id: {run.wandb.run_id}')
        
        
    return run.model_dump()
        

def update_runs19():

    ids = _fetch_ids()
    
    q = Que()
    
    # r = q.old_runs[len(q.old_runs)-1].model_dump()
    # z = fix_split(r, ids)
    # print(ids)
    # print(json.dumps(z['admin'], indent=4))
    key_set = []
    q.update_runs(key_set, lambda x: fix_split(x, ids))
    fp = '/home/luke/Code/SLR/src/que/Runs_updated.json'
    q.save_state(fp)
    q.load_state(fp) #check that it can load


if __name__ == "__main__":
    # update_runs14()
    # test_read_server_state()
    update_runs19()
    # pass
    
    
    
    
