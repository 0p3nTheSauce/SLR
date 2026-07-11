from typing import Optional, Literal
import json
from pathlib import Path

# from .server import connect_manager

# from que.shell import QueShell
from src.que.core import (
    TO_RUN,
    CUR_RUN,
    OLD_RUNS,
    FAIL_RUNS,
    Que,
    _get_basic_logger
)
from src.run_types import (
    FailedExp,
    CompExpInfo,
    ExpInfo,
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

def update_runs15():
    from src.stopping import StopperInfo, EarlyStopperInfo

    with open("/home/luke/Code/SLR/src/que/Runs.json", "r") as f:
        all_runs = json.load(f)

    for loc in KEYS:
        que_list = all_runs[loc]
        new_quelist = []
        for run in que_list:
            
            stopping = run.pop('stopping', None)
            
            if stopping is None:
                raise ValueError(f"Run {run['admin']['config_path']} is missing 'stopping' information.")
            elif stopping['type'] == 'stopper':
                stopping = StopperInfo(max_epoch=stopping['max_epoch'])
            elif stopping['type'] == 'early_stopper':
                stopping = EarlyStopperInfo(
                    max_epoch=stopping['max_epoch'],
                    metric=stopping['metric'],
                    phase=stopping['phase'],
                    mode=stopping['mode'],
                    patience=stopping['patience'],
                    min_delta=stopping['min_delta']
                )
            print(stopping.model_dump())
            
            run['stopping'] = stopping.model_dump()

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

if __name__ == "__main__":
    # update_runs14()
    # test_read_server_state()
    update_runs16()
    # pass
    
    
    
    
