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


def test_update_config_file(
    default_mode: Literal["overwrite", "duplicate"] = "overwrite",
    dry_run: bool = True,
    retro_support: bool = False,
):
    from src.que.runs_to_configs import update_config_file

    with open("/home/luke/Code/SLR/src/que/Runs.json", "r") as f:
        all_runs = json.load(f)

    old_runs = all_runs[OLD_RUNS]

    run_info = old_runs[75]

    update_config_file(run_info, default_mode, dry_run, retro_support)


def test_update_all_files(
    default_mode: Literal["overwrite", "duplicate"] = "overwrite",
    dry_run: bool = True,
    retro_support: bool = False,
    output: Optional[Path] = None
):
    from src.que.runs_to_configs import update_config_file

    with open("/home/luke/Code/SLR/src/que/Runs.json", "r") as f:
        all_runs = json.load(f)

    flat_all_runs = []
    for key in KEYS:
        flat_all_runs.extend(all_runs[key])

    for run_info in flat_all_runs:
        update_config_file(run_info, default_mode, dry_run, retro_support, output = output)

    print(len(flat_all_runs))


def update_runs10():
    from configs import correct_paths

    with open("/home/luke/Code/SLR/src/que/Runs.json", "r") as f:
        all_runs = json.load(f)

    for loc in KEYS:
        que_list = all_runs[loc]
        new_quelist = []
        for run in que_list:
            run["admin"] = correct_paths(run["admin"])

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


def update_runs11():
    from src.que.runs_to_configs import _get_save_name
    q = Que()
    
    key_set = ['admin', 'config_path']
    q.update_runs(key_set, _get_save_name)
    q.save_state('/home/luke/Code/SLR/src/que/Runs_no_ini.json')



def update_runs12():
    from src.stopping import StopperInfo, EarlyStopperInfo

    with open("/home/luke/Code/SLR/src/que/Runs.json", "r") as f:
        all_runs = json.load(f)

    for loc in KEYS:
        que_list = all_runs[loc]
        new_quelist = []
        for run in que_list:
            max_epochs = run['training']['max_epoch']
            early_stopping = run.pop('early_stopping', None)
            if early_stopping is None:
                early_stopping = StopperInfo(max_epoch=max_epochs)
            else:
                early_stopping = EarlyStopperInfo(
                    max_epoch=max_epochs,
                    metric=early_stopping['metric'][1],
                    phase=early_stopping['metric'][0],
                    mode=early_stopping['mode'],
                    patience=early_stopping['patience'],
                    min_delta=early_stopping['min_delta']
                )
            print(early_stopping.model_dump())
            
            run['stopping'] = early_stopping.model_dump()

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
    
def rem_updated(name:str) -> str:
    return name.replace('_updated', '')
        
def update_runs13():
    
    q = Que()
    
    key_set = ['admin', 'config_path']
    q.update_runs(key_set, rem_updated)
    q.save_state('/home/luke/Code/SLR/src/que/Runs_no_updated.json')
    

def update_runs14():
    from src.stopping import StopperInfo, EarlyStopperInfo

    with open("/home/luke/Code/SLR/src/que/Runs_old.json", "r") as f:
        all_runs = json.load(f)

    for loc in KEYS:
        que_list = all_runs[loc]
        new_quelist = []
        for run in que_list:
            
            run['admin']['config_path'] = rem_updated(run['admin']['config_path'])
            
            max_epochs = run['training']['max_epoch']
            early_stopping = run.pop('early_stopping', None)
            if early_stopping is None:
                early_stopping = StopperInfo(max_epoch=max_epochs)
            else:
                early_stopping = EarlyStopperInfo(
                    max_epoch=max_epochs,
                    metric=early_stopping['metric'][1],
                    phase=early_stopping['metric'][0],
                    mode=early_stopping['mode'],
                    patience=early_stopping['patience'],
                    min_delta=early_stopping['min_delta']
                )
            print(early_stopping.model_dump())
            
            run['stopping'] = early_stopping.model_dump()

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

def test_read_server_state():
    from src.que.core import read_server_state
    
    s = read_server_state()
    print(s.model_dump())

if __name__ == "__main__":
    # update_runs14()
    # test_read_server_state()
    pass
    
    
    
    
