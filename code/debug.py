from typing import Dict, List, Any
from que.core import Que, QueEmpty, connect_manager, _get_basic_logger
import json
import subprocess
import sys
import logging
import torch
import video_dataset
import preprocess
import time
logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	)
logger = logging.getLogger(__name__)

def test_dump_peak():
    #works fine
    q = Que(logger)
    run = q.peak_run('cur_run', 0)
    print(json.dumps(run, indent=4))
    
def test_dump_peak_server():
    #works fine with non underscore method
    man = connect_manager()
    q = man.get_que()
    run = q.peak_run('to_run', 0)
    print(json.dumps(run, indent=4))
    
def test_subprocess():
    proc = subprocess.Popen(
        ['python', 'extable.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = proc.communicate()
    print("STDOUT:")
    print(stdout)
    print("STDERR:")
    print(stderr)
    
def test_subrocess2():
    proc = subprocess.Popen(
        [sys.executable, '-u', '-m', 'que.daemon'],
        stdout=open('./que/Daemon.log', 'a'),
        stderr=subprocess.STDOUT,
        bufsize=0
    )
    return_code = proc.wait()
    
def test_ex():
    try:
        raise QueEmpty('cur_run')
    except Exception as e:
        print(e)

def test_clear():
    man = connect_manager()
    q = man.get_que()
    try:
        q.clear_runs('cur_run')
    except Exception as e:
        print(e)
        
def test_clear2():
    q = Que(logger)
    try:
        q.clear_runs('cur_run')
    except Exception as e:
        print(e)
        


def test_instance_typegaurd():
    valid_dict = preprocess.InstanceDict(
        bbox=[0, 0, 100, 100],
        frame_end=10,
        frame_start=0,
        instance_id=0,
        signer_id=0,
        source='video1',
        split='train',
        url='path/to/video1',
        variation_id=0,
        video_id='video1',
        label_name='label1',
        label_num=0,
    )
    invalid_dict = {
        'video_id': 'video1',
        'frame_start': 0,
        'frame_end': 10,
        'label_name': 'label1',
        'label_num': 0,
        # 'bbox' key is missing
    }
    print(video_dataset.is_instance_dict(valid_dict))  # Should print True
    print(video_dataset.is_instance_dict(invalid_dict))  # Should print False


def time_instance_typegaurd():
    
    json_path = '/home/luke/Code/SLR/code/preprocessed/labels/asl2000/train_instances_fixed_frange_bboxes_len.json'

    #time how long it takes to load the json and check each dict with the type guard
    start_time = time.time()
    data = video_dataset.load_data_from_json(json_path, policy='strict')
    end_time = time.time()
    print(f"Time taken to load and check all items: {end_time - start_time} seconds")
    # Time taken to load and check all items: 0.019509077072143555 seconds

    #Conclusion: The type guard check is very fast, and does not add significant overhead to loading the data. 


def test_find_runs():
    logger = _get_basic_logger()
    
    
    q = Que(logger)
    key_set = [
        ['admin', 'exp_no'],
        ['admin', 'model'],
        ['training', "batch_size_equivalent"],
        ["optimizer", "eps"],
        ["optimizer", "backbone_init_lr"],
        ["optimizer", "backbone_weight_decay"],
        ["optimizer", "classifier_init_lr"],
        ["optimizer", "classifier_weight_decay"],
        ["model_params", "drop_p"],
        ["data", "num_frames"],
        ["data", "frame_size"],
        ["scheduler", "type"],
        ["scheduler", "t0"],
        ["scheduler", "tmult"],
        ["scheduler", "eta_min"],
    ]
    criterions = [
        lambda x: (x != '007') and (x != '046') ,
        lambda x: x == 'S3D' or x == 'MViTv2_S',
        lambda x: x == 8,
        lambda x: x == 1e-05,
        lambda x: x == 0.0001,
        lambda x: x == 0.001,
        lambda x: x == 0.001,
        lambda x: x == 0.001,
        lambda x: x == 0.5,
        lambda x: x == 16,
        lambda x: x == 224,
        lambda x: x == "CosineAnnealingWarmRestarts",
        lambda x: x == 10,
        lambda x: x == 1,
        lambda x: x == 0,
    ]
    _, runs = q.find_runs('old_runs', key_set, criterions)
    print(len(runs))
    all_exps = {'asl100':{}, 'asl300':{}, 'asl1000':{}, 'asl2000':{}}
    for run in runs:
        split_name = run['admin']['split'] 
        model_name = run['admin']['model']
        exp_no = run['admin']['exp_no']
        cur_split = all_exps[split_name]
        if model_name in cur_split:
            all_exps[split_name][model_name].append(exp_no)
        else:
            all_exps[split_name][model_name] = [exp_no]
      
    with open('./results/wlasl_saicist.json', 'w') as f:
        json.dump(all_exps,f,indent=2)
        

def test_find_S3D_runs():
    logger = _get_basic_logger()
    
    
    q = Que(logger)
    key_set = [
        ['admin', 'model'],
        ['training', "batch_size_equivalent"],
        ["optimizer", "eps"],
        ["optimizer", "backbone_init_lr"],
        ["optimizer", "backbone_weight_decay"],
        ["optimizer", "classifier_init_lr"],
        ["optimizer", "classifier_weight_decay"],
        ["model_params", "drop_p"],
        ["data", "num_frames"],
        ["data", "frame_size"],
    #     ["scheduler", "type"],
    #     ["scheduler", "t0"],
    #     ["scheduler", "tmult"],
    #     ["scheduler", "eta_min"],
    ]
    criterions = [
        lambda x: x == 'S3D',
        lambda x: x == 8,
        lambda x: x == 1e-05,
        lambda x: x == 0.0001,
        lambda x: x == 0.001,
        lambda x: x == 0.001,
        lambda x: x == 0.001,
        lambda x: x == 0.5,
        lambda x: x == 32,
        lambda x: x == 224,
        # lambda x: x == "CosineAnnealingWarmRestarts",
        # lambda x: x == 10,
        # lambda x: x == 1,
        # lambda x: x == 0,
    ]
    idxs, runs = q.find_runs('old_runs', key_set, criterions)
    print(len(runs))
    all_exps = {'asl100':{}, 'asl300':{}, 'asl1000':{}, 'asl2000':{}}
    for run in runs:
        all_exps[run['admin']['split']][run['admin']['model']] = [run['admin']['exp_no']]
    
    
    with open('./S3D_32_SAICIST.json', 'w') as f:
        json.dump(all_exps,f,indent=2)
    print(json.dumps(all_exps, indent=4))
    print(idxs)

class CleverDict(Dict):
    def __init__(self, dict: Dict[Any, Any]):
        self.dict = dict
        
    def __getitem__(self, keys: List[Any]) -> Any:
        d = self.dict.copy()
        for key in keys:
            d = d[key]
        return d
    
    def __setitem__(self, keys: List[Any], val: Any):
        self.dict = self._set_inplace(self.dict, keys[0], keys[1:], val)

    def _set_inplace(self, d:Dict[Any, Any], k:Any,ks:List[Any], val:Any) -> Dict[Any, Any]:
        if len(ks) == 0:
            d[k] = val
            return d
        else:
            next_key = ks.pop(0)
            d[k] = self._set_inplace(d[k],next_key, ks, val)
            return d          
        
    def to_dict(self) -> Dict[Any, Any]:
        return self.dict.copy()
    
    
    def __str__(self) -> str:
        return str(self.dict)
    
    def __delitem__(self, key):
        raise NotImplementedError
        

def test_set_nested():
    a = {a:b for a, b in zip('abcdef', range(5))}
    b = a.copy()
    c = a.copy()
    d = {
        'z':a,
        'y':b,
        'x':c
    }
    print(json.dumps(d, indent=4))    
    cd = CleverDict(d)
    ks = ['z', 'a']
    v = 200
    cd[ks] = v
    ks = ['x', 'e']
    v = 300
    ks = ['x', 'e']
    print(json.dumps(cd.to_dict(), indent=4)) 
    print(cd) 

def test_extend_classifier():
    from models import extend_classifier, get_model

    model = get_model('MViTv2_S', 100, 0.5)
    print(model.classifier)
    checkpoint = torch.load('runs/asl100/MViTv2_S_exp011/checkpoints/best.pth')
    model.load_state_dict(checkpoint)
    model = extend_classifier(model, 300)
    print(model.classifier)
    

if __name__ == '__main__':
    # test_dump_peak()
    # test_dump_peak_server()
    # test_subprocess()
    # test_subrocess2()
    # test_ex()
    # test_clear()
    # test_instance_typegaurd()
    # time_instance_typegaurd()
    # test_find_runs()
    test_find_S3D_runs()
    # test_set_nested()
    # test_extend_classifier()
    