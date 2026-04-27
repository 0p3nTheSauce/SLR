from typing import Dict, Any, List, Tuple, Callable, cast
try:
    import tomllib  # type: ignore
except ImportError:
    import tomli as tomllib

from pathlib import Path
from argparse import ArgumentParser
import json
import inspect
# locals
from que.core import Que, _get_basic_logger
from run_types import GenInfo, ResSet, RunRes, CompExpInfo
from results.saicair.saicair import key_set, criterions, frames, out_suffix

FILE_TYPE = ".toml"
logger = _get_basic_logger()


def load_config(result_dir: str) -> GenInfo:
    """Load config from .toml file and merge with AdminInfo from CLI.
    Supports backward compatibility with .ini files, using old loading mechanism.

    Args:
        admin: Parsed admin info from command line arguments.

    Returns:
        RunInfo: Fully validated config model for the run.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        pydantic.ValidationError: If config values fail validation.
    """
    conf_path = Path(result_dir)
    if not conf_path.exists():
        raise FileNotFoundError(f"{conf_path} not found")

    with open(conf_path, "rb") as f:
        raw = tomllib.load(f)

    return GenInfo.model_validate(raw)


def make_keys_and_criterions(specifications: Dict[str, Any]) -> Tuple[List[List[str]], List[Callable[[Any], bool]]]:
    key_set = []
    criterions = []
    spec = specifications
    
    for key in spec:
        
        d = spec[key]
        if isinstance(d, Dict):
            sub_keys, sub_crits = make_keys_and_criterions(d)
        
            for sub_key, sub_crit in zip(sub_keys, sub_crits):
                key_set.append([key] + sub_key)
                criterions.append(sub_crit)
        else:
            key_chain = []
            key_chain.append(key)
            
            key_set.append(key_chain)
            criterions.append(lambda x: x == d)
        
        
    return key_set, criterions
            
                
def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=4))



def main():
    parser = ArgumentParser()
    parser.add_argument("--result_dir", "-d", help="Path to result directory", type=str)
    
    parser.add_argument("--out_path", "-o", help=f"Path to output json file, default is same as config path with {out_suffix}", type=str, default=None)
    parser.add_argument("--num_frames", "-f", help="Number of frames to filter by, default is 16", type=int, default=frames)

    args = parser.parse_args()

    resdir = Path(args.result_dir)
    if not resdir.exists():
        raise FileNotFoundError(f"{resdir} not found")
    
    if args.out_path is not None:
        out_path = Path(args.out_path)
    else:
        out_path = resdir / f"{out_suffix}"
    print(f"Output path: {out_path}")
    # print_json(load_config(args.result_dir).model_dump())
    # key_set, crits = make_keys_and_criterions(
    #         load_config(args.result_dir).model_dump()
    #     )
    
    run_set = []
    
    # print('[')
    # for key_chain in key_set:
    #     print(f'    {key_chain},')
    # print(']')

    q = Que(logger)
    _, runs = q.find_loc_runs(
        loc='old_runs',
        key_set=key_set,
        criterions=criterions  
    )
    
    for run in runs:
        run = cast(CompExpInfo, run)
        run_set.append(RunRes(admin=run.admin, results=run.results))


    data_override = runs[0].data
    if data_override.train_augs is not None:
        data_override.train_augs.norm_dict = None
    if data_override.test_augs is not None:
        data_override.test_augs.norm_dict = None


    res_set = ResSet(
        training=runs[0].training,
        optimizer=runs[0].optimizer,
        model_params=runs[0].model_params,
        data=runs[0].data,
        scheduler=runs[0].scheduler,
        results=run_set
    )
    
    # with open(out_path, "w") as f:
    #     json.dump([run.model_dump() for run in runs], f, indent=4)

    print(len(run_set))

    with open(out_path, "w") as f:
        json.dump(res_set.model_dump(), f, indent=4)

if __name__ == "__main__":
    main()
