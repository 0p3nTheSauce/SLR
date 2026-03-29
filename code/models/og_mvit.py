from typing import Optional, Union
from pathlib import Path

import torch

from .mvit.slowfast.models.video_model_builder import MViT
from .mvit.slowfast.utils.parser import load_model_config
import torch.nn as nn


CONF_PATH = Path('/home/luke/Code/SLR/code/models/mvit/configs/MVITv2_S_16x4.yaml')
WEIGHTS_PATH = Path('/home/luke/Code/SLR/code/models/mvit/weights/MViTv2_S_16x4_k400_f302660347.pyth')

class MVITv2_S_16x4(nn.Module):
    def __init__(self, num_classes: int=100, drop_p:float=0.5, weights_path:Optional[Union[str, Path]]=None, cfg_path: Union[str, Path] = CONF_PATH):
        super().__init__()
        cfg_path = Path(cfg_path)
        assert cfg_path.is_file(), f"Config file {cfg_path} does not exist."
        cfg = load_model_config(path_to_config=str(cfg_path))
        self.model = MViT(cfg)
        
        if weights_path is not None:
            weights_path = Path(weights_path)
            state_dict = torch.load(weights_path, map_location='cpu')['model_state']
            self.model.load_state_dict(state_dict)
            
if __name__ == "__main__":
    model = MVITv2_S_16x4(num_classes=100, drop_p=0.5, weights_path=WEIGHTS_PATH, cfg_path=CONF_PATH)
    print(model)