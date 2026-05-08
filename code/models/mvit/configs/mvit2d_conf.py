from functools import partial
import torch.nn as nn
from pydantic import BaseModel
    
    
class MViT_Conf(BaseModel):
    type: str = '2d'
    
class MViT_t_3x(MViT_Conf):
    embed_dim: int=96
    depth: int=10
    num_heads: int=1
    last_block_indexes: tuple[int, int, int, int]=(0, 2, 7, 9)
    residual_pooling: bool=True
    drop_path_rate: float=0.2
    norm_layer=partial(nn.LayerNorm, eps=1e-6)
    out_features: list[str]=["scale2", "scale3", "scale4", "scale5"]
    pretrain_path: str = '/home/luke/Code/SLR/code/models/mvit/weights/MViTV2-T_IN1K.pkl'
    
class MViT_s_3x(MViT_t_3x):
    depth: int = 16
    last_block_indexes: tuple[int, int, int, int] = (0, 2, 13, 15)
    weight_path: str = '/home/luke/Code/SLR/code/models/mvit/weights/MViTV2-S_IN1K.pkl'
    
class MViT_b_3x(MViT_t_3x):
    depth: int = 24
    last_block_indexes: tuple[int, int, int, int] = (1, 4, 20, 23)
    drop_path_rate: float = 0.4
    weight_path: str = '/home/luke/Code/SLR/code/models/mvit/weights/MViTV2-B_IN1K.pkl'