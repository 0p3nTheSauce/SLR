from typing import Optional
from pathlib import Path
import torch
from .mvit.slowfast.models.video_model_builder import MViT
from .mvit.slowfast.utils.parser import load_model_config
from .mvit.slowfast.models import head_helper
import torch.nn as nn

CONF_PATH_16x4 = Path(__file__).parent.parent / 'configs' / 'MVITv2_S_16x4.yaml'
CONF_PATH_32x3 = Path(__file__).parent.parent / 'configs' / 'MVITv2_B_32x3.yaml'
WEIGHTS_PATH_16x4 = Path(__file__).parent.parent / 'weights' / 'MViTv2_S_16x4_k400_f302660347.pyth'
WEIGHTS_PATH_32x3 = Path(__file__).parent.parent / 'weights' / 'MViTv2_B_32x3_k400_f304025456.pyth'

class MVITv2_basic(MViT):
    def __init__(self, num_classes: int, pretrain_path: Path, cfg_path: Path, drop_p: Optional[float] = None):
        cfg = load_model_config(str(cfg_path))
        super().__init__(cfg)

        if pretrain_path is not None:
            state_dict = torch.load(pretrain_path, map_location='cpu', weights_only=True)['model_state']
            self.load_state_dict(state_dict)

        drop_prob = drop_p if drop_p is not None else cfg.MODEL.DROPOUT_RATE

        embed_dim = self.head.projection.in_features
        self.head = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(embed_dim, num_classes),
        )
        self.head.apply(self._init_weights)
        self.head[1].weight.data.mul_(self.cfg.MVIT.HEAD_INIT_SCALE)
        self.head[1].bias.data.mul_(self.cfg.MVIT.HEAD_INIT_SCALE)

        # Parameter group handles — backbone deduplicates by tensor identity
        self.backbone = nn.ModuleList([self.patch_embed, self.blocks, self.norm])
        self.classifier = self.head

    def forward(self, x, bboxes=None, return_attn=False):
        return super().forward([x], bboxes, return_attn)

class MVITv2_S_16x4_basic(MVITv2_basic):
    def __init__(self, num_classes: int, drop_p: Optional[float] = None, pretrain_path: Path = WEIGHTS_PATH_16x4, cfg_path: Path = CONF_PATH_16x4):
        super().__init__(num_classes, pretrain_path, cfg_path, drop_p)
        
class MVITv2_B_32x3_basic(MVITv2_basic):
    def __init__(self, num_classes: int, drop_p: Optional[float] = None, pretrain_path: Path = WEIGHTS_PATH_32x3, cfg_path: Path = CONF_PATH_32x3):
        super().__init__(num_classes, pretrain_path, cfg_path, drop_p)

def get_num_parameters(model: nn.Module) -> int:
    """Get total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    model = MVITv2_S_16x4_basic(num_classes=100, drop_p=0.0, pretrain_path=WEIGHTS_PATH_16x4, cfg_path=CONF_PATH_16x4)
    # print(model)
    # model = MVITv2_B_32x3_basic(num_classes=100, drop_p=0.0, pretrain_path=WEIGHTS_PATH_32x3, cfg_path=CONF_PATH_32x3)
    print(get_num_parameters(model))