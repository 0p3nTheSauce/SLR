from typing import Optional
from pathlib import Path
import torch
from .mvit.slowfast.models.video_model_builder import MViT
from .mvit.slowfast.utils.parser import load_model_config

# from .mvit.slowfast.models import head_helper
import torch.nn as nn
import torch.nn.functional as F

base_conf_path = Path(__file__).parent / "mvit" / "configs"
base_weights_path = Path(__file__).parent / "mvit" / "weights"


# 3D MViT configs and weights
CONF_PATH_16x4 = base_conf_path / "MVITv2_S_16x4.yaml"
CONF_PATH_32x3 = base_conf_path / "MVITv2_B_32x3.yaml"
WEIGHTS_PATH_16x4 = base_weights_path / "MViTv2_S_16x4_k400_f302660347.pyth"
WEIGHTS_PATH_32x3 = base_weights_path / "MViTv2_B_32x3_k400_f304025456.pyth"

# #Image classification configs and weights
# CONF_PATH_2D_T = base_conf_path / 'MVITv2_T_2D.yaml'
# WEIGHTS_PATH_2D_T = base_weights_path / 'MViTv2_T_2D_IN1K_ic.pyth'

"""I am aware this is a bit of a ad hco solution"""


class MVITv2_basic(MViT):
    def __init__(
        self,
        num_classes: int,
        pretrain_path: Path,
        cfg_path: Path,
        drop_p: Optional[float] = None,
    ):
        cfg = load_model_config(str(cfg_path))
        super().__init__(cfg)

        if pretrain_path is not None:
            state_dict = torch.load(
                pretrain_path, map_location="cpu", weights_only=True
            )["model_state"]
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
    def __init__(
        self,
        num_classes: int,
        drop_p: Optional[float] = None,
        pretrain_path: Path = WEIGHTS_PATH_16x4,
        cfg_path: Path = CONF_PATH_16x4,
    ):
        super().__init__(num_classes, pretrain_path, cfg_path, drop_p)


class MVITv2_S_16x4_extended(MVITv2_basic):
    def __init__(
        self,
        num_classes: int,
        drop_p: Optional[float] = None,
        pretrain_path: Path = WEIGHTS_PATH_16x4,
        cfg_path: Path = CONF_PATH_16x4,
        num_frames: int = 32,
    ):
        """SLowFast MViTv2-S_16x4 at 32 frames"""
        super().__init__(num_classes, pretrain_path, cfg_path, drop_p)
        self._adapt_pos_encoding(num_frames=num_frames)

    def _adapt_pos_encoding(self, num_frames: int):
        temporal_stride = self.patch_stride[0]
        new_T = num_frames // temporal_stride
        assert new_T >= 1, (
            f"num_frames={num_frames} too small for temporal_stride={temporal_stride}"
        )
        new_len = 2 * new_T - 1

        for module in self.blocks.modules():
            if not (hasattr(module, "rel_pos_t") and module.rel_pos_t is not None):
                continue
            old_rpt = module.rel_pos_t.data
            if old_rpt.shape[0] == new_len:
                continue
            rpt = old_rpt.permute(1, 0).unsqueeze(0)
            rpt = F.interpolate(rpt, size=new_len, mode="linear", align_corners=False)
            rpt = rpt.squeeze(0).permute(1, 0)
            module.rel_pos_t = nn.Parameter(rpt)

        self.T = new_T


class MVITv2_B_32x3_basic(MVITv2_basic):
    def __init__(
        self,
        num_classes: int,
        drop_p: Optional[float] = None,
        pretrain_path: Path = WEIGHTS_PATH_32x3,
        cfg_path: Path = CONF_PATH_32x3,
    ):
        super().__init__(num_classes, pretrain_path, cfg_path, drop_p)


# class MVITv2_T_2D_basic(MVITv2_basic):
#     def __init__(self, num_classes: int, drop_p: Optional[float] = None, pretrain_path: Path = WEIGHTS_PATH_2D_T, cfg_path: Path = CONF_PATH_2D_T):
#         super().__init__(num_classes, pretrain_path, cfg_path, drop_p)


class MVITv2_B_32x3_reduced(MVITv2_basic):
    def __init__(
        self,
        num_classes: int,
        drop_p: Optional[float] = None,
        pretrain_path: Path = WEIGHTS_PATH_32x3,
        cfg_path: Path = CONF_PATH_32x3,
        num_frames: int = 16,
    ):
        """SLowFast MViTv2-B_132xc at 16 frames"""
        super().__init__(num_classes, pretrain_path, cfg_path, drop_p)
        self._adapt_pos_encoding(num_frames=num_frames)

    def _adapt_pos_encoding(self, num_frames: int):
        temporal_stride = self.patch_stride[0]
        new_T = num_frames // temporal_stride
        assert new_T >= 1, (
            f"num_frames={num_frames} too small for temporal_stride={temporal_stride}"
        )
        new_len = 2 * new_T - 1

        for module in self.blocks.modules():
            if not (hasattr(module, "rel_pos_t") and module.rel_pos_t is not None):
                continue
            old_rpt = module.rel_pos_t.data
            if old_rpt.shape[0] == new_len:
                continue
            rpt = old_rpt.permute(1, 0).unsqueeze(0)
            rpt = F.interpolate(rpt, size=new_len, mode="linear", align_corners=False)
            rpt = rpt.squeeze(0).permute(1, 0)
            module.rel_pos_t = nn.Parameter(rpt)

        self.T = new_T


def get_num_parameters(model: nn.Module) -> int:
    """Get total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print(Path(__file__))
    # model = MVITv2_T_2D_basic(num_classes=100)
    # print(f"Number of parameters: {get_num_parameters(model)}")
